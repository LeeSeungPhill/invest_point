"""
dart_client.py
==============
OpenDART 원본 공시(사업/반기/분기보고서)에서 "II. 사업의 내용" 본문을 추출하고,
주요 재무계정(매출액/영업이익/당기순이익) 시계열을 가져오는 클라이언트.

OpenDART의 정형 API는 재무 '숫자'는 잘 주지만 사업의 내용 같은 '서술형 텍스트'는
주지 않는다. 그래서 본문은 document.xml(원본 ZIP)을 받아 직접 파싱한다.

필요 환경변수:
    OPENDART_API_KEY   # https://opendart.fss.or.kr 에서 발급

호출 제한: OpenDART는 일 10,000회 제한이 있으므로 corpCode는 디스크 캐시한다.
"""

from __future__ import annotations

import io
import os
import re
import time
import zipfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# .env가 있으면 자동 로드 (python-dotenv 미설치여도 죽지 않게 처리)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("dart_client")

BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = Path(os.getenv("DART_CACHE_DIR", Path.home() / ".cache" / "dart_mvp"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 정기보고서 종류 코드 (reprt_code)
REPRT_CODE = {
    "11011": "사업보고서",
    "11012": "반기보고서",
    "11013": "1분기보고서",
    "11014": "3분기보고서",
}

# DART 본문은 전각 로마숫자(Ⅰ Ⅱ Ⅲ ...)를 쓰는 경우가 많다. 정규화용 매핑.
_ROMAN = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V",
          "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X",
          "Ⅺ": "XI", "Ⅻ": "XII"}


class DartError(RuntimeError):
    pass


@dataclass
class Filing:
    corp_code: str
    corp_name: str
    rcept_no: str          # 접수번호 (원본 다운로드 키)
    report_nm: str         # 보고서명
    rcept_dt: str          # 접수일자 YYYYMMDD


@dataclass
class FinancialSeries:
    # 연도/기간 라벨 -> 금액(원). 최신이 뒤로 가도록 정렬해서 채운다.
    revenue: dict = field(default_factory=dict)        # 매출액
    operating_profit: dict = field(default_factory=dict)  # 영업이익
    net_income: dict = field(default_factory=dict)     # 당기순이익


class DartClient:
    def __init__(self, api_key: Optional[str] = None, request_pause: float = 0.3):
        # DART_API_KEY / OPENDART_API_KEY 둘 다 허용
        self.api_key = (api_key
                        or os.getenv("DART_API_KEY")
                        or os.getenv("OPENDART_API_KEY"))
        if not self.api_key:
            raise DartError("DART_API_KEY(또는 OPENDART_API_KEY)가 설정되지 않았습니다.")
        self.pause = request_pause
        self._corp_map: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # low-level
    # ------------------------------------------------------------------ #
    def _get(self, endpoint: str, **params) -> requests.Response:
        params["crtfc_key"] = self.api_key
        url = f"{BASE}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        time.sleep(self.pause)  # 호출 제한 보호
        if resp.status_code != 200:
            raise DartError(f"{endpoint} HTTP {resp.status_code}")
        return resp

    @staticmethod
    def _check_json_status(data: dict, ctx: str):
        # OpenDART는 정상 '000', 데이터없음 '013' 등 상태코드를 status에 담아준다.
        status = data.get("status")
        if status == "000":
            return
        if status == "013":  # 조회 데이터 없음
            raise DartError(f"{ctx}: 조회된 데이터가 없습니다 (status 013)")
        raise DartError(f"{ctx}: status={status} msg={data.get('message')}")

    # ------------------------------------------------------------------ #
    # 1) 종목코드 -> 고유번호(corp_code)
    # ------------------------------------------------------------------ #
    def _load_corp_map(self) -> dict:
        """corpCode.xml(전체 기업 매핑)을 받아 stock_code -> (corp_code, corp_name) 캐시."""
        if self._corp_map is not None:
            return self._corp_map

        cache_file = CACHE_DIR / "corp_map.tsv"
        # 캐시가 7일 이내면 재사용
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 7 * 86400:
            self._corp_map = {}
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                stock, code, name = line.split("\t")
                self._corp_map[stock] = (code, name)
            logger.info("corp_map 캐시 로드 (%d건)", len(self._corp_map))
            return self._corp_map

        logger.info("corpCode.xml 다운로드 중...")
        resp = self._get("corpCode.xml")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_bytes = zf.read(zf.namelist()[0])
        xml_text = xml_bytes.decode("utf-8")

        mapping: dict = {}
        # <list><corp_code>..</corp_code><corp_name>..</corp_name><stock_code>..</stock_code>..
        for m in re.finditer(r"<list>(.*?)</list>", xml_text, re.S):
            block = m.group(1)
            stock = _tag(block, "stock_code").strip()
            if not stock:           # 비상장은 종목코드가 비어있음 -> 스킵
                continue
            code = _tag(block, "corp_code").strip()
            name = _tag(block, "corp_name").strip()
            mapping[stock] = (code, name)

        cache_file.write_text(
            "\n".join(f"{s}\t{c}\t{n}" for s, (c, n) in mapping.items()),
            encoding="utf-8",
        )
        self._corp_map = mapping
        logger.info("corp_map 생성 (%d건)", len(mapping))
        return mapping

    def resolve(self, stock_code: str) -> tuple[str, str]:
        """'005930' -> ('00126380', '삼성전자')"""
        stock_code = stock_code.zfill(6)
        m = self._load_corp_map()
        if stock_code not in m:
            raise DartError(f"종목코드 {stock_code} 에 해당하는 고유번호를 찾지 못했습니다.")
        return m[stock_code]

    # ------------------------------------------------------------------ #
    # 2) 최신 정기보고서 찾기
    # ------------------------------------------------------------------ #
    def latest_periodic_report(self, corp_code: str, lookback_days: int = 420) -> Filing:
        """pblntf_ty='A'(정기공시) 중 가장 최근 사업/반기/분기 보고서 1건."""
        end = time.strftime("%Y%m%d")
        bgn = time.strftime("%Y%m%d", time.localtime(time.time() - lookback_days * 86400))
        data = self._get(
            "list.json",
            corp_code=corp_code,
            bgn_de=bgn,
            end_de=end,
            pblntf_ty="A",
            page_count=100,
        ).json()
        self._check_json_status(data, "list.json")

        wanted = ("사업보고서", "반기보고서", "분기보고서")
        for item in data.get("list", []):   # 최신순 정렬되어 내려옴
            nm = item.get("report_nm", "")
            if any(w in nm for w in wanted):
                return Filing(
                    corp_code=corp_code,
                    corp_name=item.get("corp_name", ""),
                    rcept_no=item["rcept_no"],
                    report_nm=nm,
                    rcept_dt=item.get("rcept_dt", ""),
                )
        raise DartError("최근 정기보고서를 찾지 못했습니다.")

    # ------------------------------------------------------------------ #
    # 3) 원본 다운로드 + "사업의 내용" 추출  <<< 핵심
    # ------------------------------------------------------------------ #
    def fetch_business_section(self, rcept_no: str, max_chars: int = 40000) -> str:
        """document.xml(원본 ZIP)을 받아 II. 사업의 내용 본문 텍스트를 반환."""
        resp = self._get("document.xml", rcept_no=rcept_no)

        # 응답이 ZIP이 아니라 에러 XML일 수 있다(키오류/없음). 먼저 점검.
        if not resp.content[:2] == b"PK":
            head = resp.content[:300].decode("utf-8", "ignore")
            raise DartError(f"원본이 ZIP이 아닙니다. 응답 일부: {head}")

        # ZIP 안의 XML들을 디코드해 합친다(본문이 여러 파일로 쪼개진 경우 대비).
        raw_docs: list[str] = []
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith((".xml", ".html", ".htm")):
                    continue
                raw = zf.read(name)
                raw_docs.append(_decode(raw))

        if not raw_docs:
            raise DartError("ZIP 내부에 파싱할 문서가 없습니다.")

        # 가장 큰 문서를 본문으로 본다(첨부보다 본문이 보통 가장 큼).
        full = max(raw_docs, key=len)
        text = _xml_to_text(full)
        section = _slice_business_section(text)
        return section[:max_chars]

    # ------------------------------------------------------------------ #
    # 4) 주요 재무계정 시계열
    # ------------------------------------------------------------------ #
    def financial_series(self, corp_code: str, bsns_year: str,
                         reprt_code: str = "11011") -> FinancialSeries:
        """단일회사 주요계정. 당기/전기/전전기 3개년 매출/영업이익/순이익."""
        data = self._get(
            "fnlttSinglAcnt.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
            fs_div="CFS",   # 연결재무제표 우선 (없으면 OFS로 재시도)
        ).json()
        if data.get("status") == "013":
            data = self._get(
                "fnlttSinglAcnt.json", corp_code=corp_code,
                bsns_year=bsns_year, reprt_code=reprt_code, fs_div="OFS",
            ).json()
        self._check_json_status(data, "fnlttSinglAcnt.json")

        fs = FinancialSeries()
        target = {
            "매출액": fs.revenue,
            "영업이익": fs.operating_profit,
            "당기순이익": fs.net_income,
        }
        # 각 계정은 당기/전기/전전기 금액을 함께 제공한다.
        period_cols = [
            ("thstrm_nm", "thstrm_amount"),     # 당기
            ("frmtrm_nm", "frmtrm_amount"),     # 전기
            ("bfefrmtrm_nm", "bfefrmtrm_amount"),  # 전전기
        ]
        for row in data.get("list", []):
            acct = row.get("account_nm", "").strip()
            bucket = target.get(acct)
            if bucket is None:
                continue
            for nm_key, amt_key in period_cols:
                label = (row.get(nm_key) or "").strip()
                amount = _to_int(row.get(amt_key))
                if label and amount is not None:
                    bucket[label] = amount
        return fs


# ---------------------------------------------------------------------- #
# 모듈 레벨 헬퍼 (파싱 로직)
# ---------------------------------------------------------------------- #
def _tag(block: str, tag: str) -> str:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", block, re.S)
    return m.group(1) if m else ""


def _decode(raw: bytes) -> str:
    """DART 원본은 보통 UTF-8, 구 공시는 CP949(EUC-KR)인 경우가 있다."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def _xml_to_text(doc: str) -> str:
    """DART 문서 XML/HTML -> 가독성 있는 평문.
    표는 셀/행 경계를 탭/줄바꿈으로 살려 숫자가 붙어버리는 것을 줄인다."""
    import html

    # 전각 로마숫자 정규화 (Ⅱ -> II) : 섹션 슬라이싱을 단순화
    for k, v in _ROMAN.items():
        doc = doc.replace(k, v)

    # 표 구조를 약간 보존
    doc = re.sub(r"</T[DH]>", "\t", doc, flags=re.I)
    doc = re.sub(r"</TR>", "\n", doc, flags=re.I)
    doc = re.sub(r"<(BR|P|DIV|TITLE|SECTION-\d)[^>]*>", "\n", doc, flags=re.I)

    # 나머지 모든 태그 제거
    doc = re.sub(r"<[^>]+>", "", doc)
    doc = html.unescape(doc)

    # 공백 정리
    doc = re.sub(r"[ \t]+", " ", doc)
    doc = re.sub(r"\n[ \t]*\n+", "\n\n", doc)
    return doc.strip()


def _slice_business_section(text: str) -> str:
    """'사업의 내용' 시작 ~ '재무에 관한 사항' 시작 구간을 잘라낸다.

    목차(맨 앞)에도 같은 문구가 있으므로, 시작/종료 후보쌍 중
    가장 긴 구간(=실제 본문)을 택하는 휴리스틱을 쓴다.
    기업/연도별로 소제목 체계가 제각각이라 본문 전체를 통으로 넘기고
    세부 분류는 상위 LLM 노드에 맡긴다.
    """
    starts = [m.start() for m in re.finditer(r"사업의\s*내용", text)]
    ends = [m.start() for m in re.finditer(r"재무에\s*관한\s*사항", text)]

    if not starts:
        # 헤더를 못 찾으면 빈 문자열 대신 앞부분이라도 반환하지 않고 신호를 준다.
        return ""

    best = ""
    for s in starts:
        later_ends = [e for e in ends if e > s]
        end = min(later_ends) if later_ends else len(text)
        chunk = text[s:end]
        if len(chunk) > len(best):
            best = chunk
    # 끝에 다음 섹션의 로마숫자 머리(예: 'III.')가 묻어 들어오면 제거
    best = re.sub(r"\b(I{1,3}|IV|V|VI{0,3}|IX|XI{0,2})\.?\s*$", "", best.strip())
    return best.strip()


def _to_int(s) -> Optional[int]:
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None
