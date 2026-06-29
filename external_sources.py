"""
external_sources.py
===================
DART 외 외부 소스 수집기.

  🟢 fetch_naver_news()        : 네이버 공식 뉴스 검색 API (합법, 권장)
  🟡 fetch_naver_research()    : 네이버증권 리포트 '목록 메타데이터' (제목/증권사/
                                목표주가/투자의견/날짜/링크). 본문 PDF는 받지 않음.
  🟢 aggregate_consensus()     : 위 리포트들의 목표주가·투자의견을 직접 집계한
                                'self-built 컨센서스'. (FnGuide 컨센서스 상품의 대체)

  🔴 fetch_fnguide_consensus() : comp.fnguide.com 스크래핑은 ToS 위반(데이터베이스화
                                금지, 유료 라이선스 데이터)이라 구현하지 않는다.
                                정식 경로는 FnSpace 라이선스 API.

안정성 설계: 모든 네트워크 호출에 타임아웃 + 지수 백오프 재시도 + 디스크 캐시.
호출자(노드)는 예외를 잡아 state['errors']/sources_status로 흘려보낸다.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import hashlib
import logging
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("external_sources")

CACHE_DIR = Path(os.getenv("SRC_CACHE_DIR", Path.home() / ".cache" / "dart_mvp" / "ext"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (compatible; personal-research/1.0)"


# ---------------------------------------------------------------------- #
# 공통: 재시도 + 캐시
# ---------------------------------------------------------------------- #
class SourceError(RuntimeError):
    pass


def _request(method: str, url: str, *, headers=None, params=None,
             timeout: int = 12, retries: int = 3, backoff: float = 1.5) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.request(method, url, headers=headers, params=params, timeout=timeout)
            # 429/5xx는 재시도 대상
            if r.status_code in (429, 500, 502, 503, 504):
                raise SourceError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.RequestException, SourceError) as e:
            last = e
            sleep = backoff ** attempt
            logger.warning("요청 실패(%s) 재시도 %d/%d, %.1fs 대기: %s",
                           url, attempt + 1, retries, sleep, e)
            time.sleep(sleep)
    raise SourceError(f"요청 최종 실패: {url} ({last})")


def _cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def _cache_get(key: str, ttl: int) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _cache_put(key: str, value: dict):
    _cache_path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _strip_html(s: str) -> str:
    # 먼저 엔티티 해제 후 태그 제거 (실제 태그/이스케이프된 태그 모두 처리)
    return re.sub(r"<[^>]+>", "", html.unescape(s or "")).strip()


# ---------------------------------------------------------------------- #
# 🟢 네이버 뉴스 검색 API
# ---------------------------------------------------------------------- #
@dataclass
class NewsItem:
    title: str
    link: str
    pub_date: str
    summary: str


def fetch_naver_news(query: str, *, display: int = 20, sort: str = "date",
                     client_id: Optional[str] = None,
                     client_secret: Optional[str] = None,
                     cache_ttl: int = 3600) -> list[dict]:
    """공식 API. display<=100, sort in {sim,date}. 결과는 dict 리스트."""
    cid = client_id or os.getenv("NAVER_CLIENT_ID")
    csec = client_secret or os.getenv("NAVER_CLIENT_SECRET")
    if not cid or not csec:
        raise SourceError("NAVER_CLIENT_ID/SECRET 미설정 (developers.naver.com 발급)")

    ck = f"news::{query}::{display}::{sort}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached["items"]

    r = _request(
        "GET", "https://openapi.naver.com/v1/search/news.json",
        headers={"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec, "User-Agent": UA},
        params={"query": query, "display": min(display, 100), "sort": sort},
    )
    items = []
    for it in r.json().get("items", []):
        items.append(asdict(NewsItem(
            title=_strip_html(it.get("title", "")),
            link=it.get("originallink") or it.get("link", ""),
            pub_date=it.get("pubDate", ""),
            summary=_strip_html(it.get("description", "")),
        )))
    _cache_put(ck, {"items": items})
    return items


# ---------------------------------------------------------------------- #
# 🟡 네이버증권 리포트 목록 (메타데이터만)
# ---------------------------------------------------------------------- #
@dataclass
class Report:
    title: str
    broker: str = ""
    date: str = ""
    target_price: Optional[int] = None
    opinion: str = ""
    report_url: str = ""
    pdf_url: str = ""


def fetch_naver_research(stock_code: str, *, cache_ttl: int = 21600) -> list[dict]:
    """finance.naver.com 종목 리서치 목록을 긁어 메타데이터만 추출(본문 PDF 미수집).

    주의: 네이버 페이지 구조/컬럼은 수시로 바뀐다. 아래 파서는 방어적으로
    '있는 만큼' 뽑으며, 컬럼 매칭이 어긋나면 비는 값이 생긴다.
    => 이 어긋남을 잡아내는 게 바로 '파이프라인 안정성 검증' 단계의 목적.
    개인 리서치 용도. 저작권 있는 리포트 PDF 본문은 재생산/저장하지 않는다.
    """
    code = stock_code.zfill(6)
    ck = f"research::{code}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached["reports"]

    url = ("https://finance.naver.com/research/company_list.naver"
           f"?searchType=itemCode&itemCode={code}")
    r = _request("GET", url, headers={"User-Agent": UA})
    # 네이버 금융은 EUC-KR(cp949)
    r.encoding = "euc-kr"
    soup_html = r.text

    reports = _parse_research_rows(soup_html)
    _cache_put(ck, {"reports": [asdict(x) if isinstance(x, Report) else x for x in reports]})
    return [asdict(x) if isinstance(x, Report) else x for x in reports]


def _parse_research_rows(page: str) -> list[Report]:
    """리서치 목록 테이블 파싱. BeautifulSoup이 있으면 쓰고, 없으면 정규식 폴백."""
    out: list[Report] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page, "html.parser")
        # 종목분석 리포트 목록은 table.type_1 형태
        table = soup.select_one("table.type_1") or soup.find("table")
        if not table:
            return out
        for tr in table.select("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            cells = [td.get_text(strip=True) for td in tds]
            links = {a.get_text(strip=True): a.get("href", "") for a in tr.find_all("a")}
            # 제목/증권사/날짜/PDF를 위치+휴리스틱으로 추출
            title = next((t for t in links if t), cells[0] if cells else "")
            report_url = links.get(title, "")
            pdf_url = next((h for h in links.values() if h.lower().endswith(".pdf")), "")
            # 날짜처럼 보이는 셀
            date = next((c for c in cells if re.match(r"\d{2,4}[.\-/]\d{1,2}[.\-/]\d{1,2}", c)), "")
            # 증권사: '증권'/'투자'/'리서치' 포함 셀 우선, 없으면 휴리스틱
            broker = next((c for c in cells if ("증권" in c or "투자" in c or "리서치" in c)
                           and len(c) <= 12), "")
            if not broker:
                broker = next((c for c in cells
                               if c and c != title and c != date
                               and not c.replace(",", "").isdigit() and len(c) <= 12), "")
            # 목표주가처럼 보이는 셀(콤마 포함 4자리 이상 숫자)
            tp = None
            for c in cells:
                m = re.fullmatch(r"[\d,]{4,}", c)
                if m:
                    tp = int(c.replace(",", ""))
                    break
            if title:
                out.append(Report(title=title, broker=broker, date=date,
                                  target_price=tp, report_url=report_url, pdf_url=pdf_url))
    except ImportError:
        # bs4 미설치 폴백: 링크/날짜만 거칠게
        for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', page):
            href, text = m.group(1), _strip_html(m.group(2))
            if text and len(text) > 6:
                out.append(Report(title=text, report_url=href))
    return out


# ---------------------------------------------------------------------- #
# 🟢 self-built 컨센서스 (FnGuide 컨센서스 상품의 합법 대체)
# ---------------------------------------------------------------------- #
def aggregate_consensus(reports: list[dict]) -> dict:
    """수집한 리포트들의 목표주가/투자의견을 직접 집계.
    개별 증권사가 공개한 의견을 출처와 함께 모은 것 — FnGuide 집계상품 복제 아님."""
    tps = [r["target_price"] for r in reports if r.get("target_price")]
    opinions = [r.get("opinion", "") for r in reports if r.get("opinion")]
    brokers = sorted({r.get("broker", "") for r in reports if r.get("broker")})
    result = {
        "n_reports": len(reports),
        "n_target_prices": len(tps),
        "brokers": brokers,
    }
    if tps:
        result.update({
            "target_price_mean": round(statistics.mean(tps)),
            "target_price_median": round(statistics.median(tps)),
            "target_price_min": min(tps),
            "target_price_max": max(tps),
        })
    if opinions:
        dist: dict = {}
        for o in opinions:
            dist[o] = dist.get(o, 0) + 1
        result["opinion_distribution"] = dist
    return result


# ---------------------------------------------------------------------- #
# comp.fnguide.com 예상실적(컨센서스 추정) — 사용자 지시로 구현
#   주의: FnGuide ToS는 무단 사용/데이터베이스화를 제한한다. 법적 리스크는
#   사용자 책임 하에, 호출 최소화(긴 캐시·종목당 1회·UA·rate limit)로 운용.
#   값 단위는 '억원'(Financial Highlight 기준) — DART(원)와 섞지 말 것.
# ---------------------------------------------------------------------- #
def _to_num(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "N/A", "nan"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("%", "")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def fetch_fnguide_estimates(stock_code: str, *, cache_ttl: int = 86400) -> dict:
    """Snapshot의 Financial Highlight에서 연간 (E)=추정 컬럼의
    매출액/영업이익/당기순이익을 추출. 단위: 억원."""
    code = stock_code.zfill(6)
    ck = f"fnguide_est::{code}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached

    try:
        import pandas as pd
    except ImportError:
        raise SourceError("pandas/lxml 필요: pip install pandas lxml")

    url = ("https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp"
           f"?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=11&stkGb=701")
    r = _request("GET", url, headers={"User-Agent": UA})
    html_text = r.text

    import io
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except Exception as e:  # noqa: BLE001  read_html 내부 ValueError/IndexError 등 흡수
        raise SourceError(f"표 파싱 실패(종목/페이지 구조 확인): {type(e).__name__}")
    target_rows = ("매출액", "영업이익", "당기순이익")

    fin_tbl = None
    for t in tables:
        # 빈/비정형 테이블 방어 (작은 종목 페이지에 자주 섞임)
        if t is None or t.shape[1] == 0 or len(t) == 0:
            continue
        try:
            first_col = t.iloc[:, 0].astype(str).str.replace(" ", "")
        except (IndexError, KeyError):
            continue
        if first_col.str.contains("매출액").any() and first_col.str.contains("영업이익").any():
            fin_tbl = t
            break
    if fin_tbl is None:
        raise SourceError("Financial Highlight 표를 찾지 못했습니다(페이지 구조 변경 가능).")

    # 컬럼 라벨 평탄화
    cols = ["/".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
            for col in fin_tbl.columns]
    est_cols = [i for i, c in enumerate(cols) if "(E)" in c.replace(" ", "")]

    annual: dict = {}
    estimates: dict = {}
    name_col = fin_tbl.columns[0]
    for _, row in fin_tbl.iterrows():
        acct = str(row[name_col]).replace(" ", "")
        acct = next((a for a in target_rows if a in acct), None)
        if not acct:
            continue
        for i, c in enumerate(cols):
            if i == 0:
                continue
            val = _to_num(row.iloc[i])
            if val is None:
                continue
            annual.setdefault(c, {})[acct] = val
            if i in est_cols:
                estimates.setdefault(c, {})[acct] = val

    result = {
        "source": "comp.fnguide.com",
        "unit": "억원",
        "fetched_at": time.strftime("%Y-%m-%d"),
        "annual": annual,
        "estimates": estimates,
    }
    _cache_put(ck, result)
    return result
