"""
external_sources.py
===================
DART 외 외부 소스 수집기.

  🟢 fetch_naver_news()        : 네이버 공식 뉴스 검색 API (합법, 권장)
  🟡 fetch_naver_research()    : 네이버 모바일 증권 JSON API로 최근 90일 종목분석
                                리포트 최신 10건(제목/증권사/작성일/투자의견/목표주가/
                                직전 목표주가/본문 요약/PDF 링크). PDF 본문은 받지 않음.
  🟢 aggregate_consensus()     : 위 리포트들의 목표주가/투자의견을 증권사별로 직접 집계.
  🟡 fetch_naver_price()       : 네이버 모바일 증권 JSON API(m.stock.naver.com)로
                                현재가·52주 최고/최저·PER/PBR 조회. HTML 스크래핑이
                                아니라 네이버 앱이 쓰는 API라 페이지 구조 변경에 강함.

  🟡 FnGuide 실적(연간/분기/예상)·제품비중·컨센서스는 comp.fnguide.com /
     navercomp.wisereport.co.kr을 직접 스크래핑하는 fnguide_sources.py로 분리했다
     (개인 리서치 목적 한정, 상업적 재배포·DB화 금지, 캐시로 호출 최소화).
     예전에는 ToS 우려로 이 파일에서 구현을 보류했으나, 옆 프로젝트
     simul_server.py에서 이미 같은 방식으로 운용 중인 것과 동일한 개인 사용
     범위로 판단해 fnguide_sources.fetch_fnguide()로 이식했다.

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
# 🟡 네이버증권 종목 리포트 (모바일 JSON API — 목록 + 리포트별 상세)
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
    research_id: Optional[int] = None
    prev_target_price: Optional[int] = None   # 직전 목표주가(상향/하향 판단용)
    price_at_write: Optional[int] = None      # 작성일 주가
    summary: str = ""                         # 증권사가 직접 쓴 본문 요약(네이버 제공 텍스트)


_NAVER_RESEARCH_LIST = "https://m.stock.naver.com/api/research/stock/{code}"
_NAVER_RESEARCH_DETAIL = "https://m.stock.naver.com/api/research/company/{rid}"


def _int_or_none(v) -> Optional[int]:
    n = _strip_unit(v)
    return int(n) if n else None   # 0/빈값은 '목표가 미제시'로 보고 None


def fetch_naver_research(stock_code: str, *, days: int = 90, max_reports: int = 10,
                         cache_ttl: int = 21600) -> list[dict]:
    """네이버 모바일 증권 API로 최근 `days`일 이내 종목분석 리포트 최신 `max_reports`건을
    수집한다. 목록(/api/research/stock/{code})으로 대상을 고르고, 리포트별 상세
    (/api/research/company/{researchId})에서 투자의견·목표주가·직전 목표주가·작성일
    주가·본문 요약·PDF 링크를 채운다.

    예전 구현은 finance.naver.com/research/company_list.naver 목록 HTML을 긁었는데,
    페이지 구조가 바뀌어 항상 0건을 반환하고 있었다(삼성전자 등으로 확인). 목록 HTML
    에는 목표주가/의견 컬럼도 없었다. JSON API로 바꾸면서 둘 다 해결된다.

    저작권: PDF 본문은 받지 않는다(pdf_url 링크만 보관). summary는 네이버가 리포트
    상세 화면에 공개하는 요약 텍스트다. 상세 조회가 실패한 건은 목록 메타데이터만 남긴다.
    """
    from datetime import date, timedelta

    code = stock_code.zfill(6)
    ck = f"research_m::{code}::{days}::{max_reports}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached["reports"]

    items = _naver_mobile_json(_NAVER_RESEARCH_LIST.format(code=code) + "?pageSize=30&page=1")
    if not isinstance(items, list):
        raise SourceError(f"리포트 목록 응답 형식 변경: {type(items).__name__}")

    since = (date.today() - timedelta(days=days)).isoformat()
    recent = [it for it in items if str(it.get("writeDate", "")) >= since]
    recent.sort(key=lambda it: (str(it.get("writeDate", "")), it.get("researchId") or 0), reverse=True)

    reports: list[Report] = []
    for it in recent[:max_reports]:
        rid = it.get("researchId")
        rep = Report(
            title=str(it.get("title", "")).strip(),
            broker=str(it.get("brokerName", "")).strip(),
            date=str(it.get("writeDate", "")),
            research_id=rid,
            report_url=f"https://m.stock.naver.com/research/company/{rid}" if rid else "",
        )
        if rid:
            try:
                rc = (_naver_mobile_json(_NAVER_RESEARCH_DETAIL.format(rid=rid))
                      .get("researchContent") or {})
                rep.opinion = str(rc.get("opinion") or "").strip()
                rep.target_price = _int_or_none(rc.get("goalPrice"))
                rep.prev_target_price = _int_or_none(rc.get("prevGoalPrice"))
                rep.price_at_write = _int_or_none(rc.get("priceAtWriteDate"))
                rep.pdf_url = str(rc.get("attachUrl") or "")
                # 태그를 공백으로 치환 — 제목 블록과 본문 첫 문장이 붙어버리는 것 방지
                rep.summary = re.sub(r"\s+", " ", re.sub(
                    r"<[^>]+>", " ", html.unescape(rc.get("content") or ""))).strip()
            except Exception as e:  # noqa: BLE001  (상세 1건 실패 — 목록 메타데이터는 유지)
                logger.warning("리포트 상세 조회 실패(%s/%s): %s", code, rid, e)
        reports.append(rep)

    out = [asdict(r) for r in reports]
    _cache_put(ck, {"reports": out})
    return out


# ---------------------------------------------------------------------- #
# 🟢 self-built 컨센서스 (FnGuide 컨센서스 상품의 합법 대체)
# ---------------------------------------------------------------------- #
_OPINION_ALIASES = {
    "강력매수": ("STRONGBUY", "STRONG BUY", "강력매수", "적극매수"),
    "매수": ("BUY", "매수", "TRADINGBUY", "TRADING BUY", "OUTPERFORM", "OVERWEIGHT",
             "비중확대", "단기매수", "ADD", "ACCUMULATE"),
    "중립": ("HOLD", "NEUTRAL", "중립", "보유", "MARKETPERFORM", "MARKET PERFORM",
             "NOTRATED", "NOT RATED", "N/R", "NR"),
    "매도": ("SELL", "매도", "UNDERPERFORM", "UNDERWEIGHT", "비중축소", "REDUCE"),
    "강력매도": ("STRONGSELL", "STRONG SELL", "강력매도"),
}
_OPINION_LOOKUP = {a.replace(" ", ""): label for label, alias in _OPINION_ALIASES.items()
                   for a in alias}


def normalize_opinion(op: str) -> str:
    """증권사별 투자의견 표기를 FnGuide 라벨(강력매수/매수/중립/매도/강력매도)로 통일.
    모르는 표기는 원문 그대로 둔다(집계에서 별도 항목으로 보이게)."""
    raw = (op or "").strip()
    return _OPINION_LOOKUP.get(raw.upper().replace(" ", ""), raw)


def aggregate_consensus(reports: list[dict]) -> dict:
    """수집한 리포트들의 목표주가/투자의견을 직접 집계.
    개별 증권사가 공개한 의견을 출처와 함께 모은 것 — FnGuide 집계상품 복제 아님."""
    tps = [r["target_price"] for r in reports if r.get("target_price")]
    # 증권사마다 '매수'/'Buy'/'BUY'/'StrongBuy'처럼 표기가 달라 그대로 세면 같은 의견이
    # 쪼개진다 — FnGuide 라벨 체계(강력매수/매수/중립/매도/강력매도)로 통일해서 집계
    # (cross_check.check_opinion이 FnGuide 라벨과 다수결을 문자열 비교하기 때문).
    opinions = [normalize_opinion(r.get("opinion", "")) for r in reports if r.get("opinion")]
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
# 🟡 네이버 모바일 증권 시세 (m.stock.naver.com JSON API)
#   가치 관점(2단계) 판단에 필요한 '주가 위치'(52주 밴드 내 어디인지)를 위한 소스.
#   HTML 파싱이 아니라 네이버 앱이 쓰는 JSON 엔드포인트라 페이지 구조 변경에 강하다.
# ---------------------------------------------------------------------- #
def _strip_unit(s) -> Optional[float]:
    """'25.02배', '309,500', '8.22%' 등에서 숫자만 뽑는다."""
    if s is None:
        return None
    s = str(s).replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def _naver_mobile_json(url: str) -> dict:
    headers = {"User-Agent": UA, "Referer": "https://m.stock.naver.com/"}
    r = _request("GET", url, headers=headers, timeout=10)
    try:
        return r.json()
    except ValueError as e:
        raise SourceError(f"응답이 JSON이 아님(엔드포인트 변경 가능): {e}") from e


def fetch_naver_price(stock_code: str, *, cache_ttl: int = 600) -> dict:
    """현재가·52주 최고/최저·PER/PBR·컨센서스 목표주가. 52주 밴드 내 위치
    (band_position, 0=최저~1=최고)를 함께 계산한다(가치 관점: 실적 추정 상승 +
    주가 하단 위치 판단에 사용). closePrice는 /basic, 52주 밴드·컨센서스는
    /integration 엔드포인트에 있어(스키마 상이) 두 곳을 조회해 합친다."""
    code = stock_code.zfill(6)
    ck = f"naver_price::{code}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached

    basic = _naver_mobile_json(f"https://m.stock.naver.com/api/stock/{code}/basic")
    integ = _naver_mobile_json(f"https://m.stock.naver.com/api/stock/{code}/integration")

    close = _strip_unit(basic.get("closePrice"))
    if close is None:
        deals = integ.get("dealTrendInfos") or []
        close = _strip_unit(deals[0].get("closePrice")) if deals else None

    total = {item.get("code"): item.get("value") for item in integ.get("totalInfos", [])}
    high52 = _strip_unit(total.get("highPriceOf52Weeks"))
    low52 = _strip_unit(total.get("lowPriceOf52Weeks"))

    band_pos = None
    if close is not None and high52 is not None and low52 is not None and high52 > low52:
        band_pos = round((close - low52) / (high52 - low52), 3)

    cons = integ.get("consensusInfo") or {}
    target_price = _strip_unit(cons.get("priceTargetMean"))

    result = {
        "price": close,
        "change_pct": _strip_unit(basic.get("fluctuationsRatio")),
        "high_52w": high52,
        "low_52w": low52,
        "band_position": band_pos,
        "per": _strip_unit(total.get("per")),
        "pbr": _strip_unit(total.get("pbr")),
        "cns_per": _strip_unit(total.get("cnsPer")),
        "target_price_mean": target_price,
        "target_upside_pct": (round((target_price / close - 1) * 100, 1)
                              if target_price and close else None),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M"),
        "source": "m.stock.naver.com",
    }
    if result["price"] is None:
        raise SourceError("현재가 파싱 실패(응답 스키마 변경 가능)")
    _cache_put(ck, result)
    return result


# ---------------------------------------------------------------------- #
# KRX KIND 상장법인목록 — 상장일(상장 후 경과년수 판단용)
#   DART company.json의 est_dt는 '법인 설립일'이라 상장일과 다르고, 네이버
#   모바일 API(basic/integration)에도 상장일 필드가 없어 이 공개 목록이 유일한
#   실측 경로다. Batch/fnguidePerformbot.py가 종목코드 매핑을 만들 때 쓰는
#   것과 동일한 다운로드 URL(개인 리서치 목적, 상업적 재배포 아님).
# ---------------------------------------------------------------------- #
def fetch_listing_age(stock_code: str, *, cache_ttl: int = 7 * 86400) -> dict:
    """상장일과 상장 후 경과년수. 상장일은 사실상 불변인 정적 데이터라 전체
    상장사 목록(2700여개)을 통째로 캐시해 재사용한다(종목별 개별 다운로드가
    아니라 프로세스 전체에서 사실상 한 번만 받는다). 조회 실패/미상장 종목이면
    빈 dict를 반환한다(다른 fetch_* 함수와 달리 예외를 던지지 않음 — 이 값은
    가치 시그널의 보조 조건 중 하나일 뿐 가격 데이터 자체를 막을 이유가 없다)."""
    cached = _cache_get("krx_listed_dates", cache_ttl)
    if cached is None:
        try:
            r = _request("GET", "http://kind.krx.co.kr/corpgeneral/corpList.do",
                         params={"method": "download"}, timeout=20)
            r.encoding = "EUC-KR"
            import pandas as pd
            from io import StringIO
            df = pd.read_html(StringIO(r.text), header=0)[0]
            df.columns = ["name", "market", "code", "industry", "product",
                          "listed_date", "settle_month", "ceo", "homepage", "region"]
            df["code"] = df["code"].astype(str).str.zfill(6)
            cached = dict(zip(df["code"], df["listed_date"].astype(str)))
            _cache_put("krx_listed_dates", cached)
        except Exception as e:  # noqa: BLE001
            logger.warning("KRX 상장법인목록 조회 실패: %s", e)
            return {}

    listed_date = cached.get(stock_code.zfill(6))
    if not listed_date:
        return {}
    try:
        from datetime import date, datetime
        d = datetime.strptime(listed_date, "%Y-%m-%d").date()
    except ValueError:
        return {}
    return {"listed_date": listed_date, "years_listed": round((date.today() - d).days / 365.25, 1)}


# ---------------------------------------------------------------------- #
# 네이버 종목분석 > Financial Summary (출처: navercomp.wisereport.co.kr / FnGuide)
#   연간/분기/예상((E)) 매출액·영업이익·당기순이익. 단위: 억원.
#   주의: 데이터 저작권은 FnGuide. 자동수집/DB화 ToS 리스크는 사용자 책임.
#   호출 최소화(긴 캐시·종목당 1회·UA/Referer).
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


def _flatten_cols(tbl) -> list[str]:
    return ["/".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
            for col in tbl.columns]


def _parse_finsummary(tables) -> tuple[dict, dict, dict]:
    """매출액·영업이익 행을 가진 표(들)에서 연간/분기/예상을 추출.
    분기표: 컬럼 라벨에 03/06/09 월이 섞여 있음. 연간표: 대부분 12월.
    (E) 포함 컬럼은 추정치로 분리."""
    target = ("매출액", "영업이익", "당기순이익")
    annual: dict = {}
    quarter: dict = {}
    estimates: dict = {}

    for t in tables:
        if t is None or t.shape[1] < 2 or len(t) == 0:
            continue
        try:
            first = t.iloc[:, 0].astype(str).str.replace(" ", "")
        except (IndexError, KeyError):
            continue
        if not (first.str.contains("매출액").any() and first.str.contains("영업이익").any()):
            continue

        cols = _flatten_cols(t)
        # 분기표 판별: 데이터 컬럼 라벨에 03/06/09가 보이면 분기
        months = re.findall(r"/(\d{2})", " ".join(cols))
        is_quarter = any(mm in ("03", "06", "09") for mm in months)
        bucket = quarter if is_quarter else annual
        name_col = t.columns[0]

        for _, row in t.iterrows():
            acct = str(row[name_col]).replace(" ", "")
            acct = next((a for a in target if a in acct), None)
            if not acct:
                continue
            for i, c in enumerate(cols):
                if i == 0:
                    continue
                val = _to_num(row.iloc[i])
                if val is None:
                    continue
                bucket.setdefault(c, {})[acct] = val
                if "(E)" in c.replace(" ", ""):
                    estimates.setdefault(c, {})[acct] = val
    return annual, quarter, estimates


def fetch_naver_financial_summary(stock_code: str, *, cache_ttl: int = 86400) -> dict:
    """네이버 종목분석의 Financial Summary(WISEreport)에서
    연간/분기/예상 매출액·영업이익·당기순이익을 추출. 단위: 억원."""
    code = stock_code.zfill(6)
    ck = f"naver_finsum::{code}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached

    try:
        import pandas as pd
    except ImportError:
        raise SourceError("pandas/lxml 필요: pip install pandas lxml")

    base = os.getenv("NAVER_FINSUM_URL",
                     "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx")
    url = f"{base}?cmp_cd={code}&cn="
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Referer": "https://finance.naver.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    r = _request("GET", url, headers=headers)
    r.encoding = r.apparent_encoding or "utf-8"
    html_text = r.text
    if len(html_text) < 2000:
        raise SourceError(f"WISEreport 응답이 짧음({len(html_text)}B) — 엔드포인트/종목 확인 필요.")

    import io
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except Exception as e:  # noqa: BLE001  read_html 내부 오류 흡수
        raise SourceError(f"표 파싱 실패(페이지 구조 확인): {type(e).__name__}")

    annual, quarter, estimates = _parse_finsummary(tables)
    if not (annual or quarter):
        raise SourceError("Financial Summary 표를 찾지 못했습니다(페이지 구조 변경 가능).")

    result = {
        "source": "finance.naver.com (WISEreport/FnGuide)",
        "unit": "억원",
        "fetched_at": time.strftime("%Y-%m-%d"),
        "annual": annual,
        "quarter": quarter,
        "estimates": estimates,
    }
    _cache_put(ck, result)
    return result
