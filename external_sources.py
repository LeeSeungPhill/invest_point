"""
external_sources.py
===================
DART 외 외부 소스 수집기.

  🟢 fetch_naver_news()        : 네이버 공식 뉴스 검색 API (합법, 권장)
  🟡 fetch_naver_research()    : 네이버증권 리포트 '목록 메타데이터' (제목/증권사/
                                작성일/링크). 본문 PDF는 받지 않음. 이 목록 페이지에는
                                목표주가/투자의견 컬럼이 없어(조회수만 있음) 채우지 않는다
                                — 목표주가는 fetch_naver_price()/fnguide consensus 사용.
  🟢 aggregate_consensus()     : 위 리포트들에 target_price가 있는 경우에만(현재는
                                거의 없음) 집계하는 자리— 실질적 목표주가 컨센서스는
                                fetch_naver_price()의 target_price_mean을 사용하라.
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

    주의: company_list.naver 목록 페이지의 실제 컬럼은 종목명/제목/증권사/첨부/작성일/
    조회수 6개뿐이며 목표주가·투자의견 컬럼은 없다(개별 리포트 PDF 본문에만 있음).
    그래서 target_price/opinion은 이 함수에서 채우지 않는다 — 대신 목표주가 컨센서스는
    fetch_naver_price()(네이버 자체 컨센서스 API)나 fnguide_sources.fetch_fnguide()의
    consensus(WiseReport cTB15)를 사용하라. 과거 버전은 '조회수' 숫자 컬럼을 목표주가로
    오인해 집계하는 버그가 있었다(예: 004000에서 평균 2,838원처럼 실제 주가와 무관한 값).

    네이버 페이지 구조/컬럼은 수시로 바뀐다. 아래 파서는 방어적으로 '있는 만큼' 뽑으며,
    컬럼 매칭이 어긋나면 비는 값이 생긴다 => 이 어긋남을 잡아내는 게 바로
    '파이프라인 안정성 검증' 단계의 목적. 개인 리서치 용도. 저작권 있는 리포트 PDF
    본문은 재생산/저장하지 않는다.
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
            # 주의: 이 목록 페이지에는 목표주가 컬럼이 없다(마지막 숫자 셀은 조회수).
            # target_price는 채우지 않는다 — fetch_naver_price()/fnguide consensus 사용.
            if title:
                out.append(Report(title=title, broker=broker, date=date,
                                  report_url=report_url, pdf_url=pdf_url))
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
