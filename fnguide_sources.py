"""
fnguide_sources.py
===================
FnGuide(comp.fnguide.com) / WiseReport(navercomp.wisereport.co.kr) 직접 스크래핑.

옆 프로젝트 simul_server.py(_fnguide_data/_wr_cf1002_estimates)에서 이미 검증된
파서를 개인 리서치 파이프라인용으로 이식했다. FnGuide/WiseReport 데이터는
유료 라이선스 상품이므로 상업적 재배포·DB화는 하지 않고, 여기서는 개인 분석
파이프라인의 입력(연간/분기 실적+추정, 제품비중, 컨센서스)으로만 사용한다.
호출 최소화를 위해 디스크 캐시를 쓴다.

제공 함수:
  fetch_fnguide(code) -> dict
    {
      'products': [{'name','pct'}], 'market_shares': [...], 'keywords': [...],
      'annual_highlight': [...], 'financial_highlight': [...](분기, FnGuide 하이라이트 표),
      'cf1002': {'freq': 'quarter'|'annual'|'none', 'rows': [...]}
                (실측+추정이 같은 주기로 나란히 있는 단일 시계열 — 성장률 계산 기준),
      'consensus': {'opinion_label','target_price','eps','per','analyst_count','date'} | None,
      'source': 'FnGuide'|'WiseReport-fallback'|'none',
    }
  각 highlight row: {period, is_estimate, revenue, op_profit, net_profit, op_margin, op_growth}
  (단위: 억원)

안정성: 개별 하위 수집 단계는 각각 try/except로 감싸 부분 실패해도 나머지는 반환한다
(파이프라인 안정성 검증의 취지와 동일 — 전부 실패해야 빈 결과).
"""
from __future__ import annotations

import html as html_mod
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from external_sources import SourceError, _cache_get, _cache_put, _request

logger = logging.getLogger("fnguide_sources")

# FnGuide/WiseReport는 비-브라우저 UA를 종종 차단하므로 브라우저 UA를 쓴다
# (external_sources.UA는 뉴스/DART용 경량 UA라 여기선 별도 사용).
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_FNG_HDR = {"User-Agent": _BROWSER_UA, "Referer": "https://comp.fnguide.com/",
            "Accept-Language": "ko-KR,ko;q=0.9"}
_WR_HDR = {"User-Agent": _BROWSER_UA, "Referer": "https://navercomp.wisereport.co.kr/",
           "Accept-Language": "ko-KR,ko;q=0.9"}


def _growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev in (None, 0):
        return None
    return round((curr - prev) / abs(prev) * 100, 1)


def _fng_fetch(url: str) -> str:
    r = _request("GET", url, headers=_FNG_HDR, timeout=15, retries=2)
    ct = r.headers.get("content-type", "").lower()
    r.encoding = "utf-8" if "utf" in ct else "euc-kr"
    return r.text


def _wr_fetch(url: str) -> str:
    r = _request("GET", url, headers=_WR_HDR, timeout=12, retries=2)
    r.encoding = "utf-8"
    return r.text


def _extract_table(html_text: str, tid: str, window: int = 25000) -> str:
    p = html_text.find(f'id="{tid}"')
    if p < 0:
        return ""
    e = html_text.find("</table>", p)
    return html_text[p:e + 8] if e > 0 else html_text[p:p + window]


def _extract_div_table(html_text: str, div_id: str) -> str:
    """div_id 내 첫 번째 <table>...</table> 추출."""
    p = html_text.find(f'id="{div_id}"')
    if p < 0:
        return ""
    ts = html_text.find("<table", p)
    if ts < 0 or ts > p + 1000:
        return ""
    te = html_text.find("</table>", ts)
    return html_text[ts:te + 8] if te > 0 else ""


def _parse_fng_table(tbl_html: str, today_ym: str) -> list[dict]:
    """FnGuide SVD_Main / WiseReport highlight table 파싱 -> [{period, is_estimate, ...}]."""
    if not tbl_html:
        return []
    thead_m = re.search(r"<thead[^>]*>(.*?)</thead>", tbl_html, re.DOTALL)
    thead_content = thead_m.group(1) if thead_m else tbl_html[:2000]
    all_th = re.findall(r"<th[^>]*>(.*?)</th>", thead_content, re.DOTALL)
    periods = []
    for th in all_th:
        plain = re.sub(r"<[^>]+>", "", th).replace("\xa0", "").strip()
        plain = re.sub(r"\s+", " ", plain).strip()
        m = re.search(r"(\d{4}/\d{2})", plain)
        if m:
            p = m.group(1)
            is_est = "(E)" in plain or "(e)" in plain.lower()
            periods.append((p, is_est))
    if not periods:
        return []

    fin_data: dict = {}
    for row_m in re.finditer(r"<tr[^>]*>(.*?)</tr>", tbl_html, re.DOTALL):
        row_html = row_m.group(1)
        th_m = re.search(r'<th[^>]*scope=["\']row["\'][^>]*>(.*?)</th>', row_html, re.DOTALL)
        if not th_m:
            continue
        th_content = th_m.group(1)
        a_m = re.search(r"<a[^>]*>\s*([^<]+?)\s*</a>", th_content)
        if a_m:
            name = a_m.group(1).strip()
        else:
            raw = re.sub(r"<[^>]+>", " ", th_content)
            raw = re.sub(r"\s+", " ", raw).strip()
            name = re.sub(r"\s*\([^)]+\)\s*$", "", raw).strip()
        name = re.sub(r"\s+", " ", name)
        if not name:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
        vals = []
        for td in tds:
            sp = re.search(r"<span[^>]*>([-\d,\.]+)</span>", td)
            if sp:
                try:
                    vals.append(float(sp.group(1).replace(",", "")))
                except ValueError:
                    vals.append(None)
            else:
                clean = re.sub(r"<[^>]+>", "", td).strip().replace(",", "")
                if clean and re.match(r"^-?[\d.]+$", clean):
                    try:
                        vals.append(float(clean))
                    except ValueError:
                        vals.append(None)
                else:
                    vals.append(None)
        fin_data[name] = vals

    def _pick(*names):
        for n in names:
            if n in fin_data and any(v is not None for v in fin_data[n]):
                return fin_data[n]
        return []

    rev_v = _pick("매출액", "영업수익", "이자수익", "보험료수익")
    op_v = _pick("영업이익", "영업이익(발표기준)", "영업이익(손실)")
    ni_v = _pick("당기순이익", "당기순이익(지배)", "당기순이익(지배주주)",
                 "지배주주순이익", "당기순이익(손실)")

    fh = []
    for i, (period, is_est) in enumerate(periods):
        rev = rev_v[i] if i < len(rev_v) else None
        op = op_v[i] if i < len(op_v) else None
        ni = ni_v[i] if i < len(ni_v) else None
        if not any(v is not None for v in (rev, op, ni)):
            continue
        fh.append({
            "period": period,
            "is_estimate": is_est or period > today_ym,
            "revenue": rev, "op_profit": op, "net_profit": ni,
            "op_margin": round(op / rev * 100, 1) if op and rev and rev > 0 else None,
            "op_growth": None,
        })
    return fh


def _extract_js_obj(text: str, key: str) -> Optional[dict]:
    """`information.init({... key: {...}, ...})`처럼 스크립트에 인라인 주입된 JS
    객체 리터럴에서 key에 해당하는 값(중첩 객체)만 정확히 잘라 json.loads로 판다.
    정규식만으로는 중첩된 {}를 안전하게 못 자른다(예: salesPercent 안에 data/
    expData가 또 배열 속 객체를 담고 있어 '{...}를 최소 매칭'해선 잘림) — 그래서
    문자 단위로 중괄호 깊이를 세되, 문자열 리터럴("...") 안의 {}는 건너뛴다.
    이 페이지의 값들은 키가 전부 큰따옴표로 감싸인 정상 JSON이라 json.loads로
    바로 파싱된다(JS 전용 문법 없음 — 실측 확인)."""
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', text)
    if not m:
        return None
    start = m.end() - 1  # '{' 위치
    depth = 0
    in_str = False
    str_ch = ""
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str, str_ch = True, ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        i += 1
    return None


def _fetch_products_xml(code: str) -> dict:
    """제품비율/시장점유율/키워드.

    2026년 FnGuide가 구버전 사이트(comp.fnguide.com)의 XML 엔드포인트를 폐쇄하고
    신버전(wcomp.fnguide.com)으로 이전했다(실측 확인). 신버전은 별도 XML이 아니라
    회사정보 페이지(CompanyInfo/Information) HTML 안에 <script>로
    `information.init({ salesPercent: {...}, marketShare: {...}, ... })`처럼
    데이터를 인라인 주입한다.

    salesPercent.data는 여러 분기를 한 번에 담고 있고(FY1~FY4_AMT 키, header가
    각 키의 실제 기간 라벨(YYYY/MM)을 알려줌 — FY1이 가장 최근), 세그먼트별로
    분기마다 결측(null)이 섞여 있어 가장 최근 기간(header상 NM이 제일 큰 값)의
    값만 골라 쓴다. marketShare는 단일 기간(YYMM)이라 그대로 쓴다."""
    out = {"products": [], "market_shares": [], "keywords": []}
    try:
        r = _request("GET", f"https://wcomp.fnguide.com/CompanyInfo/Information?gicode=A{code}",
                     headers=_FNG_HDR, timeout=12, retries=2)
        text = r.content.decode("utf-8", errors="replace")

        # 실측 확인된 문제: ?gicode=A{code} 쿼리스트링이 실제로는 라우팅에 반영되지
        # 않고 코드와 무관하게 항상 동일한(관찰된 사례: 삼성전자 005930) 페이로드가
        # 내려오는 경우가 있다(캐시버스팅 파라미터를 붙여도 동일 — 프록시/CDN 캐시가
        # 아니라 서버 자체가 쿼리스트링을 무시하는 것으로 보임). 이 상태로 그냥
        # 파싱하면 '조용히 빈 값' 대신 '조용히 엉뚱한 회사 데이터'를 반환하는 더 나쁜
        # 실패가 되므로, 페이로드의 cmp_cd가 요청한 code와 실제로 일치하는지 반드시
        # 검증하고, 다르면 신뢰하지 않는다.
        cmp_m = re.search(r"cmp_cd\s*:\s*['\"](\d+)['\"]", text)
        if not cmp_m or cmp_m.group(1).zfill(6) != code.zfill(6):
            logger.warning(
                "FnGuide 제품정보(%s): 응답의 cmp_cd(%s)가 요청 코드와 불일치 — "
                "?gicode= 쿼리스트링이 서버에서 무시되고 있는 것으로 보임(신버전 "
                "사이트의 알려진 문제, 실측 확인). 잘못된 회사 데이터를 쓰지 않도록 "
                "빈 값으로 유지.", code, cmp_m.group(1) if cmp_m else "없음")
            return out

        sales = _extract_js_obj(text, "salesPercent")
        market = _extract_js_obj(text, "marketShare")

        if sales:
            header = sales.get("header") or []
            latest = max(header, key=lambda h: h.get("NM") or "", default=None)
            latest_id = latest.get("ID") if latest else None
            if latest_id:
                for rec in sales.get("data") or []:
                    name = (rec.get("COMM_NM") or "").strip()
                    val = rec.get(latest_id)
                    if not name or val is None or "내부거래" in name or name.startswith("기타"):
                        continue
                    if val > 0:
                        out["products"].append({"name": name, "pct": round(float(val), 1)})

        if market:
            for rec in market.get("data") or []:
                pl = (rec.get("PRD_NM") or "").strip()
                pv = rec.get("OCCU_RATIO")
                if pl and pv is not None:
                    out["market_shares"].append({"product": pl, "share": round(float(pv), 1)})

        seen, keywords = set(), []
        for p in out["products"]:
            kw = re.split(r"[,·ㆍ/]", p["name"])[0].strip()
            kw = re.sub(r"\s*(등|및)$", "", kw).strip()
            if kw and 2 <= len(kw) <= 15 and kw not in seen:
                seen.add(kw)
                keywords.append(kw)
        out["keywords"] = keywords[:4]

        if not sales and not market:
            logger.warning(
                "FnGuide 제품정보(%s): CompanyInfo/Information 페이지에서 "
                "salesPercent/marketShare 인라인 데이터를 찾지 못함 — 페이지 "
                "구조가 다시 바뀌었을 수 있음.", code)
    except Exception as e:  # noqa: BLE001
        logger.warning("FnGuide 제품정보 수집 실패(%s): %s", code, e)
    return out


def _fetch_highlights(code: str, cf_html: str, c1_html: str) -> tuple[list, list, str]:
    """분기/연간 하이라이트(실적+추정). 반환: (annual_fh, quarter_fh, source_label)."""
    today_ym = datetime.now().strftime("%Y/%m")

    qtr_tbl, tried_q = "", set()
    for qid in ("highlight_D_E", "highlight_D_Q", "highlight_A_E", "highlight_A_Q"):
        tried_q.add(qid)
        qtr_tbl = _extract_div_table(cf_html, qid)
        if qtr_tbl:
            break
    if not qtr_tbl:
        for dm in re.finditer(r'id="(highlight_[^"]+)"', cf_html):
            did = dm.group(1)
            if did in tried_q:
                continue
            tried_q.add(did)
            t = _extract_div_table(cf_html, did)
            if not t:
                continue
            fh_t = _parse_fng_table(t, today_ym)
            if not fh_t:
                continue
            months = {p.split("/")[1] for p in (r["period"] for r in fh_t) if "/" in p}
            if len(months) > 1:
                qtr_tbl = t
                break

    ann_tbl, tried_a = "", set()
    for aid in ("highlight_D_A", "highlight_A_A"):
        tried_a.add(aid)
        ann_tbl = _extract_div_table(cf_html, aid)
        if ann_tbl:
            break
    if not ann_tbl:
        for dm in re.finditer(r'id="(highlight_[^"]+)"', cf_html):
            did = dm.group(1)
            if did in tried_q or did in tried_a:
                continue
            tried_a.add(did)
            t = _extract_div_table(cf_html, did)
            if not t:
                continue
            fh_t = _parse_fng_table(t, today_ym)
            if not fh_t:
                continue
            months = {p.split("/")[1] for p in (r["period"] for r in fh_t) if "/" in p}
            if len(months) == 1:
                ann_tbl = t
                break

    qtr_fh = _parse_fng_table(qtr_tbl, today_ym)
    ann_fh = _parse_fng_table(ann_tbl, today_ym)
    source = "FnGuide" if (qtr_tbl or ann_tbl) else "none"

    if not qtr_fh and not ann_fh:
        ann_fh, qtr_fh, source = _fallback_wisereport_highlights(code, c1_html, today_ym)

    if qtr_fh or ann_fh:
        for j in range(1, len(qtr_fh)):
            qtr_fh[j]["op_growth"] = _growth(qtr_fh[j].get("op_profit"),
                                             qtr_fh[j - 1].get("op_profit"))
    return ann_fh, qtr_fh, source


def _fallback_wisereport_highlights(code: str, c1_html: str, today_ym: str) -> tuple[list, list, str]:
    """FnGuide SVD_Main에서 못 찾으면 WiseReport cF1001(cTB26) 등으로 폴백."""
    try:
        wr_html = _wr_fetch(
            f"https://navercomp.wisereport.co.kr/v2/company/cF1001.aspx?cmp_cd={code}&cn=")
    except Exception as e:  # noqa: BLE001
        logger.warning("WiseReport cF1001 수집 실패(%s): %s", code, e)
        return [], [], "none"

    tb26 = _extract_table(wr_html, "cTB26")
    if not tb26:
        return [], [], "none"

    all_periods = re.findall(r"(\d{4}/\d{2})<br", tb26)
    ann_periods = all_periods[:4]
    fin_data_wr: dict = {}
    for row_m in re.finditer(r"<th scope='row'[^>]*>\s*([^<]+?)\s*</th>(.*?)</tr>", tb26, re.DOTALL):
        nm = re.sub(r"\s+", " ", row_m.group(1)).strip()
        tds_html = re.findall(r"<td[^>]*>(.*?)</td>", row_m.group(2), re.DOTALL)
        vals = []
        for td in tds_html:
            sp = re.search(r"<span[^>]*>([-\d,\.]+)</span>", td)
            if sp:
                try:
                    vals.append(float(sp.group(1).replace(",", "")))
                except ValueError:
                    vals.append(None)
            else:
                vals.append(None)
        if nm:
            fin_data_wr[nm] = vals

    def _pk(d, *names):
        for n in names:
            if n in d and any(v is not None for v in d[n]):
                return d[n]
        return []

    rev_a = _pk(fin_data_wr, "매출액", "영업수익", "이자수익", "보험료수익")
    op_a = _pk(fin_data_wr, "영업이익", "영업이익(발표기준)", "영업이익(손실)")
    ni_a = _pk(fin_data_wr, "당기순이익", "당기순이익(지배)", "당기순이익(지배주주)",
               "지배주주순이익", "당기순이익(손실)")

    def _row(rv, ov, nv, period, i):
        rev = rv[i] if i < len(rv) else None
        op = ov[i] if i < len(ov) else None
        ni = nv[i] if i < len(nv) else None
        if not any(v is not None for v in (rev, op, ni)):
            return None
        return {
            "period": period, "is_estimate": period > today_ym,
            "revenue": rev, "op_profit": op, "net_profit": ni,
            "op_margin": round(op / rev * 100, 1) if op and rev and rev > 0 else None,
            "op_growth": None,
        }

    ann_fh = [r for r in (_row(rev_a, op_a, ni_a, p, i) for i, p in enumerate(ann_periods)) if r]
    qtr_fh: list = []  # cTB26 분기 레이블은 신뢰 불가(DART로 보완하는 것을 권장)

    # (E) 보완: c1_html 내 다른 cTBxx 테이블 탐색
    def _scan(html_text: str, skip_ids: set) -> tuple[list, list]:
        a, q = [], []
        for tm in re.finditer(r'id="(cTB\d+)"', html_text):
            tid = tm.group(1)
            if tid in skip_ids:
                continue
            t = _extract_table(html_text, tid)
            if not t:
                continue
            fh = _parse_fng_table(t, today_ym)
            if not fh or len(fh) < 2 or not any(r.get("revenue") is not None for r in fh):
                continue
            mo = {p.split("/")[1] for p in (r["period"] for r in fh) if "/" in p}
            if len(mo) == 1 and not a:
                a = fh
            elif len(mo) > 1 and not q:
                q = fh
            if a and q:
                break
        return a, q

    ex_ann, ex_qtr = _scan(c1_html, {"cTB15"})
    if ex_ann:
        ex_periods = {r["period"] for r in ex_ann}
        extra = [r for r in ann_fh if r["period"] not in ex_periods]
        ann_fh = sorted(extra + ex_ann, key=lambda x: x["period"])
    if ex_qtr:
        qtr_fh = ex_qtr

    return ann_fh, qtr_fh, "WiseReport-fallback"


def _fetch_consensus(code: str, c1_html: str) -> Optional[dict]:
    """cTB15: 투자의견/목표주가/EPS/PER/추정기관수."""
    pos = c1_html.find('id="cTB15"')
    if pos < 0:
        return None
    tb15 = _extract_table(c1_html, "cTB15", window=3000)
    if not tb15:
        return None

    cons_date = None
    before = c1_html[max(0, pos - 8000):pos]
    date_hits = re.findall(r"\[.{0,6}:\s*(\d{4}\.\d{2}\.\d{2})\]", before)
    if date_hits:
        cons_date = date_hits[-1]

    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tb15, re.DOTALL)
    data_trs = [t for t in trs if "<td" in t]
    if not data_trs:
        return None
    tds = re.findall(r"<td[^>]*>(.*?)</td>", data_trs[-1], re.DOTALL)

    def _clean(raw):
        t = re.sub(r"<[^>]+>", "", raw)
        t = html_mod.unescape(t).replace("\xa0", "").strip()
        return t if t and t not in ("-", "–", "—") else None

    cells = [_clean(td) for td in tds]
    opin_lbl = {"1": "강력매도", "2": "매도", "3": "중립", "4": "매수", "5": "강력매수"}
    opin_rev = {v: k for k, v in opin_lbl.items()}
    op_raw = cells[0] if cells else None
    op_num = ""
    if op_raw:
        if op_raw in opin_rev:
            op_num = opin_rev[op_raw]
        elif op_raw.replace(".", "").lstrip("-").isdigit():
            op_num = str(round(float(op_raw)))
    cons = {
        "opinion": op_num,
        "opinion_label": opin_lbl.get(op_num, op_raw or ""),
        "target_price": cells[1] if len(cells) > 1 else None,
        "eps": cells[2] if len(cells) > 2 else None,
        "per": cells[3] if len(cells) > 3 else None,
        "analyst_count": cells[4] if len(cells) > 4 else None,
        "date": cons_date,
    }
    return cons if any(v for v in cons.values() if v) else None


def _fetch_cf1002_series(code: str, frq: str) -> list[dict]:
    """WiseReport cF1002.aspx(cTB25): 실적 3기 + 추정(E) 2기, 같은 주기(frq)로 통일된
    시계열 전체(실측+추정)를 반환한다. frq: '0'=연간 '1'=분기.

    이 표는 재무년월 컬럼 하나 + 지표별 고정 컬럼이라 cTB26/하이라이트 표보다
    파싱이 안전하고, 무엇보다 실측과 추정이 '같은 주기'로 나란히 있어
    성장률 계산 시 연간 실적과 분기 추정을 잘못 비교하는 실수를 막아준다."""
    try:
        html_text = _wr_fetch(
            f"https://navercomp.wisereport.co.kr/v2/company/cF1002.aspx"
            f"?cmp_cd={code}&finGubun=MAIN&frq={frq}&rpt=0&finAcctClass=&cn=")
    except Exception as e:  # noqa: BLE001
        logger.warning("WiseReport cF1002 수집 실패(%s,frq=%s): %s", code, frq, e)
        return []

    def _num(raw):
        s = re.sub(r"<[^>]+>", "", raw).replace(",", "").strip()
        if not s or s in ("N/A", "-"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    rows = []
    for row_m in re.finditer(r"<td class='center'>([^<]+)</td>(.*?)</tr>", html_text, re.DOTALL):
        period_raw = row_m.group(1).strip()
        is_est = "(E)" in period_raw
        period = period_raw.replace("(A)", "").replace("(E)", "").strip()
        tds = re.findall(r"<td class='num'>(.*?)</td>", row_m.group(2), re.DOTALL)
        if len(tds) < 4:
            continue
        rev, op, ni = _num(tds[0]), _num(tds[2]), _num(tds[3])
        if rev is not None and rev < 0:
            # 실측: WiseReport 원본 표 자체에 매출액이 음수로 찍힌 사례 확인(182360
            # 2025.12 매출 -594억원 — 전후 분기는 +501/+151로 정상). 매출은 특이한
            # 회계 조정이 아닌 이상 음수일 수 없으므로, 이건 우리 파싱이 아니라
            # 소스 데이터 자체의 오류로 보고 결측치로 처리한다(잘못된 값을 그대로
            # 성장률 계산·LLM 프롬프트에 흘려보내지 않기 위함).
            logger.warning("WiseReport cF1002 매출액 음수 이상값 무시(%s, %s): %s",
                           code, period_raw, rev)
            rev = None
        if rev is None and op is None and ni is None:
            continue
        rows.append({
            "period": period, "is_estimate": is_est,
            "revenue": rev, "op_profit": op, "net_profit": ni,
            "op_margin": round(op / rev * 100, 1) if op is not None and rev else None,
        })
    rows.sort(key=lambda r: r["period"])
    return rows


def fetch_fnguide(stock_code: str, *, cache_ttl: int = 21600) -> dict:
    """FnGuide/WiseReport에서 제품비중·시장점유율·연간/분기 실적+추정·컨센서스 수집.
    전부 실패해도 예외를 던지지 않고 빈 결과를 반환한다(부분 실패는 허용)."""
    code = stock_code.zfill(6)
    ck = f"fnguide::{code}"
    cached = _cache_get(ck, cache_ttl)
    if cached:
        return cached

    result: dict = {
        "products": [], "market_shares": [], "keywords": [],
        "annual_highlight": [], "financial_highlight": [],
        # cf1002: 실측+추정이 같은 주기(freq)로 나란히 있는 단일 시계열.
        # 성장률 계산은 이 필드를 우선 쓴다(연간 실적 vs 분기 추정처럼 주기가
        # 다른 값을 비교하는 오류를 피하기 위해) — invest_point.py 참조.
        "cf1002": {"freq": "none", "rows": []},
        "consensus": None, "source": "none",
    }

    result.update(_fetch_products_xml(code))

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_cf = ex.submit(
                _fng_fetch,
                f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp"
                f"?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701")
            f_c1 = ex.submit(
                _wr_fetch,
                f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&cn=")
        cf_html = f_cf.result()
        c1_html = f_c1.result()
    except Exception as e:  # noqa: BLE001
        logger.warning("FnGuide/WiseReport 메인 페이지 수집 실패(%s): %s", code, e)
        cf_html = c1_html = ""

    if cf_html or c1_html:
        try:
            ann_fh, qtr_fh, source = _fetch_highlights(code, cf_html, c1_html)
            result["annual_highlight"] = ann_fh
            result["financial_highlight"] = qtr_fh
            result["source"] = source
        except Exception as e:  # noqa: BLE001
            logger.warning("FnGuide 하이라이트 파싱 실패(%s): %s", code, e)

        try:
            result["consensus"] = _fetch_consensus(code, c1_html)
        except Exception as e:  # noqa: BLE001
            logger.warning("FnGuide 컨센서스 파싱 실패(%s): %s", code, e)

    try:
        qtr_series = _fetch_cf1002_series(code, "1")
        if any(r["is_estimate"] for r in qtr_series):
            result["cf1002"] = {"freq": "quarter", "rows": qtr_series}
        else:
            ann_series = _fetch_cf1002_series(code, "0")
            if any(r["is_estimate"] for r in ann_series):
                result["cf1002"] = {"freq": "annual", "rows": ann_series}
            elif qtr_series or ann_series:
                # 추정치는 없지만 실측 시계열은 있음 — 성장 방향 계산엔 못 쓰지만
                # 추이 표시용으로 남겨둔다.
                result["cf1002"] = {"freq": "quarter" if qtr_series else "annual",
                                    "rows": qtr_series or ann_series}
    except Exception as e:  # noqa: BLE001
        logger.warning("WiseReport cF1002 예상실적 수집 실패(%s): %s", code, e)

    _cache_put(ck, result)
    return result
