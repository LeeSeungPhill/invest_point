"""
dividend_metrics.py
====================
invest_mng.dividend_rate(배당율)와 dividend_check(배당 안정성 점수)를 계산하는
순수 함수 모음. 네트워크 호출은 하지 않는다(수집은 dart_client, 호출은
mvp_graph.build_dividend 노드).

배당율 = 실효 DPS / 현재가 (사용자 확정 2026-11-02):
  1. 기본: 최근 12개월(TTM) 주당배당금 합계. DART '현금ㆍ현물배당결정' 공시의
     배당기준일로 묶는다. 12개월은 오늘이 아니라 가장 최근 배당기준일부터 거꾸로
     잡는다 — 오늘 기준이면 결산배당 공시가 아직 안 나온 1~2월에 분기배당 3건만
     잡혀 수익률이 일시적으로 떨어진다.
  2. 특별배당: 공시 양식에 특별배당 칸이 없다(배당구분은 결산/분기/중간뿐, 실측:
     삼성전자 2026년 결산배당 566원에 '정기 분기배당금에 1.3조원을 추가'한 금액이
     기타 투자판단 사항 서술에만 적혀 있음). 그래서
       (a) 서술에 특별·추가·일회성 배당 표현이 있으면 플래그만 세우고 금액은 3번에 맡기고,
       (b) 분기배당 회사의 결산배당이 정기 분기배당보다 크면 초과분을 특별분으로 추정해 뺀다.
     같은 배당에 둘 다 해당하면 (a)를 우선한다(차감하지 않음).
  3. 이상치: 특별분을 뺀 TTM이 5년 평균 DPS의 1.5배를 넘으면 플래그를 세우고 둘 중
     작은 값을 쓴다.

배당 안정성 점수(0~100, 수익률과 섞지 않음): 연속배당 연수 30 + 감배 횟수 30 +
배당성향 20 + DPS CAGR 20. 연간 DPS는 DART '배당에 관한 사항'(약 10년)을 쓰되
  - 액면분할/병합은 같은 응답의 주당액면가액으로 환산한다(실측: 삼성전자 2016년
    28,500원·액면 5,000원 ↔ 2019년 1,416원·액면 100원 — 그대로면 95% 감배로 오판).
  - 직전 해가 그 전해의 1.5배를 넘는 급증(특별배당 추정)이었으면 그 다음 해 감소는
    감배로 세지 않는다(실측: 삼성전자 2020년 특별배당 2,994원 → 2021년 1,444원).
"""
from __future__ import annotations

import re
import statistics
from datetime import date, datetime, timedelta
from typing import Optional

TTM_DAYS = 365
STALE_DAYS = 450                 # 최근 배당기준일이 이보다 오래되면 배당 중단으로 보고 TTM=0
OUTLIER_RATIO = 1.5
QUARTERLY_EXCESS_TOL = 0.10      # (b) 정기 분기배당보다 10% 넘게 클 때만 초과분 차감
CUT_TOL = 0.05                   # 감배: 전년 대비 5% 넘게 줄었을 때
SPIKE_RATIO = 1.5                # 감배 판정에서 제외할 직전 해 급증 기준
AUTO_PREFIX = "[자동]"           # dividend_check 자동값 표시(사용자 직접 입력과 구분)

# '배당금에 ... 추가': 문장 끝 마침표는 넘지 않되, 숫자 소수점('1.3조원')은 허용한다
# (실측: 삼성전자 '정기 분기배당금에 1.3조원을 추가하여'가 소수점에서 끊겨 놓쳤었다).
_SPECIAL_RE = re.compile(
    r"특별\s*배당|추가\s*배당|일회성\s*배당|배당금?에(?:[^.\n]|\.(?=\d)){0,30}추가")


def _num(s) -> Optional[float]:
    s = str(s or "").strip().replace(",", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _date(s) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


# --------------------------------------------------------------------- #
# 1~3. 배당결정 공시 → TTM DPS → 실효 DPS
# --------------------------------------------------------------------- #
def parse_dividend_doc(text: str) -> dict:
    """'현금ㆍ현물배당결정' 공시 본문(태그 제거한 평문)에서 배당구분·보통주 1주당
    배당금·배당기준일과 특별배당 언급 여부를 뽑는다. 못 찾은 항목은 None."""
    t = re.sub(r"\s+", " ", text or "")
    kind = re.search(r"배당구분\s*([가-힣]+배당)", t)
    dps = re.search(r"1주당\s*배당금\s*\(원\)\s*보통주식?\s*([\d,\.]+|-)", t)
    rd = re.search(r"배당기준일\s*(\d{4}-\d{2}-\d{2})", t)
    etc = t.split("기타 투자판단", 1)[1] if "기타 투자판단" in t else ""
    etc = etc.split("※ 관련공시", 1)[0]
    m = _SPECIAL_RE.search(etc)
    return {
        "div_type": kind.group(1) if kind else None,
        "dps": _num(dps.group(1)) if dps else None,
        "record_date": rd.group(1) if rd else None,
        "special_mention": m.group(0) if m else None,
    }


def ttm_dps(events: list, today: Optional[date] = None) -> dict:
    """배당결정 공시 목록(접수 오래된순, 각 parse_dividend_doc 결과 + rcept_no)으로
    TTM DPS를 계산한다. 정정공시는 같은 (기준일, 배당구분)의 마지막 공시로 대체.
    반환: {ttm_raw, ttm_adj, anchor, events, flags}. 공시를 하나도 해석하지 못하면
    ttm_raw/ttm_adj가 None(알 수 없음 — 0으로 단정하지 않는다)."""
    today = today or date.today()
    latest: dict = {}
    for e in events:
        rd = _date(e.get("record_date"))
        if rd is None or e.get("dps") is None:
            continue
        latest[(rd, e.get("div_type"))] = {**e, "rd": rd}
    valid = [e for e in latest.values() if e["dps"] > 0]
    if not valid:
        return {"ttm_raw": None, "ttm_adj": None, "anchor": None, "events": [],
                "flags": ["배당결정 공시 해석 불가"]}

    anchor = max(e["rd"] for e in valid)
    if (today - anchor).days > STALE_DAYS:
        return {"ttm_raw": 0.0, "ttm_adj": 0.0, "anchor": anchor.isoformat(), "events": [],
                "flags": [f"최근 배당기준일 {anchor.isoformat()} 이후 배당 없음"]}

    window = sorted((e for e in valid if (anchor - e["rd"]).days < TTM_DAYS),
                    key=lambda e: e["rd"])
    quarterly = [e["dps"] for e in window if e.get("div_type") == "분기배당"]
    regular_q = statistics.median(quarterly) if quarterly else None

    flags, deducted, used = [], 0.0, []
    for e in window:
        rd_s = e["rd"].isoformat()
        if e.get("special_mention"):
            flags.append(f"(a) {rd_s} {e.get('div_type')} {e['dps']:,.0f}원: 공시에 "
                         f"'{e['special_mention']}' 언급(금액은 이상치 보정에 맡김)")
        elif (e.get("div_type") == "결산배당" and regular_q
              and e["dps"] > regular_q * (1 + QUARTERLY_EXCESS_TOL)):
            excess = e["dps"] - regular_q
            deducted += excess
            flags.append(f"(b) {rd_s} 결산배당 {e['dps']:,.0f}원 중 정기 분기배당 "
                         f"{regular_q:,.0f}원 초과분 {excess:,.0f}원을 특별분으로 추정해 차감")
        used.append({"record_date": rd_s, "div_type": e.get("div_type"), "dps": e["dps"]})

    raw = round(sum(e["dps"] for e in window), 1)
    return {"ttm_raw": raw, "ttm_adj": round(raw - deducted, 1),
            "anchor": anchor.isoformat(), "events": used, "flags": flags}


def effective_dps(ttm_adj: Optional[float], avg5: Optional[float]) -> tuple[Optional[float], list]:
    """3. 이상치: TTM(특별분 차감 후)이 5년 평균의 1.5배를 넘으면 플래그 + 작은 값."""
    if ttm_adj is None:
        return None, []
    if avg5 and ttm_adj > avg5 * OUTLIER_RATIO:
        return round(min(ttm_adj, avg5), 1), [
            f"이상치: TTM {ttm_adj:,.0f}원 > 5년 평균 {avg5:,.0f}원 x {OUTLIER_RATIO} "
            f"→ 5년 평균 사용"]
    return ttm_adj, []


# --------------------------------------------------------------------- #
# 4. 배당 안정성 점수
# --------------------------------------------------------------------- #
def stability_score(history: dict) -> dict:
    """history: {연도(int): {dps, par, payout}} (DART 배당에 관한 사항). 반환:
    {score(0~100 또는 None), streak, cuts, payout, cagr, first_year, last_year, text}."""
    years = sorted(y for y, v in (history or {}).items() if v.get("dps") is not None)
    if not years:
        return {"score": None, "text": None}

    last = years[-1]
    par_ref = history[last].get("par")
    adj = {}
    for y in years:
        dps, par = history[y]["dps"], history[y].get("par")
        adj[y] = dps * par_ref / par if par and par_ref and par != par_ref else dps

    streak, expected = 0, last
    for y in reversed(years):
        if y != expected or adj[y] <= 0:
            break
        streak += 1
        expected -= 1

    cuts = 0
    for i in range(1, len(years)):
        prev, cur = adj[years[i - 1]], adj[years[i]]
        if prev <= 0 or cur >= prev * (1 - CUT_TOL):
            continue
        prevprev = adj[years[i - 2]] if i >= 2 else None
        if prevprev and prev > prevprev * SPIKE_RATIO:
            continue  # 직전 해 특별배당 급증 이후 정상화 — 감배로 보지 않음
        cuts += 1

    payout = history[last].get("payout")
    first = years[-streak] if streak else None
    cagr = None
    if streak >= 3 and adj[first] > 0:
        n = last - first
        cagr = round(((adj[last] / adj[first]) ** (1 / n) - 1) * 100, 1) if n > 0 else None

    s_streak = min(streak, 10) / 10 * 30
    s_cuts = {0: 30, 1: 20, 2: 10}.get(cuts, 0)
    if payout is None or payout <= 0:
        s_payout = 0
    elif 20 <= payout <= 60:
        s_payout = 20
    elif 10 <= payout < 20 or 60 < payout <= 80:
        s_payout = 10
    else:
        s_payout = 0
    s_cagr = 0 if cagr is None or cagr <= 0 else min(cagr, 5) / 5 * 20
    score = int(round(s_streak + s_cuts + s_payout + s_cagr))

    text = (f"{AUTO_PREFIX} 배당안정성 {score}/100 · 연속배당 {streak}년 · 감배 {cuts}회 · "
            f"배당성향 {payout if payout is not None else '-'}% · DPS CAGR "
            f"{f'{cagr}%' if cagr is not None else '-'}"
            f" ({years[0]}~{last}, 액면변경 환산)")
    return {"score": score, "streak": streak, "cuts": cuts, "payout": payout, "cagr": cagr,
            "first_year": years[0], "last_year": last, "text": text}
