"""
invest_point.py
================
2단계: 성장 관점 + 가치 관점 정량 시그널.

설계 원칙(mvp_graph.py와 동일):
  - 여기서는 코드가 fnguide_sources/external_sources가 준 실측·추정 숫자만 가지고
    사칙연산을 한다. LLM은 이 결과를 '해석'만 하고 숫자를 새로 만들지 않는다.
  - 데이터가 부족하면 조용히 지어내지 않고 해당 필드를 None + notes에 사유를 남긴다.

판단 기준:
  성장(growth)  : 최근 실적 대비 다음 예상(E) 영업이익 방향(증가/둔화/턴어라운드/역성장).
                  분기 하이라이트가 있으면 분기 기준, 없으면 연간 기준, 그것도 없으면
                  cF1002 next_earnings(분기 우선)로 대체.
  가치(value)   : 52주 밴드 내 주가 위치(band_position, 0=최저~1=최고)가 낮은데
                  (<= LOW_BAND_THRESHOLD) 예상 영업이익이 직전 실적보다 높으면(est_rising)
                  '실적 상승 + 주가 하단' 시그널(signal=True).

(성장점수/가치점수는 실사용 결과 제거했다 — 대신 아래 growth/valuation의 raw 필드
 (growth_trend/op_yoy_forward/value_signal/band_position/target_upside_pct)를
 그대로 리포트에 노출한다. SCORE_MAX/REVENUE_WEIGHT/OP_PROFIT_WEIGHT는
 stability_score.py의 실적안정성 계산(변동계수 가중평균)이 계속 재사용한다.)
"""
from __future__ import annotations

from typing import Optional

LOW_BAND_THRESHOLD = 0.4  # 52주 밴드 하위 40% 이내를 '하단'으로 본다

SCORE_MAX = 100          # stability_score.py가 재사용(실적/재무안정성 점수 스케일)
REVENUE_WEIGHT = 0.7     # stability_score.py가 재사용(변동계수 가중평균 시 매출 가중치)
OP_PROFIT_WEIGHT = 0.3   # stability_score.py가 재사용(변동계수 가중평균 시 영업이익 가중치)


def _growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev in (None, 0):
        return None
    return round((curr - prev) / abs(prev) * 100, 1)


_FREQ_LABEL = {"quarter": "분기", "annual": "연간", "none": "알수없음"}


def _pick_series(fnguide: dict) -> tuple[list, str]:
    """성장률 계산은 반드시 '같은 주기'의 실측·추정이 나란히 있는 계열만 쓴다.
    연간 실적과 분기 추정을 섞어 비교하면 규모가 달라 증감률이 왜곡되므로,
    cf1002(WiseReport, 실측+추정 동일 주기)를 우선하고 없을 때만 FnGuide
    하이라이트 표(분기/연간 각각 단독, 서로 안 섞음)로 대체한다."""
    cf1002 = fnguide.get("cf1002") or {}
    rows = cf1002.get("rows") or []
    if len(rows) >= 2 and any(r.get("is_estimate") for r in rows):
        return rows, _FREQ_LABEL.get(cf1002.get("freq"), "분기")

    qtr = fnguide.get("financial_highlight") or []
    if len(qtr) >= 2 and any(r.get("is_estimate") for r in qtr):
        return qtr, "분기"
    ann = fnguide.get("annual_highlight") or []
    if len(ann) >= 2 and any(r.get("is_estimate") for r in ann):
        return ann, "연간"
    # 추정치가 전혀 없으면 추이 표시용으로만 실측 시계열을 반환(성장 판단은 불가)
    return rows or qtr or ann, _FREQ_LABEL.get(cf1002.get("freq"), "분기")


def build_growth_signal(fnguide: dict) -> dict:
    """실적 시나리오의 성장성 판단. 반환: {basis, latest, prev, next_est,
    op_yoy_actual, op_yoy_forward, turn_to_profit, trend, notes}."""
    notes: list = []
    rows, basis = _pick_series(fnguide)
    actual_rows = [r for r in rows if not r.get("is_estimate")]
    est_rows = [r for r in rows if r.get("is_estimate")]

    if not actual_rows:
        return {"basis": basis, "latest": None, "prev": None, "next_est": None,
                "op_yoy_actual": None, "op_yoy_forward": None,
                "turn_to_profit": None, "trend": None, "actual_rows": [],
                "notes": ["실적 실측치 없음 — 자료상 확인 불가"]}

    latest = actual_rows[-1]
    prev = actual_rows[-2] if len(actual_rows) >= 2 else None
    next_est = est_rows[0] if est_rows else None
    next_est2 = est_rows[1] if len(est_rows) >= 2 else None

    op_yoy_actual = _growth(latest.get("op_profit"), prev.get("op_profit")) if prev else None
    op_yoy_forward = _growth(next_est.get("op_profit"), latest.get("op_profit")) if next_est else None
    if next_est is None:
        notes.append("예상(E) 실적 없음 — 자료상 확인 불가")
    if prev is None:
        notes.append("직전 실적 없음(비교 불가) — 자료상 확인 불가")

    turn_to_profit = None
    if latest.get("op_profit") is not None and next_est and next_est.get("op_profit") is not None:
        turn_to_profit = latest["op_profit"] <= 0 < next_est["op_profit"]

    trend = None
    if op_yoy_forward is not None:
        if turn_to_profit:
            trend = "실적 턴어라운드(적자→흑자 전환 예상)"
        elif op_yoy_forward > 0 and (op_yoy_actual is None or op_yoy_actual > 0):
            trend = "성장 가속" if (op_yoy_actual is not None and op_yoy_forward > op_yoy_actual) \
                else "성장 지속"
        elif op_yoy_forward > 0 >= (op_yoy_actual or 0):
            trend = "실적 개선(저점 통과 추정)"
        elif op_yoy_forward <= 0:
            trend = "역성장/둔화"

    return {
        "basis": basis,
        "latest": latest, "prev": prev, "next_est": next_est, "next_est2": next_est2,
        "op_yoy_actual": op_yoy_actual, "op_yoy_forward": op_yoy_forward,
        "turn_to_profit": turn_to_profit, "trend": trend,
        "actual_rows": actual_rows,
        "notes": notes,
    }


LISTED_YEARS_THRESHOLD = 5     # analysis_history.value_signal 보조조건: 상장 후 경과년수
DEEP_UPSIDE_THRESHOLD_PCT = 50  # analysis_history.value_signal 보조조건: 목표주가 상승여력
# analysis_history.value_signal 보조조건: 최근 실적 개선 관찰 기간 — 현재년도
# 기준 과거 실측 RECENT_EARNINGS_PAST_YEARS개년 + 예상실적 RECENT_EARNINGS_
# FUTURE_YEARS개년(사용자 요청, 기존 '실측 3개년'에서 변경).
RECENT_EARNINGS_PAST_YEARS = 2
RECENT_EARNINGS_FUTURE_YEARS = 1
# analysis_history.value_signal 보조조건: 매출액 상승 관찰 기간 — 현재년도 기준
# 과거 실측 REVENUE_PAST_YEARS개년 + 예상실적 REVENUE_FUTURE_YEARS개년(사용자
# 요청, 기존 '실측 5개년'에서 변경 — 실측 개수만 4로 줄고 예상 1개년이 더해져
# 총 관찰 기간 자체는 5개년으로 동일).
REVENUE_PAST_YEARS = 4
REVENUE_FUTURE_YEARS = 1


def _annual_actual_series(annual_highlight: Optional[list]) -> list:
    """annual_highlight(실측+추정, 오래된순)에서 실측 연도만, 오래된 순서 그대로."""
    if not annual_highlight:
        return []
    return [r for r in annual_highlight if not r.get("is_estimate")]


def _annual_estimate_series(annual_highlight: Optional[list]) -> list:
    """annual_highlight(실측+추정, 오래된순)에서 추정 연도만, 가까운 미래부터."""
    if not annual_highlight:
        return []
    return [r for r in annual_highlight if r.get("is_estimate")]


def _rising_or_none(vals: list) -> Optional[bool]:
    """모든 값이 있어야 판단 가능. 하나라도 비면 None(불명), 값이 다 있으면
    연속 증가 여부(bool)를 반환한다."""
    if any(v is None for v in vals):
        return None
    return all(vals[i] > vals[i - 1] for i in range(1, len(vals)))


def _revenue_consistently_rising(actual: list, estimate: list) -> Optional[bool]:
    """매출액 상승 관찰: 현재년도 기준 과거 REVENUE_PAST_YEARS개년(실측) +
    예상실적 REVENUE_FUTURE_YEARS개년(추정 중 가장 가까운 해)을 합한 기간
    동안 매출액이 예외 없이 연속 증가했는지(사용자 요청 — 기존 '실측
    5개년'에서 '실측 과거 4개년+예상 1개년'으로 변경). 필요한 연도 수(실측
    4개년/예상 1개년)를 못 채우거나 값이 비어 있으면 None(불명)을 반환한다."""
    recent_actual = actual[-REVENUE_PAST_YEARS:]
    recent_est = estimate[:REVENUE_FUTURE_YEARS]
    if len(recent_actual) < REVENUE_PAST_YEARS or len(recent_est) < REVENUE_FUTURE_YEARS:
        return None
    recent = recent_actual + recent_est
    return _rising_or_none([r.get("revenue") for r in recent])


def _recent_earnings_rising(actual: list, estimate: list) -> Optional[bool]:
    """최근 실적 개선 관찰: 현재년도 기준 과거 RECENT_EARNINGS_PAST_YEARS개년
    (실측) + 예상실적 RECENT_EARNINGS_FUTURE_YEARS개년(추정 중 가장 가까운
    해)을 합한 기간 동안 영업이익 '또는' 당기순이익 중 하나라도 예외 없이
    연속 증가하면 True('실적이 좋아지는' 조건 — 사용자 요청으로 실측 3개년
    전부+AND에서, 실측 과거 2개년+예상 1개년+OR로 변경). 한쪽 지표만 값이
    있어도(다른 쪽이 결측이어도) 있는 지표만으로 판단한다 — 둘 다 결측이거나
    필요한 연도 수(실측 2개년/예상 1개년)를 못 채우면 None(불명)을 반환한다.
    적자 여부 자체는 안 보고 방향(증가 추세)만 본다 — 적자 배제는 이 조건에서
    의도적으로 뺐다(사용자 요청)."""
    recent_actual = actual[-RECENT_EARNINGS_PAST_YEARS:]
    recent_est = estimate[:RECENT_EARNINGS_FUTURE_YEARS]
    if (len(recent_actual) < RECENT_EARNINGS_PAST_YEARS
            or len(recent_est) < RECENT_EARNINGS_FUTURE_YEARS):
        return None
    recent = recent_actual + recent_est
    op_rising = _rising_or_none([r.get("op_profit") for r in recent])
    ni_rising = _rising_or_none([r.get("net_profit") for r in recent])
    if op_rising is None and ni_rising is None:
        return None
    return bool(op_rising) or bool(ni_rising)


def build_value_signal(price: dict, growth: dict, annual_highlight: Optional[list] = None) -> dict:
    """가치 관점: 예상실적 상승 + 주가가 52주 밴드 하단인지(signal — LLM 리포트
    [가치 포지션]/scenario_check.py가 그대로 참조하는 기존 정의, 변경하지 않음).

    quality_signal은 여기에 얹는 별개 시그널이다: 상장 5년 이상 + 매출액 상승
    (현재년도 기준 과거 REVENUE_PAST_YEARS개년 실측 + 예상 REVENUE_FUTURE_YEARS
    개년을 합해 연속 상승) + 최근 실적 개선(현재년도 기준 과거 RECENT_EARNINGS_
    PAST_YEARS개년 실측 + 예상 RECENT_EARNINGS_FUTURE_YEARS개년을 합해 영업이익
    또는 당기순이익 중 하나라도 연속 상승) + 목표주가 상승여력 50% 이상을 모두
    만족하면 True. 목표주가 상승여력은 invest_mng.high_price(사용자 지정
    목표가, price['mng_target_upside_pct']로 전달됨)가 있으면 그것을 컨센서스
    목표주가(target_upside_pct)보다 우선 쓴다(사용자 요청) — target_upside_pct
    필드 자체는 리포트/DB의 다른 곳(투자포인트 표시, 컨센서스 대비 등)에서
    그대로 쓰이므로 컨센서스 값 그대로 유지하고, quality_signal 판단에서만
    갈아 끼운다. signal과 달리 LLM 프롬프트/scenario_check는 이 필드를
    보지 않는다 —
    analysis_history.save_run()이 DB의 value_signal 컬럼 값을 만들 때 signal
    대신 이 값을 그대로 쓴다(사용자 요청: 기존 조건을 '상장 5년+, 매출 지속
    상승, 최근 실적(영업이익·순이익) 상승, 목표가 대비 50%+ 구간'으로 대체).
    기존 signal 자체의 정의(밴드+성장 AND)는 그대로 둔 이유는, 이미 프롬프트
    규칙5·scenario_check.check_band_threshold가 '정확히 이 두 조건의 AND'라고
    전제하고 있어 여기에 다른 정의를 섞으면 그 검증들이 오탐하기 때문이다."""
    notes: list = []
    band_pos = price.get("band_position") if price else None
    op_yoy_forward = growth.get("op_yoy_forward")

    if band_pos is None:
        notes.append("52주 밴드 계산 불가(시세 데이터 부족) — 자료상 확인 불가")
    if op_yoy_forward is None:
        notes.append("예상실적 방향 불명 — 자료상 확인 불가")

    est_rising = op_yoy_forward is not None and op_yoy_forward > 0
    is_low_band = band_pos is not None and band_pos <= LOW_BAND_THRESHOLD
    signal = bool(est_rising and is_low_band)

    target_upside_pct = price.get("target_upside_pct") if price else None
    mng_target_upside_pct = price.get("mng_target_upside_pct") if price else None
    # quality_signal 판단용 상승여력: invest_mng.high_price가 있으면 그것을
    # 우선하고(사용자 요청), 없으면 기존 컨센서스 목표주가로 대체한다.
    deep_upside_pct_used = (mng_target_upside_pct if mng_target_upside_pct is not None
                            else target_upside_pct)
    deep_upside_source = ("invest_mng" if mng_target_upside_pct is not None
                          else "consensus" if target_upside_pct is not None else None)
    years_listed = price.get("years_listed") if price else None
    actual = _annual_actual_series(annual_highlight)
    estimate = _annual_estimate_series(annual_highlight)
    listed_5y = years_listed is not None and years_listed >= LISTED_YEARS_THRESHOLD
    revenue_up = _revenue_consistently_rising(actual, estimate)
    earnings_rising = _recent_earnings_rising(actual, estimate)
    deep_upside = (deep_upside_pct_used is not None
                  and deep_upside_pct_used >= DEEP_UPSIDE_THRESHOLD_PCT)
    quality_signal = bool(listed_5y and revenue_up and earnings_rising and deep_upside)

    return {
        "band_position": band_pos,
        "low_band_threshold": LOW_BAND_THRESHOLD,
        "est_rising": est_rising,
        "is_low_band": is_low_band,
        "target_upside_pct": target_upside_pct,
        "per": price.get("per") if price else None,
        "cns_per": price.get("cns_per") if price else None,
        "signal": signal,
        "years_listed": years_listed,
        "listed_5y": listed_5y,
        "revenue_consistently_rising": revenue_up,
        "recent_earnings_rising": earnings_rising,
        "deep_upside_pct_used": deep_upside_pct_used,
        "deep_upside_source": deep_upside_source,
        "deep_upside_50pct": deep_upside,
        "quality_signal": quality_signal,
        "notes": notes,
    }


def explain_quality_signal(valuation: dict) -> str:
    """quality_signal이 False인 이유를 사람이 읽을 수 있는 문장으로 설명한다
    (호출부인 mvp_graph.build_invest_point가 로그로 남긴다 — 사용자 요청).
    quality_signal이 True면 빈 문자열을 반환한다. 조건별로 '판단 결과 False'와
    '판단 자체가 불가(자료 부족)'를 구분해서 알려준다 — 둘은 원인이 달라서
    (전자는 실적/매출 방향이 실제로 안 좋은 것, 후자는 데이터가 없어서 아예
    확인이 안 된 것) 뭉뚱그리면 어디를 봐야 할지 알 수 없다."""
    if valuation.get("quality_signal"):
        return ""

    reasons = []
    years_listed = valuation.get("years_listed")
    if not valuation.get("listed_5y"):
        reasons.append(f"상장경과 {years_listed}년 < {LISTED_YEARS_THRESHOLD}년" if years_listed is not None
                       else "상장일 확인 불가")

    revenue_up = valuation.get("revenue_consistently_rising")
    if revenue_up is not True:
        reasons.append(f"매출액 상승(과거{REVENUE_PAST_YEARS}+예상{REVENUE_FUTURE_YEARS}개년) 미충족"
                       if revenue_up is False else "매출액 상승 여부 판단 불가(자료 부족)")

    earnings_up = valuation.get("recent_earnings_rising")
    if earnings_up is not True:
        reasons.append(f"최근실적(영업이익/순이익) 상승(과거{RECENT_EARNINGS_PAST_YEARS}+"
                       f"예상{RECENT_EARNINGS_FUTURE_YEARS}개년) 미충족" if earnings_up is False
                       else "최근실적 상승 여부 판단 불가(자료 부족)")

    upside_pct = valuation.get("deep_upside_pct_used")
    upside_source = valuation.get("deep_upside_source")
    if not valuation.get("deep_upside_50pct"):
        reasons.append(
            f"목표가 상승여력 {upside_pct}%({upside_source}) < {DEEP_UPSIDE_THRESHOLD_PCT}%"
            if upside_pct is not None else "목표가 상승여력 확인 불가")

    return "; ".join(reasons)


def build_invest_point(fnguide: Optional[dict], price: Optional[dict]) -> dict:
    """성장 + 가치 시그널을 합쳐 정량 투자포인트를 만든다. LLM 프롬프트에
    그대로 제시."""
    fnguide = fnguide or {}
    growth = build_growth_signal(fnguide)
    value = build_value_signal(price or {}, growth, fnguide.get("annual_highlight"))
    return {"growth": growth, "valuation": value, "signal": value["signal"]}


def format_invest_point_block(ip: dict) -> str:
    """analyze 노드 프롬프트에 넣을 사람이 읽기 쉬운 텍스트 블록."""
    g, v = ip["growth"], ip["valuation"]
    lines = [f"(기준: {g['basis']})"]

    def _fmt_row(label, row):
        if not row:
            return f"- {label}: 자료 없음"
        return (f"- {label} {row.get('period','')}"
                f"{'(E)' if row.get('is_estimate') else ''}: "
                f"매출 {row.get('revenue')}, 영업이익 {row.get('op_profit')}, "
                f"순이익 {row.get('net_profit')} (단위 억원)")

    lines.append(_fmt_row("직전실적", g["latest"]))
    if g["prev"]:
        lines.append(_fmt_row("그전실적", g["prev"]))
    lines.append(_fmt_row("다음예상", g["next_est"]))
    if g.get("next_est2"):
        lines.append(_fmt_row("그다음예상", g["next_est2"]))
    lines.append(f"- 영업이익 증감률(직전 대비 예상): "
                f"{g['op_yoy_forward']}%" if g["op_yoy_forward"] is not None else
                "- 영업이익 증감률(직전 대비 예상): 자료상 확인 불가")
    lines.append(f"- 성장 판단: {g['trend'] or '자료상 확인 불가'}")

    if v["band_position"] is not None:
        lines.append(f"- 52주 밴드 내 위치: {v['band_position']*100:.1f}% "
                     f"(0%=52주최저, 100%=52주최고, 하단 기준 {v['low_band_threshold']*100:.0f}% 이하)")
    else:
        lines.append("- 52주 밴드 내 위치: 자료상 확인 불가")
    if v["target_upside_pct"] is not None:
        lines.append(f"- 컨센서스 목표주가 대비 상승여력: {v['target_upside_pct']}%")
    lines.append(f"- 가치 시그널(예상실적 상승 & 주가 하단 동시 충족): "
                f"{'예' if v['signal'] else '아니오'}")

    notes = g.get("notes", []) + v.get("notes", [])
    if notes:
        lines.append("- 데이터 공백: " + "; ".join(notes))
    return "\n".join(lines)
