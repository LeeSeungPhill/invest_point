"""
stability_score.py
===================
실적안정성(매출·영업이익 변동성) + 재무안정성(유동비율/부채비율/자기자본비율)
점수화(0~100).

실적안정성은 FnGuide annual_highlight(연간, 이미 fetch_fnguide로 수집 중이라 추가
API 불필요)를 쓴다 — DART 주요계정 API의 3개년 대신 FnGuide가 주는 4개년 실측을
쓰면 표본이 하나 더 늘어난다. 처음엔 '향후 분기(next_est/next_est2)'로 통째로
바꾸는 안도 검토했으나, 그 두 값은 컨센서스 단일 추정치일 뿐(분산을 잴 수 있는
분포가 아님)이라 '변동성'을 재대로 측정할 수 없고, WiseReport cf1002 분기
시계열은 실측이 3분기뿐이라(112610 실측 확인) 계절성 보정에 필요한 전년동기
페어조차 못 만든다 — 그래서 향후분기는 별도의 '급변 위험' 보조지표로만 반영한다
(메인 변동계수 계산에는 섞지 않음).

재무안정성은 여전히 DART 3개년(최근 확정 사업보고서 기준, dart_client.FinancialSeries)
을 쓴다 — 유동자산/유동부채/자본총계/부채총계/자산총계는 FnGuide 스크레이핑에 없다.

설계 원칙(invest_point.py와 동일):
  - 코드가 실측 숫자만으로 사칙연산한다. 데이터 부족(3개년 미만, 계정 결측)이면
    조용히 지어내지 않고 None + notes에 사유를 남긴다.
  - 업종 상대비교는 전체 피어사 데이터 수집이 아니라 가벼운 카테고리(경기민감/방어주,
    자본집약적/자본경박) 임계값 테이블로 대체한다(사용자 확정 — 피어 데이터 수집 없음).
"""
from __future__ import annotations

import re
import statistics
from typing import Optional

from invest_point import REVENUE_WEIGHT, OP_PROFIT_WEIGHT, SCORE_MAX, _growth

# --- 실적안정성 파라미터(초기값, 실사용 후 분포 보고 조정 가능) ---
CV_THRESHOLDS = {  # 변동계수(CV): (만점 기준=낮음, 0점 기준=높음)
    "경기민감": (0.30, 0.80),
    "중립": (0.15, 0.50),
    "방어주": (0.08, 0.30),
}
LOSS_YEAR_PENALTY = 20  # 적자연도(영업이익<0) 1개당 감점
EARNINGS_TREND_ADJ = 5  # 영업이익률(최신 vs 최고령) 개선/악화 추세 보정폭
MIN_ACTUAL_PERIODS = 3  # 실적안정성 계산에 필요한 최소 실측 연도(분기) 수
FORWARD_RISK_PENALTY = 10  # 향후분기(next_est->next_est2) 급변 위험 감점
FORWARD_RISK_RATE_THRESHOLD = 100.0  # 이 증감률(%) 이상이면 '급변'으로 본다

# --- 재무안정성 파라미터 ---
CURRENT_RATIO_THRESHOLDS = {  # 유동비율(%): (0점 기준, 만점 기준)
    "자본집약적": (70.0, 150.0), "중립": (100.0, 180.0), "자본경박": (120.0, 220.0),
}
DEBT_RATIO_THRESHOLDS = {  # 부채비율(%): (0점 기준=높음, 만점 기준=낮음 — 역방향)
    "자본집약적": (280.0, 110.0), "중립": (200.0, 50.0), "자본경박": (150.0, 30.0),
}
EQUITY_RATIO_THRESHOLDS = {  # 자기자본비율(%): (0점 기준, 만점 기준)
    "자본집약적": (10.0, 35.0), "중립": (20.0, 45.0), "자본경박": (30.0, 55.0),
}
FINANCIAL_TREND_ADJ = 5  # 지표(유동/부채/자기자본비율)당 개선/악화 추세 보정폭


def _period_num(label: str) -> Optional[int]:
    """'제 20 기' -> 20. 오래된->최신 정렬용."""
    m = re.search(r"(\d+)", label or "")
    return int(m.group(1)) if m else None


def _sorted_series(d: Optional[dict]) -> list:
    """{기수라벨: 금액} -> [(기수라벨, 금액), ...] 오래된 순 정렬."""
    items = [(k, v) for k, v in (d or {}).items() if _period_num(k) is not None]
    items.sort(key=lambda kv: _period_num(kv[0]))
    return items


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _linear_score(value: float, zero_at: float, full_at: float) -> int:
    """value가 zero_at이면 0점, full_at이면 만점, 사이는 선형. full_at<zero_at이면
    (부채비율처럼) 값이 작을수록 좋은 역방향 지표도 그대로 처리된다."""
    if full_at == zero_at:
        return 0
    frac = (value - zero_at) / (full_at - zero_at)
    return round(SCORE_MAX * _clip01(frac))


def _stability_series(fnguide: Optional[dict]) -> list:
    """실적안정성 전용 시계열 선택: annual_highlight(연간, 계절성 없음, FnGuide가
    보통 3~4개년 실측을 준다)을 우선하고, 실측이 부족하면 financial_highlight
    (분기, 계절성 왜곡 감수)로 대체한다. cf1002(WiseReport)는 실측 구간이 짧아
    (보통 3분기, 112610으로 실측 확인) 최후순위."""
    fnguide = fnguide or {}
    for key in ("annual_highlight", "financial_highlight"):
        rows = fnguide.get(key) or []
        actual = [r for r in rows if not r.get("is_estimate")
                 and r.get("revenue") is not None and r.get("op_profit") is not None]
        if len(actual) >= MIN_ACTUAL_PERIODS:
            return rows
    cf = (fnguide.get("cf1002") or {}).get("rows") or []
    return cf


def _forward_risk(next_est: Optional[dict], next_est2: Optional[dict]) -> dict:
    """향후분기(next_est->next_est2) 급변 위험: 흑자/적자 전환 또는 영업이익
    증감률이 FORWARD_RISK_RATE_THRESHOLD(%) 이상. 예측 두 포인트로는 '변동성'
    자체를 잴 수 없으므로(분산을 낼 분포가 아니라 컨센서스 단일값) CV 계산에
    섞지 않고, 급변 신호가 있을 때만 별도 감점+표시한다."""
    if not next_est or not next_est2:
        return {"flag": False, "penalty": 0, "note": None}
    op1, op2 = next_est.get("op_profit"), next_est2.get("op_profit")
    if op1 is None or op2 is None:
        return {"flag": False, "penalty": 0, "note": None}
    sign_flip = (op1 <= 0 < op2) or (op1 > 0 >= op2)
    rate = _growth(op2, op1)
    big_swing = rate is not None and abs(rate) >= FORWARD_RISK_RATE_THRESHOLD
    if not (sign_flip or big_swing):
        return {"flag": False, "penalty": 0, "note": None}
    p1, p2 = next_est.get("period"), next_est2.get("period")
    if sign_flip:
        note = f"{p1}→{p2} 영업이익 흑자/적자 전환 예상"
    else:
        note = f"{p1}→{p2} 영업이익 급변 예상({rate}%)"
    return {"flag": True, "penalty": -FORWARD_RISK_PENALTY, "note": note}


def build_earnings_stability(fnguide: Optional[dict], next_est: Optional[dict],
                             next_est2: Optional[dict], category: Optional[dict]) -> dict:
    """실적안정성: 매출·영업이익 변동계수(CV, 매출:영업이익=7:3 가중, FnGuide 연간
    실측 기준) + 적자연도수 페널티 + 영업이익률 추세 보정 + 향후분기 급변위험
    감점. 실측 데이터가 MIN_ACTUAL_PERIODS 미만이면 None+notes."""
    notes: list = []
    rows = _stability_series(fnguide)
    actual_rows = sorted(
        [r for r in rows if not r.get("is_estimate")
         and r.get("revenue") is not None and r.get("op_profit") is not None],
        key=lambda r: r.get("period") or "")

    if len(actual_rows) < MIN_ACTUAL_PERIODS:
        notes.append(f"실측 실적 데이터 부족({len(actual_rows)}개, "
                     f"{MIN_ACTUAL_PERIODS}개 이상 필요) — 실적안정성 계산 불가")
        return {"score": None, "notes": notes}

    rev_vals = [r["revenue"] for r in actual_rows]
    op_vals = [r["op_profit"] for r in actual_rows]

    rev_mean = statistics.mean(rev_vals)
    op_mean_abs = abs(statistics.mean(op_vals))
    if rev_mean <= 0 or op_mean_abs == 0:
        notes.append("매출/영업이익 평균이 0 이하 — 변동계수 계산 불가")
        return {"score": None, "notes": notes}

    rev_cv = statistics.stdev(rev_vals) / rev_mean
    op_cv = statistics.stdev(op_vals) / op_mean_abs
    blended_cv = REVENUE_WEIGHT * rev_cv + OP_PROFIT_WEIGHT * op_cv

    cyclicality = (category or {}).get("cyclicality", "중립")
    low, high = CV_THRESHOLDS.get(cyclicality, CV_THRESHOLDS["중립"])
    cv_score = round(SCORE_MAX * _clip01((high - blended_cv) / (high - low)))

    loss_periods = sum(1 for v in op_vals if v < 0)
    loss_penalty = -LOSS_YEAR_PENALTY * loss_periods

    oldest_margin = (op_vals[0] / rev_vals[0]) if rev_vals[0] else None
    latest_margin = (op_vals[-1] / rev_vals[-1]) if rev_vals[-1] else None
    trend_adj = 0
    if oldest_margin is not None and latest_margin is not None and oldest_margin != latest_margin:
        trend_adj = EARNINGS_TREND_ADJ if latest_margin > oldest_margin else -EARNINGS_TREND_ADJ

    forward_risk = _forward_risk(next_est, next_est2)

    score = round(max(0, min(SCORE_MAX,
                             cv_score + loss_penalty + trend_adj + forward_risk["penalty"])))
    if loss_periods:
        notes.append(f"최근 {len(actual_rows)}개 기간 중 {loss_periods}개 영업적자")
    if forward_risk["note"]:
        notes.append(f"향후분기 급변위험: {forward_risk['note']}")

    return {"score": score, "cyclicality": cyclicality, "revenue_cv": round(rev_cv, 3),
            "op_profit_cv": round(op_cv, 3), "blended_cv": round(blended_cv, 3),
            "n_periods": len(actual_rows), "loss_periods": loss_periods,
            "trend_adj": trend_adj, "forward_risk": forward_risk["flag"], "notes": notes}


def _ratio_series(num_series: list, den_series: list, mul: float = 100.0) -> list:
    den_by_period = dict(den_series)
    out = []
    for period, num in num_series:
        den = den_by_period.get(period)
        if den:
            out.append((period, num / den * mul))
    return out


def _debt_ratio_rating(debt_ratio: float, capital_intensity: str) -> str:
    """부채비율을 업종군(자본집약도) 임계값과 비교해 안정/적정/위험 3단계로 판정.
    DEBT_RATIO_THRESHOLDS의 만점 기준(더 낮음, 더 좋음) 이하면 안정, 0점 기준
    (더 높음, 더 나쁨) 이상이면 위험, 그 사이는 적정 — 새 임계값을 따로 두지 않고
    이미 점수화에 쓰는 업종별 기준을 그대로 재사용해 두 표시가 항상 일치하게 한다."""
    zero_at, full_at = DEBT_RATIO_THRESHOLDS.get(capital_intensity, DEBT_RATIO_THRESHOLDS["중립"])
    if debt_ratio <= full_at:
        return "안정"
    if debt_ratio >= zero_at:
        return "위험"
    return "적정"


def build_financial_stability(fs, category: Optional[dict]) -> dict:
    """재무안정성: 최신 연도 유동비율/부채비율/자기자본비율(업종 카테고리별 임계값)
    + 3개년 추세(최신 vs 최고령 방향) 보정. 계정 결측이면 None+notes."""
    notes: list = []
    ca = _sorted_series(getattr(fs, "current_assets", None))
    cl = _sorted_series(getattr(fs, "current_liabilities", None))
    eq = _sorted_series(getattr(fs, "equity", None))
    li = _sorted_series(getattr(fs, "liabilities", None))
    at = _sorted_series(getattr(fs, "assets", None))

    if not (ca and cl and eq and li and at):
        notes.append("재무상태표 계정(유동자산/유동부채/자본총계/부채총계/자산총계) "
                     "결측 — 재무안정성 계산 불가")
        return {"score": None, "notes": notes}

    current_ratio_series = _ratio_series(ca, cl)
    debt_ratio_series = _ratio_series(li, eq)
    equity_ratio_series = _ratio_series(eq, at)

    if not (current_ratio_series and debt_ratio_series and equity_ratio_series):
        notes.append("비율 계산 가능한 공통 기간 없음 — 재무안정성 계산 불가")
        return {"score": None, "notes": notes}

    capital_intensity = (category or {}).get("capital_intensity", "중립")
    cr_zero, cr_full = CURRENT_RATIO_THRESHOLDS.get(capital_intensity, CURRENT_RATIO_THRESHOLDS["중립"])
    dr_zero, dr_full = DEBT_RATIO_THRESHOLDS.get(capital_intensity, DEBT_RATIO_THRESHOLDS["중립"])
    er_zero, er_full = EQUITY_RATIO_THRESHOLDS.get(capital_intensity, EQUITY_RATIO_THRESHOLDS["중립"])

    latest_cr = current_ratio_series[-1][1]
    latest_dr = debt_ratio_series[-1][1]
    latest_er = equity_ratio_series[-1][1]

    cr_score = _linear_score(latest_cr, cr_zero, cr_full)
    dr_score = _linear_score(latest_dr, dr_zero, dr_full)
    er_score = _linear_score(latest_er, er_zero, er_full)
    base_score = round((cr_score + dr_score + er_score) / 3)

    trend_adj = 0
    if len(current_ratio_series) >= 2 and current_ratio_series[-1][1] != current_ratio_series[0][1]:
        trend_adj += FINANCIAL_TREND_ADJ if current_ratio_series[-1][1] > current_ratio_series[0][1] else -FINANCIAL_TREND_ADJ
    if len(debt_ratio_series) >= 2 and debt_ratio_series[-1][1] != debt_ratio_series[0][1]:
        trend_adj += FINANCIAL_TREND_ADJ if debt_ratio_series[-1][1] < debt_ratio_series[0][1] else -FINANCIAL_TREND_ADJ
    if len(equity_ratio_series) >= 2 and equity_ratio_series[-1][1] != equity_ratio_series[0][1]:
        trend_adj += FINANCIAL_TREND_ADJ if equity_ratio_series[-1][1] > equity_ratio_series[0][1] else -FINANCIAL_TREND_ADJ

    score = round(max(0, min(SCORE_MAX, base_score + trend_adj)))

    return {"score": score, "capital_intensity": capital_intensity,
            "current_ratio": round(latest_cr, 1), "debt_ratio": round(latest_dr, 1),
            "debt_ratio_rating": _debt_ratio_rating(latest_dr, capital_intensity),
            "equity_ratio": round(latest_er, 1), "trend_adj": trend_adj, "notes": notes}


def format_stability_block(earnings: dict, financial: dict) -> str:
    """analyze 노드 프롬프트에 넣을 사람이 읽기 쉬운 텍스트 블록."""
    lines = []
    if earnings.get("score") is not None:
        lines.append(
            f"- 실적안정성: {earnings['score']} / {SCORE_MAX} "
            f"(업종 {earnings['cyclicality']}, 변동계수 {earnings['blended_cv']}"
            f"[매출{earnings['revenue_cv']}/영업이익{earnings['op_profit_cv']}], "
            f"최근{earnings['n_periods']}개 기간 중 적자 {earnings['loss_periods']}회, "
            f"추세보정 {earnings['trend_adj']:+d}점"
            f"{', 향후분기 급변위험 반영' if earnings.get('forward_risk') else ''})")
    else:
        lines.append("- 실적안정성: 자료상 확인 불가"
                     + (f" ({'; '.join(earnings.get('notes', []))})" if earnings.get("notes") else ""))

    if financial.get("score") is not None:
        lines.append(
            f"- 재무안정성: {financial['score']} / {SCORE_MAX} "
            f"(업종 {financial['capital_intensity']}, 유동비율 {financial['current_ratio']}%, "
            f"부채비율 {financial['debt_ratio']}%[업종 대비 {financial['debt_ratio_rating']}], "
            f"자기자본비율 {financial['equity_ratio']}%, 추세보정 {financial['trend_adj']:+d}점)")
    else:
        lines.append("- 재무안정성: 자료상 확인 불가"
                     + (f" ({'; '.join(financial.get('notes', []))})" if financial.get("notes") else ""))
    return "\n".join(lines)
