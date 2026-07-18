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


def build_value_signal(price: dict, growth: dict) -> dict:
    """가치 관점: 예상실적 상승 + 주가가 52주 밴드 하단인지."""
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

    return {
        "band_position": band_pos,
        "low_band_threshold": LOW_BAND_THRESHOLD,
        "est_rising": est_rising,
        "is_low_band": is_low_band,
        "target_upside_pct": price.get("target_upside_pct") if price else None,
        "per": price.get("per") if price else None,
        "cns_per": price.get("cns_per") if price else None,
        "signal": signal,
        "notes": notes,
    }


def build_invest_point(fnguide: Optional[dict], price: Optional[dict]) -> dict:
    """성장 + 가치 시그널을 합쳐 정량 투자포인트를 만든다. LLM 프롬프트에
    그대로 제시."""
    fnguide = fnguide or {}
    growth = build_growth_signal(fnguide)
    value = build_value_signal(price or {}, growth)
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
