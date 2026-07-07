"""
cross_check.py
===============
3단계: 서로 다른 소스가 준 핵심 값이 서로 부합하는지 '분석 전' 교차검증한다.

동기: 실제로 발생했던 사고 — 자체 집계 컨센서스가 리서치 목록 페이지의 조회수
컬럼을 목표주가로 오인해 '평균 목표주가 2,838원' 같은 완전히 틀린 값을 만들어낸
적이 있다. 소스 하나만 믿고 LLM에게 넘기면 이런 오류가 그대로 리포트에 실린다.
여기서는 독립된 두 소스가 있을 때만 비교하고, 하나뿐이면 검증을 생략한다
(생략은 '통과'가 아니라 '비교 불가'이며 discrepancy로 취급하지 않는다).
"""
from __future__ import annotations

from typing import Optional

TARGET_PRICE_TOLERANCE_PCT = 20.0   # 이 이상 차이나면 discrepancy


def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("원", "").strip())
    except ValueError:
        return None


def _pct_diff(a: float, b: float) -> float:
    base = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / base * 100


def check_target_price(price: Optional[dict], fnguide: Optional[dict],
                       consensus: Optional[dict]) -> dict:
    """네이버 API 컨센서스 vs FnGuide cTB15 컨센서스 vs 자체 집계(있다면) 목표주가."""
    p_naver = _num((price or {}).get("target_price_mean"))
    p_fng = _num((fnguide or {}).get("consensus", {}).get("target_price"))
    p_self = _num((consensus or {}).get("target_price_mean"))
    vals = {k: v for k, v in (("naver_api", p_naver), ("fnguide_cTB15", p_fng),
                              ("자체집계", p_self)) if v is not None}
    if len(vals) < 2:
        return {"name": "목표주가 교차검증", "ok": True,
                "detail": "비교 가능한 독립 소스 2개 미만 — 검증 생략", "values": vals}

    ks = list(vals.keys())
    mismatches = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = _pct_diff(vals[ks[i]], vals[ks[j]])
            if d > TARGET_PRICE_TOLERANCE_PCT:
                mismatches.append(
                    f"{ks[i]}({vals[ks[i]]:,.0f}) vs {ks[j]}({vals[ks[j]]:,.0f}) 차이 {d:.0f}%")
    ok = not mismatches
    detail = "; ".join(mismatches) if mismatches else (
        "일치(" + ", ".join(f"{k}={v:,.0f}" for k, v in vals.items()) + ")")
    return {"name": "목표주가 교차검증", "ok": ok, "detail": detail, "values": vals}


def check_opinion(fnguide: Optional[dict], consensus: Optional[dict]) -> dict:
    """FnGuide 투자의견 vs 자체 집계 리포트들의 투자의견 분포 다수결."""
    op_fng = ((fnguide or {}).get("consensus") or {}).get("opinion_label", "")
    dist = (consensus or {}).get("opinion_distribution") or {}
    if not op_fng or not dist:
        return {"name": "투자의견 교차검증", "ok": True,
                "detail": "비교 가능한 독립 소스 부족 — 검증 생략"}
    self_majority = max(dist, key=dist.get)
    ok = (op_fng == self_majority)
    return {"name": "투자의견 교차검증", "ok": ok,
            "detail": f"FnGuide={op_fng} / 자체집계 다수={self_majority}"}


def check_price_band(price: Optional[dict]) -> dict:
    """현재가가 52주 밴드 범위 안에 있는지(사소한 갭 허용) — 단일 소스 내부 정합성."""
    p = price or {}
    lo, hi, cur = p.get("low_52w"), p.get("high_52w"), p.get("price")
    if lo is None or hi is None or cur is None:
        return {"name": "52주 밴드 정합성", "ok": True, "detail": "데이터 부족 — 검증 생략"}
    ok = lo <= hi and (lo - hi * 0.02) <= cur <= (hi + hi * 0.02)
    return {"name": "52주 밴드 정합성", "ok": ok,
            "detail": f"현재가 {cur:,.0f} vs 밴드 {lo:,.0f}~{hi:,.0f}"}


def run_cross_check(*, price: Optional[dict], fnguide: Optional[dict],
                    consensus: Optional[dict]) -> dict:
    checks = [
        check_target_price(price, fnguide, consensus),
        check_opinion(fnguide, consensus),
        check_price_band(price),
    ]
    discrepancies = [c["detail"] for c in checks if not c["ok"]]
    return {"checks": checks, "all_ok": not discrepancies, "discrepancies": discrepancies}


def format_cross_check_block(cc: dict) -> str:
    if not cc or not cc.get("checks"):
        return "(교차검증 대상 없음)"
    lines = []
    for c in cc["checks"]:
        mark = "✅" if c["ok"] else "❌"
        lines.append(f"{mark} {c['name']}: {c['detail']}")
    return "\n".join(lines)
