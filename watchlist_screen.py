"""
watchlist_screen.py
====================
트레이딩 계획서 2장 '관심종목 선정'의 편입 유형·체크리스트·제외 조건을 코드로
판정한다. invest_point.py와 같은 원칙 — 여기서는 이미 수집된 숫자로 판정만 하고
네트워크/DB 호출은 하지 않는다(수집은 mvp_graph.screen_watchlist 노드가 담당).

사용자 확정 범위(2026-10-12):
  - 주가 신호는 모든 유형 공통으로 '52주 고점 대비 -30% 이상 하락'.
  - E(기술)·F(정책)는 정성 판단이라 편입 대상에서 제외.
  - B는 전년 동기(YoY) 비교에 필요한 분기 데이터가 없어(실측 3개 분기뿐) 그
    3개 분기 안에서 매출·영업이익이 연속 증가했는지로 판정.
  - C는 PER·PBR 과거 밴드 조건을 뺀다(과거 밴드 데이터 없음).
  - D는 실적 발표(정기보고서 접수일) 이후 주가를 일별 시세로 확인한다.
  - 체크리스트의 일평균 거래대금 조건은 뺀다.

계획서에 숫자가 없어 여기서 정한 기준(바꾸려면 아래 상수만 고치면 된다):
  - A '의견·목표가 상향': 가장 최근 목표가 또는 EPS 리비전이 상향이고 그
    리포트가 최근 REVISION_RECENT_DAYS일 안에 나왔을 것.
  - D '하락 또는 횡보': 접수일 종가 대비 현재가 상승이 POST_RELEASE_SIDEWAYS_PCT% 이하.
  - '1년 내 유동성 위기 징후 없음': 유동비율 CURRENT_RATIO_MIN% 이상으로 대신 본다.
  - '반복적 유상증자·CB': 최근 2년 희석성 발행결정이 DILUTION_MAX_COUNT건 초과.
  - 투자주의는 하루짜리 지정이라 최근 7일 내 지정분만 제외(external_sources 참조).
  - '가치함정(실적 훼손이 하락 이유)'은 자동 판정하지 않는다 — B/C/D가 이미 실적
    조건을 보고 있고, 원인 판단은 사람 몫으로 둔다.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

DRAWDOWN_THRESHOLD_PCT = -30.0
REVISION_RECENT_DAYS = 90
OP_FLAT_BAND_PCT = 10.0
POST_RELEASE_SIDEWAYS_PCT = 5.0
DEBT_RATIO_MAX = 200.0
CURRENT_RATIO_MIN = 100.0
DILUTION_MAX_COUNT = 1

TYPE_LABELS = {
    "A": "투자의견 개선",
    "B": "실적 개선 + 주가 하락",
    "C": "실적 유지 + 주가 하락",
    "D": "턴어라운드 + 주가 하락",
}


def drawdown_pct(price: Optional[dict]) -> Optional[float]:
    """현재가의 52주 고점 대비 등락률(%). 음수가 하락."""
    p, h = (price or {}).get("price"), (price or {}).get("high_52w")
    if not p or not h:
        return None
    return round((p / h - 1) * 100, 1)


def quarterly_actuals(fnguide: Optional[dict], n: int = 3) -> list:
    """분기 실측 최근 n개(오래된순). cf1002가 분기 주기면 그것을, 아니면 FnGuide
    분기 하이라이트를 쓴다(invest_point._pick_series와 같은 우선순위)."""
    fng = fnguide or {}
    cf = fng.get("cf1002") or {}
    rows = (cf.get("rows") if cf.get("freq") == "quarter" else None) or \
        fng.get("financial_highlight") or []
    return [r for r in rows if not r.get("is_estimate")][-n:]


def _pct(curr, prev) -> Optional[float]:
    if curr is None or prev in (None, 0):
        return None
    return round((curr - prev) / abs(prev) * 100, 1)


def _parse_date(s) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def type_a(revisions: Optional[dict]) -> tuple[bool, str]:
    """투자의견 개선: 가장 최근 목표가(없으면 EPS) 리비전이 상향이고 최근 리포트."""
    for label, key in (("목표가", "target_price"), ("EPS", "eps")):
        rows = [r for r in (revisions or {}).get(key) or [] if r.get("previous") is not None]
        if not rows:
            continue
        last = rows[-1]
        d = _parse_date(last.get("date"))
        if d is None or (date.today() - d).days > REVISION_RECENT_DAYS:
            continue
        if last["current"] > last["previous"]:
            return True, (f"{label} 상향 {last['previous']:,.0f}→{last['current']:,.0f}"
                          f"({last['date']} {last.get('broker', '')})")
    return False, ""


def type_b(q: list) -> tuple[bool, str]:
    """실적 개선: 실측 3개 분기 동안 매출·영업이익이 모두 연속 증가."""
    if len(q) < 3:
        return False, ""
    rev = [r.get("revenue") for r in q]
    op = [r.get("op_profit") for r in q]
    if any(v is None for v in rev + op):
        return False, ""
    if all(rev[i] > rev[i - 1] for i in (1, 2)) and all(op[i] > op[i - 1] for i in (1, 2)):
        periods = "→".join(r.get("period", "") for r in q)
        return True, (f"{periods} 매출 {rev[0]:,.0f}→{rev[1]:,.0f}→{rev[2]:,.0f}, "
                      f"영업이익 {op[0]:,.0f}→{op[1]:,.0f}→{op[2]:,.0f}(억원)")
    return False, ""


def type_c(q: list) -> tuple[bool, str]:
    """실적 유지: 직전 분기 대비 최근 분기 영업이익 증감이 ±OP_FLAT_BAND_PCT% 이내(흑자 유지)."""
    if len(q) < 2:
        return False, ""
    prev, latest = q[-2].get("op_profit"), q[-1].get("op_profit")
    if prev is None or latest is None or prev <= 0:
        return False, ""
    chg = _pct(latest, prev)
    if chg is not None and abs(chg) <= OP_FLAT_BAND_PCT:
        return True, (f"영업이익 {q[-2].get('period')} {prev:,.0f}→{q[-1].get('period')} "
                      f"{latest:,.0f}억원({chg:+.1f}%)")
    return False, ""


def type_d_fundamental(q: list) -> tuple[bool, str]:
    """턴어라운드: 직전 분기 영업적자에서 최근 분기 흑자전환 또는 적자 축소."""
    if len(q) < 2:
        return False, ""
    prev, latest = q[-2].get("op_profit"), q[-1].get("op_profit")
    if prev is None or latest is None or prev >= 0:
        return False, ""
    if latest > 0:
        kind = "흑자전환"
    elif latest > prev:
        kind = "적자축소"
    else:
        return False, ""
    return True, (f"영업이익 {kind} {q[-2].get('period')} {prev:,.0f}→"
                  f"{q[-1].get('period')} {latest:,.0f}억원")


def release_matches_latest_quarter(report_nm: Optional[str], q: list) -> bool:
    """접수일을 '실적 발표일'로 쓰려면 그 정기보고서가 최근 실측 분기의 보고서여야
    한다(예: '반기보고서 (2026.06)' ↔ 분기 '2026.06'). FnGuide가 잠정실적으로
    다음 분기를 먼저 반영한 경우 등 어긋나면 발표일을 알 수 없다고 본다."""
    if not report_nm or not q:
        return False
    m = re.search(r"\((\d{4})\.(\d{2})\)", report_nm)
    digits = re.sub(r"\D", "", str(q[-1].get("period", "")))
    return bool(m) and digits[:6] == m.group(1) + m.group(2)


def post_release_change_pct(daily_closes: list, rcept_dt: Optional[str]) -> Optional[float]:
    """접수일(이후 첫 거래일) 종가 대비 최근 종가 등락률(%)."""
    d = _parse_date(rcept_dt)
    if d is None or not daily_closes:
        return None
    after = [r for r in daily_closes if r["date"] >= d.isoformat()]
    if not after:
        return None
    return _pct(daily_closes[-1]["close"], after[0]["close"])


def evaluate_checklist(financial: Optional[dict], audit_opinions: Optional[list]) -> dict:
    """편입 체크리스트(거래대금·2군 항목 제외). 값: (통과 여부 또는 None=확인불가, 설명)."""
    fin = financial or {}
    dr, cr = fin.get("debt_ratio"), fin.get("current_ratio")
    ops = audit_opinions or []
    audit_ok = bool(ops) and all(o.replace(" ", "").startswith("적정") for o in ops)
    return {
        "부채비율": (dr <= DEBT_RATIO_MAX if dr is not None else None,
                 f"{dr}% (기준 {DEBT_RATIO_MAX:.0f}% 이하)" if dr is not None else "확인 불가"),
        "유동성": (cr >= CURRENT_RATIO_MIN if cr is not None else None,
                f"유동비율 {cr}% (기준 {CURRENT_RATIO_MIN:.0f}% 이상)" if cr is not None else "확인 불가"),
        # '부적정의견'도 '적정'을 포함하므로 부분 문자열이 아니라 접두어로 판정한다.
        "감사의견": (audit_ok if ops else None,
                 ", ".join(sorted(set(ops))) if ops else "확인 불가"),
    }


def evaluate_exclusions(stock_code: str, warnings: Optional[dict],
                        dilution_count: Optional[int]) -> list[str]:
    """제외 사유 목록(비어 있으면 제외 아님). 조회 실패(None)는 사유가 아니라
    '확인 불가'로 남겨, 편입 판정에서 통과로 치지 않도록 호출부가 처리한다."""
    code = (stock_code or "").zfill(6)
    w = warnings or {}
    reasons = []
    for key, label in (("admin", "관리종목"), ("caution", "투자주의(최근 7일)"),
                       ("warning", "투자경고"), ("risk", "투자위험")):
        if code in (w.get(key) or set()):
            reasons.append(label)
    if dilution_count is not None and dilution_count > DILUTION_MAX_COUNT:
        reasons.append(f"최근 2년 유상증자·CB 등 {dilution_count}건")
    return reasons


def finalize(matched: list, drawdown: Optional[float], checklist: dict,
             exclusions: list, unknown_sources: list) -> dict:
    """편입 여부 확정. 유형이 하나 이상 맞고, 체크리스트가 전부 통과(확인불가는
    통과 아님)하고, 제외 사유와 조회 실패가 없어야 편입."""
    failed = [f"{k}: {v[1]}" for k, v in checklist.items() if v[0] is not True]
    eligible = bool(matched) and not failed and not exclusions and not unknown_sources
    return {
        "eligible": eligible,
        "types": [m["type"] for m in matched] if eligible else [],
        "matched": matched,
        "drawdown_pct": drawdown,
        "checklist": checklist,
        "checklist_failed": failed,
        "exclusions": exclusions,
        "unknown_sources": unknown_sources,
    }


def explain(result: dict) -> str:
    """편입되지 않은 이유(로그용)."""
    if result.get("eligible"):
        return ""
    dd = result.get("drawdown_pct")
    if dd is None:
        return "52주 고점 대비 하락률 계산 불가(시세 없음)"
    if dd > DRAWDOWN_THRESHOLD_PCT:
        return f"52주 고점 대비 {dd}% (기준 {DRAWDOWN_THRESHOLD_PCT:.0f}% 이하 미달)"
    if not result.get("matched"):
        return f"52주 고점 대비 {dd}%이나 A~D 펀더멘털 조건 미충족"
    parts = [f"유형 {','.join(m['type'] for m in result['matched'])} 충족"]
    if result.get("checklist_failed"):
        parts.append("체크리스트 미통과: " + "; ".join(result["checklist_failed"]))
    if result.get("exclusions"):
        parts.append("제외 조건: " + ", ".join(result["exclusions"]))
    if result.get("unknown_sources"):
        parts.append("조회 실패: " + ", ".join(result["unknown_sources"]))
    return " / ".join(parts)


def format_alert(corp_name: Optional[str], stock_code: str, result: dict,
                 new_types: list, price: Optional[dict]) -> str:
    """텔레그램 본문. 계획서 체크리스트 첫 항목('편입 유형과 근거 숫자를 한 줄로')을
    그대로 유형별 한 줄로 만든다."""
    p = price or {}
    lines = [f"[관심종목 편입] {corp_name}({stock_code})",
             f"현재가 {p.get('price'):,.0f}원 · 52주 고점 {p.get('high_52w'):,.0f}원 "
             f"대비 {result['drawdown_pct']}%"]
    for m in result["matched"]:
        mark = " (신규)" if m["type"] in new_types else ""
        lines.append(f"· {m['type']}. {TYPE_LABELS[m['type']]}{mark}: {m['evidence']}")
    lines.append("체크: " + ", ".join(f"{k} {v[1]}" for k, v in result["checklist"].items()))
    return "\n".join(lines)
