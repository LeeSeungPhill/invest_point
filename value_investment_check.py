"""
value_investment_check.py
==========================
4단계 이후 보조 체크: invest_point.build_value_signal()의 quality_signal이
True인 종목만, 사용자가 제공한 가치주 점검 체크리스트("이 종목이 싼 이유가
일시적인가 구조적인가", "싼 상태를 끝낼 촉매가 있는가")를 정리해
analysis_history.value_invest에 저장할 텍스트를 만든다(실제 LLM 호출/프롬프트
조립은 mvp_graph.check_value_investment가 담당 — 여기는 그 전 단계인
'코드로 계산 가능한 정량 통계'만 순수 함수로 만든다. invest_point.py/
stability_score.py와 같은 이유 — LLM은 숫자를 새로 만들지 않고 여기서
계산한 값만 인용한다).

범위(사용자 확정 MVP): 자본시장연구원/산업연구원/KDI/관세청/한국은행 BSI 등
macro·정책 자료는 구조화 API가 없어 자동화하지 않는다. 대신:
  - 한경컨센서스(hankyung_consensus.py)의 목표주가/EPS '현재/이전' 이력으로
    추정치 리비전 방향·리포트 편차·커버리지 증감을 여기서 코드로 계산한다.
  - 산업 사이클 위치/구조적 쇠퇴 신호/경쟁 구도/매크로 민감도는 이미 RAG로
    수집된 사업보고서 청크에 실제 근거가 있을 때만 LLM이 인용하며 서술하게
    하고(mvp_graph 프롬프트), 없으면 '자료상 확인 불가'라고 쓰게 한다 —
    analyze() 노드와 같은 인용 규율.
"""
from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from typing import Optional


def _parse_date(s: Optional[str]):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def summarize_revisions(rows: list[dict]) -> dict:
    """한경컨센서스 목표주가/EPS 이력(오래된순, {date,current,previous})에서
    방향(상향/하향/유지) 건수·최근 3건 편차·최근 6개월 대 이전 6개월 리포트
    건수(커버리지 증감 근사치)를 계산한다. rows가 비어 있으면 n=0만 반환한다."""
    if not rows:
        return {"n": 0}

    comparable = [r for r in rows if r.get("previous") is not None]
    ups = sum(1 for r in comparable if r["current"] > r["previous"])
    downs = sum(1 for r in comparable if r["current"] < r["previous"])
    flats = len(comparable) - ups - downs

    latest_direction = None
    if comparable:
        last = comparable[-1]
        latest_direction = ("상향" if last["current"] > last["previous"]
                            else "하향" if last["current"] < last["previous"] else "유지")

    recent = rows[-3:]
    dispersion_pct = None
    if len(recent) >= 2:
        vals = [r["current"] for r in recent]
        m = statistics.mean(vals)
        if m:
            dispersion_pct = round((max(vals) - min(vals)) / m * 100, 1)

    today = date.today()
    def _in_window(r, start_days, end_days):
        d = _parse_date(r.get("date"))
        if d is None:
            return False
        age = (today - d).days
        return end_days <= age < start_days

    coverage_recent_6m = sum(1 for r in rows if _in_window(r, 10 ** 6, 180))
    coverage_prior_6m = sum(1 for r in rows if _in_window(r, 365, 180))

    return {
        "n": len(rows), "up": ups, "down": downs, "flat": flats,
        "latest_direction": latest_direction,
        "recent_dispersion_pct": dispersion_pct,
        "coverage_recent_6m": coverage_recent_6m,
        "coverage_prior_6m": coverage_prior_6m,
    }


def format_revision_block(revisions: Optional[dict]) -> str:
    """analyze()류 프롬프트에 그대로 넣을 텍스트 블록. LLM은 이 숫자를 그대로
    인용만 하고 새로 계산하면 안 된다(mvp_graph 프롬프트 규칙에서 강제)."""
    revisions = revisions or {}
    tp = summarize_revisions(revisions.get("target_price") or [])
    eps = summarize_revisions(revisions.get("eps") or [])
    lines = []

    if tp.get("n"):
        line = (f"- 한경컨센서스 목표주가 리비전(최근 {tp['n']}건): "
                f"상향 {tp['up']}건 / 하향 {tp['down']}건 / 유지 {tp['flat']}건, "
                f"가장 최근 방향: {tp.get('latest_direction') or '불명'}")
        if tp.get("recent_dispersion_pct") is not None:
            line += f", 최근 3건 목표주가 편차 {tp['recent_dispersion_pct']}%"
        lines.append(line)
        lines.append(f"- 목표주가 리포트 커버리지: 최근 6개월 {tp['coverage_recent_6m']}건 "
                     f"vs 이전 6개월 {tp['coverage_prior_6m']}건")
    else:
        lines.append("- 한경컨센서스 목표주가 리비전 이력: 자료상 확인 불가(검색 결과 없음)")

    if eps.get("n"):
        lines.append(f"- 한경컨센서스 EPS 추정치 리비전(최근 {eps['n']}건): "
                     f"상향 {eps['up']}건 / 하향 {eps['down']}건 / 유지 {eps['flat']}건, "
                     f"가장 최근 방향: {eps.get('latest_direction') or '불명'}")
    else:
        lines.append("- 한경컨센서스 EPS 추정치 리비전 이력: 자료상 확인 불가(검색 결과 없음)")

    return "\n".join(lines)
