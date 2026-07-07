"""
scenario_check.py
==================
3단계 두 번째 검증 레이어: citation.py가 '청크 인용이 실재하고 그 청크와
관련 있는지'(그라운딩)를 본다면, 여기서는 LLM 리포트가

  1) invest_point.py가 계산한 성장 판단/가치 시그널과 실제로 같은 방향을 말하는지
     (예: 시그널이 '아니오'인데 '충족'이라고 쓰면 심각한 오류)
  2) 프롬프트에 제시된 적 없는 파생 수치(YoY%, 배수 등)를 새로 지어내지 않았는지

를 대조한다. 둘 다 휴리스틱(완벽한 사실검증 아님)이며, 불일치는 사람이
재확인하거나(플래그) 1회 자동 재생성의 트리거로 쓰인다.
"""
from __future__ import annotations

import re
from typing import Optional

_GROWTH_UP_KW = ("성장 가속", "성장 지속", "실적 개선", "턴어라운드", "흑자 전환")
_GROWTH_DOWN_KW = ("역성장", "둔화")

_SIGNAL_YES_KW = ("시그널: 예", "시그널 충족", "동시 충족", "성립")
_SIGNAL_NO_KW = ("시그널: 아니오", "미충족", "불성립")

_SECTION = re.compile(r"\[([^\]]+)\](.*?)(?=\[[^\]]+\]|$)", re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*\s*%?")
_NOISE_TAGS = {"핵심", "이슈", "투자포인트", "리스크", "성장", "시나리오",
              "가치", "포지션", "컨센서스", "대비", "데이터", "공백"}


def _sections(report: str) -> dict:
    return {m.group(1).strip(): m.group(2).strip() for m in _SECTION.finditer(report)}


def check_trend(report_sections: dict, expected_trend: Optional[str]) -> Optional[dict]:
    if not expected_trend:
        return None
    section = report_sections.get("성장 시나리오") or report_sections.get("투자포인트") or ""
    if not section:
        return None
    expects_up = any(k in expected_trend for k in ("가속", "지속", "개선", "턴어라운드"))
    expects_down = any(k in expected_trend for k in ("역성장", "둔화")) and not expects_up
    said_up = any(k in section for k in _GROWTH_UP_KW)
    said_down = any(k in section for k in _GROWTH_DOWN_KW)

    match = True
    if expects_up and said_down and not said_up:
        match = False
    if expects_down and said_up and not said_down:
        match = False
    return {"expected": expected_trend, "match": match, "section": section[:200]}


def check_signal(report_sections: dict, expected_signal: Optional[bool]) -> Optional[dict]:
    if expected_signal is None:
        return None
    section = report_sections.get("가치 포지션") or ""
    if not section:
        return None
    said_yes = any(k in section for k in _SIGNAL_YES_KW)
    said_no = any(k in section for k in _SIGNAL_NO_KW)

    if expected_signal:
        match = said_yes or not said_no
    else:
        match = said_no or not said_yes
    return {"expected": expected_signal, "match": match, "section": section[:200]}


def _trusted_numbers(blocks: list) -> set:
    nums = set()
    for b in blocks:
        for m in _NUM.finditer(b or ""):
            tok = m.group().replace(",", "").replace(" ", "").rstrip("%")
            if tok and tok not in ("-",):
                nums.add(tok)
    return nums


def _is_noise(tok: str) -> bool:
    """항목 번호·짧은 숫자·연도 파편 등은 파생수치 검사에서 제외."""
    digits = tok.replace("-", "").replace(".", "")
    return len(digits) < 3


def find_unverified_numbers(report_sections: dict, trusted_blocks: list) -> list:
    """[성장 시나리오]/[가치 포지션]/[핵심 이슈]/[투자포인트] 절의 숫자만 검사
    (리스크·공시 등 서술 절의 예시 숫자까지 검사하면 오탐이 늘어난다)."""
    trusted = _trusted_numbers(trusted_blocks)
    trusted_floats = []
    for t in trusted:
        try:
            trusted_floats.append(float(t))
        except ValueError:
            continue

    target_sections = ["핵심 이슈", "투자포인트", "성장 시나리오", "가치 포지션", "컨센서스 대비"]
    unverified = []
    for name in target_sections:
        text = report_sections.get(name, "")
        for m in _NUM.finditer(text):
            tok = m.group().replace(",", "").replace(" ", "").rstrip("%")
            if not tok or _is_noise(tok):
                continue
            try:
                val = float(tok)
            except ValueError:
                continue
            found = any(abs(val - t) <= max(0.05 * abs(val), 0.5) for t in trusted_floats)
            if not found:
                unverified.append(f"{tok}({name})")
    return unverified


def check(report: str, invest_point: dict, trusted_blocks: list) -> dict:
    """report(LLM 최종 텍스트) 검증. invest_point는 mvp_graph의 build_invest_point 결과,
    trusted_blocks는 analyze()가 프롬프트에 실제로 제시한 구조화 블록 문자열 리스트."""
    sections = _sections(report)
    growth = (invest_point or {}).get("growth", {})
    valuation = (invest_point or {}).get("valuation", {})

    trend_chk = check_trend(sections, growth.get("trend"))
    signal_chk = check_signal(sections, valuation.get("signal"))
    unverified = find_unverified_numbers(sections, trusted_blocks)

    problems = []
    if trend_chk and not trend_chk["match"]:
        problems.append(f"성장 판단 불일치(기대: {trend_chk['expected']})")
    if signal_chk and not signal_chk["match"]:
        problems.append(
            f"가치 시그널 불일치(기대: {'예' if valuation.get('signal') else '아니오'})")
    if unverified:
        problems.append(f"미검증 파생수치 {len(unverified)}건: {', '.join(unverified[:5])}")

    consistent = not problems
    return {
        "trend_check": trend_chk,
        "signal_check": signal_chk,
        "unverified_numbers": unverified,
        "problems": problems,
        "consistent": consistent,
        "verdict": "✅ 시나리오 일관" if consistent else ("⚠️ 불일치: " + "; ".join(problems)),
    }
