"""
citation.py
===========
LLM이 단 [Cxxx] 인용이 (1) 실재하는 청크인지, (2) 주장 내용이 그 청크에
실제로 근거하는지(용어 중첩 기반 grounding)를 검증한다.

목적: '인용 품질 검증'. 환각 인용(없는 청크 참조)과 근거 빈약 주장을 잡아낸다.
grounding은 휴리스틱(완벽한 사실검증 아님)이며, 낮은 점수는 사람이 재확인하라는 신호.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE = re.compile(r"\[(C\d{3})\]")
# 한국어 2글자+ 토큰 / 영문/숫자 토큰 (조사·기호 무시)
_TOKEN = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d{2,}")


@dataclass
class CitationReport:
    n_lines_with_cite: int = 0
    n_citations: int = 0
    n_valid: int = 0           # 실재하는 청크 참조
    n_invalid: int = 0         # 환각: 없는 청크 참조
    invalid_ids: list = field(default_factory=list)
    avg_grounding: float = 0.0   # 유효 인용의 평균 용어중첩(0~1)
    weak_lines: list = field(default_factory=list)  # grounding 낮은 (line, score, id)
    uncited_claim_lines: list = field(default_factory=list)  # 인용 없는 단정 문장

    def quality_verdict(self) -> str:
        if self.n_citations == 0:
            return "❌ 인용 없음"
        valid_rate = self.n_valid / self.n_citations
        if self.n_invalid > 0 or valid_rate < 0.9:
            return "❌ 환각 인용 존재"
        if self.avg_grounding < 0.25:
            return "⚠️ 근거 약함(재확인 필요)"
        return "✅ 인용 신뢰"


def _tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(s))


def _grounding(claim: str, chunk_text: str) -> float:
    ct = _tokens(claim)
    if not ct:
        return 0.0
    cht = _tokens(chunk_text)
    return len(ct & cht) / len(ct)


# 단정적 어미(인용이 있어야 바람직한 문장 탐지용)
_ASSERTIVE = re.compile(r"(다|함|음|됨|이다|한다|된다|예상|전망|증가|감소|확대|개선)\.?\s*$")


def verify(llm_text: str, get_chunk) -> CitationReport:
    """llm_text의 줄별 인용을 검증. get_chunk(chunk_id)->chunk dict|None."""
    rep = CitationReport()
    groundings = []

    for raw in llm_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        cites = CITE.findall(line)
        if cites:
            rep.n_lines_with_cite += 1
            for cid in cites:
                rep.n_citations += 1
                ch = get_chunk(cid)
                if ch is None:
                    rep.n_invalid += 1
                    rep.invalid_ids.append(cid)
                    continue
                rep.n_valid += 1
                g = _grounding(re.sub(CITE, "", line), ch.get("text", ""))
                groundings.append(g)
                if g < 0.25:
                    rep.weak_lines.append({"line": line[:80], "id": cid, "grounding": round(g, 2)})
        else:
            # 인용 없는데 단정적이면 '근거 미표시 주장'으로 표시
            content = re.sub(r"^\[[^\]]+\]\s*", "", line)  # [핵심 이슈] 같은 머리표 제거
            if _ASSERTIVE.search(content) and len(content) > 10:
                rep.uncited_claim_lines.append(content[:80])

    rep.avg_grounding = round(sum(groundings) / len(groundings), 3) if groundings else 0.0
    return rep
