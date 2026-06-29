"""
mvp_graph.py
============
1단계 MVP: 단일 종목에 대해
  종목코드 -> 고유번호 -> 최신 정기보고서 -> 사업의 내용 + 재무 시계열
  -> LLM이 이슈/투자포인트 + 시나리오(낙관/기본/보수) 초안 작성
하는 LangGraph 골격.

설계 원칙(앞선 검토와 일관):
  - LLM에게 매출을 '포인트 숫자'로 예측시키지 않는다.
  - 정량 베이스(재무 시계열)는 코드가 만들고, LLM은 '근거 구조화 + 정성 해석 +
    낙관/기본/보수 시나리오'만 담당한다.
  - 모든 판단은 어떤 데이터(공시 문장/재무 항목)에 근거했는지 밝히게 한다.
  - 각 노드는 실패해도 그래프를 죽이지 않고 state['errors']에 적재한다.

필요 환경변수:
  OPENDART_API_KEY
  LLM_BACKEND         # ollama(기본) | openai_compat | google  — llm_backend.py 참조
                      # ollama면 추가 키 불필요(로컬). 자세한 설정은 llm_backend.py

실행:
  python mvp_graph.py 112610      # 예: CS WIND
"""

from __future__ import annotations

import os
import sys
import time
import logging
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from dart_client import DartClient, DartError, REPRT_CODE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
log = logging.getLogger("mvp")


# ---------------------------------------------------------------------- #
# 그래프 상태
# ---------------------------------------------------------------------- #
class AnalysisState(TypedDict, total=False):
    stock_code: str
    corp_code: str
    corp_name: str
    report_nm: str
    rcept_dt: str
    business_text: str          # 사업의 내용 본문
    financials: dict            # {revenue:{...}, operating_profit:{...}, net_income:{...}}
    news: list                  # [{title, link, pub_date, summary}]
    reports: list               # 애널리스트 리포트 메타데이터
    consensus: dict             # self-built 컨센서스(목표주가/의견 집계)
    estimates: dict             # comp.fnguide.com 예상실적(매출/영업이익/순이익, 억원)
    n_chunks: int               # RAG 청크 수
    retrieved_chunks: list      # LLM에 제시한 근거 청크(인용 검증 대상)
    rag_context: str            # 청크들을 [Cxxx] 라벨로 포맷한 컨텍스트
    citation_report: dict       # 인용 품질 검증 결과
    report: str                 # 최종 LLM 리포트
    # 병렬 노드들이 같은 superstep에서 동시에 쓰므로 reducer 지정
    errors: Annotated[list, operator.add]
    sources_status: Annotated[dict, lambda a, b: {**(a or {}), **(b or {})}]


def _append_error(state: AnalysisState, where: str, exc: Exception) -> list:
    log.warning("노드 오류 %s: %s", where, exc)
    # reducer(operator.add)가 기존 리스트에 더해주므로 '추가분'만 반환
    return [f"[{where}] {type(exc).__name__}: {exc}"]


# 클라이언트는 노드 간 공유 (corp_map 캐시 재사용)
_dart = DartClient()


# ---------------------------------------------------------------------- #
# 노드들
# ---------------------------------------------------------------------- #
def resolve_corp(state: AnalysisState) -> AnalysisState:
    try:
        corp_code, corp_name = _dart.resolve(state["stock_code"])
        return {"corp_code": corp_code, "corp_name": corp_name}
    except DartError as e:
        return {"errors": _append_error(state, "resolve_corp", e)}


def fetch_business(state: AnalysisState) -> AnalysisState:
    """최신 정기보고서를 찾아 '사업의 내용' 본문을 추출."""
    if not state.get("corp_code"):
        return {"sources_status": {"dart_business": "skip"}}
    try:
        filing = _dart.latest_periodic_report(state["corp_code"])
        text = _dart.fetch_business_section(filing.rcept_no)
        if not text:
            raise DartError("사업의 내용 섹션을 찾지 못했습니다(문서 구조 상이 가능).")
        return {
            "report_nm": filing.report_nm,
            "rcept_dt": filing.rcept_dt,
            "business_text": text,
            "sources_status": {"dart_business": "ok"},
        }
    except DartError as e:
        return {"errors": _append_error(state, "fetch_business", e),
                "sources_status": {"dart_business": "error"}}


def fetch_financials(state: AnalysisState) -> AnalysisState:
    """최근 3개년 매출/영업이익/순이익. 사업보고서(11011) 기준으로 시도."""
    if not state.get("corp_code"):
        return {"sources_status": {"dart_financials": "skip"}}
    try:
        # fetch_business와 동일한 우선순위(분기→반기→사업)로 고른 보고서 기준
        filing = _dart.latest_periodic_report(state["corp_code"])
        fs = _dart.financial_series(state["corp_code"], filing.bsns_year, filing.reprt_code)
        # 해당 보고서에서 매출이 비면 직전 사업보고서(연간)로 폴백
        if not fs.revenue:
            prev = str(int(filing.bsns_year) - 1) if filing.bsns_year.isdigit() else filing.bsns_year
            fs = _dart.financial_series(state["corp_code"], prev, "11011")
        if not fs.revenue:
            raise DartError("주요계정(매출/영업이익) 조회 실패.")
        return {"financials": {
            "basis": filing.report_nm,   # 어느 보고서 기준인지(분기=누적 주의)
            "revenue": fs.revenue,
            "operating_profit": fs.operating_profit,
            "net_income": fs.net_income,
        }, "sources_status": {"dart_financials": "ok"}}
    except DartError as e:
        return {"errors": _append_error(state, "fetch_financials", e),
                "sources_status": {"dart_financials": "error"}}


def fetch_news(state: AnalysisState) -> AnalysisState:
    """네이버 공식 뉴스 API로 최근 종목 이슈 수집."""
    name = state.get("corp_name")
    if not name:
        return {"sources_status": {"news": "skip"}}
    try:
        from external_sources import fetch_naver_news
        items = fetch_naver_news(name, display=20, sort="date")
        return {"news": items,
                "sources_status": {"news": "ok" if items else "empty"}}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "fetch_news", e),
                "sources_status": {"news": "error"}}


def fetch_research(state: AnalysisState) -> AnalysisState:
    """네이버증권 리포트 목록(메타데이터) + self-built 컨센서스."""
    if not state.get("stock_code"):
        return {"sources_status": {"research": "skip"}}
    try:
        from external_sources import fetch_naver_research, aggregate_consensus
        reports = fetch_naver_research(state["stock_code"])
        consensus = aggregate_consensus(reports)
        return {"reports": reports, "consensus": consensus,
                "sources_status": {"research": "ok" if reports else "empty"}}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "fetch_research", e),
                "sources_status": {"research": "error"}}


def fetch_estimates(state: AnalysisState) -> AnalysisState:
    """comp.fnguide.com 예상실적(매출/영업이익/순이익 추정). 단위 억원."""
    if not state.get("stock_code"):
        return {"sources_status": {"estimates": "skip"}}
    try:
        from external_sources import fetch_fnguide_estimates
        est = fetch_fnguide_estimates(state["stock_code"])
        has = bool(est.get("estimates"))
        return {"estimates": est,
                "sources_status": {"estimates": "ok" if has else "empty"}}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "fetch_estimates", e),
                "sources_status": {"estimates": "error"}}


# 사업의 내용에서 뽑을 관점들 (각 질의로 관련 청크를 검색)
_ASPECTS = {
    "핵심이슈": "회사의 최근 핵심 이슈, 업황 변화, 주요 사건과 환경 변화",
    "투자포인트": "성장 동력, 경쟁력, 수주잔고, 신사업, 시장 점유율, 생산능력 증설",
    "리스크": "위험 요인, 원자재 가격, 환율, 규제, 소송, 경쟁 심화, 전방산업 부진",
    "매출동인": "매출과 실적을 좌우하는 제품군, 판가, 물량, 가동률, 전방 수요",
}


def rag_retrieve(state: AnalysisState) -> AnalysisState:
    """사업의 내용을 청킹·임베딩하고, 관점별로 관련 청크를 검색해 컨텍스트 구성."""
    body = state.get("business_text")
    if not body:
        return {"sources_status": {"rag": "skip"}}
    try:
        from rag import chunk_text, Retriever
        chunks = chunk_text(body)
        retr = Retriever()  # OllamaEmbedder(bge-m3) — 맥 Ollama 사용
        cache_key = f"{state.get('stock_code')}:{state.get('rcept_dt')}"
        retr.build(chunks, cache_key=cache_key)

        picked: dict = {}
        for q in _ASPECTS.values():
            for h in retr.query(q, k=4):
                picked[h["chunk_id"]] = h   # 중복 제거(같은 청크 1회만)
        retrieved = list(picked.values())
        context = "\n\n".join(
            f"[{c['chunk_id']}] ({c.get('heading','')})\n{c['text']}" for c in retrieved)
        return {
            "n_chunks": len(chunks),
            "retrieved_chunks": retrieved,
            "rag_context": context,
            "sources_status": {"rag": "ok" if retrieved else "empty"},
        }
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "rag_retrieve", e),
                "sources_status": {"rag": "error"}}


def analyze(state: AnalysisState) -> AnalysisState:
    """수집물 -> 이슈/투자포인트 + 낙관/기본/보수 시나리오 초안 (LLM)."""
    if not state.get("rag_context") and not state.get("financials"):
        return {"report": "(분석 불가: 사업의 내용/재무 데이터를 확보하지 못했습니다.)"}

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from llm_backend import get_chat_model
    except ImportError as e:
        return {"errors": _append_error(state, "analyze(import)", e),
                "report": "(LLM 백엔드 모듈/패키지 미설치)"}

    try:
        llm = get_chat_model(temperature=0.2, max_tokens=2000)
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "analyze(init)", e),
                "report": "(LLM 백엔드 초기화 실패 — LLM_BACKEND 설정 확인)"}

    system = (
        "너는 한국 주식 애널리스트의 리서치 보조다. 아래 '사업보고서 근거 청크', "
        "재무 시계열, 애널리스트 컨센서스(자체 집계), 리포트 목록, 뉴스만을 근거로 "
        "분석한다. 추론 과정은 출력하지 말고 결과만 한국어로 써라. 규칙(엄수):\n"
        "1) [핵심 이슈][투자포인트][리스크]의 각 항목은 사업보고서 청크에 근거해야 하며, "
        "문장 끝에 반드시 그 근거 청크 id를 [C012] 형식으로 붙여라. 예: "
        "'DR 시험 이행률이 105%로 높다 [C034]'. 인용 없는 항목은 쓰지 마라.\n"
        "2) 제공된 '사용 가능 청크 id' 목록에 있는 id만 인용하라. 목록에 없는 id를 "
        "지어내면 심각한 오류다.\n"
        "3) 금액 숫자(매출·이익 등)를 새로 지어내지 마라. [재무 시계열]이나 [예상 실적]에 "
        "실제로 제시된 값만 사용하고, 없으면 '구체 수치 자료 없음'이라 쓰고 방향"
        "(증가/감소/유지)만 기술하라. 임의의 억원 수치를 만들어내는 것을 금지한다.\n"
        "4) [매출/실적 시나리오]는 fnguide 예상실적이 있으면 그 값을 기준점으로만 쓰고, "
        "없으면 숫자 없이 낙관/기본/보수의 '방향과 가정'만 서술하라.\n"
        "5) 근거 청크에 없는 내용은 '자료상 확인 불가'라고 명시하라.\n"
        "6) 출력 형식: [핵심 이슈] [투자포인트] [리스크] [컨센서스 대비] "
        "[매출/실적 시나리오] [데이터 공백]"
    )

    fin = state.get("financials", {})
    fin_lines = []
    for label, d in (("매출액", fin.get("revenue")),
                     ("영업이익", fin.get("operating_profit")),
                     ("당기순이익", fin.get("net_income"))):
        if d:
            fin_lines.append(f"- {label}: " + ", ".join(f"{k}={v:,}원" for k, v in d.items()))
    fin_block = "\n".join(fin_lines) or "(재무 데이터 없음)"
    if fin.get("basis"):
        fin_block = f"(기준: {fin['basis']} — 분기/반기는 누적 수치 주의)\n" + fin_block

    news = state.get("news") or []
    news_block = "\n".join(f"- ({n.get('pub_date','')}) {n.get('title')}" for n in news[:15]) \
        or "(뉴스 없음)"

    cons = state.get("consensus") or {}
    if cons.get("n_target_prices"):
        cons_block = (
            f"- 목표주가: 평균 {cons.get('target_price_mean'):,} / "
            f"중앙값 {cons.get('target_price_median'):,} / "
            f"범위 {cons.get('target_price_min'):,}~{cons.get('target_price_max'):,} "
            f"(증권사 {cons.get('n_target_prices')}곳)\n"
            f"- 투자의견 분포: {cons.get('opinion_distribution', '자료 부족')}\n"
            f"- 참여 증권사: {', '.join(cons.get('brokers', [])) or '미상'}"
        )
    else:
        cons_block = "(수집된 목표주가 컨센서스 없음 — 자료상 확인 불가)"

    reports = state.get("reports") or []
    rep_block = "\n".join(
        f"- {r.get('date','')} {r.get('broker','')} | {r.get('title','')}"
        for r in reports[:15]) or "(리포트 목록 없음)"

    est = state.get("estimates") or {}
    est_lines = []
    for col, vals in (est.get("estimates") or {}).items():
        parts = ", ".join(f"{k} {v:,.0f}" for k, v in vals.items())
        est_lines.append(f"- {col}: {parts}")
    est_block = ("(단위: 억원, 출처 comp.fnguide.com)\n" + "\n".join(est_lines)) \
        if est_lines else "(예상실적 없음)"

    retrieved = state.get("retrieved_chunks") or []
    valid_ids = ", ".join(c["chunk_id"] for c in retrieved) or "(없음)"

    human = (
        f"종목: {state.get('corp_name')} ({state.get('stock_code')})\n"
        f"근거 공시: {state.get('report_nm')} (접수 {state.get('rcept_dt')})\n\n"
        f"[재무 시계열(실적, DART)]\n{fin_block}\n\n"
        f"[예상 실적(컨센서스 추정, fnguide)]\n{est_block}\n\n"
        f"[애널리스트 컨센서스(자체 집계)]\n{cons_block}\n\n"
        f"[최근 리포트 목록]\n{rep_block}\n\n"
        f"[최근 뉴스]\n{news_block}\n\n"
        f"[사용 가능 청크 id — 이 중에서만 인용]\n{valid_ids}\n\n"
        f"[사업보고서 근거 청크 (이 id만 인용 가능)]\n{state.get('rag_context', '(없음)')}"
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"report": resp.content if isinstance(resp.content, str)
                else str(resp.content)}
    except Exception as e:  # noqa: BLE001  (네트워크/키 등 광범위)
        return {"errors": _append_error(state, "analyze(llm)", e),
                "report": "(LLM 호출 실패)"}


def verify_citations(state: AnalysisState) -> AnalysisState:
    """LLM 리포트의 [Cxxx] 인용이 실재 청크인지 + 근거가 맞는지 검증."""
    report = state.get("report", "")
    retrieved = state.get("retrieved_chunks") or []
    if not report or not retrieved:
        return {"citation_report": {"note": "검증할 인용/청크 없음"}}
    try:
        from dataclasses import asdict
        import citation
        lookup = {c["chunk_id"]: c for c in retrieved}
        rep = citation.verify(report, lambda cid: lookup.get(cid))
        data = asdict(rep)
        data["verdict"] = rep.quality_verdict()
        return {"citation_report": data}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "verify_citations", e)}


# ---------------------------------------------------------------------- #
# 그래프 조립
# ---------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(AnalysisState)
    g.add_node("resolve_corp", resolve_corp)
    g.add_node("fetch_business", fetch_business)
    g.add_node("rag_retrieve", rag_retrieve)
    g.add_node("fetch_financials", fetch_financials)
    g.add_node("fetch_news", fetch_news)
    g.add_node("fetch_research", fetch_research)
    g.add_node("fetch_estimates", fetch_estimates)
    g.add_node("analyze", analyze)
    g.add_node("verify_citations", verify_citations)

    g.add_edge(START, "resolve_corp")
    # 사업보고서 체인: 본문 → RAG 검색 → analyze
    g.add_edge("resolve_corp", "fetch_business")
    g.add_edge("fetch_business", "rag_retrieve")
    g.add_edge("rag_retrieve", "analyze")
    # 나머지 수집 노드는 병렬로 analyze에 fan-in
    g.add_edge("resolve_corp", "fetch_financials")
    g.add_edge("resolve_corp", "fetch_news")
    g.add_edge("resolve_corp", "fetch_research")
    g.add_edge("resolve_corp", "fetch_estimates")
    g.add_edge("fetch_financials", "analyze")
    g.add_edge("fetch_news", "analyze")
    g.add_edge("fetch_research", "analyze")
    g.add_edge("fetch_estimates", "analyze")
    # 분석 후 인용 검증
    g.add_edge("analyze", "verify_citations")
    g.add_edge("verify_citations", END)
    return g.compile()


def run(stock_code: str) -> AnalysisState:
    graph = build_graph()
    final = graph.invoke({"stock_code": stock_code, "errors": []})
    return final


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "112610"
    result = run(code)

    print("\n" + "=" * 70)
    print(f"종목: {result.get('corp_name')} ({code})  근거: {result.get('report_nm')}")
    print(f"소스 상태: {result.get('sources_status', {})}  | RAG 청크: {result.get('n_chunks', 0)}")
    print("=" * 70)
    print(result.get("report", "(리포트 없음)"))

    cr = result.get("citation_report", {})
    if cr and "verdict" in cr:
        print("\n--- 인용 품질 검증 ---")
        print(f" 판정: {cr['verdict']} | 인용 {cr['n_citations']}건 "
              f"(유효 {cr['n_valid']}, 환각 {cr['n_invalid']}) | "
              f"평균 grounding {cr['avg_grounding']}")
        if str(cr["verdict"]).startswith("❌"):
            print(" ⚠️  경고: 인용 검증 실패 — 이 리포트는 근거가 불충분하여 신뢰 불가."
                  " 매매 판단에 사용하지 마세요.")
        if cr.get("invalid_ids"):
            print(f" 환각 인용 id: {cr['invalid_ids']}")
        if cr.get("weak_lines"):
            print(" 근거 약한 문장:")
            for w in cr["weak_lines"][:5]:
                print(f"   - [{w['id']} g={w['grounding']}] {w['line']}")
        if cr.get("uncited_claim_lines"):
            print(" 근거 미표시 단정:")
            for u in cr["uncited_claim_lines"][:5]:
                print(f"   - {u}")

    if result.get("errors"):
        print("\n--- 수집/분석 경고 ---")
        for e in result["errors"]:
            print(" •", e)
