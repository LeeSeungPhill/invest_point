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
    news: list                  # [{title, url, date, summary}]  (MVP에서는 stub)
    report: str                 # 최종 LLM 리포트
    # 병렬 노드들이 같은 superstep에서 동시에 append 할 수 있으므로 reducer 지정
    errors: Annotated[list, operator.add]


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
        return {}
    try:
        filing = _dart.latest_periodic_report(state["corp_code"])
        text = _dart.fetch_business_section(filing.rcept_no)
        if not text:
            raise DartError("사업의 내용 섹션을 찾지 못했습니다(문서 구조 상이 가능).")
        return {
            "report_nm": filing.report_nm,
            "rcept_dt": filing.rcept_dt,
            "business_text": text,
        }
    except DartError as e:
        return {"errors": _append_error(state, "fetch_business", e)}


def fetch_financials(state: AnalysisState) -> AnalysisState:
    """최근 3개년 매출/영업이익/순이익. 사업보고서(11011) 기준으로 시도."""
    if not state.get("corp_code"):
        return {}
    try:
        year = time.strftime("%Y")
        # 가장 최근에 확정된 사업보고서는 보통 전년도분이므로 한 해 뒤로
        fs = None
        for y in (str(int(year) - 1), str(int(year) - 2)):
            try:
                fs = _dart.financial_series(state["corp_code"], y, "11011")
                if fs.revenue:
                    break
            except DartError:
                continue
        if not fs or not fs.revenue:
            raise DartError("주요계정(매출/영업이익) 조회 실패.")
        return {"financials": {
            "revenue": fs.revenue,
            "operating_profit": fs.operating_profit,
            "net_income": fs.net_income,
        }}
    except DartError as e:
        return {"errors": _append_error(state, "fetch_financials", e)}


def fetch_news(state: AnalysisState) -> AnalysisState:
    """뉴스 수집 (MVP stub).
    실제 구현 시 네이버 검색 API / RSS / 기존 FastAPI 크롤러를 연결.
    여기서는 자리만 잡아두고 빈 리스트를 반환한다.
    """
    # TODO: 네이버 뉴스검색 API (X-Naver-Client-Id/Secret) 또는 사내 크롤러 연결
    return {"news": []}


def analyze(state: AnalysisState) -> AnalysisState:
    """수집물 -> 이슈/투자포인트 + 낙관/기본/보수 시나리오 초안 (LLM)."""
    if not state.get("business_text") and not state.get("financials"):
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
        "너는 한국 주식 애널리스트의 리서치 보조다. 아래 DART 사업보고서 본문과 "
        "재무 시계열, (있다면) 뉴스만을 근거로 분석한다. 규칙:\n"
        "1) 매출·실적을 단일 숫자로 단정해 예측하지 말 것. 대신 낙관/기본/보수 "
        "3개 시나리오로 방향과 가정을 서술.\n"
        "2) 모든 핵심 주장 뒤에 근거(어느 데이터/문장에서 나왔는지)를 괄호로 표기.\n"
        "3) 본문에 근거가 없으면 '자료상 확인 불가'라고 명시. 추측 금지.\n"
        "4) 출력 형식: [핵심 이슈] [투자포인트] [리스크] [매출/실적 시나리오] [데이터 공백]"
    )

    fin = state.get("financials", {})
    fin_lines = []
    for label, d in (("매출액", fin.get("revenue")),
                     ("영업이익", fin.get("operating_profit")),
                     ("당기순이익", fin.get("net_income"))):
        if d:
            fin_lines.append(f"- {label}: " + ", ".join(f"{k}={v:,}원" for k, v in d.items()))
    fin_block = "\n".join(fin_lines) or "(재무 데이터 없음)"

    news = state.get("news") or []
    news_block = "\n".join(f"- {n.get('title')}" for n in news) or "(뉴스 없음)"

    human = (
        f"종목: {state.get('corp_name')} ({state.get('stock_code')})\n"
        f"근거 공시: {state.get('report_nm')} (접수 {state.get('rcept_dt')})\n\n"
        f"[재무 시계열]\n{fin_block}\n\n"
        f"[뉴스]\n{news_block}\n\n"
        f"[사업의 내용 본문(발췌)]\n{state.get('business_text', '(없음)')[:12000]}"
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"report": resp.content if isinstance(resp.content, str)
                else str(resp.content)}
    except Exception as e:  # noqa: BLE001  (네트워크/키 등 광범위)
        return {"errors": _append_error(state, "analyze(llm)", e),
                "report": "(LLM 호출 실패)"}


# ---------------------------------------------------------------------- #
# 그래프 조립
# ---------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(AnalysisState)
    g.add_node("resolve_corp", resolve_corp)
    g.add_node("fetch_business", fetch_business)
    g.add_node("fetch_financials", fetch_financials)
    g.add_node("fetch_news", fetch_news)
    g.add_node("analyze", analyze)

    g.add_edge(START, "resolve_corp")
    # resolve 이후 3개 수집 노드를 병렬 fan-out
    g.add_edge("resolve_corp", "fetch_business")
    g.add_edge("resolve_corp", "fetch_financials")
    g.add_edge("resolve_corp", "fetch_news")
    # 3개가 모두 끝나면 analyze로 fan-in (LangGraph가 자동 join)
    g.add_edge("fetch_business", "analyze")
    g.add_edge("fetch_financials", "analyze")
    g.add_edge("fetch_news", "analyze")
    g.add_edge("analyze", END)
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
    print("=" * 70)
    print(result.get("report", "(리포트 없음)"))
    if result.get("errors"):
        print("\n--- 수집/분석 경고 ---")
        for e in result["errors"]:
            print(" •", e)
