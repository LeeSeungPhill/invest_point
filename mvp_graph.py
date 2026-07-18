"""
mvp_graph.py
============
1단계: 종목코드 -> 고유번호 -> 최신 정기보고서 -> 사업의 내용/개요 + 재무 시계열
       (연간/분기/예상) + 공시 + 뉴스를 안정적으로 수집(pipeline_stability.py로 검증).
2단계: 수집물로 성장/가치 정량 시그널(invest_point.py)을 코드가 계산하고,
       LLM이 이슈/투자포인트 + 성장 시나리오/가치 포지션 해석을 작성한 뒤
       citation.py로 원문 인용 품질을 검증하는 LangGraph 골격.
3단계: 분석 '전' 서로 다른 소스 간 교차검증(cross_check.py)으로 잘못된 값이
       LLM에게 넘어가는 것을 막고, 분석 '후' LLM 리포트가 invest_point 시그널과
       실제로 같은 방향을 말하는지 + 제공되지 않은 파생 수치를 지어내지 않았는지
       검증(scenario_check.py)한다. 불일치 발견 시 1회 자동 재생성 후 그래도
       남아있으면 최종 결과에 경고를 남긴다(무한 재시도 금지). 또한 같은 종목의
       과거 분석 결과를 PostgreSQL에 저장·조회해(analysis_history.py) LLM이
       '지난 분석 대비' 서술을 실제 과거 숫자로만 하게 하고, 근거 수치는 그대로인데
       판단만 뒤집힌 경우를 참고 신호(history_drift)로 남긴다.
4단계: 컨센서스(자체 집계 + FnGuide, 코드 계산값)와 3단계에서 검증까지 끝난 LLM
       리포트의 정성 절([투자포인트]/[가치 포지션]/[컨센서스 대비])을 하나로 묶어
       '투자포인트 요약'을 만든다(build_investment_summary). 새 숫자를 만들지 않고
       이미 검증된 컨센서스 수치 + 리포트 문장을 재구성만 하므로 추가 LLM 호출이나
       환각 위험이 없다. 결과는 analysis_history에도 함께 저장된다.

설계 원칙(앞선 검토와 일관):
  - LLM에게 매출을 '포인트 숫자'로 예측시키지 않는다.
  - 정량 베이스(재무 시계열·성장률·주가 위치)는 코드가 만들고, LLM은 '근거 구조화 +
    정성 해석 + 시나리오' 작성만 담당한다.
  - 모든 판단은 어떤 데이터(공시 문장/재무 항목)에 근거했는지 밝히게 한다.
  - 각 노드는 실패해도 그래프를 죽이지 않고 state['errors']에 적재한다.
  - 재생성 루프는 LangGraph의 fan-in(여러 노드가 analyze로 모이는 구조) 때문에
    analyze로 직접 되돌아가는 사이클을 만들지 않는다(다른 선행 노드들이 다시
    실행되지 않아 join이 멈출 수 있음). 대신 회귀 전용 노드를 별도로 두어
    '최대 1회'가 사이클이 아니라 조건부 분기로 구조적으로 보장되게 한다.

필요 환경변수:
  OPENDART_API_KEY
  LLM_BACKEND         # ollama(기본) | openai_compat | google  — llm_backend.py 참조
                      # ollama면 추가 키 불필요(로컬). 자세한 설정은 llm_backend.py

실행:
  python mvp_graph.py 112610      # 예: CS WIND
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
import operator
from typing import Annotated, Optional, TypedDict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, START, END

from dart_client import DartClient, DartError, REPRT_CODE
from invest_point import build_invest_point as build_invest_point_calc
from invest_point import format_invest_point_block
from invest_point import build_growth_signal
from cross_check import run_cross_check, format_cross_check_block
from scenario_check import SECTION_TAGS
from industry_category import classify_industry
import stability_score
import analysis_history

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
    business_text: str          # 사업의 내용 본문(RAG용 원문)
    biz_summary: dict           # DART 정확 섹션 기반 '사업의 개요'(문단+표 요약)
    financials: dict            # {revenue:{...}, operating_profit:{...}, net_income:{...}} (DART, 원단위)
    news: list                  # [{title, link, pub_date, summary}]
    reports: list               # 애널리스트 리포트 메타데이터
    consensus: dict             # self-built 컨센서스(목표주가/의견 집계)
    fnguide: dict                # FnGuide/WiseReport 연간·분기 실적+예상, 제품비중, 컨센서스(억원)
    disclosures: list           # 최근 DART 공시 목록(반복성 공시 제외)
    price: dict                 # 현재가·52주 밴드·PER/PBR·컨센서스 목표주가
    invest_point: dict          # 성장/가치 정량 시그널(invest_point.py 계산)
    financial_history: dict     # DART 최근 확정 연간 사업보고서 기준 3개년 재무제표(FinancialSeries)
    industry_category: dict     # 업종 카테고리(경기민감도/자본집약도, KSIC 기반)
    stability: dict             # 실적안정성/재무안정성 점수(stability_score.py 계산)
    cross_check: dict           # 3단계: 소스 간 교차검증 결과(목표주가/투자의견/밴드 정합성)
    n_chunks: int               # RAG 청크 수
    retrieved_chunks: list      # LLM에 제시한 근거 청크(인용 검증 대상)
    rag_context: str            # 청크들을 [Cxxx] 라벨로 포맷한 컨텍스트
    citation_report: dict       # 인용 품질 검증 결과(청크 그라운딩)
    scenario_consistency: dict  # 3단계: 시나리오 방향/파생수치 일관성 검증 결과
    regenerated: bool           # 1회 자동 재생성이 실제로 발생했는지
    history: list               # 3단계: 같은 종목의 과거 분석 이력(최신순, PostgreSQL)
    history_drift: dict         # 3단계: 과거 대비 '근거 수치는 그대로인데 판단만 뒤집힘' 참고신호
    report: str                 # 최종 LLM 리포트
    investment_summary: str     # 4단계: 컨센서스 + 리포트 결합 투자포인트 요약
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


def fetch_fnguide(state: AnalysisState) -> AnalysisState:
    """FnGuide/WiseReport 연간·분기 실적+예상(억원), 제품비중, 컨센서스.
    DISABLE_FNGUIDE=1 로 끌 수 있다(개인 리서치 목적 직접 스크래핑 — fnguide_sources.py 참조)."""
    if os.getenv("DISABLE_FNGUIDE", "0") == "1":
        return {"sources_status": {"fnguide": "off"}}
    if not state.get("stock_code"):
        return {"sources_status": {"fnguide": "skip"}}
    try:
        from fnguide_sources import fetch_fnguide as _fetch_fnguide
        fng = _fetch_fnguide(state["stock_code"])
        has = bool(fng.get("annual_highlight") or fng.get("financial_highlight")
                   or (fng.get("cf1002") or {}).get("rows"))
        return {"fnguide": fng, "sources_status": {"fnguide": "ok" if has else "empty"}}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "fetch_fnguide", e),
                "sources_status": {"fnguide": "error"}}


def fetch_price(state: AnalysisState) -> AnalysisState:
    """현재가·52주 밴드·PER/PBR·컨센서스 목표주가(가치 관점 판단용)."""
    if not state.get("stock_code"):
        return {"sources_status": {"price": "skip"}}
    try:
        from external_sources import fetch_naver_price
        price = fetch_naver_price(state["stock_code"])
        return {"price": price, "sources_status": {"price": "ok"}}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "fetch_price", e),
                "sources_status": {"price": "error"}}


def fetch_financial_history(state: AnalysisState) -> AnalysisState:
    """최근 확정 연간 사업보고서(reprt_code=11011) 기준 3개년(당기/전기/전전기)
    재무제표 — 실적안정성/재무안정성 점수용. fetch_financials가 쓰는 '최신 정기
    보고서'는 분기일 수 있어(그러면 당기/전기/전전기가 같은 분기끼리 비교가 되어
    연간 변동성 계산에 부적합) 여기서는 연간 보고서로 고정 조회한다. 아직 그 해
    사업보고서가 안 나왔을 수도 있어 작년→재작년 순으로 시도한다."""
    if not state.get("corp_code"):
        return {"sources_status": {"financial_history": "skip"}}
    this_year = int(time.strftime("%Y"))
    last_err: Optional[Exception] = None
    for candidate_year in (this_year - 1, this_year - 2):
        try:
            fs = _dart.financial_series(state["corp_code"], str(candidate_year), "11011")
            if fs.revenue:
                return {"financial_history": fs, "sources_status": {"financial_history": "ok"}}
        except DartError as e:
            last_err = e
            continue
    return {"errors": _append_error(state, "fetch_financial_history",
                                    last_err or DartError("연간 사업보고서 재무 데이터 없음")),
            "sources_status": {"financial_history": "error"}}


def fetch_industry_category(state: AnalysisState) -> AnalysisState:
    """DART company.json의 induty_code(KSIC)로 실적/재무안정성 점수의 업종
    카테고리(경기민감도/자본집약도)를 분류한다."""
    if not state.get("corp_code"):
        return {"sources_status": {"industry_category": "skip"}}
    try:
        overview = _dart.company_overview(state["corp_code"])
        category = classify_industry(overview.get("induty_code"))
        return {"industry_category": category, "sources_status": {"industry_category": "ok"}}
    except DartError as e:
        return {"errors": _append_error(state, "fetch_industry_category", e),
                "sources_status": {"industry_category": "error"}}


def build_stability(state: AnalysisState) -> AnalysisState:
    """3단계: 실적안정성(매출·영업이익 변동성, FnGuide 연간 실측 기준) + 재무안정성
    (유동/부채/자기자본비율, DART 연간 기준) 점수 계산. fetch_fnguide/
    fetch_financial_history/fetch_industry_category 결과만으로 순수 계산(숫자
    생성 없음). next_est/next_est2(향후분기 급변위험 판단용)는 fnguide로 growth
    시그널을 한 번 더 계산해 얻는다(build_invest_point와는 독립적인 병렬 노드라
    그 결과에 의존할 수 없음 — analyze의 fan-in 깊이를 안 건드리기 위함)."""
    fnguide = state.get("fnguide")
    fs = state.get("financial_history")
    category = state.get("industry_category")
    growth = build_growth_signal(fnguide or {})
    earnings = stability_score.build_earnings_stability(
        fnguide, growth.get("next_est"), growth.get("next_est2"), category)
    financial = stability_score.build_financial_stability(fs, category)
    return {"stability": {"earnings": earnings, "financial": financial}}


def fetch_disclosures(state: AnalysisState) -> AnalysisState:
    """최근 DART 공시 목록(지분/임원 등 반복성 공시 제외)."""
    if not state.get("corp_code"):
        return {"sources_status": {"disclosures": "skip"}}
    try:
        items = _dart.recent_disclosures(state["corp_code"])
        return {"disclosures": items,
                "sources_status": {"disclosures": "ok" if items else "empty"}}
    except DartError as e:
        return {"errors": _append_error(state, "fetch_disclosures", e),
                "sources_status": {"disclosures": "error"}}


def fetch_biz_summary(state: AnalysisState) -> AnalysisState:
    """DART 문서뷰어 offset 기반 '사업의 개요'(문단+제품/사업현황 표 요약)."""
    if not state.get("corp_code"):
        return {"sources_status": {"biz_summary": "skip"}}
    try:
        summary = _dart.biz_summary(state["corp_code"])
        has = bool(summary.get("sentences"))
        return {"biz_summary": summary,
                "sources_status": {"biz_summary": "ok" if has else "empty"}}
    except DartError as e:
        return {"errors": _append_error(state, "fetch_biz_summary", e),
                "sources_status": {"biz_summary": "error"}}


def build_invest_point(state: AnalysisState) -> AnalysisState:
    """2단계: 성장(예상실적 방향) + 가치(컨센서스 목표주가 상승여력) 정량 시그널
    계산. fetch_fnguide/fetch_price 결과만으로 순수 계산(숫자 생성 없음)."""
    ip = build_invest_point_calc(state.get("fnguide"), state.get("price"))
    return {"invest_point": ip}


def cross_check_sources(state: AnalysisState) -> AnalysisState:
    """3단계: LLM에게 넘기기 전에 목표주가/투자의견/52주 밴드를 소스 간 교차검증.
    실제 사고 사례(조회수를 목표주가로 오인)를 데이터 단계에서 잡아내는 게 목적."""
    try:
        cc = run_cross_check(price=state.get("price"), fnguide=state.get("fnguide"),
                             consensus=state.get("consensus"))
        return {"cross_check": cc}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "cross_check_sources", e)}


def load_history(state: AnalysisState) -> AnalysisState:
    """3단계: 같은 종목의 과거 분석 이력(PostgreSQL)을 불러와 analyze 프롬프트에
    참조 자료로 제공한다. DB 접속 불가 시에도 그래프는 계속 진행(빈 이력 취급)."""
    if not state.get("stock_code"):
        return {"sources_status": {"history": "skip"}}
    if not analysis_history.is_enabled():
        return {"sources_status": {"history": "off"}}
    try:
        hist = analysis_history.get_recent(state["stock_code"], limit=5)
        return {"history": hist, "sources_status": {"history": "ok" if hist else "empty"}}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "load_history", e),
                "sources_status": {"history": "error"}}


def save_history(state: AnalysisState) -> AnalysisState:
    """3단계: 이번 실행의 핵심 결과를 다음 분석이 참조할 수 있도록 저장.
    두 종료 경로(재생성 없음 / 1회 재생성 후) 모두 이 노드를 거쳐 END로 간다."""
    try:
        analysis_history.save_run(
            stock_code=state.get("stock_code"), corp_name=state.get("corp_name"),
            report_nm=state.get("report_nm"), rcept_dt=state.get("rcept_dt"),
            invest_point=state.get("invest_point"), price=state.get("price"),
            citation_report=state.get("citation_report"), cross_check=state.get("cross_check"),
            scenario_consistency=state.get("scenario_consistency"),
            regenerated=bool(state.get("regenerated")), report=state.get("report"),
            investment_summary=state.get("investment_summary"),
            stability=state.get("stability"),
        )
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "save_history", e)}
    return {}


_SECTION_TAG_ALT = "|".join(re.escape(t) for t in SECTION_TAGS)


def _extract_report_section(report: str, tag: str) -> str:
    """LLM 리포트 텍스트에서 '[tag] ... (다음 절 제목 전까지)' 절만 추출.
    system 프롬프트가 정의한 실제 절 제목(SECTION_TAGS)만 절 경계로 인정한다.
    LLM이 본문 중간에 '[C012]'나 '[투자포인트(정량)]' 같은 인용/유사인용 태그를
    붙이는 경우 이를 절 경계로 오인하면 뒤 문장이 통째로 잘려나가는 버그가 있었다
    (예: [가치 포지션]의 핵심 결론 문장이 중간의 '[투자포인트(정량)]' 태그 때문에
    검증 대상에서 누락된 실사례 — scenario_check.py의 동일 버그와 함께 수정)."""
    if not report:
        return ""
    m = re.search(rf"\[{re.escape(tag)}\](.*?)(?=\[(?:{_SECTION_TAG_ALT})\]|$)", report, re.S)
    return m.group(1).strip() if m else ""


def build_investment_summary(state: AnalysisState) -> AnalysisState:
    """4단계: 3단계 검증까지 끝난 LLM 리포트에서 [핵심 이슈]/[투자포인트]/[리스크]
    절만 뽑아 investment_summary로 남긴다(사용자 확정 — 이 3개 절만). 여기서는
    새 숫자를 만들지 않고 이미 검증된 리포트 문장만 재구성한다(추가 LLM 호출 없음)."""
    report = state.get("report") or ""

    lines = []
    for tag in ("핵심 이슈", "투자포인트", "리스크"):
        section = _extract_report_section(report, tag)
        if section:
            lines.append(f"[{tag}]\n{section}")

    # analyze() 실패 시 report는 빈 문자열이 아니라 "(...)" 플레이스홀더로 채워진다
    # (데이터 부족/LLM 호출 실패/빈 응답). 두 경우 모두 실제 절이 하나도 안 뽑혔을
    # 것이므로, '리포트 미생성'임을 명확히 표시한다(조용히 빈 값을 남기면 실패를
    # 정상 결과처럼 오인하기 쉽다).
    if not report or report.strip().startswith("("):
        lines.append(f"(리포트 미생성: {report or '리포트 없음'})")

    return {"investment_summary": "\n\n".join(lines)}


# 사업의 내용에서 뽑을 관점들 (각 질의로 관련 청크를 검색)
_ASPECTS = {
    "핵심이슈": "회사의 최근 핵심 이슈, 업황 변화, 주요 사건과 환경 변화, 매출과 실적을 좌우하는 제품군, 판가, 물량, 가동률, 전방 수요, 성장 동력, 경쟁력, 수주잔고, 신사업, 시장 점유율, 생산능력 증설",
    "리스크": "위험 요인, 원자재 가격, 환율, 규제, 소송, 경쟁 심화, 전방산업 부진",
}


def _biz_summary_chunks(state: AnalysisState, start_idx: int) -> list:
    """'사업의 개요' 문장들을 [Cxxx] 인용 체계에 맞는 합성 청크로 변환.
    Ollama 임베딩 없이도(즉 rag_retrieve의 벡터검색 성패와 무관하게) 항상
    citation.py로 검증 가능한 근거로 제공한다 — 안정성 강화 포인트."""
    summary = state.get("biz_summary") or {}
    sentences = summary.get("sentences") or []
    out = []
    for i, s in enumerate(sentences):
        out.append({
            "chunk_id": f"C{start_idx + i:03d}",
            "text": s,
            "heading": "사업의 개요/제품·사업현황",
            "start": -1, "end": -1,
        })
    return out


def rag_retrieve(state: AnalysisState) -> AnalysisState:
    """사업의 내용을 청킹·임베딩하고, 관점별로 관련 청크를 검색해 컨텍스트 구성.
    '사업의 개요' 요약 문장도 함께 합성 청크로 편입해(임베딩 불필요) 항상 인용
    가능하게 한다."""
    body = state.get("business_text")
    retrieved: list = []
    n_chunks = 0
    rag_error: Optional[Exception] = None

    if body:
        try:
            from rag import chunk_text, Retriever
            chunks = chunk_text(body)
            n_chunks = len(chunks)
            retr = Retriever()  # OllamaEmbedder(bge-m3) — 로컬/사설 Ollama 사용
            cache_key = f"{state.get('stock_code')}:{state.get('rcept_dt')}"
            retr.build(chunks, cache_key=cache_key)

            picked: dict = {}
            for q in _ASPECTS.values():
                for h in retr.query(q, k=4):
                    picked[h["chunk_id"]] = h   # 중복 제거(같은 청크 1회만)
            retrieved = list(picked.values())
        except Exception as e:  # noqa: BLE001
            rag_error = e

    biz_chunks = _biz_summary_chunks(state, start_idx=len(retrieved) + 1)
    retrieved = retrieved + biz_chunks

    if not retrieved:
        result: AnalysisState = {"n_chunks": n_chunks, "sources_status": {"rag": "empty" if not body else "error"}}
        if rag_error:
            result["errors"] = _append_error(state, "rag_retrieve", rag_error)
        return result

    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c.get('heading','')})\n{c['text']}" for c in retrieved)
    result = {
        "n_chunks": n_chunks,
        "retrieved_chunks": retrieved,
        "rag_context": context,
        "sources_status": {"rag": "ok"},
    }
    if rag_error:
        # 임베딩 검색은 실패했지만 사업의 개요 합성 청크는 확보됨 — 부분 성공으로 기록
        result["errors"] = _append_error(state, "rag_retrieve(embed)", rag_error)
        result["sources_status"] = {"rag": "partial"}
    return result


def _build_prompt_blocks(state: AnalysisState) -> dict:
    """analyze()가 LLM에 제시하는 구조화 블록들을 만든다. verify_scenario도 같은
    블록을 '신뢰 가능한 숫자 사전'으로 재사용해(scenario_check.py) 프롬프트 구성과
    환각 검증이 서로 다른 데이터를 보는 일이 없게 한다."""
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
        cons_block = ("(자체 집계 목표주가 컨센서스 없음 — company_list.naver 목록 페이지에는 "
                     "목표주가 컬럼이 없어 리포트별 집계 불가. 아래 [주가/밸류에이션]의 "
                     "컨센서스 목표주가를 참고하라.)")

    fng_cons = (state.get("fnguide") or {}).get("consensus") or {}
    if fng_cons.get("target_price") or fng_cons.get("opinion_label"):
        fng_cons_block = (
            f"- 투자의견: {fng_cons.get('opinion_label') or '자료 없음'}\n"
            f"- 목표주가: {fng_cons.get('target_price') or '자료 없음'}\n"
            f"- EPS/PER: {fng_cons.get('eps') or '-'}/{fng_cons.get('per') or '-'}\n"
            f"- 추정 참여 기관수: {fng_cons.get('analyst_count') or '자료 없음'} "
            f"(기준일 {fng_cons.get('date') or '미상'})"
        )
    else:
        fng_cons_block = "(FnGuide/WiseReport 컨센서스 없음 — 자료상 확인 불가)"

    reports = state.get("reports") or []
    rep_block = "\n".join(
        f"- {r.get('date','')} {r.get('broker','')} | {r.get('title','')}"
        for r in reports[:15]) or "(리포트 목록 없음)"

    fng = state.get("fnguide") or {}

    def _fmt_rows(rows: list, limit: int = 8) -> str:
        lines = []
        for r in rows[-limit:]:
            tag = "(E)" if r.get("is_estimate") else ""
            lines.append(f"  · {r.get('period')}{tag}: 매출 {r.get('revenue')}, "
                         f"영업이익 {r.get('op_profit')}, 순이익 {r.get('net_profit')}")
        return "\n".join(lines)

    ann_rows = fng.get("annual_highlight") or []
    qtr_rows = fng.get("financial_highlight") or []
    cf1002_rows = (fng.get("cf1002") or {}).get("rows") or []
    if ann_rows or qtr_rows or cf1002_rows:
        blocks = ["(단위: 억원, 출처 FnGuide/WiseReport — 직접 스크래핑, 개인 리서치 목적)"]
        if ann_rows:
            blocks.append("연간:\n" + _fmt_rows(ann_rows))
        if qtr_rows:
            blocks.append("분기(FnGuide 하이라이트):\n" + _fmt_rows(qtr_rows))
        if cf1002_rows:
            blocks.append(f"실측+예상 동일주기 시계열({(fng.get('cf1002') or {}).get('freq')}, "
                          f"WiseReport cF1002):\n" + _fmt_rows(cf1002_rows))
        est_block = "\n".join(blocks)
    else:
        est_block = "(연간/분기/예상 실적 없음)"

    disclosures = state.get("disclosures") or []
    discl_block = "\n".join(f"- ({d.get('date','')}) {d.get('title')}" for d in disclosures) \
        or "(최근 공시 없음)"

    price = state.get("price") or {}
    if price.get("price") is not None:
        price_block = (
            f"- 현재가: {price.get('price'):,.0f}원 (전일대비 {price.get('change_pct')}%)\n"
            f"- 52주 밴드: {price.get('low_52w'):,.0f} ~ {price.get('high_52w'):,.0f}원\n"
            f"- PER: {price.get('per')} / PBR: {price.get('pbr')} / "
            f"추정PER: {price.get('cns_per')}\n"
            f"- 컨센서스 목표주가: {price.get('target_price_mean')}"
        )
    else:
        price_block = "(시세 데이터 없음 — 자료상 확인 불가)"

    ip = state.get("invest_point") or {}
    ip_block = format_invest_point_block(ip) if ip else "(정량 투자포인트 계산 불가)"

    stability = state.get("stability") or {}
    stability_block = (
        stability_score.format_stability_block(stability["earnings"], stability["financial"])
        if stability else "(실적안정성/재무안정성 계산 불가)"
    )

    cc = state.get("cross_check") or {}
    cross_check_block = format_cross_check_block(cc) if cc else "(교차검증 미실행)"

    history = state.get("history") or []
    history_block = analysis_history.format_history_block(history)
    history_drift = analysis_history.check_drift(ip, history) if history else None

    retrieved = state.get("retrieved_chunks") or []
    valid_ids = ", ".join(c["chunk_id"] for c in retrieved) or "(없음)"

    return {
        "fin_block": fin_block, "news_block": news_block, "cons_block": cons_block,
        "fng_cons_block": fng_cons_block, "rep_block": rep_block, "est_block": est_block,
        "discl_block": discl_block, "price_block": price_block, "ip_block": ip_block,
        "stability_block": stability_block,
        "cross_check_block": cross_check_block, "history_block": history_block,
        "history_drift": history_drift, "valid_ids": valid_ids,
    }


def _analyze_core(state: AnalysisState, *, feedback: Optional[dict] = None) -> AnalysisState:
    """수집물 -> 이슈/투자포인트 + 성장/가치 시나리오 초안 (LLM).
    feedback이 주어지면(=1회 자동 재생성) 직전 시도에서 발견된 불일치를 콕 집어
    프롬프트에 덧붙인다."""
    if not state.get("rag_context") and not state.get("financials"):
        return {"report": "(분석 불가: 사업의 내용/재무 데이터를 확보하지 못했습니다.)"}

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from llm_backend import get_chat_model
    except ImportError as e:
        return {"errors": _append_error(state, "analyze(import)", e),
                "report": "(LLM 백엔드 모듈/패키지 미설치)"}

    try:
        # qwen3 등 'thinking' 모델은 내부 추론에 토큰을 상당히 쓰므로(특히 피드백
        # 블록이 붙는 재생성 시도), 2000으로는 최종 답변이 빈 문자열로 잘리는 경우가
        # 실측됐다. 4000으로 여유를 둔다(그래도 비면 위의 빈 응답 처리가 잡아준다).
        llm = get_chat_model(temperature=0.2, max_tokens=4000)
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "analyze(init)", e),
                "report": "(LLM 백엔드 초기화 실패 — LLM_BACKEND 설정 확인)"}

    system = (
        "너는 한국 주식 애널리스트의 리서치 보조다. 아래 '사업보고서 근거 청크'(사업의 "
        "내용 + 사업의 개요/제품·사업현황 요약 포함), 재무 시계열, 정량 투자포인트(성장/"
        "가치), 컨센서스, 공시, 뉴스만을 근거로 분석한다. 추론 과정은 출력하지 말고 "
        "결과만 한국어로 써라. 규칙(엄수):\n"
        "1) [핵심 이슈][투자포인트][리스크]에서 사업보고서 청크(사업의 내용/사업의 개요/"
        "제품·사업현황)의 서술 내용에 실제로 근거한 정성적 문장에만, 그 문장 끝에 진짜로 "
        "관련된 청크 id를 [C012] 형식으로 붙여라. 예: 'DR 시험 이행률이 105%로 높다 "
        "[C034]'. 인용 없는 정성적 단정 문장은 쓰지 마라. 관련 청크가 없는 정성적 주장은 "
        "아예 쓰지 마라(숫자 사실은 규칙3 참조). 매출/이익 증감을 말할 때는 반드시 어느 "
        "기간과 비교한 것인지 명시하라(예: '직전분기 대비 감소', '2026년(E) 대비 증가'). "
        "과거 실적은 부진했지만 예상 실적은 개선되는 경우(저점 통과) 두 서술이 방향은 "
        "달라도 정상이다 — 문제는 방향이 다른 게 아니라 기간을 명시하지 않아 어느 쪽 "
        "얘기인지 불분명한 것이니, 기간을 항상 같이 써서 모호함을 없애라. [투자포인트]에는 "
        "성장 시나리오, 가치 포지션과 함께 정량 투자포인트의 영업이익증감률·"
        "밴드위치·목표주가상승여력과 안정성(정량)의 실적안정성·재무안정성 점수까지 모두 "
        "그대로(대괄호 태그 없이) 반드시 포함하라(자료상 확인 불가면 그렇게만 써라). 부채비율과 "
        "그 업종 대비 등급(안정/적정/위험)은 안정성(정량) 데이터에서 그대로 가져와, 이미 "
        "청크를 인용한 문장 뒤에 이어 붙여 서술하라(수치·등급 자체에는 대괄호 태그를 붙이지 "
        "말 것 — 규칙3).\n"
        "2) 제공된 '사용 가능 청크 id' 목록에 있는 id만, 그리고 그 청크 내용과 실제로 "
        "관련된 문장에만 인용하라. 관련 없는 청크 id를 숫자 뒤에 습관적으로 붙이는 것도 "
        "환각 인용과 같은 수준의 오류다.\n"
        "3) 금액·비율 숫자(매출·이익·성장률·주가 등)는 투자포인트 정량 데이터, 재무 시계열, "
        "연간/분기/예상 실적에 실제로 제시된 값만 그대로 쓰고 새로 지어내지 마라. 이 숫자들은 "
        "사업보고서 청크가 아니라 코드 계산값/공시 수치가 출처이므로 숫자 자체 뒤에는 "
        "어떤 대괄호 표시도 붙이지 마라(청크 인용 [Cxxx]도, 블록 이름을 흉내 낸 다른 "
        "대괄호 태그도 금지 — 예를 들어 '[투자포인트(정량)]'처럼 데이터 블록 제목을 "
        "그대로 따와 문장 끝에 붙이는 것은 실재하지 않는 인용을 만드는 것과 같은 오류다). "
        "없으면 '구체 수치 자료 없음'이라 쓰고 방향(증가/감소/유지)만 기술하라.\n"
        "4) [성장 시나리오]는 투자포인트 정량 데이터의 '성장 판단'과 영업이익 증감률 숫자를 "
        "그대로(대괄호 태그 없이) 서술하라(자료상 확인 불가면 그렇게만 써라). 그 방향을 "
        "뒷받침하는 사업보고서상의 정성적 근거(증설/수주/신사업 등)가 있으면 별도 문장으로 "
        "덧붙이며 그 문장 끝에만 실제 관련 청크 id를 붙여라. 관련 청크가 없으면 정성 근거 "
        "문장 자체를 쓰지 말고 '정성적 근거는 자료상 확인 불가'라고만 써라.\n"
        "5) [가치 포지션]은 투자포인트 정량 데이터의 52주 밴드 위치·가치 시그널 여부를 "
        "그대로(대괄호 태그 없이) 서술하되, 반드시 숫자로 직접 비교해서 판단하라 — "
        "예를 들어 '밴드위치 X%가 하단 기준 Y% 이하'라고 쓰려면 X<=Y를 실제로 확인하고, "
        "X>Y이면 '하단 기준을 살짝 초과'처럼 정확하게 써라. 가치 시그널 필드가 '아니오'이면 "
        "그 이유(밴드 기준 미충족 또는 예상실적 방향 불명 등 구체적 이유)까지 정확히 설명하고, "
        "밴드 기준을 충족했다고 서술하면 안 된다. 시그널이 '아니오'인데 '충족'/'성립'이라고 "
        "서술하면 같은 절 안에서 스스로 모순되는 심각한 오류다.\n"
        "6) 근거 청크에 없는 내용은 '자료상 확인 불가'라고 명시하라.\n"
        "7) 제공된 블록에 없는 파생 수치(YoY%, 배수 등)를 직접 계산해 새로 만들지 마라. "
        "제공된 숫자만 그대로 쓰고, 시점 비교가 필요하면 '2026.03 대비'처럼 어느 기간 "
        "대비인지만 서술하되 새 계산값을 만들지 마라.\n"
        "8) 한 문장 또는 한 항목에는 그 내용과 직접 관련된 청크 id 하나만 붙여라. 여러 "
        "문장을 이어 쓰고 맨 끝에 인용 하나만 붙이지 마라 — 문장마다 그 문장의 실제 "
        "근거 청크를 따로 표시하라.\n"
        "9) 사업보고서 청크에서 금액 숫자를 인용할 때는 그 청크에 적힌 단위(예: 백만원, "
        "천원, 억원)를 반드시 그대로 확인하고 그 단위째로 써라. 다른 블록([재무 시계열], "
        "[투자포인트(정량) 등])은 억원 단위지만, 청크 원문 표는 백만원 단위인 경우가 많다 "
        "— 단위를 확인하지 않고 숫자만 그대로 '억원'이라 쓰면 100배 부풀리는 심각한 오류다. "
        "청크에 적힌 단위를 그대로 밝혀 쓰거나(예: '수주잔고 4,774억원(원문 477,440백만원)'), "
        "단위가 불확실하면 숫자를 쓰지 말고 '단위 확인 필요'라고 하라.\n"
        "10) 3단계 교차검증에서 [교차검증]의 어느 항목이 ❌(불일치)면, 그 항목의 숫자는 "
        "어느 쪽이 맞는지 스스로 판단하지 말고 '소스 간 불일치 — 사람 확인 필요'라고 "
        "그대로 밝혀라.\n"
        "11) [컨센서스 대비]에서 과거와 비교하려면 [과거 분석 이력]에 실제로 적힌 "
        "날짜·수치만 근거로 '지난 분석(YYYY-MM-DD) 대비 개선/악화' 식으로 서술하라. "
        "이력이 없거나('과거 분석 이력 없음') 비교할 수치가 없으면 비교 서술 자체를 "
        "하지 말고 '과거 이력 없음'이라고만 써라 — 이력에 없는 추세를 지어내지 마라.\n"
        "12) 출력 형식: [핵심 이슈] [투자포인트] [리스크] [성장 시나리오] [가치 포지션] "
        "[컨센서스 대비] [데이터 공백]"
    )

    b = _build_prompt_blocks(state)

    feedback_block = ""
    if feedback and not feedback.get("consistent", True):
        problems = "; ".join(feedback.get("problems", [])) or "불일치 발견"
        feedback_block = (
            "\n\n[이전 시도 검증 결과 — 반드시 수정]\n"
            f"직전 답변에서 다음 불일치가 발견되었다: {problems}\n"
            "[투자포인트(정량)]의 성장 판단/가치 시그널 값과 정확히 같은 방향으로 "
            "다시 서술하고, 어느 블록에도 없는 파생 수치를 새로 계산해 넣지 마라."
        )

    human = (
        f"종목: {state.get('corp_name')} ({state.get('stock_code')})\n"
        f"근거 공시: {state.get('report_nm')} (접수 {state.get('rcept_dt')})\n\n"
        f"[재무 시계열(실적, DART)]\n{b['fin_block']}\n\n"
        f"[연간/분기/예상 실적(FnGuide/WiseReport)]\n{b['est_block']}\n\n"
        f"[투자포인트(정량) — 성장/가치, 코드 계산값 그대로 인용]\n{b['ip_block']}\n\n"
        f"[안정성(정량) — 실적안정성/재무안정성, 코드 계산값 그대로 인용]\n{b['stability_block']}\n\n"
        f"[주가/밸류에이션]\n{b['price_block']}\n\n"
        f"[교차검증 — 서로 다른 소스 간 정합성]\n{b['cross_check_block']}\n\n"
        f"[과거 분석 이력 — 최신순, 이 안의 날짜·수치만 비교에 사용 가능]\n{b['history_block']}\n\n"
        f"[애널리스트 컨센서스(자체 집계)]\n{b['cons_block']}\n\n"
        f"[FnGuide/WiseReport 컨센서스]\n{b['fng_cons_block']}\n\n"
        f"[최근 리포트 목록]\n{b['rep_block']}\n\n"
        f"[최근 공시]\n{b['discl_block']}\n\n"
        f"[최근 뉴스]\n{b['news_block']}\n\n"
        f"[사용 가능 청크 id — 이 중에서만 인용]\n{b['valid_ids']}\n\n"
        f"[사업보고서 근거 청크 (이 id만 인용 가능, 사업의 개요/제품·사업현황 포함)]\n"
        f"{state.get('rag_context', '(없음)')}"
        f"{feedback_block}"
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        if not text.strip():
            # 실측 원인: qwen3 등 'thinking' 모델이 재생성 시도(피드백 블록으로 프롬프트가
            # 길어짐)에서 내부 추론만 하다 max_tokens를 다 써버려 최종 답변이 빈 문자열로
            # 오는 경우가 있다. 빈 문자열을 그대로 report로 넘기면 verify_scenario가
            # '검증 대상 없음(consistent=True)'로 잘못 통과시켜 실패가 성공처럼 보인다.
            return {"errors": _append_error(
                        state, "analyze(llm)",
                        RuntimeError("LLM 응답이 비어 있음(추론만 하고 답변 미생성 가능성)")),
                    "report": "(LLM 응답 비어있음 — 재시도 필요)"}
        return {"report": text, "history_drift": b.get("history_drift")}
    except Exception as e:  # noqa: BLE001  (네트워크/키 등 광범위)
        return {"errors": _append_error(state, "analyze(llm)", e),
                "report": "(LLM 호출 실패)"}


def analyze(state: AnalysisState) -> AnalysisState:
    return _analyze_core(state)


def regenerate_analysis(state: AnalysisState) -> AnalysisState:
    """3단계: scenario_check가 불일치를 발견했을 때 1회만 호출되는 재생성 노드.
    analyze로 직접 되돌아가는 사이클 대신 별도 노드로 둬서(그래프 조립 참고)
    '최대 1회'를 구조적으로 보장한다."""
    result = _analyze_core(state, feedback=state.get("scenario_consistency"))
    result["regenerated"] = True
    return result


def _verify_citations_core(state: AnalysisState) -> AnalysisState:
    """LLM 리포트의 [Cxxx] 인용이 실재 청크인지 + 근거가 맞는지 검증(그라운딩)."""
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


def verify_citations(state: AnalysisState) -> AnalysisState:
    return _verify_citations_core(state)


def verify_citations_2(state: AnalysisState) -> AnalysisState:
    return _verify_citations_core(state)


def _verify_scenario_core(state: AnalysisState) -> AnalysisState:
    """3단계 2차 검증 레이어: LLM 리포트가 invest_point 시그널과 같은 방향을
    말하는지 + 제공되지 않은 파생 수치를 지어내지 않았는지(환각 통제)."""
    report = state.get("report", "")
    # analyze()가 실패하면 항상 "(...)" 형태의 플레이스홀더를 report에 넣는다
    # (데이터 부족/LLM 호출 실패/빈 응답 등). 이걸 '검증할 게 없으니 일관됨(True)'
    # 으로 처리하면 생성 실패가 통과처럼 보이는 실측 버그가 있었다 — 명확히
    # 실패로 표시해서 재생성 트리거/최종 경고까지 이어지게 한다.
    if not report or report.strip().startswith("("):
        return {"scenario_consistency": {
            "consistent": False,
            "problems": ["리포트 생성 실패"],
            "verdict": f"❌ 리포트 생성 실패: {report or '(리포트 없음)'}",
        }}
    try:
        import scenario_check
        blocks = _build_prompt_blocks(state)
        text_blocks = [v for v in blocks.values() if isinstance(v, str)]
        result = scenario_check.check(report, state.get("invest_point") or {}, text_blocks)
        return {"scenario_consistency": result}
    except Exception as e:  # noqa: BLE001
        return {"errors": _append_error(state, "verify_scenario", e)}


def verify_scenario(state: AnalysisState) -> AnalysisState:
    return _verify_scenario_core(state)


def verify_scenario_2(state: AnalysisState) -> AnalysisState:
    return _verify_scenario_core(state)


def _route_after_scenario(state: AnalysisState) -> str:
    """불일치가 있고 아직 재생성한 적 없으면 딱 1회 regenerate_analysis로,
    그 외에는 END로. regenerated 플래그로 재귀 방지(사이클이 아니라 별도
    노드이므로 원래도 1회 이상 못 돌지만, 라우팅 의도를 명시적으로 남긴다)."""
    sc = state.get("scenario_consistency") or {}
    if sc.get("consistent", True):
        return "end"
    if state.get("regenerated"):
        return "end"
    return "regenerate"


# ---------------------------------------------------------------------- #
# 그래프 조립
# ---------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(AnalysisState)
    g.add_node("resolve_corp", resolve_corp)
    g.add_node("fetch_business", fetch_business)
    g.add_node("fetch_biz_summary", fetch_biz_summary)
    g.add_node("rag_retrieve", rag_retrieve)
    g.add_node("fetch_financials", fetch_financials)
    g.add_node("fetch_news", fetch_news)
    g.add_node("fetch_research", fetch_research)
    g.add_node("fetch_fnguide", fetch_fnguide)
    g.add_node("fetch_price", fetch_price)
    g.add_node("fetch_financial_history", fetch_financial_history)
    g.add_node("fetch_industry_category", fetch_industry_category)
    g.add_node("build_stability", build_stability)
    g.add_node("fetch_disclosures", fetch_disclosures)
    g.add_node("build_invest_point", build_invest_point)
    g.add_node("cross_check_sources", cross_check_sources)
    g.add_node("load_history", load_history)
    # analyze는 서로 다른 홉 깊이의 가지(1홉: fetch_news/fetch_disclosures/load_history,
    # 2홉: rag_retrieve/build_invest_point/cross_check_sources)가 합류하는 지점이다.
    # LangGraph는 기본적으로 이런 비대칭 fan-in에서 먼저 도착한 얕은 가지들만으로
    # 노드를 한 번 실행한 뒤, 늦게 도착한 깊은 가지가 채워지면 다시 한 번 더
    # 실행한다(실측 확인 — analyze가 LLM을 두 번 호출하고 save_history가 행을
    # 두 번 쓰는 원인이었음). defer=True로 '이 실행에 남은 다른 작업이 없을 때까지'
    # 대기시켜 정확히 1회만 실행되게 한다.
    g.add_node("analyze", analyze, defer=True)
    g.add_node("verify_citations", verify_citations)
    g.add_node("verify_scenario", verify_scenario)
    g.add_node("regenerate_analysis", regenerate_analysis)
    g.add_node("verify_citations_2", verify_citations_2)
    g.add_node("verify_scenario_2", verify_scenario_2)
    # 4단계: 두 종료 경로 각각 전용 노드 인스턴스로 분리(save_history와 동일한 이유 —
    # 공유 노드는 두 경로에서 두 번 실행되는 문제가 실측 확인됨).
    g.add_node("build_summary", build_investment_summary)
    g.add_node("build_summary_2", build_investment_summary)
    # save_history를 두 종료 경로가 공유하면 LangGraph가 노드를 두 번(각 경로당
    # 한 번씩) 실행해 DB에 중복 저장하는 문제가 실측 확인됨 — verify_citations와
    # 동일하게 경로별 전용 노드 인스턴스로 분리해 각 경로가 정확히 1회만 저장하게 한다.
    g.add_node("save_history", save_history)
    g.add_node("save_history_2", save_history)

    g.add_edge(START, "resolve_corp")
    # 사업보고서 체인: 본문(RAG) + 사업의 개요(합성 청크) → rag_retrieve에서 병합 → analyze
    g.add_edge("resolve_corp", "fetch_business")
    g.add_edge("resolve_corp", "fetch_biz_summary")
    g.add_edge("fetch_business", "rag_retrieve")
    g.add_edge("fetch_biz_summary", "rag_retrieve")
    g.add_edge("rag_retrieve", "analyze")
    # 2단계 정량 시그널 체인: fnguide + 시세 → build_invest_point → analyze.
    # analyze에 새 엣지를 추가하지 않는다 — analyze는 defer=True로 등록된 비대칭
    # fan-in 지점이라, 새 수집 노드를 여기 직접 연결하면 fan-in 깊이가 또 바뀌어
    # 더블 실행 버그가 재발할 위험이 있다(위 주석 참조).
    g.add_edge("resolve_corp", "fetch_fnguide")
    g.add_edge("resolve_corp", "fetch_price")
    g.add_edge("fetch_fnguide", "build_invest_point")
    g.add_edge("fetch_price", "build_invest_point")
    g.add_edge("build_invest_point", "analyze")
    # 3단계 안정성 점수 체인: 연간 재무제표 + 업종카테고리 + fnguide(실적안정성용
    # 연간 실측·향후분기) → build_stability → analyze. build_invest_point와 동일한
    # 이유로 depth 2를 유지하고 analyze에는 직접 연결하지 않는다. fetch_fnguide는
    # 이미 depth 1(resolve_corp 바로 다음)이라 여기 엣지를 추가해도 depth는 안 바뀐다.
    g.add_edge("resolve_corp", "fetch_financial_history")
    g.add_edge("resolve_corp", "fetch_industry_category")
    g.add_edge("fetch_fnguide", "build_stability")
    g.add_edge("fetch_financial_history", "build_stability")
    g.add_edge("fetch_industry_category", "build_stability")
    g.add_edge("build_stability", "analyze")
    # 나머지 수집 노드는 병렬로 analyze에 fan-in
    g.add_edge("resolve_corp", "fetch_financials")
    g.add_edge("resolve_corp", "fetch_news")
    g.add_edge("resolve_corp", "fetch_research")
    g.add_edge("resolve_corp", "fetch_disclosures")
    g.add_edge("fetch_news", "analyze")
    g.add_edge("fetch_disclosures", "analyze")
    # 3단계 교차검증: 목표주가/투자의견/가격밴드를 소스 간 대조 → analyze에도 블록으로 제시
    g.add_edge("fetch_financials", "cross_check_sources")
    g.add_edge("fetch_fnguide", "cross_check_sources")
    g.add_edge("fetch_price", "cross_check_sources")
    g.add_edge("fetch_research", "cross_check_sources")
    g.add_edge("cross_check_sources", "analyze")
    # 3단계 이력: 같은 종목의 과거 분석을 불러와 analyze에 참조 자료로 제공
    g.add_edge("resolve_corp", "load_history")
    g.add_edge("load_history", "analyze")
    # 분석 → 인용 그라운딩 검증 → 시나리오 일관성/환각 검증
    g.add_edge("analyze", "verify_citations")
    g.add_edge("verify_citations", "verify_scenario")
    # 불일치 발견 시 딱 1회만 재생성(사이클 아님 — 전용 노드로 선형 분기)
    g.add_conditional_edges("verify_scenario", _route_after_scenario,
                            {"regenerate": "regenerate_analysis", "end": "build_summary"})
    g.add_edge("regenerate_analysis", "verify_citations_2")
    g.add_edge("verify_citations_2", "verify_scenario_2")
    # 두 종료 경로(재생성 없음 / 1회 재생성 후) 각각 4단계 요약 → 저장 노드를 거쳐 END로
    g.add_edge("build_summary", "save_history")
    g.add_edge("save_history", END)
    g.add_edge("verify_scenario_2", "build_summary_2")
    g.add_edge("build_summary_2", "save_history_2")
    g.add_edge("save_history_2", END)
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

    ip = result.get("invest_point")
    if ip:
        print("\n--- 정량 투자포인트(성장/가치) ---")
        print(format_invest_point_block(ip))

    cc = result.get("cross_check")
    if cc:
        print("\n--- 교차검증(3단계) ---")
        print(format_cross_check_block(cc))
        if not cc.get("all_ok", True):
            print(" ⚠️  소스 간 불일치 발견 — 위 표시된 항목은 사람이 직접 확인하세요.")

    hist = result.get("history")
    print("\n--- 과거 분석 이력(3단계) ---")
    print(analysis_history.format_history_block(hist or []))
    drift = result.get("history_drift")
    if drift:
        print(f" ⚠️  과거({drift['prev_run_at']}) 대비 근거 수치는 유지되는데 판단만 "
              f"바뀐 항목이 있습니다:")
        for f in drift["flips"]:
            print(f"   - {f}")

    if result.get("regenerated"):
        print("\n(※ 시나리오 일관성 검증에서 불일치가 발견되어 1회 자동 재생성됨)")

    print()
    print(result.get("report", "(리포트 없음)"))

    inv_summary = result.get("investment_summary")
    if inv_summary:
        print("\n--- 4단계: 컨센서스+리포트 결합 투자포인트 요약 ---")
        print(inv_summary)

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

    sc = result.get("scenario_consistency", {})
    if sc and "verdict" in sc:
        print("\n--- 시나리오 일관성/환각 통제 검증(3단계) ---")
        print(f" 판정: {sc['verdict']}")
        if not sc.get("consistent", True):
            tag = "1회 재생성 후에도 불일치 지속" if result.get("regenerated") else "재생성 미실행"
            print(f" ⚠️  경고: {tag} — 성장/가치 서술을 정량 시그널과 대조해 수동 확인하세요.")
        if sc.get("trend_check") and not sc["trend_check"]["match"]:
            print(f"   - 성장 판단 기대값: {sc['trend_check']['expected']} / "
                  f"리포트 절: {sc['trend_check']['section'][:80]}")
        if sc.get("signal_check") and not sc["signal_check"]["match"]:
            print(f"   - 가치 시그널 기대값: "
                  f"{'예' if sc['signal_check']['expected'] else '아니오'} / "
                  f"리포트 절: {sc['signal_check']['section'][:80]}")
        if sc.get("unverified_numbers"):
            print(f"   - 미검증 파생수치: {sc['unverified_numbers'][:5]}")

    if result.get("errors"):
        print("\n--- 수집/분석 경고 ---")
        for e in result["errors"]:
            print(" •", e)
