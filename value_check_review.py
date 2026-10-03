"""
value_check_review.py
=====================
투자관리 화면에서 담당자가 입력한 '가치주 체크 사항'을 근거 자료와 대조해,
기존 가치투자 검토문(analysis_history.value_invest)을 보완하는 검토 섹션을 만든다.

근거 자료(기존 경로 + 추가):
  - 가치투자 검토 원본(analysis_history.value_invest)
  - 담당자 체크 사항(value_check)
  - 최근 90일 증권사 리포트 최신 10건(external_sources.fetch_naver_research — 의견·
    목표주가·증권사 요약. PDF 본문 없음)
  - 체크 사항 키워드로 검색한 종목 뉴스(external_sources.fetch_naver_news)

잘못 요약할 위험을 줄이는 장치(mvp_graph.check_value_investment와 같은 규율):
  1) 원본 검토문은 다시 쓰지 않는다 — 보완 섹션만 생성해 원본 뒤에 붙인다.
  2) 리포트·뉴스에 [R1]/[N1] id를 붙이고 목록에 있는 id만 인용하게 한다.
  3) 생성 후 코드로 검증: 필수 태그 / 목록 밖 인용 id / 자료에 없는 수치.
  4) LLM 자기검증: 자료에 근거하지 않은 단정 문장을 나열하게 한다.
  5) 3)·4)에서 문제가 나오면 피드백을 붙여 딱 1회 재생성. 그래도 3)이 실패하면
     자동 검토를 포기하고 체크 사항만 덧붙이는 안전한 형태로 폴백한다.
     4)만 남으면 해당 문장을 표시해 담당자가 확인하게 한다(최종 반영은 사람이 결정).

simul_server.py가 백그라운드 작업으로 run_review()를 호출한다.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

try:  # simul_server처럼 다른 작업 디렉터리에서 import돼도 이 프로젝트 .env를 읽도록
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:  # noqa: BLE001
    pass

logger = logging.getLogger("value_check_review")

NEWS_DAYS = 90
NEWS_LIMIT = 10
REVIEW_TAGS = ("체크사항 검토", "보완 내용", "종합 결론")
CONCLUSIONS = ("일시적 저평가로 추정", "구조적 저평가(가치함정 우려)", "판단 보류(근거 부족)")

# 주가·지수 움직임 위주 기사(simul_server._naver_issue_news와 같은 기준 + 특징주류)
_MARKET_MOVE_RE = re.compile(r"코스피|코스닥|증시|지수|서킷브레이커|특징주|상한가|하한가")
_CITE_RE = re.compile(r"\[([RN])(\d+)\]")
_ANY_CITE_RE = re.compile(r"\[(?:[RNC])\d+\]")
# 숫자 + (있으면) 바로 뒤 단위 첫 글자. '1,686억원'을 '1,686원'으로 옮기는 단위 누락도
# 잡기 위해 단위까지 비교한다(실측: qwen3:14b가 억 단위를 빠뜨린 사례).
_NUM_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만|천|원|%|배|주)?")


# ---------------------------------------------------------------------- #
# LLM 호출 공통
# ---------------------------------------------------------------------- #
class _OllamaChat:
    """Ollama /api/chat를 requests로 직접 호출하는 최소 클라이언트.

    simul_server가 이 모듈을 자기 프로세스에서 import하는데, 그 Python 환경에는
    langchain_ollama/langchain_core가 없을 수 있다(실측: "No module named
    'langchain_ollama'"로 검토 작업 실패). 이 기능은 단순 챗 호출뿐이라 langchain
    없이 requests만으로 호출해 의존성을 없앤다. 설정값(OLLAMA_MODEL/BASE_URL/
    NUM_CTX)은 llm_backend.get_chat_model과 같은 환경변수를 쓴다."""

    def __init__(self, temperature: float, max_tokens: int, think: bool,
                 timeout: int = 900):
        import os
        self.url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.options = {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384"))}
        self.think = think
        self.timeout = timeout

    def chat(self, system: str, human: str) -> str:
        import requests
        payload = {"model": self.model, "stream": False, "think": self.think,
                   "options": self.options,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": human}]}
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama 호출 실패(HTTP {r.status_code}): {r.text[:300]}")
        return ((r.json().get("message") or {}).get("content") or "")


def _llm_text(llm, system: str, human: str) -> str:
    if hasattr(llm, "chat"):              # _OllamaChat(기본)
        text = llm.chat(system, human)
    else:                                 # LangChain 챗 모델(ollama 외 백엔드)
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
    # think 분리를 지원하지 않는 Ollama/모델 대비: 본문에 섞인 think 블록 제거.
    # 화면은 일반 텍스트 영역이라 마크다운 굵게(**)는 지운다.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.replace("**", "").strip()


def _default_llm(kind: str):
    import os
    review = kind == "review"
    if os.getenv("LLM_BACKEND", "ollama").lower() == "ollama":
        return _OllamaChat(temperature=0.2 if review else 0.0,
                           max_tokens=3000 if review else 800, think=review)
    from llm_backend import get_chat_model
    return get_chat_model(temperature=0.2 if review else 0.0, max_tokens=3000 if review else 800)


# ---------------------------------------------------------------------- #
# 1) 체크 사항 → 뉴스 검색 키워드
# ---------------------------------------------------------------------- #
_KEYWORD_SYSTEM = (
    "너는 뉴스 검색어를 뽑는 도우미다. 담당자 메모를 읽고 해당 종목 뉴스 검색에 쓸 "
    "핵심 키워드 2~4개를 쉼표로만 구분해 한 줄로 출력해라. 규칙: 종목명은 넣지 마라. "
    "각 키워드는 2~12자 명사구. 번호·따옴표·설명 없이 키워드만."
)


def _fallback_keywords(value_check: str, corp_name: str) -> list[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]{2,12}", value_check)
    stop = {"있는지", "있음", "없음", "확인", "필요", "여부", "관련", "때문", "경우", "대한", corp_name}
    out = []
    for w in words:
        if w not in stop and w not in out:
            out.append(w)
    return out[:3]


def extract_keywords(value_check: str, corp_name: str, llm=None) -> list[str]:
    try:
        llm = llm or _default_llm("small")
        raw = _llm_text(llm, _KEYWORD_SYSTEM, f"종목명: {corp_name}\n담당자 메모:\n{value_check}")
    except Exception as e:  # noqa: BLE001  (키워드만 못 뽑은 것 — 단순 추출로 대체)
        logger.warning("키워드 추출 LLM 실패: %s", e)
        return _fallback_keywords(value_check, corp_name)
    out = []
    for k in re.split(r"[,\n、/]", raw):
        k = re.sub(r"^[\s\-\d.)·*\"'#]+|[\s\"'.]+$", "", k).replace(corp_name, "").strip()
        if 2 <= len(k) <= 15 and k not in out:
            out.append(k)
    return out[:4] or _fallback_keywords(value_check, corp_name)


# ---------------------------------------------------------------------- #
# 2) 키워드 뉴스 수집
# ---------------------------------------------------------------------- #
def _parse_pub_date(s: str) -> Optional[datetime]:
    try:
        return parsedate_to_datetime(s)
    except Exception:  # noqa: BLE001
        return None


def collect_news(corp_name: str, keywords: list[str], *, days: int = NEWS_DAYS,
                 limit: int = NEWS_LIMIT, fetch: Optional[Callable] = None) -> list[dict]:
    """종목명+키워드로 검색한 뉴스 중, 최근 days일·종목명 언급·시세성 기사 제외 조건을
    통과하고 키워드가 1개 이상 들어간 기사만 키워드 일치 수 → 최신순으로 limit건."""
    if fetch is None:
        from external_sources import fetch_naver_news as fetch
    since = datetime.now(timezone.utc) - timedelta(days=days)
    ko_name = re.sub(r"[^가-힣]", "", corp_name)
    names = [n for n in {corp_name, ko_name} if len(n) >= 2]

    pool: dict[str, dict] = {}
    for kw in keywords:
        try:
            items = fetch(f"{corp_name} {kw}", display=30, sort="date")
        except Exception as e:  # noqa: BLE001
            logger.warning("뉴스 검색 실패(%s %s): %s", corp_name, kw, e)
            continue
        for it in items:
            title, summary = it.get("title", ""), it.get("summary", "")
            pub = _parse_pub_date(it.get("pub_date", ""))
            if not pub or pub < since:
                continue
            if not any(n in title or n in summary for n in names):
                continue
            if _MARKET_MOVE_RE.search(title):
                continue
            key = re.sub(r"\W", "", title)[:40]
            if key in pool:
                continue
            text = f"{title} {summary}"
            matched = [k for k in keywords if k in text]
            if not matched:
                continue
            pool[key] = {"title": title, "summary": summary, "link": it.get("link", ""),
                         "date": pub.strftime("%Y-%m-%d"), "matched": matched, "_ts": pub.timestamp(),
                         "_in_title": any(n in title for n in names)}
    # 제목에 종목명이 있는 기사 우선 — 업종 나열 기사(요약에만 종목명이 스치듯 나오는 것)는
    # 실측상 무관한 내용이 많아(예: 다른 회사 원전 수출 기사) 뒤로 보낸다.
    ranked = sorted(pool.values(), key=lambda n: (n["_in_title"], len(n["matched"]), n["_ts"]),
                    reverse=True)[:limit]
    for n in ranked:
        n.pop("_ts", None)
        n.pop("_in_title", None)
    return ranked


# ---------------------------------------------------------------------- #
# 3) 프롬프트
# ---------------------------------------------------------------------- #
_REVIEW_SYSTEM = (
    "너는 가치투자 검토문을 보완하는 편집자다. 아래 자료만 근거로, 담당자 체크 사항을 "
    "검토하고 기존 가치투자 검토문에 덧붙일 보완 섹션을 한국어로 써라. 기존 검토문을 "
    "다시 쓰거나 요약하지 마라. 추론 과정은 출력하지 말고 결과만 써라. 규칙(엄수):\n"
    "1) 근거는 [가치투자 검토 원본], [담당자 체크 사항], [최근 리포트], [체크 사항 관련 "
    "뉴스]에 실제로 적힌 내용뿐이다. 어디에도 없는 사실·수치·전망을 새로 만들지 마라.\n"
    "2) 리포트·뉴스 내용을 쓸 때는 그 문장 끝에 [R1], [N3] 형식으로 인용하라. '사용 가능 "
    "id' 목록에 없는 id는 절대 쓰지 마라.\n"
    "3) 리포트는 제목·의견·목표주가·증권사 요약만 제공됐다. 제공되지 않은 리포트 본문을 "
    "추측하지 마라.\n"
    "4) 숫자는 자료에 적힌 표기 그대로만 옮겨라. 새로 계산하거나 단위를 바꾸지 마라.\n"
    "5) 담당자 체크 사항의 항목마다 자료가 '뒷받침' / '반박' / '자료상 확인 불가' 중 "
    "무엇인지 밝혀라. 자료와 담당자 의견이 상충하면 둘 다 적고 '(담당자 의견)'을 표시하라.\n"
    "6) 출력 형식: [체크사항 검토] [보완 내용] [종합 결론] — 이 3개 대괄호 태그를 정확히 "
    "이 글자 그대로, 각 내용 앞에 붙여라.\n"
    "   [체크사항 검토]: 체크 사항 항목별 판정과 근거.\n"
    "   [보완 내용]: 기존 검토문에 추가·수정이 필요한 점(근거 인용 포함). 없으면 '추가 "
    "보완 사항 없음'.\n"
    "   [종합 결론]: '일시적 저평가로 추정' / '구조적 저평가(가치함정 우려)' / '판단 "
    "보류(근거 부족)' 중 하나를 고르고 이유를 한 문장으로. 근거가 대부분 확인 불가면 "
    "'판단 보류(근거 부족)'."
)


def _fmt_report(i: int, r: dict) -> str:
    meta = []
    if r.get("opinion"):
        meta.append(f"의견 {r['opinion']}")
    tp, prev = r.get("target_price"), r.get("prev_target_price")
    if tp:
        chg = ""
        if prev and prev != tp:
            chg = f", 직전 {prev:,}원 대비 {'상향' if tp > prev else '하향'}"
        meta.append(f"목표주가 {tp:,}원{chg}")
    head = f"[R{i}] {r.get('date', '')} {r.get('broker', '')} | {r.get('title', '')}"
    if meta:
        head += f" ({'; '.join(meta)})"
    summ = (r.get("summary") or "").strip()
    if summ:
        head += f"\n     요약: {summ[:300]}"
    return head


def _fmt_news(i: int, n: dict) -> str:
    summ = (n.get("summary") or "").strip()
    return f"[N{i}] {n.get('date', '')} {n.get('title', '')}" + (f"\n     요약: {summ[:250]}" if summ else "")


def build_source_blocks(original: str, value_check: str, reports: list[dict],
                        news: list[dict]) -> dict:
    rep_block = "\n".join(_fmt_report(i, r) for i, r in enumerate(reports, 1)) or "(최근 90일 리포트 없음)"
    news_block = "\n".join(_fmt_news(i, n) for i, n in enumerate(news, 1)) or "(체크 사항 관련 뉴스 없음)"
    valid = [f"R{i}" for i in range(1, len(reports) + 1)] + [f"N{i}" for i in range(1, len(news) + 1)]
    return {
        "original": (original or "").strip() or "(가치투자 검토 원본 없음)",
        "value_check": value_check.strip(),
        "rep_block": rep_block,
        "news_block": news_block,
        "valid_ids": ", ".join(valid) or "(없음 — 인용하지 마라)",
        "valid_set": set(valid),
    }


def _human(b: dict, corp_name: str, stock_code: str, feedback: str = "") -> str:
    fb = f"\n\n[이전 답변의 문제 — 반드시 고쳐서 다시 써라]\n{feedback}" if feedback else ""
    return (
        f"종목: {corp_name} ({stock_code})\n\n"
        f"[가치투자 검토 원본]\n{b['original']}\n\n"
        f"[담당자 체크 사항]\n{b['value_check']}\n\n"
        f"[최근 리포트 — 최근 90일, 본문 없음]\n{b['rep_block']}\n\n"
        f"[체크 사항 관련 뉴스]\n{b['news_block']}\n\n"
        f"[사용 가능 id — 이 중에서만 인용]\n{b['valid_ids']}"
        f"{fb}"
    )


# ---------------------------------------------------------------------- #
# 4) 검증
# ---------------------------------------------------------------------- #
def _norm_num(s: str) -> str:
    """천단위 쉼표 제거 + 숫자와 단위 사이 공백 제거('396 억원' → '396억원')."""
    return re.sub(r"(\d)\s+(?=[조억만천원%배주])", r"\1", s.replace(",", ""))


def check_output(text: str, b: dict) -> list[str]:
    """코드 검증: 필수 태그, 목록 밖 인용 id, 자료에 없는 수치. 문제 목록을 반환."""
    problems = []
    if not text.strip():
        return ["응답이 비어 있음"]
    missing = [t for t in REVIEW_TAGS if f"[{t}]" not in text]
    if missing:
        problems.append("필수 태그 누락: " + ", ".join(f"[{t}]" for t in missing))
    bad_ids = sorted({f"{k}{n}" for k, n in _CITE_RE.findall(text)} - b["valid_set"])
    if bad_ids:
        problems.append("목록에 없는 인용 id 사용: " + ", ".join(bad_ids))

    source = _norm_num(" ".join([b["original"], b["value_check"], b["rep_block"], b["news_block"]]))
    body = _ANY_CITE_RE.sub(" ", text)
    unknown = []
    for num, unit in _NUM_RE.findall(body):
        n = _norm_num(num).rstrip(".")
        if len(n.replace(".", "")) < 2:      # 한 자리 숫자(항목 번호 등)는 제외
            continue
        token = f"{n}{unit or ''}"
        # 숫자 앞뒤가 다른 숫자에 붙은 부분일치('1686'이 '11686' 안에 있는 경우)는 제외
        found = re.search(rf"(?<![\d.]){re.escape(token)}", source) if unit else \
            re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", source)
        if not found and token not in unknown:
            unknown.append(token)
    if unknown:
        problems.append("자료에 없는 수치: " + ", ".join(unknown[:10]))
    return problems


_SELF_CHECK_SYSTEM = (
    "너는 사실 검증 담당자다. [검토 결과]의 각 문장이 [자료]에 근거하는지 확인해라. "
    "자료에 없는 사실·수치·전망을 단정한 문장만 한 줄에 하나씩 '- ' 로 시작해 그대로 "
    "인용해 나열하라. 판단·결론 문장이나 '자료상 확인 불가'라고 쓴 문장은 문제가 아니다. "
    "문제가 없으면 '없음' 한 단어만 출력해라."
)


def self_check(text: str, b: dict, llm) -> list[str]:
    human = (
        f"[자료]\n가치투자 검토 원본:\n{b['original']}\n\n담당자 체크 사항:\n{b['value_check']}\n\n"
        f"최근 리포트:\n{b['rep_block']}\n\n관련 뉴스:\n{b['news_block']}\n\n"
        f"[검토 결과]\n{text}"
    )
    raw = _llm_text(llm, _SELF_CHECK_SYSTEM, human)
    if raw.strip().startswith("없음"):
        return []
    return [ln.lstrip("-• ").strip() for ln in raw.splitlines()
            if ln.strip().startswith(("-", "•")) and len(ln.strip()) > 3][:8]


# ---------------------------------------------------------------------- #
# 5) 실행
# ---------------------------------------------------------------------- #
def _sources_section(text: str, reports: list[dict], news: list[dict]) -> str:
    """보완 섹션이 실제 인용한 R/N id만 출처 목록으로 남긴다(저장 후에도 인용 추적 가능)."""
    cited = {f"{k}{n}" for k, n in _CITE_RE.findall(text)}
    lines = []
    for i, r in enumerate(reports, 1):
        if f"R{i}" in cited:
            lines.append(f"[R{i}] {r.get('date', '')} {r.get('broker', '')} \"{r.get('title', '')}\" {r.get('report_url', '')}".rstrip())
    for i, n in enumerate(news, 1):
        if f"N{i}" in cited:
            lines.append(f"[N{i}] {n.get('date', '')} \"{n.get('title', '')}\" {n.get('link', '')}".rstrip())
    return "\n".join(lines)


def run_review(stock_code: str, corp_name: str, original: str, value_check: str, *,
               llm_review=None, llm_small=None, fetch_reports: Optional[Callable] = None,
               fetch_news: Optional[Callable] = None) -> dict:
    """반환: {status: ok|warn|fallback, review(저장용 보완 섹션 전체), body(LLM 본문),
    keywords, reports, news, problems(코드 검증), flagged(자기검증 지적), regenerated,
    elapsed_sec}"""
    t0 = time.time()
    value_check = (value_check or "").strip()
    if not value_check:
        raise ValueError("가치주 체크 사항이 비어 있습니다.")
    if fetch_reports is None:
        from external_sources import fetch_naver_research as fetch_reports

    try:
        reports = fetch_reports(stock_code)
    except Exception as e:  # noqa: BLE001  (리포트 없이도 검토는 진행)
        logger.warning("리포트 수집 실패(%s): %s", stock_code, e)
        reports = []

    llm_small = llm_small or _default_llm("small")
    keywords = extract_keywords(value_check, corp_name, llm_small)
    news = collect_news(corp_name, keywords, fetch=fetch_news)
    b = build_source_blocks(original, value_check, reports, news)

    llm_review = llm_review or _default_llm("review")
    body = _llm_text(llm_review, _REVIEW_SYSTEM, _human(b, corp_name, stock_code))
    problems = check_output(body, b)
    flagged = self_check(body, b, llm_small) if not problems else []
    regenerated = False

    if problems or flagged:
        regenerated = True
        feedback = "\n".join([f"- {p}" for p in problems] +
                             [f"- 자료에 근거 없는 문장: {f}" for f in flagged])
        body2 = _llm_text(llm_review, _REVIEW_SYSTEM, _human(b, corp_name, stock_code, feedback))
        problems = check_output(body2, b)
        flagged = self_check(body2, b, llm_small) if not problems else []
        body = body2

    today = date.today().isoformat()
    header = f"[담당자 체크사항 반영 검토 — {today}]"
    if problems:
        status = "fallback"
        review = (f"{header}\n[담당자 체크 사항]\n{value_check}\n"
                  f"(자동 검토 결과가 검증을 통과하지 못해 체크 사항만 기록함: {'; '.join(problems)})")
    else:
        status = "warn" if flagged else "ok"
        sources = _sources_section(body, reports, news)
        review = f"{header}\n[담당자 체크 사항]\n{value_check}\n\n{body}"
        if sources:
            review += f"\n\n[근거 자료]\n{sources}"

    return {
        "status": status, "review": review, "body": body, "keywords": keywords,
        "reports": [{"id": f"R{i}", **{k: r.get(k) for k in ("date", "broker", "title", "opinion",
                     "target_price", "prev_target_price", "report_url")}}
                    for i, r in enumerate(reports, 1)],
        "news": [{"id": f"N{i}", **n} for i, n in enumerate(news, 1)],
        "problems": problems, "flagged": flagged, "regenerated": regenerated,
        "elapsed_sec": round(time.time() - t0, 1),
    }
