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

# 실측 오탐 사례: "예상실적 상승 & 주가 하단 동시 충족 미달"에서 '동시 충족'만
# 부분 문자열로 잡으면 뒤의 '미달'(부정)을 놓쳐 정답('아니오')을 '예'로 오판했다.
# "시그널: 아니오"처럼 콜론을 요구하면 LLM이 실제로 즐겨 쓰는 "시그널 '아니오'"
# (따옴표, 콜론 없음) 표기를 못 잡는다. 그래서 '시그널' 뒤 일정 구간(window) 안에서
# 부정어를 긍정어보다 먼저 검사해 우선순위를 준다(부정어가 있으면 무조건 '아니오').
# "충족하지 못했다"처럼 긍정 단어(충족) 뒤에 '못했-'가 붙어 실제로는 부정인
# 표현을 놓치면(336260 실측 사례), 밴드 하단 조건이 실제로는 충족(<=40%)인데
# 리포트가 '미달'로 잘못 서술해도 못 잡는다. '못했'을 부정어에 추가한다.
# "가치 시그널 충족 불가"(437730 실측)처럼 '충족' 뒤에 '불가'가 붙는 표현도
# 실제로는 부정(아니오)인데 '불가'가 부정어 목록에 없어 '충족'만 보고 오판했다.
_NEG_WORDS = ("아니오", "아니다", "미달", "미충족", "불충족", "불성립", "안됨", "안 됨",
             "못했", "못함", "불가")
_POS_WORDS = ("충족", "성립", "예")
_SIGNAL_WINDOW = 60

# analyze()의 출력 형식(system 프롬프트 규칙)에 정의된 실제 절 제목만 절 경계로
# 인정한다. LLM이 본문 중간에 "[투자포인트(정량)]" 같은 유사-인용 태그를 습관적으로
# 붙이는 경우가 있는데, 예전에는 '[...]'를 전부 절 경계로 오인해 그 뒤 문장이
# 통째로 잘려나가는 버그가 있었다(예: [가치 포지션]의 '가치 시그널: 아니오' 문장이
# 중간의 '[투자포인트(정량)]' 태그 때문에 잘려서 검증 대상에서 누락됨).
SECTION_TAGS = ("핵심 이슈", "투자포인트", "리스크", "성장 시나리오",
               "가치 포지션", "컨센서스 대비", "데이터 공백")
_TAG_ALT = "|".join(re.escape(t) for t in SECTION_TAGS)
_SECTION = re.compile(rf"\[({_TAG_ALT})\](.*?)(?=\[(?:{_TAG_ALT})\]|$)", re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*\s*%?")


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

    idx = section.find("시그널")
    window = section[idx: idx + _SIGNAL_WINDOW] if idx >= 0 else section
    # 부정어 우선: "동시 충족 미달"처럼 긍정 단어(충족) 뒤에 부정어(미달)가 붙는
    # 표현이 흔해서, 부정어가 하나라도 있으면 최종 판정은 '아니오'로 본다.
    said_no = any(w in window for w in _NEG_WORDS)
    said_yes = (not said_no) and any(w in window for w in _POS_WORDS)

    if said_no:
        determined = False
    elif said_yes:
        determined = True
    else:
        determined = None  # 판단 불가 — 관대하게 통과(오탐 방지)

    match = determined is None or determined == expected_signal
    return {"expected": expected_signal, "match": match, "section": section[:200]}


def check_band_threshold(report_sections: dict, valuation: dict) -> Optional[dict]:
    """[가치 포지션]의 '밴드 하단 기준 이하 충족/미달' 서술이 실제 is_low_band와
    맞는지 본다. check_signal은 최종 '가치 시그널 예/아니오'만 보는데, 그 답이
    다른 조건(예상실적 상승 여부) 때문에 우연히 맞아도 밴드 조건 자체에 대한
    서술은 틀릴 수 있다 — 실측(336260, 437730): band_position이 29.4%/34.8%로
    LOW_BAND_THRESHOLD(40%) 이하라 실제로는 조건을 충족하는데, 리포트가 각각
    '충족하지 못했다'/'미달'로 정반대로 서술했다(최종 시그널 '아니오'는 예상실적
    데이터 부재로 우연히 일치해 check_signal은 놓침)."""
    is_low_band = valuation.get("is_low_band")
    if is_low_band is None:
        return None
    section = report_sections.get("가치 포지션") or ""
    if not section:
        return None

    idx = section.find("이하")
    if idx < 0:
        return None
    start = max(0, idx - 5)
    window = section[start: idx + _SIGNAL_WINDOW]
    said_no = any(w in window for w in _NEG_WORDS)
    said_yes = (not said_no) and any(w in window for w in _POS_WORDS)

    if said_no:
        determined = False
    elif said_yes:
        determined = True
    else:
        determined = None

    match = determined is None or determined == is_low_band
    return {"expected": is_low_band, "match": match, "section": section[:200]}


# 매출 증감 방향 검사(check_revenue_direction)는 도입 후 실사용에서 제거했다:
# '과거 실적은 하락, 향후 예상은 회복'(저점 통과/턴어라운드) 서사는 정상적으로
# 두 방향이 공존하는데, backward-actual과 forward-estimate 언급을 구분하지
# 못해 정상 리포트를 오탐(false positive)하는 사례가 실측됐다(예: 394280,
# 059210). 그 오탐이 잡으려던 실제 사례(394280의 조작된 수치)는 이미
# find_unverified_numbers/find_scale_implausible_numbers가 독립적으로 잡고
# 있었으므로, 신뢰도 낮은 이 체크 없이도 안전망은 유지된다.
#
# 그런데 이후 실측(394280, 2026-07-12)에서 [핵심 이슈] 절이 실제와 정반대
# 방향("매출 감소 및 영업손실 확대")을 단정적으로 서술한 사례가 나왔다
# (실측: 매출 153→161억원 증가, 다음예상도 315억원으로 더 증가 — '감소'는
# backward/forward 어느 쪽으로 봐도 틀림). check_trend는 [성장 시나리오]/
# [투자포인트]만 보고 [핵심 이슈]는 아예 스캔하지 않아 놓쳤다. 아래
# check_headline_direction은 [핵심 이슈]만 대상으로 하되, 과거 실적(latest
# vs prev)과 예상(next_est vs latest) 두 방향을 모두 계산해서 — 둘 다 있는데
# 서로 갈리면(저점통과처럼) 판단을 보류하고, 방향이 일치하거나 한쪽만 있을
# 때만 비교한다. 그래서 removed된 check_revenue_direction과 달리 backward/
# forward가 실제로 갈리는 정상 서사는 오탐하지 않는다.
# 실측(394280, 2회차): 주어와 방향어 사이에 원단위 숫자가 끼어드는 문장이 흔해서
# ("매출액 2,673,764,184원(제10기)으로 전기(...) 대비 43% 감소") 고정 14자 창은
# 방향어에 닿지 못했다. 다음 구두점(,/./;)까지, 상한 80자로 창을 넓힌다 — 절
# 경계를 넘지 않아 다른 주어의 방향어를 잘못 끌어오는 오탐은 막으면서도 숫자가
# 끼어든 실제 문장 길이는 커버한다.
_UP_KW = ("증가", "성장", "확대", "상승")
_DOWN_KW = ("감소", "축소", "하락", "둔화")
# 절대극성 단어(주어가 손실/적자여도 뜻이 안 뒤집힘): 개선/호전=항상 좋아짐,
# 악화/심화=항상 나빠짐. 위 상대극성 단어(확대/축소 등)만 손실 주어일 때 반전한다.
_ABS_BETTER_KW = ("개선", "호전")
_ABS_WORSE_KW = ("악화", "심화")
_LOSS_SUBJ = ("영업손실", "손실", "적자")
_HEADLINE_WINDOW_MAX = 80
# 억/원 단위 숫자의 천단위 구분 콤마("9,128,010,811")나 소수점("43.5%")을 절
# 경계로 오인하면 안 되므로, 바로 뒤에 숫자가 오는 콤마/마침표는 경계에서 뺀다.
_CLAUSE_END_RE = re.compile(r"[,.](?!\d)|;")


def _claimed_direction(section: str, subj_idx: int, is_loss_subject: bool,
                       other_subject_words: tuple = ()) -> Optional[str]:
    # 창 끝 후보: 다음 구두점, 상한 80자, 그리고 (구두점 없이 '및'/'이며' 등으로
    # 이어지는 경우를 위해) 다른 주어 단어가 다음으로 등장하는 위치 중 가장 가까운
    # 것 — 그래야 "매출 감소 및 영업손실 확대"에서 매출 쪽 창이 영업손실 쪽 '확대'를
    # 잘못 끌어와 상쇄(오탐→미판정)되는 걸 막는다.
    end_candidates = [subj_idx + _HEADLINE_WINDOW_MAX]
    m = _CLAUSE_END_RE.search(section, subj_idx)
    if m:
        end_candidates.append(m.start() + 1)
    for w in other_subject_words:
        idx = section.find(w, subj_idx + 1)
        if idx >= 0:
            end_candidates.append(idx)
    window = section[subj_idx:min(end_candidates)]
    up = any(k in window for k in _UP_KW)
    down = any(k in window for k in _DOWN_KW)
    if is_loss_subject:
        # 손실 확대=악화(down), 손실 축소/감소=개선(up) — 상대극성 어휘와 뜻이 반대.
        up, down = down, up
    if any(k in window for k in _ABS_BETTER_KW):
        up = True
    if any(k in window for k in _ABS_WORSE_KW):
        down = True
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def _actual_direction(latest_val, prev_val, next_val) -> Optional[str]:
    back = None
    if latest_val is not None and prev_val is not None:
        back = "up" if latest_val > prev_val else ("down" if latest_val < prev_val else None)
    fwd = None
    if next_val is not None and latest_val is not None:
        fwd = "up" if next_val > latest_val else ("down" if next_val < latest_val else None)
    if back and fwd and back != fwd:
        return None  # 방향 혼재(저점통과 등) — 판단 보류(오탐 방지)
    return back or fwd


def check_headline_direction(report_sections: dict, growth: dict) -> Optional[dict]:
    """[핵심 이슈]의 매출/영업손익 방향 단정이 실측(과거·예상)과 정반대인지만 잡는다."""
    section = report_sections.get("핵심 이슈") or ""
    if not section:
        return None
    latest, prev, next_est = growth.get("latest"), growth.get("prev"), growth.get("next_est")
    if not latest or not prev:
        return None
    problems = []

    rev_idx = section.find("매출")
    if rev_idx >= 0:
        claimed = _claimed_direction(section, rev_idx, is_loss_subject=False,
                                     other_subject_words=_LOSS_SUBJ + ("영업이익",))
        actual = _actual_direction(latest.get("revenue"), prev.get("revenue"),
                                   (next_est or {}).get("revenue"))
        if claimed and actual and claimed != actual:
            problems.append(
                f"매출 {'증가' if claimed=='up' else '감소'} 서술이나 실측 방향은 "
                f"{'증가' if actual=='up' else '감소'}"
                f"({prev.get('revenue')}→{latest.get('revenue')}억원)")

    loss_idx = -1
    for w in _LOSS_SUBJ:
        idx = section.find(w)
        if idx >= 0:
            loss_idx = idx
            break
    is_loss = loss_idx >= 0
    op_idx = loss_idx if is_loss else section.find("영업이익")
    if op_idx < 0:
        op_idx = section.find("이익")
    if op_idx >= 0:
        claimed = _claimed_direction(section, op_idx, is_loss_subject=is_loss,
                                     other_subject_words=("매출",))
        actual = _actual_direction(latest.get("op_profit"), prev.get("op_profit"),
                                   (next_est or {}).get("op_profit"))
        if claimed and actual and claimed != actual:
            problems.append(
                f"영업손익 {'개선' if claimed=='up' else '악화'} 서술이나 실측 방향은 "
                f"{'개선' if actual=='up' else '악화'}"
                f"({prev.get('op_profit')}→{latest.get('op_profit')}억원)")

    if not problems:
        return None
    return {"match": False, "problems": problems, "section": section[:200]}


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


_UNIT_ERROR_RATIO = 50.0  # 실측 매출(억원 단위)의 이 배수를 넘으면 단위 오류 의심

# 삼성전자/SK하이닉스처럼 매출 규모가 큰 종목은 LLM이 '133조 8,734억원'처럼
# 조/억원을 나눠 쓴다. 이 표기의 뒷부분(8,734)만 떼어 놓고 보면 ip_block의
# '1338734'(억원)와 전혀 안 맞는 것처럼 보여 오탐이 났다 — 133*10000+8734=
# 1338734로 정확히 일치하는데도. 검증 전에 '조 X 억 Y'를 단일 억원 값으로
# 합쳐서, 뒷부분만 뚝 떼어 보고 오판하지 않게 한다.
_JO_EOK_RE = re.compile(r"(\d[\d,]*)\s*조\s*(\d[\d,]*(?:\.\d+)?)\s*억")


def _merge_jo_eok(text: str) -> str:
    def _combine(m: re.Match) -> str:
        jo = float(m.group(1).replace(",", ""))
        eok = float(m.group(2).replace(",", ""))
        return f"{jo * 10000 + eok:.0f}억"
    return _JO_EOK_RE.sub(_combine, text)


# 규칙9(mvp_graph.py)가 '원문 단위를 괄호로 같이 밝혀라'라고 요구한 결과, LLM이
# "36.98억원(3,698,095,942원)"처럼 원단위 원본값을, 또는 "수주잔고 477,440백만원"
# 처럼 청크 표에 적힌 백만원/천원 단위를 그대로 인용하는 경우가 흔하다(둘 다
# 규칙9가 원한 정확한 행동). 억원 단위만 아는 검증기가 이 숫자들을 그대로 스캔하면
# 자릿수가 커 보여 정상 인용을 '단위오류의심'으로 오탐한다 — 실제 사례 둘 다 확인:
#   · 3,698,095,942원 = 36.98억원 (raw 원)
#   · 477,440백만원 = 4,774.4억원 (백만원, 수주잔고 표 원문 단위 그대로 인용한 정상 케이스)
# 검증 전에 단위별로 억원 환산값을 미리 만들어둔다. raw 원은 1억원 미만이면
# 주가처럼 원래 원단위 표기가 자연스러우므로(예: "현재가 1,160원") 건드리지 않는다.
_WON_UNIT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(백만|천)?\s*원")
_WON_UNIT_DIVISOR = {"백만": 100.0, "천": 100_000.0, None: 100_000_000.0}


def _normalize_won_units(text: str) -> str:
    def _convert(m: re.Match) -> str:
        val = float(m.group(1).replace(",", ""))
        unit = m.group(2)
        if unit is None and val < 100_000_000:
            return m.group(0)
        return f"{val / _WON_UNIT_DIVISOR[unit]:.2f}억원"
    return _WON_UNIT_RE.sub(_convert, text)


def find_unverified_numbers(report_sections: dict, trusted_blocks: list,
                            revenue_scale: Optional[float] = None) -> list:
    """[성장 시나리오]/[가치 포지션]/[핵심 이슈]/[투자포인트] 절의 숫자만 검사
    (리스크·공시 등 서술 절의 예시 숫자까지 검사하면 오탐이 늘어난다).

    revenue_scale(직전 분기/연간 매출, 억원)이 주어지면, 미검증 숫자가 그 50배를
    넘을 때 '단위 오류 의심(백만원을 억원으로 잘못 읽음 등)'을 함께 표시한다 —
    실제로 발생한 사례(수주잔고 표가 백만원 단위인데 억원으로 인용해 100배
    부풀려진 경우)를 프롬프트 준수 여부와 무관하게 붙잡기 위함."""
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
        text = _normalize_won_units(_merge_jo_eok(report_sections.get(name, "")))
        for m in _NUM.finditer(text):
            tok = m.group().replace(",", "").replace(" ", "").rstrip("%")
            if not tok or _is_noise(tok):
                continue
            try:
                val = float(tok)
            except ValueError:
                continue
            # 대형주(SK하이닉스 등)는 억원 단위 숫자가 커서 LLM이 조원으로 환산해
            # 쓰는 게 자연스럽고 정확한데(848104억원=84.8104조원), 문자열 매칭만
            # 하면 이런 정상적인 조원 환산을 '미검증'으로 오탐한다. val을 조원으로
            # 보고 10000배 한 값도 같이 대조해 정상 환산은 통과시킨다.
            found = any(
                abs(val - t) <= max(0.05 * abs(val), 0.5)
                or abs(val * 10000 - t) <= max(0.05 * abs(val * 10000), 0.5)
                for t in trusted_floats
            )
            if not found:
                tag = f"{tok}({name})"
                if revenue_scale and abs(val) > revenue_scale * _UNIT_ERROR_RATIO:
                    tag += " ⚠단위오류의심(백만원→억원 100배 오독 등 확인)"
                unverified.append(tag)
    return unverified


# [리스크]/[데이터 공백]은 오탐 방지를 위해 일반 미검증 숫자 검사 대상에서 뺐지만
# (실제 사업보고서 청크의 진짜 숫자를 인용하는 경우가 많음), '회사 매출 규모 대비
# 비정상적으로 큰 금액'만은 예외적으로 스캔한다 — 실제 사례: DART 원단위 영업손실
# -1,630,054,490원(=-16.3억원)을 100배 부풀려 [리스크]에 '-1,630억원'으로 오기재.
_SCALE_CHECK_SECTIONS = ("리스크", "데이터 공백")


def find_scale_implausible_numbers(report_sections: dict, revenue_scale: Optional[float]) -> list:
    if not revenue_scale:
        return []
    out = []
    for name in _SCALE_CHECK_SECTIONS:
        text = _normalize_won_units(_merge_jo_eok(report_sections.get(name, "")))
        for m in _NUM.finditer(text):
            tok = m.group().replace(",", "").replace(" ", "").rstrip("%")
            if not tok or _is_noise(tok):
                continue
            try:
                val = float(tok)
            except ValueError:
                continue
            if abs(val) > revenue_scale * _UNIT_ERROR_RATIO:
                out.append(f"{tok}({name}) ⚠단위오류의심(회사 매출 규모 대비 비정상적으로 큼)")
    return out


def check(report: str, invest_point: dict, trusted_blocks: list) -> dict:
    """report(LLM 최종 텍스트) 검증. invest_point는 mvp_graph의 build_invest_point 결과,
    trusted_blocks는 analyze()가 프롬프트에 실제로 제시한 구조화 블록 문자열 리스트."""
    sections = _sections(report)
    growth = (invest_point or {}).get("growth", {})
    valuation = (invest_point or {}).get("valuation", {})

    trend_chk = check_trend(sections, growth.get("trend"))
    signal_chk = check_signal(sections, valuation.get("signal"))
    band_chk = check_band_threshold(sections, valuation)
    headline_chk = check_headline_direction(sections, growth)
    revenue_scale = (growth.get("latest") or {}).get("revenue")
    unverified = (find_unverified_numbers(sections, trusted_blocks, revenue_scale)
                 + find_scale_implausible_numbers(sections, revenue_scale))

    problems = []
    if trend_chk and not trend_chk["match"]:
        problems.append(f"성장 판단 불일치(기대: {trend_chk['expected']})")
    if signal_chk and not signal_chk["match"]:
        problems.append(
            f"가치 시그널 불일치(기대: {'예' if valuation.get('signal') else '아니오'})")
    if band_chk and not band_chk["match"]:
        problems.append(
            f"밴드 하단 기준 서술 불일치(기대: {'충족' if valuation.get('is_low_band') else '미달'})")
    if headline_chk and not headline_chk["match"]:
        problems.append("핵심 이슈 방향 불일치: " + "; ".join(headline_chk["problems"]))
    if unverified:
        problems.append(f"미검증 파생수치 {len(unverified)}건: {', '.join(unverified[:5])}")

    consistent = not problems
    return {
        "trend_check": trend_chk,
        "signal_check": signal_chk,
        "band_check": band_chk,
        "headline_check": headline_chk,
        "unverified_numbers": unverified,
        "problems": problems,
        "consistent": consistent,
        "verdict": "✅ 시나리오 일관" if consistent else ("⚠️ 불일치: " + "; ".join(problems)),
    }
