"""
hankyung_consensus.py
======================
한경컨센서스(consensus.hankyung.com)에서 목표주가·EPS 추정치의 '현재/이전' 값
변경 이력을 가져온다. 개별 리포트 PDF 본문은 받지 않는다(저작권) — 목록
페이지(skinType=stock_good, up_down_type=1|4)에 이미 표로 나와 있는 현재/이전
숫자만 파싱한다. value_investment_check.py가 이 숫자로 '추정치 리비전 방향·
리포트 편차·커버리지 증감'을 코드로 직접 계산하는 데 쓴다(LLM이 숫자를 새로
만들지 않는다는 프로젝트 공통 원칙).

실측 확인(2026):
  - up_down_type=1: 적정주가(목표주가) 현재/이전. up_down_type=4: EPS 현재/이전.
    (2=투자의견(기업), 3=투자의견(산업)도 있지만 체크리스트가 '의견보다 추정치
    변화 방향이 중요하다'고 명시해 이 두 개는 안 쓴다.)
  - search_value(내부 BUSINESS_CODE 형식)에 6자리 종목코드를 그대로 넘기면
    결과가 0건이 된다 — search_text=회사명으로만 검색해야 한다.
  - search_text=회사명 검색은 본문에 그 이름이 언급된 '다른' 종목 리포트도
    섞어 준다(예: "삼성전자"로 검색하면 "삼성생명(032830) 삼성전자 주주환원
    영향 점검"도 나옴 — 032830 리포트이지 005930 리포트가 아님). 한경 리포트
    제목이 관행적으로 '회사명(종목코드) ...'로 시작하는 점을 이용해, 제목이
    정확히 그 종목명+코드로 시작하는 행만 남기는 후처리 필터로 걸러낸다.
"""
from __future__ import annotations

import os
import re
import time
import json
import hashlib
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("hankyung_consensus")

CACHE_DIR = Path(os.getenv("HK_CACHE_DIR", Path.home() / ".cache" / "dart_mvp" / "hankyung"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_BASE = "https://consensus.hankyung.com/analysis/list"
_HDR = {"User-Agent": "Mozilla/5.0 (compatible; personal-research/1.0)"}

_ROW_RE = re.compile(
    r'<td class="first txt_number">([\d-]+)</td>\s*'
    r'<td class="text_l">\s*<a[^>]*>([^<]+)</a>.*?'
    r'<td>([^<]*)</td>\s*<td>([^<]*)</td>\s*'
    r'<td class="text_r txt_number">([^<]*)</td>\s*'
    r'<td class="text_r txt_number">([^<]*)</td>',
    re.DOTALL)


class SourceError(RuntimeError):
    pass


def _cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def _cache_get(key: str, ttl: int) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _cache_put(key: str, value: dict):
    _cache_path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _to_num(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "")
    if not s or s in ("-", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_revision_rows(stock_code: str, corp_name: str, *, up_down_type: str,
                          months: int = 12) -> list[dict]:
    edate = date.today()
    sdate = edate - timedelta(days=months * 30)
    params = {
        "skinType": "stock_good", "up_down_type": up_down_type,
        "search_text": corp_name,
        "sdate": sdate.strftime("%Y-%m-%d"), "edate": edate.strftime("%Y-%m-%d"),
        "pagenum": 80,
    }
    try:
        r = requests.get(_BASE, headers=_HDR, params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SourceError(f"한경컨센서스 요청 실패: {e}") from e

    prefix = f"{corp_name}({stock_code.zfill(6)})"
    rows = []
    for m in _ROW_RE.finditer(r.text):
        report_date, title, writer, office, cur, prev = m.groups()
        title = title.strip()
        if not title.startswith(prefix):
            continue
        cur_v = _to_num(cur)
        if cur_v is None:
            continue
        rows.append({"date": report_date.strip(), "title": title,
                     "broker": office.strip(), "current": cur_v, "previous": _to_num(prev)})
    rows.sort(key=lambda x: x["date"])
    return rows


def fetch_estimate_revisions(stock_code: str, corp_name: str, *, cache_ttl: int = 21600) -> dict:
    """반환: {'target_price': [...], 'eps': [...]} — 각 항목
    {date, title, broker, current, previous}, 오래된순 정렬. 실패해도 예외를
    던지지 않고 빈 리스트로 남긴다(개인 리서치 목적 최선의 노력 — 이 데이터가
    없어도 가치투자 체크의 나머지 항목은 여전히 유효하다)."""
    code = stock_code.zfill(6)
    ck = f"hk_revisions::{code}::{corp_name}"
    cached = _cache_get(ck, cache_ttl)
    if cached is not None:
        return cached

    result = {"target_price": [], "eps": []}
    try:
        result["target_price"] = _fetch_revision_rows(code, corp_name, up_down_type="1")
    except Exception as e:  # noqa: BLE001
        logger.warning("한경컨센서스 목표주가 이력 조회 실패(%s): %s", code, e)
    try:
        result["eps"] = _fetch_revision_rows(code, corp_name, up_down_type="4")
    except Exception as e:  # noqa: BLE001
        logger.warning("한경컨센서스 EPS 이력 조회 실패(%s): %s", code, e)

    _cache_put(ck, result)
    return result
