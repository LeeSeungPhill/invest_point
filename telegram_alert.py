"""
telegram_alert.py
==================
analyze 노드(리포트 생성 자체) 실패 시 Batch/fnguidePerformbot.py와 같은 방식
(같은 DB의 "stockAccount_stock_account".bot_token1, 같은 chat_id)으로 텔레그램
알림을 보낸다.

왜 analyze 노드만 알리는가:
  mvp_graph.py의 설계 원칙(각 노드는 실패해도 그래프를 죽이지 않고
  state['errors']에 적재)대로, news/research/disclosures 등 부수 소스 수집
  실패는 원래도 정상적으로 다운그레이드 처리된다(리포트는 나온다). 반면 analyze
  노드(import/init/llm) 실패는 이번 실행이 리포트 없이 끝난다는 뜻이라 즉시
  알아야 하는 치명적 오류다 — 그래서 알림 대상을 analyze 노드로 한정한다
  (mvp_graph._alert_analyze_failure 참조).

왜 python-telegram-bot 라이브러리 대신 requests로 직접 호출하는가:
  Batch/fnguidePerformbot.py 등은 python-telegram-bot v13 계열의 동기 API
  (bot.send_message(...)를 그냥 호출)를 쓰지만, 이 프로젝트 환경에는 v20+
  (완전 비동기, 실측 22.8)가 설치돼 있어 같은 코드를 가져오면 코루틴이
  await 없이 생성만 되고 실제로는 전송되지 않는다. Bot API는 단순 REST라
  requests.post 한 번으로 충분하고 라이브러리 버전에 영향받지 않는다.

DISABLE_TELEGRAM_ALERT=1로 끌 수 있다(DB/텔레그램 접속 불가 환경 대비).
알림 전송 자체의 실패는 절대 파이프라인을 죽이지 않는다(로그만 남기고 삼킨다) —
호출부에서 예외를 잡을 필요가 없도록 이 모듈 안에서 전부 삼킨다.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

log = logging.getLogger(__name__)

# fnguidePerformbot.py/kis_stock_search_api.py 등 Batch 쪽 스크립트가 쓰는 것과
# 같은 봇 계정(nick_name)·알림 수신 chat_id. 필요하면 .env로 덮어쓸 수 있다.
_NICK_NAME = os.getenv("TELEGRAM_ALERT_NICKNAME", "kwphills75")
_CHAT_ID = os.getenv("TELEGRAM_ALERT_CHAT_ID", "2147256258")

_token_cache: Optional[str] = None


def is_enabled() -> bool:
    return os.getenv("DISABLE_TELEGRAM_ALERT", "0") != "1"


def _get_bot_token() -> Optional[str]:
    """analysis_history.py와 같은 DB(fund_risk_mng)에서 bot_token1을 조회한다.
    프로세스 생존 기간 동안 한 번만 조회해 캐시한다(알림은 드물게 나지만,
    조회 자체는 매 알림마다 새 DB 커넥션을 여는 게 낭비라서)."""
    global _token_cache
    if _token_cache:
        return _token_cache
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DATABASE", "fund_risk_mng"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT bot_token1 FROM "stockAccount_stock_account" WHERE nick_name = %s',
                (_NICK_NAME,),
            )
            row = cur.fetchone()
        if row and row[0]:
            _token_cache = row[0]
        return _token_cache
    finally:
        conn.close()


def send_alert(text: str) -> None:
    """실패해도 예외를 던지지 않는다 — 알림은 부가 기능이지 핵심 경로가 아니다."""
    if not is_enabled():
        return
    try:
        token = _get_bot_token()
        if not token:
            log.warning("텔레그램 알림 생략: bot_token1을 찾지 못함(nick_name=%s)", _NICK_NAME)
            return
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text},
            timeout=10,
        )
        if r.status_code != 200:
            log.warning("텔레그램 알림 전송 실패: HTTP %s %s", r.status_code, r.text[:300])
    except Exception as e:  # noqa: BLE001
        log.warning("텔레그램 알림 전송 실패: %s", e)
