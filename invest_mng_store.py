"""
invest_mng_store.py
====================
가치투자 체크(value_investment_check.py) 결과를 invest_mng.value_invest에
저장한다.

invest_mng은 이 프로젝트가 만든 테이블이 아니라(별도 외부 도구가 invest_issue/
invest_point/invest_risk/price 등을 이미 채우고 있다 — simul/dly_invest_mng_save.py
확인) 이미 존재하는 관리 테이블이므로 CREATE TABLE은 하지 않고, value_invest
컬럼이 없으면 추가만 한다. mod_dt도 같이 갱신한다 — simul 쪽 다른 갱신 로직도
값을 바꿀 때 mod_dt를 함께 쓰는 관행이라 여기서도 맞춘다.

접속정보(.env)는 analysis_history.py/weekly_batch.py와 동일한 PG_HOST 등을
그대로 쓴다(같은 DB, fund_risk_mng). DISABLE_INVEST_MNG_STORE=1로 저장을
끌 수 있다(DB 접속 불가 환경 대비).
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2


def is_enabled() -> bool:
    return os.getenv("DISABLE_INVEST_MNG_STORE", "0") != "1"


def _connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DATABASE", "fund_risk_mng"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        connect_timeout=5,
    )


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE public.invest_mng ADD COLUMN IF NOT EXISTS value_invest TEXT;")
    conn.commit()


def save_value_invest(stock_code: str, text: str) -> None:
    """invest_mng.value_invest(+mod_dt) 갱신. code가 invest_mng에 없는 종목이면
    (관리 대상 밖) 조용히 0행 갱신으로 끝난다 — 에러 아님."""
    if not is_enabled() or not stock_code or not text:
        return
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.invest_mng SET value_invest=%s, mod_dt=%s WHERE code=%s",
                (text, time.strftime("%Y-%m-%d %H:%M:%S"), stock_code.zfill(6)),
            )
        conn.commit()
    finally:
        conn.close()
