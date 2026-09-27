"""
invest_mng_lookup.py
=====================
invest_mng.high_price(사용자가 직접 관리하는 목표가) 조회 전용 — value_signal
(invest_point.build_value_signal의 quality_signal) 목표주가 상승여력 조건에서,
이 값이 있으면 컨센서스 목표주가(external_sources.fetch_naver_price의
target_price_mean 기반 target_upside_pct)보다 우선 사용한다(사용자 요청).

invest_mng은 이 프로젝트가 만든 테이블이 아니다(weekly_batch.py가 대상 종목
목록을 읽어오는 곳 — invest_issue/invest_point/invest_risk/price 등은 별도
외부 도구(simul/dly_invest_mng_save.py)가 채운다). 그래서 여기서는 읽기만
하고 스키마는 건드리지 않는다(ALTER/CREATE 없음).

접속정보(.env)는 analysis_history.py/weekly_batch.py와 동일한 PG_HOST 등을
그대로 쓴다(같은 DB, fund_risk_mng). DISABLE_INVEST_MNG_LOOKUP=1로 조회를
끌 수 있다(DB 접속 불가 환경 대비).
"""
from __future__ import annotations

import os
import logging
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2

logger = logging.getLogger("invest_mng_lookup")


def is_enabled() -> bool:
    return os.getenv("DISABLE_INVEST_MNG_LOOKUP", "0") != "1"


def _connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DATABASE", "fund_risk_mng"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        connect_timeout=5,
    )


def fetch_high_price(stock_code: str) -> Optional[float]:
    """invest_mng.high_price(사용자 지정 목표가)를 조회한다. 종목이 invest_mng에
    없거나, 값이 비어 있거나, DB 접속 자체가 안 되면 None을 반환한다(예외를
    던지지 않는다 — 이 값은 보조 소스라 조회 실패가 파이프라인 전체를 막으면
    안 된다)."""
    if not is_enabled() or not stock_code:
        return None
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT high_price FROM public.invest_mng WHERE proc_yn = 'Y' AND code=%s",
                    (stock_code.zfill(6),),
                )
                row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("invest_mng.high_price 조회 실패(%s): %s", stock_code, e)
        return None
