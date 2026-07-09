"""
weekly_batch.py
================
보유 종목(stockBalance_stock_balance)을 대상으로 mvp_graph.run()을 순차 실행하는
주간 배치. 매주 일요일 1회 cron으로 기동하는 것을 전제로 한다 — 스케줄('언제')은
cron(또는 이 스크립트를 감싸는 쉘 스크립트)이 담당하고, 이 파일은 '실행되면
무엇을 할지'만 담당한다.

대상 종목 선정: stockBalance_stock_balance에서 보유 처리 중(proc_yn='Y')이고
매수금액이 있는(purchase_amount > 0) 6자리 종목코드를 종목명 기준 중복 제거해
가져온다. 이 테이블에는 상품유형 컬럼이 없어 ETF/ETN은 종목명 브랜드 접두사로
걸러낸다(_is_etf_name). 종목별 실행은 서로 독립적으로 예외 처리되어 한 종목의
수집/분석 실패가 나머지 종목 실행을 막지 않는다. mvp_graph의 3단계(analysis_history
저장)·4단계(투자포인트 요약) 로직이 종목별 실행 안에서 그대로 재사용되므로 이번
실행 결과도 analysis_history에 남아 다음 분석이 '과거 이력'으로 참조한다.

실행:
  python weekly_batch.py                # 대상 전 종목
  python weekly_batch.py --limit 5      # 테스트용으로 앞 5종목만
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
import psycopg2.extras

import mvp_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("weekly_batch")

# 종목 간 호출 간격(초) — DART/네이버/FnGuide 등 외부 소스에 대한 배치성 부하 완화
INTER_STOCK_PAUSE_SEC = 5

# 국내 주요 ETF/ETN 브랜드 접두사 — stockBalance_stock_balance에는 상품유형 컬럼이
# 없어 종목명 패턴으로 배제한다. 새 브랜드가 생기면 이 목록에 추가한다.
_ETF_NAME_PREFIXES = (
    "KODEX", "TIGER", "KBSTAR", "ARIRANG", "HANARO", "KINDEX", "KOSEF", "SOL",
    "ACE", "PLUS", "RISE", "WOORI", "TIMEFOLIO", "FOCUS", "BNK", "KTOP", "WON",
    "VITA", "마이티", "히어로즈", "파워",
)


def _is_etf_name(name: str) -> bool:
    if not name:
        return False
    upper = name.upper()
    if "ETF" in upper or "ETN" in upper:
        return True
    return any(upper.startswith(p) for p in _ETF_NAME_PREFIXES)


def _connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DATABASE", "fund_risk_mng"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        connect_timeout=5,
    )


def list_target_stocks() -> list[dict]:
    """보유 처리 중(proc_yn='Y') & 매수금액>0 & 6자리 종목코드, ETF 제외."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                'SELECT code, name FROM public."stockBalance_stock_balance" '
                "WHERE proc_yn = %s AND purchase_amount > 0 AND length(code) = 6 "
                "GROUP BY code, name ORDER BY code",
                ("Y",),
            )
            rows = cur.fetchall()
        return [{"stock_code": r["code"], "corp_name": r["name"]}
                for r in rows if not _is_etf_name(r["name"])]
    finally:
        conn.close()


def run_batch(limit: int | None = None) -> None:
    stocks = list_target_stocks()
    if limit:
        stocks = stocks[:limit]
    if not stocks:
        log.info("배치 대상 종목이 없어 종료합니다.")
        return

    log.info("배치 대상 %d개 종목: %s", len(stocks),
             ", ".join(f"{s['stock_code']}({s.get('corp_name') or '?'})" for s in stocks))

    ok, failed = 0, []
    for i, s in enumerate(stocks, 1):
        code, name = s["stock_code"], s.get("corp_name") or "?"
        log.info("[%d/%d] %s(%s) 분석 시작", i, len(stocks), code, name)
        try:
            result = mvp_graph.run(code)
            errs = result.get("errors") or []
            log.info("[%d/%d] %s(%s) 완료 — 경고 %d건%s", i, len(stocks), code, name,
                     len(errs), " 재생성됨" if result.get("regenerated") else "")
            ok += 1
        except Exception:  # noqa: BLE001 — 한 종목 실패가 배치 전체를 죽이면 안 됨
            log.exception("[%d/%d] %s(%s) 실패", i, len(stocks), code, name)
            failed.append(code)
        if i < len(stocks):
            time.sleep(INTER_STOCK_PAUSE_SEC)

    log.info("배치 종료: 성공 %d / 실패 %d개 %s", ok, len(failed), failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="보유 종목(비-ETF) 대상 주간 mvp_graph 배치")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞 N종목만 실행")
    args = parser.parse_args()
    run_batch(limit=args.limit)
