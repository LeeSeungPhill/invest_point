"""
weekly_batch.py
================
analysis_history 테이블에 한 번이라도 저장된 모든 종목을 대상으로 mvp_graph.run()을
순차 실행하는 주간 배치. 매주 일요일 1회 cron으로 기동하는 것을 전제로 한다 —
스케줄('언제')은 cron(또는 이 스크립트를 감싸는 쉘 스크립트)이 담당하고, 이 파일은
'실행되면 무엇을 할지'만 담당한다.

대상 종목 선정: analysis_history.list_distinct_stocks() — 이력에 stock_code가
한 번이라도 기록된 종목 전체(중복 제거). 종목별 실행은 서로 독립적으로 예외
처리되어 한 종목의 수집/분석 실패가 나머지 종목 실행을 막지 않는다. mvp_graph의
3단계(analysis_history 저장)·4단계(투자포인트 요약) 로직이 종목별 실행 안에서
그대로 재사용되므로 이번 실행 결과도 다음 주 배치가 '과거 이력'으로 참조한다.

실행:
  python weekly_batch.py                # analysis_history의 전 종목
  python weekly_batch.py --limit 5      # 테스트용으로 앞 5종목만
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import analysis_history
import mvp_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("weekly_batch")

# 종목 간 호출 간격(초) — DART/네이버/FnGuide 등 외부 소스에 대한 배치성 부하 완화
INTER_STOCK_PAUSE_SEC = 5


def run_batch(limit: int | None = None) -> None:
    if not analysis_history.is_enabled():
        log.error("DISABLE_HISTORY=1 — analysis_history 조회가 꺼져 있어 배치 대상 종목을 "
                  "알 수 없습니다. 배치를 실행하려면 DISABLE_HISTORY를 해제하세요.")
        return

    stocks = analysis_history.list_distinct_stocks()
    if limit:
        stocks = stocks[:limit]
    if not stocks:
        log.info("analysis_history에 저장된 종목이 없어 배치를 종료합니다.")
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
    parser = argparse.ArgumentParser(description="analysis_history 종목 대상 주간 mvp_graph 배치")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞 N종목만 실행")
    args = parser.parse_args()
    run_batch(limit=args.limit)
