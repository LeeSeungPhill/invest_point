"""
weekly_batch.py
================
투자관리 종목(invest_mng)을 대상으로 mvp_graph.run()을 순차 실행하는
주간 배치. 매주 일요일 1회 cron으로 기동하는 것을 전제로 한다 — 스케줄('언제')은
cron(또는 이 스크립트를 감싸는 쉘 스크립트)이 담당하고, 이 파일은 '실행되면
무엇을 할지'만 담당한다.

대상 종목 선정: analysis_history에 investment_summary가 채워진 이력이 있고 그 최근 run_at 일자가 7일 이상 지난 invest_mng 의 6자리 종목코드 & proc_yn='Y' 중복 제거해 가져온다.
이 테이블에는 상품유형 컬럼이 없어 ETF/ETN은 종목명 브랜드 접두사로
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
import re
import sys
import time
from datetime import datetime

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

import analysis_history
import mvp_graph
import telegram_alert
from invest_mng_sync import update_invest_mng  # 분석 후 invest_mng 반영(simul_server와 공용)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("weekly_batch")

# 종목 간 호출 간격(초) — DART/네이버/FnGuide 등 외부 소스에 대한 배치성 부하 완화
INTER_STOCK_PAUSE_SEC = 5

# 국내 주요 ETF/ETN 브랜드 접두사 — invest_mng 상품유형 컬럼이
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
    """6자리 종목코드 & proc_yn='Y' 이면서, analysis_history에 investment_summary가
    채워진 이력이 있고 그 최근 run_at 일자가 7일 이상 지난 종목. ETF 제외.

    날짜를 'YYYYMMDD'로 자른 뒤 비교하므로 '<=' 이어야 매주 정확히 7일 간격
    cron으로 돌 때도 경계에서 스킵되지 않는다 — '<'였을 때는 '정확히 7일 지남'
    케이스가 매번 거짓이 되어 한 번 실행되면 다음 주는 항상 스킵되고 그다음 주
    (14일 뒤)에야 다시 실행되는, '주간' 배치가 사실상 격주로 도는 버그가 있었다."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT m.code, m.name FROM public.invest_mng m "
                "JOIN ("
                "  SELECT stock_code, MAX(run_at) AS last_run_at "
                "  FROM public.analysis_history "
                "  WHERE investment_summary IS NOT NULL AND investment_summary NOT LIKE '%미생성%' "
                "  GROUP BY stock_code"
                ") h ON h.stock_code = m.code "
                "WHERE length(m.code) = 6 AND m.proc_yn = 'Y' "
                "AND to_char(h.last_run_at, 'YYYYMMDD') <= to_char(date_trunc('day', current_date - interval '7 day'), 'YYYYMMDD') "
                "GROUP BY m.code, m.name ORDER BY m.code"
            )
            rows = cur.fetchall()
        return [{"stock_code": r["code"], "corp_name": r["name"]}
                for r in rows if not _is_etf_name(r["name"])]
    finally:
        conn.close()


_TG_MAX_LEN = 3800  # 텔레그램 sendMessage 4096자 제한 여유분


def _fmt_stock_list(items: list[str]) -> str:
    """종목 나열 문자열 — 너무 길면 잘라서 '외 N건'으로 표기."""
    text = ", ".join(items)
    if len(text) <= _TG_MAX_LEN:
        return text
    shown, length = [], 0
    for it in items:
        if length + len(it) + 2 > _TG_MAX_LEN:
            break
        shown.append(it)
        length += len(it) + 2
    return ", ".join(shown) + f" 외 {len(items) - len(shown)}건"


def run_batch(limit: int | None = None) -> None:
    started = time.time()
    stocks = list_target_stocks()
    if limit:
        stocks = stocks[:limit]
    if not stocks:
        log.info("배치 대상 종목이 없어 종료합니다.")
        telegram_alert.send_alert("[주간 투자포인트 배치] 대상 종목 없음 — 종료")
        return

    labels = [f"{s['stock_code']}({s.get('corp_name') or '?'})" for s in stocks]
    log.info("배치 대상 %d개 종목: %s", len(stocks), ", ".join(labels))
    telegram_alert.send_alert(
        f"[주간 투자포인트 배치] 시작 — 대상 {len(stocks)}개 종목\n"
        f"{_fmt_stock_list(labels)}")

    ok, failed = 0, []
    llm_failed, sync_failed = [], []
    for i, s in enumerate(stocks, 1):
        code, name = s["stock_code"], s.get("corp_name") or "?"
        log.info("[%d/%d] %s(%s) 분석 시작", i, len(stocks), code, name)
        try:
            result = mvp_graph.run(code)
            errs = result.get("errors") or []
            log.info("[%d/%d] %s(%s) 완료 — 경고 %d건%s", i, len(stocks), code, name,
                     len(errs), " 재생성됨" if result.get("regenerated") else "")
            if result.get("llm_error"):
                log.warning("[%d/%d] %s(%s) LLM 호출 에러: %s",
                            i, len(stocks), code, name, result["llm_error"])
                llm_failed.append(f"{code}({name})")
            # 분석이 끝났으면 최신 이력을 invest_mng에 반영(LLM 실패 시엔 이력이 비어
            # 있을 수 있어 건너뜀). 반영 실패는 분석 성공 여부와 분리해 경고만 남긴다.
            if not result.get("llm_error"):
                try:
                    if update_invest_mng(code):
                        log.info("[%d/%d] %s(%s) invest_mng 갱신 완료", i, len(stocks), code, name)
                    else:
                        log.warning("[%d/%d] %s(%s) invest_mng 갱신 대상 행 없음", i, len(stocks), code, name)
                except Exception:  # noqa: BLE001
                    log.exception("[%d/%d] %s(%s) invest_mng 갱신 실패", i, len(stocks), code, name)
                    sync_failed.append(f"{code}({name})")
            ok += 1
        except Exception:  # noqa: BLE001 — 한 종목 실패가 배치 전체를 죽이면 안 됨
            log.exception("[%d/%d] %s(%s) 실패", i, len(stocks), code, name)
            failed.append(f"{code}({name})")
        if i < len(stocks):
            time.sleep(INTER_STOCK_PAUSE_SEC)

    elapsed_min = (time.time() - started) / 60
    log.info("배치 종료: 성공 %d / 실패 %d개 %s (%.1f분)", ok, len(failed), failed, elapsed_min)

    lines = [f"[주간 투자포인트 배치] 완료 — 대상 {len(stocks)}개, 소요 {elapsed_min:.1f}분",
             f"성공 {ok}건 / 실패 {len(failed)}건"]
    if failed:
        lines.append(f"실패: {_fmt_stock_list(failed)}")
    if llm_failed:
        lines.append(f"LLM 호출 에러 {len(llm_failed)}건: {_fmt_stock_list(llm_failed)}")
    if sync_failed:
        lines.append(f"invest_mng 갱신 실패 {len(sync_failed)}건: {_fmt_stock_list(sync_failed)}")
    msg = "\n".join(lines)
    telegram_alert.send_alert(msg if len(msg) <= 4000 else msg[:4000] + "…(생략)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="투자관리 종목(비-ETF) 대상 주간 mvp_graph 배치")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞 N종목만 실행")
    args = parser.parse_args()
    run_batch(limit=args.limit)
