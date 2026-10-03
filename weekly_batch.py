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


_SUMMARY_TAGS = (("핵심 이슈", "invest_issue"), ("투자포인트", "invest_point"), ("리스크", "invest_risk"))


def _parse_investment_summary(summary: str | None) -> dict:
    """investment_summary("[핵심 이슈]\\n...\\n\\n[투자포인트]\\n...\\n\\n[리스크]\\n...")를
    3개 필드로 분리(simul_server._parse_investment_summary와 동일 규칙)."""
    out = {"invest_issue": "", "invest_point": "", "invest_risk": ""}
    if not summary:
        return out
    for tag, key in _SUMMARY_TAGS:
        m = re.search(rf"\[{re.escape(tag)}\]\s*\n(.*?)(?=\n\n\[|\Z)", summary, re.S)
        if m:
            out[key] = m.group(1).strip()
    return out


def _grew_or_similar(base, nxt, tolerance: float = -0.05) -> bool:
    """base 대비 nxt가 증가 또는 유사(-5% 이내 하락까지)한지. base가 0 이하(적자)면
    절대 개선 여부(nxt >= base)로 판정(simul_server._grew_or_similar와 동일)."""
    if base is None or nxt is None:
        return False
    try:
        base_f, nxt_f = float(base), float(nxt)
    except (TypeError, ValueError):
        return False
    if base_f <= 0:
        return nxt_f >= base_f
    return (nxt_f - base_f) / base_f >= tolerance


def update_invest_mng(code: str) -> bool:
    """분석 완료 후 analysis_history 최신 이력을 읽어 invest_mng(proc_yn='Y')의 해당
    컬럼을 갱신한다(simul_server._get_invest_point_fields + invest_mng_apply의 UPDATE와
    같은 계산). 텍스트/원천값은 비어 있으면 COALESCE로 기존 값을 유지하고, 게이트가
    걸린 파생비율(상승잔존율·배당율)은 조건 미충족 시 NULL로 갱신한다. 갱신했으면 True."""
    rows = analysis_history.get_recent(code, limit=1)
    if not rows:
        raise RuntimeError("analysis_history 이력 없음")
    row = rows[0]
    parsed = _parse_investment_summary(row.get("investment_summary"))
    price = row.get("price")
    sales_amt, ep_sales_amt = row.get("매출액-1"), row.get("매출액+1")
    sales_ok = _grew_or_similar(sales_amt, ep_sales_amt)

    # 배당율: DPS-5~DPS-1이 모두 존재·양수 + 매출 증가/유사일 때만 평균 DPS/현재가
    dps_vals = [row.get(f"DPS-{n}") for n in (5, 4, 3, 2, 1)]
    dividend_rate = None
    if all(v is not None and v > 0 for v in dps_vals) and price and sales_ok:
        dividend_rate = round(sum(float(v) for v in dps_vals) / len(dps_vals) / float(price) * 100, 2)

    # 매출증가율: 금년 매출액 대비 내년(추정) 매출액 상승률
    sales_rate = None
    if sales_amt and ep_sales_amt is not None:
        sales_rate = round((float(ep_sales_amt) - float(sales_amt)) / float(sales_amt) * 100, 1)

    # 상승잔존율(remain_rate_eligible): 상장 5년 이상 근사(매출액-5 존재) + 매출 증가/유사.
    # 값은 invest_mng.high_price 대비 현재가 기준이라 high_price는 DB 행에서 읽는다.
    remain_rate_eligible = row.get("매출액-5") is not None and sales_ok

    conn = _connect()
    try:
        with conn.cursor() as cur:
            remain_rate = None
            if remain_rate_eligible and price:
                cur.execute("SELECT high_price FROM public.invest_mng "
                            "WHERE code = %s AND proc_yn = 'Y' LIMIT 1", (code,))
                hp = cur.fetchone()
                if hp and hp[0] is not None:
                    remain_rate = round((float(hp[0]) - float(price)) / float(price) * 100, 1)

            cur.execute(
                """
                UPDATE public.invest_mng SET
                    price = COALESCE(%s, price),
                    sales_amt = COALESCE(%s, sales_amt),
                    ep_sales_amt = COALESCE(%s, ep_sales_amt),
                    remain_rate = %s,
                    dividend_rate = %s,
                    sales_rate = %s,
                    report_dt = COALESCE(%s, report_dt),
                    invest_issue = COALESCE(%s, invest_issue),
                    invest_point = COALESCE(%s, invest_point),
                    invest_risk = COALESCE(%s, invest_risk),
                    value_invest = COALESCE(%s, value_invest),
                    check_dt = %s, mod_dt = now()
                WHERE code = %s AND proc_yn = 'Y'
                """,
                (price, sales_amt, ep_sales_amt, remain_rate, dividend_rate, sales_rate,
                 row.get("rcept_dt") or None,
                 parsed["invest_issue"] or None, parsed["invest_point"] or None,
                 parsed["invest_risk"] or None, row.get("value_invest") or None,
                 datetime.now().strftime("%Y%m%d"), code),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
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
            if result.get("llm_error"):
                log.warning("[%d/%d] %s(%s) LLM 호출 에러: %s",
                            i, len(stocks), code, name, result["llm_error"])
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
            ok += 1
        except Exception:  # noqa: BLE001 — 한 종목 실패가 배치 전체를 죽이면 안 됨
            log.exception("[%d/%d] %s(%s) 실패", i, len(stocks), code, name)
            failed.append(code)
        if i < len(stocks):
            time.sleep(INTER_STOCK_PAUSE_SEC)

    log.info("배치 종료: 성공 %d / 실패 %d개 %s", ok, len(failed), failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="투자관리 종목(비-ETF) 대상 주간 mvp_graph 배치")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞 N종목만 실행")
    args = parser.parse_args()
    run_batch(limit=args.limit)
