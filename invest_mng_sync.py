"""
invest_mng_sync.py
==================
mvp_graph.run() 완료 후 analysis_history 최신 이력을 invest_mng(proc_yn='Y') 행에
반영하는 공용 로직. weekly_batch.py(주간 배치)와 simul_server.py(투자관리 화면에서
종목 조회 시 분석 이력이 오래돼 재분석한 경우)가 같은 규칙으로 갱신하도록 한 곳에 둔다.

import 부작용이 없어야 한다(simul_server가 자기 프로세스에서 import) — logging 설정이나
stdout 인코딩 변경 같은 것은 weekly_batch.py에만 둔다.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

try:  # 다른 작업 디렉터리(simul_server)에서 import돼도 이 프로젝트 .env를 읽도록
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

import psycopg2

import analysis_history


def _connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DATABASE", "fund_risk_mng"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        connect_timeout=5,
    )


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
