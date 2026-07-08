"""
analysis_history.py
====================
3단계: 과거 분석 이력을 PostgreSQL(fund_risk_mng DB)에 저장하고, 다음 분석 때 참조한다.

목적:
  - LLM이 '지난 분석 대비 개선/악화'를 실제 저장된 숫자로만 서술하게 한다
    (과거 값을 제공하지 않으면 "전보다 좋아졌다" 같은 비교를 지어낼 수 있으므로,
    참조 가능한 과거 기록을 명시적으로 제공 — 없는 비교는 하지 말라고 프롬프트에서 요구).
  - check_drift()로 '근거 수치(증감률/밴드위치)는 거의 그대로인데 판단(성장 방향/
    가치 시그널)만 뒤집힌' 경우를 잡아 참고 신호로 남긴다. 단, 과거 이력은 이번
    실행의 직접 근거가 아니므로 자동 재생성(regenerate) 트리거로는 쓰지 않는다
    — scenario_check.py의 '단일 실행 내 일관성' 검증과 역할을 분리한다.

접속정보(.env): PG_HOST/PG_PORT/PG_DATABASE/PG_USER/PG_PASSWORD.
DISABLE_HISTORY=1 로 저장/조회를 모두 끌 수 있다(DB 접속 불가 환경 대비).
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
import psycopg2.extras

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id SERIAL PRIMARY KEY,
    stock_code TEXT NOT NULL,
    corp_name TEXT,
    run_at TIMESTAMP NOT NULL,
    report_nm TEXT,
    rcept_dt TEXT,
    growth_trend TEXT,
    op_yoy_forward DOUBLE PRECISION,
    value_signal BOOLEAN,
    band_position DOUBLE PRECISION,
    target_upside_pct DOUBLE PRECISION,
    price DOUBLE PRECISION,
    citation_verdict TEXT,
    cross_check_ok BOOLEAN,
    scenario_verdict TEXT,
    regenerated BOOLEAN,
    report TEXT
);
CREATE INDEX IF NOT EXISTS idx_hist_stock_time ON analysis_history(stock_code, run_at DESC);
"""


def is_enabled() -> bool:
    return os.getenv("DISABLE_HISTORY", "0") != "1"


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
        cur.execute(_SCHEMA)
    conn.commit()


def save_run(*, stock_code: str, corp_name: Optional[str] = None,
            report_nm: Optional[str] = None, rcept_dt: Optional[str] = None,
            invest_point: Optional[dict] = None, price: Optional[dict] = None,
            citation_report: Optional[dict] = None, cross_check: Optional[dict] = None,
            scenario_consistency: Optional[dict] = None, regenerated: bool = False,
            report: Optional[str] = None) -> None:
    """이번 실행의 핵심 결과를 이력에 남긴다. 실패해도 그래프를 죽이지 않도록
    호출부(mvp_graph.save_history)에서 예외를 잡는다."""
    if not is_enabled() or not stock_code:
        return
    growth = (invest_point or {}).get("growth", {})
    valuation = (invest_point or {}).get("valuation", {})
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO analysis_history
                   (stock_code, corp_name, run_at, report_nm, rcept_dt, growth_trend,
                    op_yoy_forward, value_signal, band_position, target_upside_pct, price,
                    citation_verdict, cross_check_ok, scenario_verdict, regenerated, report)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (stock_code, corp_name, time.strftime("%Y-%m-%d %H:%M:%S"), report_nm, rcept_dt,
                 growth.get("trend"), growth.get("op_yoy_forward"),
                 bool(valuation.get("signal")), valuation.get("band_position"),
                 valuation.get("target_upside_pct"), (price or {}).get("price"),
                 (citation_report or {}).get("verdict"),
                 bool((cross_check or {}).get("all_ok", True)),
                 (scenario_consistency or {}).get("verdict"),
                 bool(regenerated), report),
            )
        conn.commit()
    finally:
        conn.close()


def get_recent(stock_code: str, *, limit: int = 5) -> list[dict]:
    """가장 최근 것부터. history[0]가 직전 실행."""
    if not is_enabled() or not stock_code:
        return []
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM analysis_history WHERE stock_code=%s "
                "ORDER BY run_at DESC LIMIT %s",
                (stock_code, limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _fmt_pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "자료없음"


def _fmt_price(v) -> str:
    return f"{v:,.0f}원" if v is not None else "자료없음"


def format_history_block(history: list[dict]) -> str:
    """analyze() 프롬프트에 넣을 과거 이력 요약. 여기 없는 숫자는 LLM이 비교에
    쓸 수 없으므로(프롬프트 규칙으로 강제), '지어낸 트렌드 서술'을 막는 역할도 한다."""
    if not history:
        return "(과거 분석 이력 없음 — 이번이 최초 분석)"
    lines = []
    for h in history:
        run_at = h.get("run_at")
        date_str = str(run_at)[:10] if run_at else "?"
        lines.append(
            f"- {date_str} ({h.get('report_nm') or '?'}): "
            f"성장판단={h.get('growth_trend') or '자료없음'}, "
            f"영업이익증감률={h.get('op_yoy_forward')}, "
            f"가치시그널={'예' if h.get('value_signal') else '아니오'}, "
            f"밴드위치={_fmt_pct(h.get('band_position'))}, "
            f"주가={_fmt_price(h.get('price'))}, "
            f"인용판정={h.get('citation_verdict') or '-'}"
        )
    return "\n".join(lines)


def check_drift(current_ip: dict, history: list[dict]) -> Optional[dict]:
    """가장 최근 과거 기록과 비교해 '근거 수치 부호는 안 바뀌었는데 판단만
    뒤집힌' 경우를 참고 신호로 남긴다(재생성 트리거 아님 — 사람이 보라는 표시)."""
    if not history:
        return None
    prev = history[0]
    growth = (current_ip or {}).get("growth", {})
    valuation = (current_ip or {}).get("valuation", {})
    flips = []

    prev_trend, cur_trend = prev.get("growth_trend"), growth.get("trend")
    prev_yoy, cur_yoy = prev.get("op_yoy_forward"), growth.get("op_yoy_forward")
    if prev_trend and cur_trend and prev_yoy is not None and cur_yoy is not None:
        prev_up = any(k in prev_trend for k in ("가속", "지속", "개선", "턴어라운드"))
        cur_up = any(k in cur_trend for k in ("가속", "지속", "개선", "턴어라운드"))
        if prev_up != cur_up and (prev_yoy >= 0) == (cur_yoy >= 0):
            flips.append(f"성장 판단 {prev_trend}→{cur_trend} "
                        f"(증감률 부호는 유지: {prev_yoy}%→{cur_yoy}%)")

    prev_signal, cur_signal = prev.get("value_signal"), valuation.get("signal")
    prev_band, cur_band = prev.get("band_position"), valuation.get("band_position")
    if (prev_signal is not None and cur_signal is not None
            and bool(prev_signal) != bool(cur_signal)
            and prev_band is not None and cur_band is not None
            and abs(prev_band - cur_band) < 0.05):
        flips.append(f"가치 시그널 {'예' if prev_signal else '아니오'}→"
                    f"{'예' if cur_signal else '아니오'} "
                    f"(밴드위치는 거의 그대로: {prev_band*100:.1f}%→{cur_band*100:.1f}%)")

    if not flips:
        return None
    return {"prev_run_at": str(prev.get("run_at")), "flips": flips}
