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

관심종목(interestItem_interest_item)도 다른 도구(simul_server 등록 화면, reservebot)가
관리하는 테이블이다. 여기서는 등록 여부·관심유형 조회와, 이미 등록된 종목의 관심유형
(watch_type) 변경만 한다 — 새 행 등록은 하지 않는다(사용자 확정: 등록에는 저항·지지가
등 가격 6개가 필요하고 매매 봇들이 그 값을 읽으므로 사람이 화면에서 등록). 관심유형을
바꿀 때는 시장구분/규모/업종/시가총액도 KIS 현재가 조회(fetch_market_meta, simul_server
_get_market_meta와 같은 기준)로 현재 시점 값으로 맞춘다.

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


_INTEREST_WHERE = ("WHERE code NOT IN ('0001', '1001') AND proc_yn = 'Y' AND code = %s ")


def normalize_watch_type(value) -> str:
    """'B, A'·['A','B'] → 'A,B'. 관심유형 코드(A~D)만 정렬해 콤마로 잇는다
    (simul_server._parse_watch_type과 같은 저장 형식)."""
    items = value if isinstance(value, (list, tuple, set)) else str(value or "").split(",")
    return ",".join(sorted({str(v).strip() for v in items} & {"A", "B", "C", "D"}))


def interest_item_watch_type(stock_code: str, *, nick_name: str = "phills75") -> tuple[Optional[bool], Optional[str]]:
    """관심종목(interestItem_interest_item, proc_yn='Y') 등록 여부와 현재 관심유형(watch_type).
    '0001'/'1001'(지수 코드)은 제외. 반환: (True, 'A,B') / (False, None) / 조회 실패 시
    (None, None) — 실패는 '미등록'으로 단정할 수 없으므로 호출부가 알림도 변경도 하지 않는다.
    같은 종목의 활성 행이 여럿이면 가장 최근에 바뀐 행의 값을 쓴다."""
    if not stock_code:
        return None, None
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT watch_type FROM public."interestItem_interest_item" '
                            + _INTEREST_WHERE + " ORDER BY last_chg_date DESC LIMIT 1",
                            (stock_code.zfill(6),))
                row = cur.fetchone()
            return (True, row[0]) if row else (False, None)
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("관심종목 조회 실패(%s): %s", stock_code, e)
        return None, None


_KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
_KIS_NICK_NAME = "phills2"  # simul_server._get_api_account와 같은 시세 조회 계좌


def _kis_account() -> dict:
    """KIS 시세 조회 계좌 인증 정보(stockAccount_stock_account, nick_name='phills2').
    simul_server._get_api_account와 같은 규칙으로, 토큰이 오늘 발급분이 아니거나
    하루가 지났으면 재발급해 DB에 저장한다(다른 도구와 토큰을 공유)."""
    import json
    from datetime import datetime
    import requests

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT acct_no, access_token, app_key, app_secret, token_publ_date '
                        'FROM "stockAccount_stock_account" WHERE nick_name = %s', (_KIS_NICK_NAME,))
            row = cur.fetchone()
        if not row:
            raise RuntimeError(f"KIS 계좌 정보 없음({_KIS_NICK_NAME})")
        acct_no, access_token, app_key, app_secret, token_publ_date = row
        now = datetime.now()
        needs_refresh = (not token_publ_date or token_publ_date[:8] != now.strftime("%Y%m%d")
                         or (now - datetime.strptime(token_publ_date, "%Y%m%d%H%M%S")).days >= 1)
        if needs_refresh:
            res = requests.post(f"{_KIS_BASE_URL}/oauth2/tokenP",
                                headers={"content-type": "application/json"},
                                data=json.dumps({"grant_type": "client_credentials",
                                                 "appkey": app_key, "appsecret": app_secret}),
                                verify=False, timeout=10)
            d = res.json()
            if "access_token" in d:
                access_token = d["access_token"]
                with conn.cursor() as cur:
                    cur.execute('UPDATE "stockAccount_stock_account" SET access_token = %s, '
                                "token_publ_date = %s, last_chg_date = %s WHERE acct_no = %s",
                                (access_token, now.strftime("%Y%m%d%H%M%S"), now, acct_no))
                conn.commit()
            else:
                logger.warning("KIS 토큰 재발급 실패: %s", d.get("error_description") or d)
        return {"app_key": app_key, "app_secret": app_secret, "access_token": access_token}
    finally:
        conn.close()


def fetch_market_meta(stock_code: str) -> dict:
    """KIS 현재가 조회로 시장구분(market)/규모(size)/업종(industry)/시가총액(mktcap, 억원)을
    현재 시점 기준으로 구한다. simul_server._get_market_meta와 같은 필드·규모 기준
    (1조 이상 대형주, 3천억 이상 중형주, 그 외 소형주)이고, 저장 형식도 관심종목 등록
    화면과 같게 mktcap은 정수 문자열로 돌려준다. 실패 시 예외."""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    ac = _kis_account()
    res = requests.get(
        f"{_KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
        headers={"Content-Type": "application/json",
                 "authorization": f"Bearer {ac['access_token']}",
                 "appkey": ac["app_key"], "appsecret": ac["app_secret"],
                 "tr_id": "FHKST01010100", "custtype": "P"},
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code.zfill(6)},
        verify=False, timeout=10,
    )
    data = res.json()
    if data.get("rt_cd") != "0" or not data.get("output"):
        raise RuntimeError(f"KIS 현재가 조회 실패: {data.get('msg1', '')}")
    out = data["output"]
    try:
        mktcap = int(str(out.get("hts_avls", "0")).replace(",", ""))
    except (TypeError, ValueError):
        mktcap = 0
    size = "대형주" if mktcap >= 10000 else "중형주" if mktcap >= 3000 else "소형주"
    return {"market": out.get("rprs_mrkt_kor_name") or None, "size": size,
            "industry": out.get("bstp_kor_isnm") or None,
            "mktcap": str(mktcap) if mktcap else None}


def update_interest_watch_type(stock_code: str, watch_type: str, *,
                               meta: Optional[dict] = None,
                               nick_name: str = "phills75") -> int:
    """등록된 관심종목(proc_yn='Y')의 관심유형을 바꾼다. 다른 갱신 경로(simul_server
    관심종목 등록, reservebot 항목 수정)처럼 last_chg_date도 함께 갱신한다. 이미 같은
    값인 행은 건드리지 않는다. meta(fetch_market_meta 결과)를 주면 시장구분/규모/업종/
    시가총액도 현재 시점 값으로 바꾸고, 빠진 값(None)은 기존 값을 유지한다.
    반환: 바뀐 행 수. 예외는 호출부가 처리한다."""
    m = meta or {}
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute('UPDATE public."interestItem_interest_item" '
                        "SET watch_type = %s, market = COALESCE(%s, market), "
                        "size = COALESCE(%s, size), industry = COALESCE(%s, industry), "
                        "mktcap = COALESCE(%s, mktcap), last_chg_date = now() "
                        + _INTEREST_WHERE + " AND watch_type IS DISTINCT FROM %s",
                        (watch_type, m.get("market"), m.get("size"), m.get("industry"),
                         m.get("mktcap"), stock_code.zfill(6), watch_type))
            n = cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()
