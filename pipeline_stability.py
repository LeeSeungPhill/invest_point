"""
pipeline_stability.py
=====================
Stage 1 합격 기준 = '데이터 파이프라인 안정성'.
여러 종목을 돌려 소스별(ok/empty/error/skip) 성공률과 소요시간을 집계한다.
Stage 3 지표(교차검증 불일치율, 자동 재생성 발생률, 시나리오 일관성 검증 분포)도
함께 집계해 3단계 통제 장치가 실제로 얼마나 자주 발동하는지 보여준다.

사용:
  python pipeline_stability.py 005930 000660 112610 035420
  (인자 없으면 기본 샘플 종목 사용)

읽어볼 지표:
  - 각 소스의 ok 비율 (낮으면 그 소스 파서/키 점검)
  - error 가 특정 소스에 몰리는지 (예: research가 자주 error면 네이버 페이지 구조 변경)
  - 종목당 소요시간 (병렬 수집이 직렬 대비 단축되는지)
  - 교차검증 불일치율(너무 높으면 소스 자체를 재점검), 재생성 발생률(너무 높으면
    프롬프트/모델 자체를 재점검)
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

from mvp_graph import run

DEFAULT = ["005930", "000660", "112610", "035420", "247540"]
SOURCES = ["dart_business", "dart_financials", "news", "research", "fnguide",
          "price", "disclosures", "biz_summary", "rag", "history",
          "financial_history", "industry_category"]


def main(codes: list[str]):
    tally: dict = {s: defaultdict(int) for s in SOURCES}
    timings = []
    cite_verdicts: dict = defaultdict(int)
    scenario_verdicts: dict = defaultdict(int)
    n_cross_discrepancy = 0
    n_regenerated = 0

    for code in codes:
        t0 = time.time()
        try:
            out = run(code)
        except Exception as e:  # noqa: BLE001  (그래프 자체가 죽는 치명적 경우)
            print(f"[{code}] 그래프 실행 실패: {e}")
            for s in SOURCES:
                tally[s]["fatal"] += 1
            continue
        dt = time.time() - t0
        timings.append(dt)
        status = out.get("sources_status", {})
        cr = out.get("citation_report", {})
        verdict = cr.get("verdict", "-")
        cite_verdicts[verdict] += 1

        cc = out.get("cross_check") or {}
        if cc and not cc.get("all_ok", True):
            n_cross_discrepancy += 1
        sc = out.get("scenario_consistency") or {}
        scenario_verdicts[sc.get("verdict", "-")] += 1
        if out.get("regenerated"):
            n_regenerated += 1

        line = " ".join(f"{s}={status.get(s, '-')}" for s in SOURCES)
        print(f"[{code}] {dt:5.1f}s | {line} | "
              f"cite={verdict} (유효{cr.get('n_valid','-')}/환각{cr.get('n_invalid','-')}) | "
              f"cross={'❌' if cc and not cc.get('all_ok', True) else '✅'} | "
              f"regen={'Y' if out.get('regenerated') else 'N'}")
        for s in SOURCES:
            tally[s][status.get(s, "missing")] += 1

    n = len(codes)
    print("\n" + "=" * 60)
    print(f"종목 {n}개 / 평균 {sum(timings)/len(timings):.1f}s" if timings else "타이밍 없음")
    print("=" * 60)
    print(f"{'source':16} {'ok':>4} {'partial':>7} {'empty':>6} {'error':>6} {'skip':>5} {'off':>4}")
    for s in SOURCES:
        t = tally[s]
        print(f"{s:16} {t['ok']:>4} {t['partial']:>7} {t['empty']:>6} {t['error']:>6} "
              f"{t['skip']:>5} {t['off']:>4}   (ok {100*t['ok']//max(n,1)}%)")

    # 합격 가이드(주관적 기준 예시)
    print("\n판정 가이드:")
    for s in SOURCES:
        ok_rate = 100 * tally[s]["ok"] // max(n, 1)
        verdict = "✅ 안정" if ok_rate >= 80 else ("⚠️ 점검" if ok_rate >= 50 else "❌ 불안정")
        print(f"  {s:16} {ok_rate:3d}%  {verdict}")

    print("\n인용 품질 분포:")
    for v, c in cite_verdicts.items():
        print(f"  {v}: {c}건")

    print("\n3단계 — 교차검증/시나리오 일관성:")
    print(f"  교차검증 불일치: {n_cross_discrepancy}/{n} 종목 "
         f"({100*n_cross_discrepancy//max(n,1)}%)")
    print(f"  자동 재생성 발생: {n_regenerated}/{n} 종목 "
         f"({100*n_regenerated//max(n,1)}%)")
    for v, c in scenario_verdicts.items():
        print(f"  시나리오 검증 '{v}': {c}건")


if __name__ == "__main__":
    codes = sys.argv[1:] or DEFAULT
    main(codes)
