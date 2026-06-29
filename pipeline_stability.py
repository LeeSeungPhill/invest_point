"""
pipeline_stability.py
=====================
Stage 1 합격 기준 = '데이터 파이프라인 안정성'.
여러 종목을 돌려 소스별(ok/empty/error/skip) 성공률과 소요시간을 집계한다.

사용:
  python pipeline_stability.py 005930 000660 112610 035420
  (인자 없으면 기본 샘플 종목 사용)

읽어볼 지표:
  - 각 소스의 ok 비율 (낮으면 그 소스 파서/키 점검)
  - error 가 특정 소스에 몰리는지 (예: research가 자주 error면 네이버 페이지 구조 변경)
  - 종목당 소요시간 (병렬 수집이 직렬 대비 단축되는지)
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

from mvp_graph import run

DEFAULT = ["005930", "000660", "112610", "035420", "247540"]
SOURCES = ["dart_business", "dart_financials", "news", "research", "estimates", "rag"]


def main(codes: list[str]):
    tally: dict = {s: defaultdict(int) for s in SOURCES}
    timings = []
    cite_verdicts: dict = defaultdict(int)

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
        line = " ".join(f"{s}={status.get(s, '-')}" for s in SOURCES)
        print(f"[{code}] {dt:5.1f}s | {line} | "
              f"cite={verdict} (유효{cr.get('n_valid','-')}/환각{cr.get('n_invalid','-')})")
        for s in SOURCES:
            tally[s][status.get(s, "missing")] += 1

    n = len(codes)
    print("\n" + "=" * 60)
    print(f"종목 {n}개 / 평균 {sum(timings)/len(timings):.1f}s" if timings else "타이밍 없음")
    print("=" * 60)
    print(f"{'source':16} {'ok':>4} {'empty':>6} {'error':>6} {'skip':>5}")
    for s in SOURCES:
        t = tally[s]
        print(f"{s:16} {t['ok']:>4} {t['empty']:>6} {t['error']:>6} {t['skip']:>5}"
              f"   (ok {100*t['ok']//max(n,1)}%)")

    # 합격 가이드(주관적 기준 예시)
    print("\n판정 가이드:")
    for s in SOURCES:
        ok_rate = 100 * tally[s]["ok"] // max(n, 1)
        verdict = "✅ 안정" if ok_rate >= 80 else ("⚠️ 점검" if ok_rate >= 50 else "❌ 불안정")
        print(f"  {s:16} {ok_rate:3d}%  {verdict}")

    print("\n인용 품질 분포:")
    for v, c in cite_verdicts.items():
        print(f"  {v}: {c}건")


if __name__ == "__main__":
    codes = sys.argv[1:] or DEFAULT
    main(codes)
