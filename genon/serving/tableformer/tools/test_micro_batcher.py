"""MicroBatcher 동작 확인: 여러 스레드가 동시에 요청을 넣어도 결과가 원본과 같고, 배치가 기대대로 묶이는지 본다.

  python test_micro_batcher.py --models-root <...> --requests '<dir>/batch_*_old.json' --device cpu

시나리오:
  1. 순차(요청 간 간격 두기)      : 요청이 1개씩 오면 배치 표 수 = 그 요청의 표 수. 지연 설정만큼만 기다린다.
  2. 동시(전부 한꺼번에)          : 몰리면 max_batch 까지 묶인다.
각 시나리오에서 결과가 원본 multi_table_predict 와 같은지(셀 구조·텍스트) 확인한다.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_batched as vb  # noqa: E402
from app.micro_batcher import MicroBatcher  # noqa: E402


def run_case(predictor, jobs, base, label, max_batch, delay_ms, gap_s):
    batcher = MicroBatcher(predictor, max_batch=max_batch, max_delay_ms=delay_ms)
    futs = [None] * len(jobs)

    def go(i):
        futs[i] = batcher.submit(copy.copy(jobs[i]))

    t = time.perf_counter()
    threads = []
    for i in range(len(jobs)):
        th = threading.Thread(target=go, args=(i,))
        th.start()
        threads.append(th)
        if gap_s:
            time.sleep(gap_s)
    for th in threads:
        th.join()
    results = [f.result(timeout=600) for f in futs]
    dt = time.perf_counter() - t
    r = vb.compare(base, results)
    st = batcher.stats()
    print(
        f"[{label}] max_batch={max_batch} delay={delay_ms}ms gap={gap_s}s : {dt:6.2f}s | "
        f"배치 {st['batches']}회 평균 {st['mean_tables_per_batch']}표/배치 히스토그램 {st['batch_size_histogram']} "
        f"평균대기 {st['mean_wait_ms']}ms | 셀 일치 {r['same_cells']}/{r['tables']} tag_seq 일치 {r['same_tag_seq']}/{r['tables']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--mode", default="accurate")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(args.requests))
    jobs = [vb.to_job(json.load(open(f))) for f in files]
    predictor = vb.load_predictor(args.models_root, args.mode, args.device, args.threads)
    predictor.multi_table_predict(copy.deepcopy(jobs[0]["iocr_page"]), [list(b) for b in jobs[0]["table_bboxes"]])
    base = [
        predictor.multi_table_predict(j["iocr_page"], [list(b) for b in j["table_bboxes"]], do_matching=j["do_matching"])
        for j in jobs
    ]
    print(f"requests={len(jobs)} tables={sum(len(j['table_bboxes']) for j in jobs)} device={args.device}")
    run_case(predictor, jobs, base, "순차", max_batch=4, delay_ms=0, gap_s=0.5)
    run_case(predictor, jobs, base, "순차+지연", max_batch=4, delay_ms=20, gap_s=0.5)
    run_case(predictor, jobs, base, "동시", max_batch=4, delay_ms=5, gap_s=0)
    run_case(predictor, jobs, base, "동시", max_batch=8, delay_ms=5, gap_s=0)
    run_case(predictor, jobs, base, "동시 delay0", max_batch=8, delay_ms=0, gap_s=0)


if __name__ == "__main__":
    main()
