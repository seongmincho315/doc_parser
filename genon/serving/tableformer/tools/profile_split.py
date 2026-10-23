"""원본 multi_table_predict 의 시간을 모델 추론(_model.predict)과 그 외(크기 조정·셀 매칭·후처리)로 나눠 본다.

  python profile_split.py --models-root <...> --requests '<dir>/batch_*_old.json' --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_batched as vb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--mode", default="accurate")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    pred = vb.load_predictor(args.models_root, args.mode, args.device, args.threads)
    files = sorted(glob.glob(args.requests))
    jobs = [vb.to_job(json.load(open(f))) for f in files]
    pred.multi_table_predict(copy.deepcopy(jobs[0]["iocr_page"]), [list(b) for b in jobs[0]["table_bboxes"]])  # 워밍업

    orig = pred._model.predict
    acc = {"model": 0.0}

    def timed(*a, **k):
        t = time.perf_counter()
        r = orig(*a, **k)
        acc["model"] += time.perf_counter() - t
        return r

    pred._model.predict = timed
    rows = []
    for f, j in zip(files, jobs):
        acc["model"] = 0.0
        t = time.perf_counter()
        out = pred.multi_table_predict(j["iocr_page"], [list(b) for b in j["table_bboxes"]], do_matching=True)
        rows.append((os.path.basename(f), len(j["table_bboxes"]), sum(len(o["tf_responses"]) for o in out),
                     len(j["iocr_page"]["tokens"]), acc["model"], time.perf_counter() - t))
    print(f"{'case':22s} tables cells tokens  model(s) total(s) model%")
    tm = tt = 0.0
    for r in rows:
        print(f"{r[0]:22s} {r[1]:6d} {r[2]:5d} {r[3]:6d}  {r[4]:7.2f}  {r[5]:7.2f}  {100 * r[4] / r[5]:5.1f}")
        tm += r[4]
        tt += r[5]
    print(f"{'합계':22s} {'':6s} {'':5s} {'':6s}  {tm:7.2f}  {tt:7.2f}  {100 * tm / tt:5.1f}")


if __name__ == "__main__":
    main()
