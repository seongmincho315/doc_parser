"""배치 추론이 원본 multi_table_predict 와 같은 결과를 내는지, 얼마나 빠른지 확인한다.

요청 JSON(서버 /table/structure 요청 형식, image_b64 = 페이지 전체)을 원본으로 한 번, 배치 크기를 바꿔가며 한 번씩 돌려 비교한다.

  python verify_batched.py --models-root <model_artifacts 가 들어 있는 폴더> --requests '<dir>/batch_*_old.json' \
      --device cpu --batch-sizes 1,2,4,8

비교 항목: 태그 시퀀스(tag_seq) 일치, 응답 셀(tf_responses)의 구조·텍스트 일치, 셀 bbox 최대 차이, 걸린 시간.
"""

from __future__ import annotations

import argparse
import base64
import copy
import glob
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/ (verify_batched 등)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 서버 루트(app 패키지)
from app.batched_predictor import TFPredictorBatched  # noqa: E402


def load_predictor(models_root: str, mode: str, device: str, threads: int) -> TFPredictorBatched:
    import docling_ibm_models.tableformer.common as c

    artifacts = os.path.join(models_root, "model_artifacts", "tableformer", mode)
    cfg = c.read_config(f"{artifacts}/tm_config.json")
    cfg["model"]["save_dir"] = artifacts
    return TFPredictorBatched(cfg, device, threads)


def to_job(req: dict) -> dict:
    image = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
    return {
        "iocr_page": {"width": req["width"], "height": req["height"], "image": np.asarray(image), "tokens": req["tokens"]},
        "table_bboxes": req["table_bboxes"],
        "do_matching": req.get("do_matching", True),
    }


def cells_key(result: dict):
    return [
        (c["start_row_offset_idx"], c["end_row_offset_idx"], c["start_col_offset_idx"], c["end_col_offset_idx"], c.get("text"))
        for c in result["tf_responses"]
    ]


def max_bbox_diff(a: dict, b: dict) -> float:
    d = 0.0
    for x, y in zip(a["tf_responses"], b["tf_responses"]):
        if x.get("bbox") and y.get("bbox"):
            d = max(d, *(abs(x["bbox"][k] - y["bbox"][k]) for k in ("l", "t", "r", "b")))
    return d


def compare(base: list, got: list) -> dict:
    same_seq = same_cells = n = 0
    worst = 0.0
    errors = 0
    for rb, rg in zip(base, got):
        if isinstance(rg, Exception) or isinstance(rb, Exception):
            errors += 1
            continue
        for tb, tg in zip(rb, rg):
            n += 1
            same_seq += tb["predict_details"]["prediction"]["tag_seq"] == tg["predict_details"]["prediction"]["tag_seq"]
            ok = cells_key(tb) == cells_key(tg)
            same_cells += ok
            worst = max(worst, max_bbox_diff(tb, tg))
    return {"tables": n, "same_tag_seq": same_seq, "same_cells": same_cells, "max_bbox_diff": worst, "job_errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--mode", default="accurate")
    ap.add_argument("--requests", required=True, help="요청 JSON glob (image_b64 = 페이지 전체)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-baseline", action="store_true", help="원본 실행·비교를 건너뛰고 배치 시간만 잰다(처리량 비교용)")
    ap.add_argument("--start-at", type=float, default=0.0, help="이 epoch 초까지 기다렸다가 시작한다(여러 프로세스 동시 시작용)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.requests))
    if args.limit:
        files = files[: args.limit]
    reqs = [json.load(open(f)) for f in files]
    predictor = load_predictor(args.models_root, args.mode, args.device, args.threads)
    jobs = [to_job(r) for r in reqs]
    n_tables = sum(len(j["table_bboxes"]) for j in jobs)
    print(f"requests={len(jobs)} tables={n_tables} device={args.device} mode={args.mode}")

    # 워밍업(첫 호출은 느리다)
    predictor.multi_table_predict(copy.deepcopy(jobs[0]["iocr_page"]), [list(b) for b in jobs[0]["table_bboxes"]])

    base, t_base = None, 0.0
    if not args.skip_baseline:
        t = time.perf_counter()
        base = [
            predictor.multi_table_predict(j["iocr_page"], [list(b) for b in j["table_bboxes"]], do_matching=j["do_matching"])
            for j in jobs
        ]
        t_base = time.perf_counter() - t
        print(f"원본 multi_table_predict : {t_base:7.2f}s  ({n_tables / t_base:5.2f} tables/s)")

    for bs in (int(x) for x in args.batch_sizes.split(",")):
        if args.start_at:
            time.sleep(max(0.0, args.start_at - time.time()))
        t = time.perf_counter()
        got = predictor.predict_jobs(jobs, max_batch=bs)
        dt = time.perf_counter() - t
        if base is None:
            print(f"배치 {bs:2d} (비교 생략)   : {dt:7.2f}s  ({n_tables / dt:5.2f} tables/s)  tables={n_tables}")
            continue
        r = compare(base, got)
        print(
            f"배치 {bs:2d}              : {dt:7.2f}s  ({n_tables / dt:5.2f} tables/s, 원본 대비 {t_base / dt:4.2f}x)  "
            f"tag_seq 일치 {r['same_tag_seq']}/{r['tables']}  셀 일치 {r['same_cells']}/{r['tables']}  "
            f"bbox 최대차 {r['max_bbox_diff']:.3f}px  job 오류 {r['job_errors']}"
        )


if __name__ == "__main__":
    main()
