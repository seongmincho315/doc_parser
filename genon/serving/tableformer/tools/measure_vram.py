"""배치 크기별 GPU 메모리(최대 할당/예약)를 잰다. 요청 전체(표 24개)를 배치 크기를 바꿔가며 돌린다.

  python measure_vram.py --models-root <...> --requests '<dir>/batch_*_old.json' --device cuda:0 --batch-sizes 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_batched as vb  # noqa: E402

MIB = 1024 * 1024


def smi_used(pid: int) -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        mine = [l for l in out.splitlines() if l.split(",")[0].strip() == str(pid)]
        return mine[0].split(",")[1].strip() + " MiB" if mine else "(컨테이너 pid 라 nvidia-smi 에서 안 보임)"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--mode", default="accurate")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    args = ap.parse_args()

    jobs = [vb.to_job(json.load(open(f))) for f in sorted(glob.glob(args.requests))]
    n_tables = sum(len(j["table_bboxes"]) for j in jobs)
    torch.cuda.reset_peak_memory_stats()
    pred = vb.load_predictor(args.models_root, args.mode, args.device, 8)
    torch.cuda.synchronize()
    print(f"모델 로드 직후        : 할당 {torch.cuda.memory_allocated() / MIB:7.0f} MiB  예약 {torch.cuda.memory_reserved() / MIB:7.0f} MiB  (표 {n_tables}개, mode={args.mode})")

    torch.cuda.reset_peak_memory_stats()
    for j in jobs:
        pred.multi_table_predict(j["iocr_page"], [list(b) for b in j["table_bboxes"]], do_matching=j["do_matching"])
    torch.cuda.synchronize()
    print(f"원본(배치 1, 순차)    : 최대 할당 {torch.cuda.max_memory_allocated() / MIB:7.0f} MiB  최대 예약 {torch.cuda.max_memory_reserved() / MIB:7.0f} MiB")

    for bs in (int(x) for x in args.batch_sizes.split(",")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pred.predict_jobs(jobs, max_batch=bs)
        torch.cuda.synchronize()
        print(f"배치 {bs:2d}              : 최대 할당 {torch.cuda.max_memory_allocated() / MIB:7.0f} MiB  최대 예약 {torch.cuda.max_memory_reserved() / MIB:7.0f} MiB")
    print(f"프로세스 전체(CUDA 컨텍스트 포함, nvidia-smi): {smi_used(os.getpid())}")


if __name__ == "__main__":
    main()
