"""떠 있는 TableFormer 서버(/table/structure)를 HTTP 로 검증한다.

요청 JSON 쌍(<prefix>_old.json = 페이지 전체, <prefix>_new.json = 표 크롭)을 여러 스레드로 동시에 보내서
  1. 페이지 전체 방식과 크롭 방식의 결과(tf_responses)가 서로 같은지,
  2. --reference-url 이 있으면 그 서버(예: 예전 운영 서버)의 결과와도 같은지,
  3. 서버가 요청을 실제로 묶었는지(/stats)
를 본다.

  python test_server.py --url http://127.0.0.1:8090 --requests '/path/batch_*_old.json' --concurrency 8 \
      --reference-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import time
import urllib.request


def post(url: str, body: bytes, timeout: float = 600):
    req = urllib.request.Request(f"{url}/table/structure", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.perf_counter() - t


def get(url: str, path: str):
    with urllib.request.urlopen(f"{url}{path}", timeout=10) as r:
        return json.loads(r.read())


def cells(resp: dict):
    return [[c for c in t["tf_responses"]] for t in resp["results"]]


def struct_text(resp: dict):
    """셀 구조(행·열 범위)와 텍스트만. 배치가 2 이상이면 GPU 연산 순서 때문에 bbox 가 1px 안쪽으로 달라질 수 있다."""
    return [
        [(c["start_row_offset_idx"], c["end_row_offset_idx"], c["start_col_offset_idx"], c["end_col_offset_idx"], c.get("text"))
         for c in t["tf_responses"]]
        for t in resp["results"]
    ]


def max_bbox_diff(a: dict, b: dict) -> float:
    d = 0.0
    for ta, tb in zip(a["results"], b["results"]):
        for x, y in zip(ta["tf_responses"], tb["tf_responses"]):
            if x.get("bbox") and y.get("bbox"):
                d = max(d, *(abs(x["bbox"][k] - y["bbox"][k]) for k in ("l", "t", "r", "b")))
    return d


def compare_line(label: str, pairs: list) -> None:
    exact = sum(cells(a) == cells(b) for a, b in pairs)
    same = sum(struct_text(a) == struct_text(b) for a, b in pairs)
    worst = max((max_bbox_diff(a, b) for a, b in pairs), default=0.0)
    print(f"{label}: 완전 일치 {exact}/{len(pairs)} | 셀 구조·텍스트 일치 {same}/{len(pairs)} | bbox 최대 차이 {worst:.3f}px")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--requests", required=True, help="*_old.json glob. 같은 접두사의 *_new.json(크롭)이 있으면 같이 쓴다")
    ap.add_argument("--reference-url", default="")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    olds = sorted(glob.glob(args.requests))
    items = []  # (이름, 방식, 본문)
    for f in olds:
        items.append((f.split("/")[-1], "page", open(f, "rb").read()))
        new = f.replace("_old.json", "_new.json")
        try:
            items.append((new.split("/")[-1], "crops", open(new, "rb").read()))
        except FileNotFoundError:
            pass
    print("health:", get(args.url, "/health"))

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futs = {pool.submit(post, args.url, body): (name, kind) for name, kind, body in items}
        got = {futs[f]: f.result() for f in cf.as_completed(futs)}
    wall = time.perf_counter() - t0
    lat = sorted(v[1] for v in got.values())
    print(f"요청 {len(items)}건 동시 {args.concurrency}: 전체 {wall:.1f}s | 요청 지연 p50 {lat[len(lat)//2]:.2f}s max {lat[-1]:.2f}s")

    ref = {}
    if args.reference_url:
        for name, kind, body in items:
            if kind == "page":  # 예전 서버는 crops 를 모른다
                ref[(name, kind)] = post(args.reference_url, body)[0]

    pc, vs_ref, crop_vs_ref = [], [], []
    for f in olds:
        base = f.split("/")[-1]
        page = got[(base, "page")][0]
        crop = got.get((base.replace("_old.json", "_new.json"), "crops"))
        if crop is not None:
            pc.append((page, crop[0]))
        if ref:
            vs_ref.append((page, ref[(base, "page")]))
            if crop is not None:
                crop_vs_ref.append((crop[0], ref[(base, "page")]))
    compare_line("페이지 전체 vs 크롭 (같은 서버)", pc)
    if ref:
        compare_line("이 서버(페이지 전체) vs 레퍼런스 서버", vs_ref)
        compare_line("이 서버(크롭) vs 레퍼런스 서버", crop_vs_ref)
    try:
        for _ in range(4):  # 워커가 여러 개면 호출마다 다른 워커 값이 나온다
            s = get(args.url, "/stats")
            print("stats:", {k: s[k] for k in ("pid", "batches", "tables", "mean_tables_per_batch", "mean_wait_ms", "batch_size_histogram")})
    except Exception as exc:  # noqa: BLE001
        print("stats 없음:", exc)


if __name__ == "__main__":
    main()
