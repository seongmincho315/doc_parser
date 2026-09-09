"""TableFormer 서빙 파드 smoke test: 합성 이미지 1장으로 /table/structure 1회 호출해
응답 스키마(results[i].tf_responses / predict_details)를 확인한다.

사용법: python smoke_test_inference.py --out /tmp/tableformer_smoke/result.json
"""

import argparse
import base64
import io
import json
import os

import requests
from PIL import Image, ImageDraw


def build_synthetic_table_image(width=400, height=200):
    """표처럼 보이는 최소한의 합성 이미지(격자선)를 만든다."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(10, 10), (width - 10, height - 10)], outline="black", width=2)
    for x in range(10, width - 10, (width - 20) // 3):
        draw.line([(x, 10), (x, height - 10)], fill="black", width=1)
    for y in range(10, height - 10, (height - 20) // 3):
        draw.line([(10, y), (width - 10, y)], fill="black", width=1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("TABLEFORMER_PORT", "8080")))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    img = build_synthetic_table_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "width": float(img.width),
        "height": float(img.height),
        "image_b64": image_b64,
        "tokens": [],
        "table_bboxes": [[10.0, 10.0, float(img.width - 10), float(img.height - 10)]],
        "do_matching": False,  # 합성 이미지엔 실제 pdf 토큰이 없으므로 matching 생략
    }

    resp = requests.post(
        f"http://127.0.0.1:{args.port}/table/structure", json=payload, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()

    assert "results" in data, f"'results' 키가 없음: {data.keys()}"
    assert len(data["results"]) == 1, f"table_bboxes 1개인데 결과가 {len(data['results'])}개"
    result = data["results"][0]
    assert "tf_responses" in result, f"'tf_responses' 없음: {result.keys()}"
    assert "predict_details" in result, f"'predict_details' 없음: {result.keys()}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[smoke] PASS: /table/structure 응답 스키마 확인 완료 → {args.out}")


if __name__ == "__main__":
    main()
