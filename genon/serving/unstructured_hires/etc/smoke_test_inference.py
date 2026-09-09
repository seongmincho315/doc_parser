"""unstructured hi_res 서빙 파드 smoke test: 합성 이미지 1장으로 /partition 1회 호출해
응답 스키마(elements[i].text/category/element_id/metadata)를 확인한다.

사용법: python smoke_test_inference.py --out /tmp/hires_smoke/result.json
"""

import argparse
import io
import json
import os

import requests
from PIL import Image, ImageDraw


def build_synthetic_image(width=400, height=200):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Hello unstructured hi_res", fill="black")
    draw.rectangle([(10, 10), (width - 10, height - 10)], outline="black", width=2)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("HIRES_PORT", "8080")))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    img = build_synthetic_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = requests.post(
        f"http://127.0.0.1:{args.port}/partition",
        files={"file": ("sample.png", buf, "image/png")},
        data={"languages": "eng", "strategy": "hi_res"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    assert "elements" in data, f"'elements' 키가 없음: {data.keys()}"
    for el in data["elements"]:
        for key in ("text", "category", "element_id", "metadata"):
            assert key in el, f"element에 '{key}' 키가 없음: {el.keys()}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[smoke] PASS: /partition 응답 스키마 확인 완료 → {args.out}")


if __name__ == "__main__":
    main()
