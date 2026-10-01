"""빌드타임 전용 스크립트: YOLOX(레이아웃 검출)/Table Transformer(표 구조 인식) 체크포인트를
이미지에 미리 받아둔다(둘 다 HuggingFace 저장소 — unstructuredio/yolo_x_layout,
microsoft/table-transformer-structure-recognition — 라 ~/.cache/huggingface 에 캐시됨).

paddle/tableformer 서빙의 "models 빌드 스테이지"와 동일한 목적: 런타임에 매 배포마다
외부 네트워크에서 받지 않도록 이미지 안에 굽는다.
"""

import os
import tempfile

import nltk
from huggingface_hub.utils import disable_progress_bars
from PIL import Image

disable_progress_bars()

if __name__ == "__main__":
    # unstructured가 요소 분류(Title/NarrativeText 등)에 쓰는 NLTK 데이터도 빌드타임에 받아둔다
    # — 안 받아두면 런타임에 raw.githubusercontent.com 으로 매번 다운로드를 시도한다(폐쇄망에서 실패).
    for pkg in ("averaged_perceptron_tagger_eng", "punkt_tab"):
        nltk.download(pkg, quiet=True)

    from unstructured.partition.auto import partition

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    Image.new("RGB", (64, 64), "white").save(tmp_path, format="PNG")
    try:
        partition(filename=tmp_path, strategy="hi_res", languages=["eng"])
        print("YOLOX/Table Transformer 체크포인트 다운로드 완료")
    finally:
        os.unlink(tmp_path)
