"""빌드타임 전용 스크립트: TableFormer 아티팩트(accurate/fast)를 이미지에 굽는다.

docling의 TableStructureModel.download_models()와 동일한 대상(ds4sd/docling-models
@ v2.2.0)을 받는다 — 이 파드는 docling 패키지에 의존하지 않는 독립 uv 프로젝트라
huggingface_hub로 직접 받는다. paddle 서빙(모델 zip을 빌드 스테이지에서 풀어두는 것)과
동일한 목적의 스테이지에서 실행됨.
"""

import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

MODEL_ROOT = os.environ.get("TABLEFORMER_MODEL_ROOT", "/models/docling-models")

if __name__ == "__main__":
    path = snapshot_download(
        repo_id="ds4sd/docling-models",
        revision="v2.2.0",
        local_dir=MODEL_ROOT,
    )
    print(f"Downloaded docling-models to {path}")
