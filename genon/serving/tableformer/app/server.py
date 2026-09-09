"""TableFormer 서빙 파드.

doc_parser 전처리기가 표 구조(row/col span, header 여부 등)를 인프로세스로 계산하던 걸
없애고 이 파드를 HTTP로 호출하도록 바꾼 것의 서버 쪽 짝. 클라이언트는
docling/models/table_structure_remote_model.py (TableStructureRemoteModel).

TFPredictor는 프로세스 시작 시 1회만 로드하고, 동시 추론 요청은 세마포어로 직렬화한다
(GPU 서빙 시 여러 요청이 동시에 몰려 OOM 나는 걸 막기 위함 — CLAUDE.md TODO #1의
"각 페이지 테이블을 한번에 밀어넣어 OOM" 문제에 대한 직접적인 대응).

응답 스키마(`results[i]` = `{"tf_responses":[...], "predict_details":{...}}`)는
docling_ibm_models 의 TFPredictor.multi_table_predict 가 원래 반환하던 것과 동일하게
그대로 통과시킨다 — 클라이언트 쪽 파싱 코드를 docling 인프로세스 시절과 거의 동일하게
재사용할 수 있게 하기 위함.
"""

import base64
import io
import logging
import os
import threading
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

_log = logging.getLogger("tableformer-serving")
logging.basicConfig(level=logging.INFO)

MODEL_ROOT = os.environ.get("TABLEFORMER_MODEL_ROOT", "/models/docling-models")
MODE = os.environ.get("TABLEFORMER_MODE", "accurate")  # "accurate" | "fast"
DEVICE_ENV = os.environ.get("TABLEFORMER_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
NUM_THREADS = int(os.environ.get("TABLEFORMER_NUM_THREADS", "8"))
MAX_CONCURRENCY = int(os.environ.get("TABLEFORMER_MAX_CONCURRENCY", "1"))

if MODE not in ("accurate", "fast"):
    _log.warning("Unknown TABLEFORMER_MODE=%r, falling back to 'accurate'", MODE)
    MODE = "accurate"


def _resolve_device(device_env: str) -> str:
    import torch

    device_env = (device_env or "auto").lower().strip()
    if device_env == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_env == "cpu":
        return "cpu"
    # auto
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_predictor():
    """docling/models/table_structure_model.py의 모델 로드 로직과 동일 — 이 파드는
    별도 프로젝트(uv 독립 pyproject)라 docling 패키지 없이 docling_ibm_models 만으로 직접
    로드한다."""
    import docling_ibm_models.tableformer.common as c
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

    artifacts_path = os.path.join(MODEL_ROOT, "model_artifacts", "tableformer", MODE)
    device = _resolve_device(DEVICE_ENV)

    tm_config = c.read_config(f"{artifacts_path}/tm_config.json")
    tm_config["model"]["save_dir"] = artifacts_path

    _log.info(
        "Loading TableFormer predictor: mode=%s device=%s num_threads=%d artifacts=%s",
        MODE, device, NUM_THREADS, artifacts_path,
    )
    predictor = TFPredictor(tm_config, device, NUM_THREADS)
    _log.info("TableFormer predictor loaded.")
    return predictor


app = FastAPI(title="doc-parser-tableformer")
_tf_predictor = None
_infer_semaphore = threading.Semaphore(MAX_CONCURRENCY)


@app.on_event("startup")
def _startup() -> None:
    global _tf_predictor
    _tf_predictor = _load_predictor()


class StructureRequest(BaseModel):
    width: float
    height: float
    image_b64: str  # base64 PNG
    tokens: List[Dict[str, Any]]  # [{"id","text","bbox"}, ...]
    table_bboxes: List[List[float]]  # [[l,t,r,b], ...], 순서 보존
    do_matching: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _tf_predictor is not None else "loading", "mode": MODE}


@app.post("/table/structure")
def structure(req: StructureRequest) -> dict:
    if _tf_predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if not req.table_bboxes:
        return {"results": []}

    try:
        image_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image_b64: {exc}") from exc

    page_input = {
        "width": req.width,
        "height": req.height,
        "image": np.asarray(image),
        "tokens": req.tokens,
    }

    with _infer_semaphore:
        tf_output = _tf_predictor.multi_table_predict(
            page_input, list(req.table_bboxes), do_matching=req.do_matching
        )

    return {"results": tf_output}
