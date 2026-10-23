"""TableFormer 서빙 파드.

doc_parser 전처리기가 표 구조(row/col span, header 여부 등)를 인프로세스로 계산하던 걸
없애고 이 파드를 HTTP로 호출하도록 바꾼 것의 서버 쪽 짝. 클라이언트는
docling/models/table_structure_remote_model.py (TableStructureRemoteModel).

동시 요청 처리
  - 워커(uvicorn --workers N, supervisord 의 TABLEFORMER_WORKERS)마다 모델을 한 벌씩 올린다. 워커끼리는 서로 독립이다.
  - 워커 안에서는 요청을 MicroBatcher 가 모아서(최대 TABLEFORMER_MAX_BATCH 개 표, 최대 TABLEFORMER_BATCH_DELAY_MS 대기)
    표 여러 개를 한 번에 모델에 넣는다(app/batched_predictor.py). 추론은 워커당 스레드 하나라 한 번에 한 배치만 GPU 에 들어간다.
    요청이 1개뿐이면 지연 시간만 기다렸다가 배치 1 로 처리한다.
  - TABLEFORMER_MAX_BATCH=1 이면 예전처럼 표를 하나씩 처리한다(결과는 같다).

입력 이미지: 페이지 전체(`image_b64`) 또는 표 영역 크롭(`crops`). 크롭은 페이지 크기의 흰 캔버스에 bbox 위치로 붙여서
기존 경로에 넣는다(multi_table_predict 가 페이지를 높이 1024 로 줄인 뒤 bbox 로 다시 자르므로 같은 크기면 결과가 같다).
크롭 둘레에 4pt 이상 여백을 붙여 보내야 경계 픽셀이 주변을 섞어 읽는 리사이즈(INTER_AREA)에서도 같은 결과가 나온다.

응답 스키마(`results[i]` = `{"tf_responses":[...], "predict_details":{...}}`)는
docling_ibm_models 의 TFPredictor.multi_table_predict 가 원래 반환하던 것과 동일하게
그대로 통과시킨다 — 클라이언트 쪽 파싱 코드를 docling 인프로세스 시절과 거의 동일하게
재사용할 수 있게 하기 위함.
"""

import base64
import io
import logging
import math
import os
from typing import Any, Dict, List, Optional

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
MAX_BATCH = int(os.environ.get("TABLEFORMER_MAX_BATCH", "8"))  # 한 번에 모델에 넣을 표 수(워커당)
BATCH_DELAY_MS = float(os.environ.get("TABLEFORMER_BATCH_DELAY_MS", "5"))  # 묶음을 기다리는 최대 시간(ms)
BBOX_CHUNK = int(os.environ.get("TABLEFORMER_BBOX_CHUNK", "64"))  # bbox 디코더 셀 묶음 크기(메모리 상한)
REQUEST_TIMEOUT_SEC = float(os.environ.get("TABLEFORMER_REQUEST_TIMEOUT_SEC", "600"))

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
    로드한다. TFPredictor 대신 여러 표를 한 번에 돌리는 TFPredictorBatched 를 쓴다."""
    import docling_ibm_models.tableformer.common as c

    from app.batched_predictor import TFPredictorBatched

    artifacts_path = os.path.join(MODEL_ROOT, "model_artifacts", "tableformer", MODE)
    device = _resolve_device(DEVICE_ENV)

    tm_config = c.read_config(f"{artifacts_path}/tm_config.json")
    tm_config["model"]["save_dir"] = artifacts_path

    _log.info(
        "Loading TableFormer predictor: mode=%s device=%s num_threads=%d artifacts=%s",
        MODE, device, NUM_THREADS, artifacts_path,
    )
    predictor = TFPredictorBatched(tm_config, device, NUM_THREADS)
    _log.info("TableFormer predictor loaded.")
    return predictor


app = FastAPI(title="doc-parser-tableformer")
_tf_predictor = None
_batcher = None


@app.on_event("startup")
def _startup() -> None:
    global _tf_predictor, _batcher
    from app.micro_batcher import MicroBatcher

    _tf_predictor = _load_predictor()
    _batcher = MicroBatcher(_tf_predictor, max_batch=MAX_BATCH, max_delay_ms=BATCH_DELAY_MS, bbox_chunk=BBOX_CHUNK)
    _log.info("MicroBatcher started: max_batch=%d delay_ms=%s bbox_chunk=%d", MAX_BATCH, BATCH_DELAY_MS, BBOX_CHUNK)


class StructureRequest(BaseModel):
    width: float
    height: float
    image_b64: str = ""  # base64 PNG, 페이지 전체. crops 를 보낼 때는 비워도 된다
    # 표 크롭만 보내는 방식: [{"bbox": [l,t,r,b] (width/height 와 같은 좌표계 px), "image_b64": 그 영역 PNG}, ...]
    crops: Optional[List[Dict[str, Any]]] = None
    tokens: List[Dict[str, Any]]  # [{"id","text","bbox"}, ...]
    table_bboxes: List[List[float]]  # [[l,t,r,b], ...], 순서 보존
    do_matching: bool = True


def _page_image(req: StructureRequest) -> np.ndarray:
    """요청의 이미지를 페이지 크기 RGB 배열로 만든다. 크롭이면 흰 캔버스에 bbox 위치로 붙인다."""
    if req.crops:
        try:
            canvas = np.full((math.ceil(req.height), math.ceil(req.width), 3), 255, dtype=np.uint8)
            for crop in req.crops:
                arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(crop["image_b64"]))).convert("RGB"))
                left, top = int(round(crop["bbox"][0])), int(round(crop["bbox"][1]))
                h = min(arr.shape[0], canvas.shape[0] - top)
                w = min(arr.shape[1], canvas.shape[1] - left)
                if h > 0 and w > 0 and left >= 0 and top >= 0:
                    canvas[top : top + h, left : left + w] = arr[:h, :w]
            return canvas
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid crops: {exc}") from exc
    try:
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image_b64: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _batcher is not None else "loading", "mode": MODE}


@app.get("/stats")
def stats() -> dict:
    """배치 통계(워커별). 워커가 여러 개면 호출할 때마다 다른 워커의 값이 나온다."""
    if _batcher is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"pid": os.getpid(), **_batcher.stats()}


@app.post("/table/structure")
def structure(req: StructureRequest) -> dict:
    if _batcher is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if not req.table_bboxes:
        return {"results": []}

    job = {
        "iocr_page": {
            "width": req.width,
            "height": req.height,
            "image": _page_image(req),
            "tokens": req.tokens,
        },
        "table_bboxes": [list(b) for b in req.table_bboxes],
        "do_matching": req.do_matching,
    }
    future = _batcher.submit(job)
    try:
        tf_output = future.result(timeout=REQUEST_TIMEOUT_SEC)
    except Exception as exc:
        _log.exception("table structure failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"results": tf_output}
