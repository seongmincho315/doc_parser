"""unstructured hi_res 서빙 파드.

doc-parser 전처리기가 이미지(JPG/PNG)·미지 확장자 첨부파일을 파싱하려고 프로세스 안에서
`unstructured-inference`(YOLOX 레이아웃 검출 + Microsoft Table Transformer 표 구조 인식,
torch 기반)를 로드하던 걸 없애고 이 파드를 HTTP로 호출하도록 바꾼 것의 서버 쪽 짝.
클라이언트는 genon/preprocessor/facade/common/loaders.py의 RemoteHiResLoader.

`unstructured.partition.auto.partition()`을 그대로 호출하고(strategy 등은 doc-parser가
기존에 UnstructuredImageLoader/UnstructuredFileLoader 호출 시 쓰던 것과 동일한 기본값을
유지 — skip_infer_table_types 등도 오버라이드하지 않아 기존 동작을 그대로 재현한다),
Element 리스트를 JSON으로 직렬화해 반환한다.
"""

import logging
import os
import tempfile
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

_log = logging.getLogger("unstructured-hires-serving")
logging.basicConfig(level=logging.INFO)

MAX_CONCURRENCY = int(os.environ.get("HIRES_MAX_CONCURRENCY", "4"))

app = FastAPI(title="doc-parser-unstructured-hires")
_infer_semaphore = threading.Semaphore(MAX_CONCURRENCY)
_ready = False


@app.on_event("startup")
def _startup() -> None:
    """가벼운 워밍업 — YOLOX/Table Transformer 체크포인트를 프로세스 메모리에 한 번 올려둔다.
    unstructured_inference 자체가 모델 싱글턴 캐싱을 하므로 이후 요청은 재로딩되지 않는다."""
    global _ready
    try:
        from unstructured.partition.auto import partition

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        Image.new("RGB", (64, 64), "white").save(tmp_path, format="PNG")
        try:
            partition(filename=tmp_path, strategy="hi_res", languages=["eng"])
        finally:
            os.unlink(tmp_path)
        _log.info("hi_res 워밍업 완료 (YOLOX/Table Transformer 체크포인트 로드됨)")
    except Exception:
        _log.warning("hi_res 워밍업 실패 — 첫 실제 요청에서 지연 로드됨", exc_info=True)
    _ready = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _ready else "loading"}


def _element_to_json(element) -> dict:
    d = element.to_dict()
    return {
        "text": d.get("text", ""),
        "category": d.get("type"),  # Element.to_dict()의 'type'이 곧 category
        "element_id": d.get("element_id"),
        "metadata": d.get("metadata") or {},
    }


@app.post("/partition")
def partition_endpoint(
    file: UploadFile = File(...),
    languages: str = Form("kor,eng"),
    strategy: str = Form("hi_res"),
) -> dict:
    from unstructured.partition.auto import partition

    languages_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    suffix = os.path.splitext(file.filename or "")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        with _infer_semaphore:
            elements = partition(
                filename=tmp_path,
                strategy=strategy,
                languages=languages_list or None,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"partition 실패: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return {"elements": [_element_to_json(el) for el in elements]}
