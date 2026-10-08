"""적재용 최소 전처리기 facade (test/model_only) -- 단일 파일, 기존 doc-parser 이미지 그대로 사용.

GenOS 는 이 파일을 /app/src/preprocessor.py 로 마운트해 `DocumentProcessor()` 를 무인자로 만들고
`await processor(request, file_path, **params)` 로 부른다(src/main.py). 이미지를 새로 빌드하지 않도록
이미지에 이미 있는 패키지(httpx, pymupdf, bs4, docling-core, genon.preprocessor.processing 의 청커)만 쓴다.
mineru-vl-utils 는 이미지에 없으므로 2단계 호출을 아래에 직접 구현했다.

파이프라인 (모델은 전부 HTTP 호출):
  1. 변환   : PDF 가 아니면 PDF 로 (hwp/hwpx -> rhwp, 그 외 -> LibreOffice)
  2. 레이아웃: MinerU2.5 VLM 또는 dots.mocr layout_only (페이지 -> 블록 bbox/type, layout_backend)
  3. 인식   : 블록 크롭을 type 별 백엔드로 -- 텍스트: mineru|paddle|ppocr, 표: mineru|paddle|tableformer,
              수식: mineru (paddle 이면 paddle, ppocr 이면 mineru), 그림: none|mineru(Image Analysis)|vlm(외부 VLM 설명)|mineru_vlm(MinerU 내용 + VLM 설명),
              차트: 그림 값 + paddle(Chart Recognition).
              묶음 블록(list 등)은 인식하지 않는다.
              tableformer 표의 셀 텍스트는 PDF 텍스트 레이어 단어, 없으면 PP-OCR 단어 박스. 실패한 표만 MinerU.
  4. 청킹   : 블록 -> DoclingDocument -> GenosSmartChunker
  5. GenOS  : 청크 -> GenOSVectorMeta, 그림 크롭/변환 PDF 업로드
설정: resource/test_processor_config.yaml (GenOS 는 MinIO <id>/resource/ 에서 /app/resource 로 받아온다).

MinerU 클라이언트 부분(레이아웃 출력 파싱, 크롭 규칙, 샘플링 값, OTSL 표 파싱)은 mineru-vl-utils 2.0.5
(Apache-2.0, opendatalab) 의 http-client 동작을 옮긴 것이다.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import itertools
import logging
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pymupdf
import yaml
from bs4 import BeautifulSoup
from docling_core.transforms.chunker import DocChunk
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    PictureItem,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import ContentLayer, DescriptionAnnotation, DocumentOrigin
from fastapi import Request
from PIL import Image
from pydantic import BaseModel

from genon.preprocessor.processing.chunking import header_path as hp
from genon.preprocessor.processing.chunking import smart_chunker as sc
from genon.preprocessor.processing.common import config_parse as cp
from genon.preprocessor.processing.common import file_probe as fp
from genon.preprocessor.processing.common import pdf_convert as pc
from genon.preprocessor.processing.common import vector_meta as vm

try:  # 파드 안에서는 /app/src 가 PYTHONPATH 에 있다
    from common.exception import GenosServiceException
except ImportError:  # 로컬 실행

    class GenosServiceException(Exception):
        def __init__(self, error_code, error_msg=None, msg_params=None, *, stage=None, error_type=None):
            super().__init__(error_msg)
            self.code, self.error_code, self.error_msg = 1, error_code, error_msg or "GenOS Service Exception"
            self.msg_params, self.stage, self.error_type = msg_params or {}, stage, error_type


try:
    from genos_utils import upload_files
except ImportError:
    upload_files = None

_log = logging.getLogger(__name__)

_CHUNK_HEADER_SEP = " > "
_CHUNK_PATH_SEP = " | "
_CHUNK_PATH_MAX_LEAVES = 5
_MIN_CHUNK_SIZE = 1024
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "resource" / "test_processor_config.yaml"


# ============================================================================ VLM 호출 공통
class VlmServer:
    """OpenAI 호환(vLLM·GenOS 게이트웨이) 서버 하나. 동시 요청 수를 세마포어로 묶는다.

    endpoint 는 서버 루트(/v1/chat/completions 를 붙인다), url 은 chat/completions 전체 주소(GenOS 서빙 게이트웨이).
    """

    def __init__(self, cfg: dict):
        self.endpoint = cfg.get("url") or cfg["endpoint"].rstrip("/") + "/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
        self.model = cfg["model"]
        self.timeout = float(cfg.get("timeout", 600))
        self.max_concurrency = int(cfg.get("max_concurrency", 64))
        self.max_tokens = cfg.get("max_tokens")
        # 세마포어는 만든 이벤트 루프에 묶이므로 루프마다 따로 둔다(워커 프로세스 안의 동시 문서들이 한도를 공유).
        self._sems: dict[int, asyncio.Semaphore] = {}

    async def ask(
        self,
        client: httpx.AsyncClient,
        image: Image.Image,
        prompt: str,
        sampling: dict,
        system_prompt: Optional[str] = None,
    ) -> str:
        sem = self._sems.setdefault(id(asyncio.get_running_loop()), asyncio.Semaphore(self.max_concurrency))
        buf = io.BytesIO()
        image.save(buf, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}],
            }
        )
        body = {"model": self.model, "messages": messages, **sampling}
        if self.max_tokens:
            body["max_tokens"] = int(self.max_tokens)
        async with sem:
            r = await client.post(self.endpoint, json=body, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class PpOcrServer:
    """PaddleX OCR 파이프라인 서빙(PP-OCRv5 검출+인식, `POST /ocr`). 줄 또는 단어 단위 (text, bbox) 를 돌려준다."""

    def __init__(self, cfg: dict):
        self.endpoint = cfg["endpoint"]
        self.timeout = float(cfg.get("timeout", 60))
        self.max_concurrency = int(cfg.get("max_concurrency", 16))
        self._sems: dict[int, asyncio.Semaphore] = {}

    async def ocr(
        self, client: httpx.AsyncClient, image: Image.Image, word_box: bool = False
    ) -> list[tuple[str, list[float]]]:
        """bbox 는 입력 이미지 픽셀 좌표 [x0, y0, x1, y1]. word_box=True 면 줄을 단어(문자 종류 경계)로 쪼갠 박스."""
        sem = self._sems.setdefault(id(asyncio.get_running_loop()), asyncio.Semaphore(self.max_concurrency))
        buf = io.BytesIO()
        image.save(buf, "PNG")
        body = {
            "file": base64.b64encode(buf.getvalue()).decode(),
            "fileType": 1,
            "visualize": False,
            "returnWordBox": word_box,
        }
        async with sem:
            r = await client.post(self.endpoint, json=body, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("errorCode") not in (0, None):
            raise RuntimeError(f"PP-OCR errorCode={j.get('errorCode')}: {j.get('errorMsg')}")
        res = j["result"]["ocrResults"][0]["prunedResult"]
        if word_box:
            return [
                (t, b)
                for words, boxes in zip(res.get("text_word", []), res.get("text_word_boxes", []))
                for t, b in zip(words, boxes)
                if t.strip()
            ]
        return [(t, b) for t, b in zip(res.get("rec_texts", []), res.get("rec_boxes", [])) if t.strip()]


# ============================================================================ MinerU 2단계 (mineru-vl-utils 이식)
# image_backend: vlm 의 기본 프롬프트(resource/prompt_image_description_default.md 와 같은 내용)
_IMAGE_DESCRIPTION_PROMPT = """문서의 일부 이미지를 설명해줘. 아래 문맥을 참고해서 핵심 정보를 2~4문장으로 간결하게 작성해줘.

[앞 문맥]
{{before_context}}

[캡션]
{{caption}}

[뒤 문맥]
{{after_context}}

요구사항:
1) 추측은 최소화하고 이미지에서 확인 가능한 사실 중심으로 작성
2) 문서 문맥과의 연결점을 포함
3) 한국어로 작성"""
# chart_backend: vlm 의 기본 프롬프트(resource/prompt_chart_description_default.md 와 같은 내용)
_CHART_DESCRIPTION_PROMPT = """Convert the image into retrieval-friendly Korean text.

[Doc Summary]
{{doc_summary}}

[Section Header]
{{section_header}}

[Caption]
{{caption}}

[Before Context]
{{before_context}}

[After Context]
{{after_context}}

Instructions:
- If the image contains a chart, graph, table, checklist, form, or structured visual data, convert the visible data into a Markdown table when possible.
- Preserve labels, units, dates, legends, axes, footnotes, and source text when visible.
- If a numeric value is not printed but can only be estimated from an axis, mark it as "추정".
- If the image is not a chart/table/form, write a concise factual description instead.
- Do not invent missing values.
- If text or numbers are unclear, write "판독 불가" or "일부 판독 불가".
- Write in Korean.

Output only the converted content or factual description."""
_PROMPT_VAR_RE = re.compile(r"\{\{(\w+)\}\}")
_CAPTION_TYPES = {"image_caption", "caption"}

_MINERU_SYSTEM_PROMPT = "You are a helpful assistant."
_MINERU_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}


def _mineru_sampling(presence: float = 0.0, frequency: float = 0.0) -> dict:
    # greedy + no_repeat_ngram_size=100(서버의 MinerULogitsProcessor 가 vllm_xargs 로 받는다)
    return {
        "temperature": 0.0,
        "top_p": 0.01,
        "top_k": 1,
        "presence_penalty": presence,
        "frequency_penalty": frequency,
        "repetition_penalty": 1.0,
        "skip_special_tokens": False,
        "vllm_xargs": {"no_repeat_ngram_size": 100, "debug": False},
    }


_MINERU_SAMPLING = {
    "table": _mineru_sampling(1.0, 0.005),
    "equation": _mineru_sampling(1.0, 0.05),
    "image": _mineru_sampling(1.0, 0.05),
    "[default]": _mineru_sampling(1.0, 0.05),
    "[layout]": _mineru_sampling(),
}
_LAYOUT_IMAGE_SIZE = (1036, 1036)
_LAYOUT_RE = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?(.*?)(?=<\|box_start\|>|$)",
    re.DOTALL,
)
_ANGLES = {"<|rotate_up|>": 0, "<|rotate_right|>": 90, "<|rotate_down|>": 180, "<|rotate_left|>": 270}
_BLOCK_TYPES = {
    "text",
    "title",
    "doc_title",
    "paragraph_title",
    "table",
    "equation",
    "formula_number",
    "code",
    "algorithm",
    "aside_text",
    "ref_text",
    "index",
    "phonetic",
    "list_item",
    "caption",
    "table_caption",
    "image_caption",
    "code_caption",
    "footnote",
    "table_footnote",
    "image_footnote",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "image",
    "chart",
    "list",
    "image_block",
    "equation_block",
}
# 인식하지 않는 블록: 묶음(list/image_block/equation_block)과 그림/차트(이미지 분석 끔)
_NOT_EXTRACTED = {"list", "image_block", "equation_block", "image", "chart"}
# 이미지 분석(Image Analysis) 대상: 가로·세로가 페이지의 10% 초과이거나 면적 1% 초과(아이콘·로고 제외)
_IMAGE_ANALYSIS_MIN_SIZE = 0.1
_IMAGE_ANALYSIS_MIN_AREA = 0.01
_IMAGE_ANALYSIS_RE = re.compile(r"<\|(caption|content)_start\|>(.*?)<\|\1_end\|>", re.DOTALL)
_MIN_IMAGE_EDGE = 28
_MAX_IMAGE_EDGE_RATIO = 50


def _convert_bbox(coords) -> Optional[list[float]]:
    x1, y1, x2, y2 = map(int, coords)
    if any(c < 0 or c > 1000 for c in (x1, y1, x2, y2)):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x1 == x2 or y1 == y2:
        return None
    return [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]


def _cover_ratio(inner, outer) -> float:
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if area == 0:
        return 0.0
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    return (w * h) / area if w > 0 and h > 0 else 0.0


def _covered(blocks: list[dict], candidates: set, containers: set, threshold: float = 0.9) -> set[int]:
    box_idx = [i for i, b in enumerate(blocks) if b["type"] in containers]
    return {
        i
        for i, b in enumerate(blocks)
        if b["type"] in candidates
        and any(j != i and _cover_ratio(b["bbox"], blocks[j]["bbox"]) >= threshold for j in box_idx)
    }


def parse_layout_output(output: str) -> list[dict]:
    blocks = []
    for m in _LAYOUT_RE.finditer(output):
        x1, y1, x2, y2, btype, rotate, _tail = m.groups()
        bbox = _convert_bbox((x1, y1, x2, y2))
        btype = btype.lower()
        if bbox is None or btype == "inline_formula":
            continue
        btype = "image" if btype == "unknown" else btype
        if btype not in _BLOCK_TYPES:
            continue
        blocks.append({"type": btype, "bbox": bbox, "angle": _ANGLES.get(rotate or "", 0), "content": None})
    # 표 안에 잡힌 텍스트/수식 블록, 그림 묶음 안의 캡션은 버린다(공식 클라이언트와 같은 규칙)
    drop = _covered(blocks, {"text", "equation", "equation_block"}, {"table"})
    drop |= _covered(blocks, {"image_caption"}, {"image", "chart", "image_block"})
    return [b for i, b in enumerate(blocks) if i not in drop]


# ---------------------------------------------------------------------------- dots.mocr 레이아웃 (layout_backend: dots)
# 공식 prompt_layout_only_en(텍스트를 생성하지 않아 폭주하지 않는다). 지능형 전처리기의 폭주 폴백과 같은 프롬프트.
_DOTS_LAYOUT_PROMPT = (
    "<|img|><|imgpad|><|endofimg|>"
    "Please output the layout information from this PDF image, including each layout's bbox and its category. "
    "The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', "
    "'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."
)
# dots 11종 -> MinerU 블록 타입(이후 인식·문서 조립은 MinerU 레이아웃과 같은 경로). 차트/그림 구분은 없다(전부 image).
_DOTS_TYPES = {
    "Caption": "caption",
    "Footnote": "footnote",
    "Formula": "equation",
    "List-item": "list_item",
    "Page-footer": "footer",
    "Page-header": "header",
    "Picture": "image",
    "Section-header": "paragraph_title",
    "Table": "table",
    "Text": "text",
    "Title": "doc_title",
}
_DOTS_ITEM_RE = re.compile(r'\{[^{}]*?"bbox"\s*:\s*\[([^\]]+)\][^{}]*?"category"\s*:\s*"([^"]+)"[^{}]*\}')
_DOTS_FACTOR, _DOTS_MIN_PIXELS, _DOTS_MAX_PIXELS = 28, 3136, 11289600


def _dots_resized_size(width: int, height: int) -> tuple[int, int]:
    """dots(Qwen2-VL 계열) 이미지 전처리 smart_resize 결과 (w, h). 출력 bbox 가 이 크기 기준 픽셀이다."""
    f = _DOTS_FACTOR
    h, w = max(f, round(height / f) * f), max(f, round(width / f) * f)
    if h * w > _DOTS_MAX_PIXELS:
        beta = math.sqrt(height * width / _DOTS_MAX_PIXELS)
        h, w = max(f, math.floor(height / beta / f) * f), max(f, math.floor(width / beta / f) * f)
    elif h * w < _DOTS_MIN_PIXELS:
        beta = math.sqrt(_DOTS_MIN_PIXELS / (height * width))
        h, w = math.ceil(height * beta / f) * f, math.ceil(width * beta / f) * f
    return w, h


def parse_dots_layout_output(output: str, image_size: tuple[int, int]) -> list[dict]:
    """dots layout_only JSON([{bbox, category}]) -> MinerU 와 같은 블록(0~1 정규화 bbox). 깨진 JSON 은 항목 단위로 건진다."""
    rw, rh = _dots_resized_size(*image_size)
    blocks = []
    for coords, category in _DOTS_ITEM_RE.findall(output or ""):
        btype = _DOTS_TYPES.get(category.strip())
        try:
            x1, y1, x2, y2 = (float(v) for v in coords.split(",")[:4])
        except ValueError:
            continue
        if btype is None:
            continue
        x1, x2 = sorted((min(max(x1 / rw, 0.0), 1.0), min(max(x2 / rw, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1 / rh, 0.0), 1.0), min(max(y2 / rh, 0.0), 1.0)))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        blocks.append({"type": btype, "bbox": [x1, y1, x2, y2], "angle": 0, "content": None})
    drop = _covered(blocks, {"text", "equation"}, {"table"})
    return [b for i, b in enumerate(blocks) if i not in drop]


def _resize_by_need(image: Image.Image) -> Image.Image:
    if max(image.size) / min(image.size) > _MAX_IMAGE_EDGE_RATIO:
        w, h = image.size
        nw, nh = (w, math.ceil(w / _MAX_IMAGE_EDGE_RATIO)) if w > h else (math.ceil(h / _MAX_IMAGE_EDGE_RATIO), h)
        canvas = Image.new(image.mode, (nw, nh), (255, 255, 255))
        canvas.paste(image, ((nw - w) // 2, (nh - h) // 2))
        image = canvas
    if min(image.size) < _MIN_IMAGE_EDGE:
        s = _MIN_IMAGE_EDGE / min(image.size)
        image = image.resize((math.ceil(image.width * s), math.ceil(image.height * s)), Image.Resampling.BICUBIC)
    return image


def image_analysis_eligible(block: dict) -> bool:
    x1, y1, x2, y2 = block["bbox"]
    w, h = x2 - x1, y2 - y1
    return (w > _IMAGE_ANALYSIS_MIN_SIZE and h > _IMAGE_ANALYSIS_MIN_SIZE) or w * h > _IMAGE_ANALYSIS_MIN_AREA


def image_analysis_text(output: str, caption: bool = True) -> str:
    """`Image Analysis:` 출력(<|class|>분류 <|caption|>설명 <|content|>표·mermaid·텍스트)에서 설명과 내용만 꺼낸다.

    caption=False 면 내용(content)만. 설명은 영어로 나오고 지어낸 말이 섞일 수 있어 VLM 설명으로 대신할 때 쓴다.
    """
    parts = {k: v.strip() for k, v in _IMAGE_ANALYSIS_RE.findall(output or "")}
    return "\n".join(v for v in (parts.get("caption") if caption else None, parts.get("content")) if v)


def crop_block(page_image: Image.Image, block: dict) -> Optional[Image.Image]:
    w, h = page_image.size
    x1, y1, x2, y2 = block["bbox"]
    crop = page_image.crop((x1 * w, y1 * h, x2 * w, y2 * h))
    if crop.width < 1 or crop.height < 1:
        return None
    if block.get("angle") in (90, 180, 270):
        crop = crop.rotate(block["angle"], expand=True)
    return _resize_by_need(crop)


# ---------------------------------------------------------------------------- OTSL 표 -> TableData
_OTSL = ("<nl>", "<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
_OTSL_RE = re.compile("(" + "|".join(map(re.escape, _OTSL)) + ")")


def otsl_to_table_data(otsl: str) -> TableData:
    """MinerU/PaddleOCR-VL 의 OTSL 표 출력 -> docling TableData (mineru-vl-utils otsl2html 의 셀 산출 이식)."""
    tokens = _OTSL_RE.findall(otsl)
    texts = [p for p in _OTSL_RE.split(otsl) if p.strip()]
    rows = [list(g) for is_nl, g in itertools.groupby(tokens, lambda t: t == "<nl>") if not is_nl]
    if not rows:
        return TableData(table_cells=[], num_rows=0, num_cols=0)
    max_cols = max(len(r) for r in rows)
    for r in rows:
        r.extend(["<ecel>"] * (max_cols - len(r)))
    # 행을 max_cols 로 채운 만큼 texts 에도 빈 셀을 맞춰 넣는다
    new_texts, ti = [], 0
    for row in rows:
        for tok in row:
            new_texts.append(tok)
            if ti < len(texts) and texts[ti] == tok:
                ti += 1
                if ti < len(texts) and texts[ti] not in _OTSL:
                    new_texts.append(texts[ti])
                    ti += 1
        new_texts.append("<nl>")
        if ti < len(texts) and texts[ti] == "<nl>":
            ti += 1
    texts = new_texts

    def run(r, c, dr, dc, which):
        n = 0
        while r < len(rows) and c < len(rows[r]) and rows[r][c] in which:
            n += 1
            r, c = r + dr, c + dc
        return n

    cells, r_idx, c_idx = [], 0, 0
    for i, t in enumerate(texts):
        if t in ("<fcel>", "<ecel>"):
            text, step = "", 1
            if t == "<fcel>" and i + 1 < len(texts) and texts[i + 1] not in _OTSL:
                text, step = texts[i + 1], 2
            col_span, row_span = 1, 1
            if i + step < len(texts) and texts[i + step] in ("<lcel>", "<xcel>"):
                col_span += run(r_idx, c_idx + 1, 0, 1, ("<lcel>", "<xcel>"))
            if r_idx + 1 < len(rows) and rows[r_idx + 1][c_idx] in ("<ucel>", "<xcel>"):
                row_span += run(r_idx + 1, c_idx, 1, 0, ("<ucel>", "<xcel>"))
            cells.append(
                TableCell(
                    text=text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                    column_header=r_idx == 0,
                )
            )
        if t in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"):
            c_idx += 1
        if t == "<nl>":
            r_idx, c_idx = r_idx + 1, 0
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=max_cols)


def html_to_table_data(html: str) -> TableData:
    """<table> HTML -> TableData. rowspan/colspan 을 점유 격자로 펼친다."""
    table = BeautifulSoup(html or "", "html.parser").find("table")
    if table is None:
        return TableData(table_cells=[], num_rows=0, num_cols=0)
    occupied, cells, num_cols = set(), [], 0
    for r, tr in enumerate(table.find_all("tr")):
        c = 0
        for td in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rs, cs = max(1, int(td.get("rowspan", 1) or 1)), max(1, int(td.get("colspan", 1) or 1))
            occupied |= {(r + dr, c + dc) for dr in range(rs) for dc in range(cs)}
            cells.append(
                TableCell(
                    text=td.get_text(" ", strip=True),
                    row_span=rs,
                    col_span=cs,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + rs,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + cs,
                    column_header=td.name == "th" or td.find_parent("thead") is not None,
                )
            )
            c += cs
            num_cols = max(num_cols, c)
    return TableData(table_cells=cells, num_rows=max([r + 1 for r, _ in occupied], default=0), num_cols=num_cols)


def table_output_to_data(text: str) -> TableData:
    text = (text or "").strip()
    if "<table" in text.lower():
        return html_to_table_data(text)
    return otsl_to_table_data(text)


# ============================================================================ TableFormer 파드
_TF_SCALE = 2.0  # 파드 쪽 가정(144dpi)과 같아야 한다


def _inside(word, bbox_pt) -> bool:
    l, t, r, b = bbox_pt
    cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
    return l <= cx <= r and t <= cy <= b


def tableformer_page_input(pdf_path: str, page_no: int, bboxes_pt: list[tuple]) -> tuple[str, list[list[tuple]]]:
    """TableFormer 입력 페이지 이미지(144dpi PNG base64)와 표별 PDF 텍스트 레이어 단어 [(x0, y0, x1, y1, text)] (pt)."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        png = page.get_pixmap(matrix=pymupdf.Matrix(_TF_SCALE, _TF_SCALE)).tobytes("png")
        words = [tuple(w[:5]) for w in page.get_text("words") if w[4].strip()]
    return base64.b64encode(png).decode("ascii"), [[w for w in words if _inside(w, bb)] for bb in bboxes_pt]


async def tableformer_tables(
    client: httpx.AsyncClient,
    endpoint: str,
    timeout: float,
    png_b64: str,
    width: float,
    height: float,
    words: list[tuple],
    bboxes_pt: list[tuple],
) -> list[TableData]:
    """한 페이지 표들을 TableFormer 파드(`POST /table/structure`)로. 셀 텍스트는 words(pt) 를 서버가 셀에 매칭."""
    tokens = [
        {
            "id": idx,
            "text": text,
            "bbox": BoundingBox(
                l=x0 * _TF_SCALE, t=y0 * _TF_SCALE, r=x1 * _TF_SCALE, b=y1 * _TF_SCALE, coord_origin=CoordOrigin.TOPLEFT
            ).model_dump(),
        }
        for idx, (x0, y0, x1, y1, text) in enumerate(words)
    ]
    payload = {
        "width": width * _TF_SCALE,
        "height": height * _TF_SCALE,
        "image_b64": png_b64,
        "tokens": tokens,
        "table_bboxes": [[round(v) * _TF_SCALE for v in bb] for bb in bboxes_pt],
        "do_matching": True,
    }
    r = await client.post(endpoint, json=payload, timeout=timeout)
    r.raise_for_status()
    tables = []
    for out in r.json()["results"]:
        cells = []
        for element in out["tf_responses"]:
            cell = TableCell.model_validate(element)
            if cell.bbox is not None:
                cell.bbox = cell.bbox.scaled(1 / _TF_SCALE)
            cells.append(cell)
        d = out.get("predict_details", {})
        tables.append(TableData(table_cells=cells, num_rows=d.get("num_rows", 0), num_cols=d.get("num_cols", 0)))
    return tables


# ============================================================================ 파싱 오케스트레이션
@dataclass
class ParsedPage:
    page_no: int
    width: float  # PDF 포인트
    height: float
    image: Image.Image
    blocks: list[dict] = field(default_factory=list)


def render_pages(pdf_path: str, dpi: int) -> list[ParsedPage]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            pages.append(
                ParsedPage(
                    i + 1,
                    page.rect.width,
                    page.rect.height,
                    Image.frombytes("RGB", (pix.width, pix.height), pix.samples),
                )
            )
    return pages


class Parser:
    def __init__(self, cfg: dict):
        self.dpi = int(cfg.get("dpi", 200))
        self.mineru = VlmServer(cfg["mineru"])
        self.layout_backend = str(cfg.get("layout_backend", "mineru")).lower()
        self.dots = VlmServer(cfg["dots"]) if cfg.get("dots", {}).get("endpoint") else None
        if self.layout_backend not in ("mineru", "dots"):
            raise ValueError(f"layout_backend 는 mineru | dots 입니다: {self.layout_backend}")
        if self.layout_backend == "dots" and self.dots is None:
            raise ValueError("layout_backend 가 dots 인데 dots.endpoint 가 없습니다")
        self.paddle = VlmServer(cfg["paddle"]) if cfg.get("paddle", {}).get("endpoint") else None
        self.ppocr = PpOcrServer(cfg["ppocr"]) if cfg.get("ppocr", {}).get("endpoint") else None
        self.text_backend = str(cfg.get("text_backend", "mineru")).lower()
        self.table_backend = str(cfg.get("table_backend", "mineru")).lower()
        self.image_backend = str(cfg.get("image_backend", "none")).lower()
        self.chart_backend = str(cfg.get("chart_backend", self.image_backend)).lower()
        vlm = cfg.get("image_vlm", {})
        self.image_vlm = VlmServer(vlm) if vlm.get("url") or vlm.get("endpoint") else None
        self.image_prompt = vlm.get("prompt") or _IMAGE_DESCRIPTION_PROMPT
        self.chart_prompt = vlm.get("chart_prompt") or _CHART_DESCRIPTION_PROMPT
        self.image_context_chars = int(vlm.get("context_chars", 300))
        tf = cfg.get("tableformer", {})
        self.tf_endpoint, self.tf_timeout = tf.get("endpoint", ""), float(tf.get("timeout", 60))
        self.tf_max_concurrency = int(tf.get("max_concurrency", 4))
        self._tf_sems: dict[int, asyncio.Semaphore] = {}
        if "paddle" in (self.text_backend, self.table_backend) and self.paddle is None:
            raise ValueError("text_backend/table_backend 가 paddle 인데 paddle.endpoint 가 없습니다")
        if self.text_backend == "ppocr" and self.ppocr is None:
            raise ValueError("text_backend 가 ppocr 인데 ppocr.endpoint 가 없습니다")
        if self.image_backend not in ("none", "mineru", "vlm", "mineru_vlm"):
            raise ValueError(f"image_backend 는 none | mineru | vlm | mineru_vlm 입니다: {self.image_backend}")
        if self.chart_backend not in ("none", "mineru", "paddle", "vlm", "mineru_vlm"):
            raise ValueError(f"chart_backend 는 none | mineru | paddle | vlm | mineru_vlm 입니다: {self.chart_backend}")
        if {"vlm", "mineru_vlm"} & {self.image_backend, self.chart_backend} and self.image_vlm is None:
            raise ValueError("image_backend/chart_backend 가 vlm 인데 image_vlm.url 이 없습니다")
        if self.chart_backend == "paddle" and self.paddle is None:
            raise ValueError("chart_backend 가 paddle 인데 paddle.endpoint 가 없습니다")

    def _picture_backend(self, block: dict) -> str:
        """그림 블록의 백엔드. 작은 그림(아이콘·로고)은 none."""
        if not image_analysis_eligible(block):
            return "none"
        return self.chart_backend if block["type"] == "chart" else self.image_backend

    async def _recognize(self, client, page: ParsedPage, block: dict, backend: Optional[str] = None):
        crop = crop_block(page.image, block)
        if crop is None:
            return
        btype = block["type"]
        if btype in _PICTURE_TYPES:
            picture_backend = self._picture_backend(block)
            if picture_backend == "paddle":  # 차트 -> 데이터 표(markdown)
                block["content"] = await self.paddle.ask(client, crop, "Chart Recognition:", {"temperature": 0.0})
            else:  # mineru 이미지 분석(설명 + 표/mermaid/텍스트). mineru_vlm 은 내용만 받고 설명은 _describe 가 붙인다
                out = await self.mineru.ask(
                    client, crop, _MINERU_PROMPTS["image"], _MINERU_SAMPLING["image"], _MINERU_SYSTEM_PROMPT
                )
                block["content"] = image_analysis_text(out, caption=picture_backend != "mineru_vlm")
            return
        kind = "table" if btype == "table" else "equation" if btype == "equation" else "text"
        backend = backend or (self.table_backend if kind == "table" else self.text_backend)
        if backend == "ppocr" and kind == "text":
            block["content"] = " ".join(t for t, _ in await self.ppocr.ocr(client, crop))
        elif backend == "paddle":
            prompt = {"table": "Table Recognition:", "equation": "Formula Recognition:"}.get(kind, "OCR:")
            block["content"] = await self.paddle.ask(client, crop, prompt, {"temperature": 0.0})
        else:  # mineru (ppocr 의 수식, tableformer 실패 표도 여기로)
            key = kind if kind in _MINERU_PROMPTS else "[default]"
            block["content"] = await self.mineru.ask(
                client, crop, _MINERU_PROMPTS[key], _MINERU_SAMPLING[key], _MINERU_SYSTEM_PROMPT
            )

    async def parse(self, pdf_path: str) -> list[ParsedPage]:
        pages = await asyncio.to_thread(render_pages, pdf_path, self.dpi)
        use_tf = self.table_backend == "tableformer" and bool(self.tf_endpoint)
        # vLLM(uvicorn) keep-alive 기본 5초 == httpx keepalive_expiry 기본 5초라, 쉬던 연결을 재사용하는 순간 서버가
        # 막 닫아 "Server disconnected without sending a response" 가 났다. 클라이언트가 먼저(2초) 버리게 한다.
        async with httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=2.0)) as client:
            if self.layout_backend == "dots":  # 원본 해상도(200dpi) 그대로. 서버가 smart_resize 한다
                layouts = await asyncio.gather(
                    *[self.dots.ask(client, p.image, _DOTS_LAYOUT_PROMPT, {"temperature": 0.0}) for p in pages]
                )
                for page, out in zip(pages, layouts):
                    page.blocks = parse_dots_layout_output(out, page.image.size)
            else:
                layouts = await asyncio.gather(
                    *[
                        self.mineru.ask(
                            client,
                            p.image.resize(_LAYOUT_IMAGE_SIZE, Image.Resampling.BICUBIC),
                            _MINERU_PROMPTS["[layout]"],
                            _MINERU_SAMPLING["[layout]"],
                            _MINERU_SYSTEM_PROMPT,
                        )
                        for p in pages
                    ]
                )
                for page, out in zip(pages, layouts):
                    page.blocks = parse_layout_output(out)
            # tableformer 면 표는 페이지 단위로 TableFormer 에 보내고(텍스트 인식과 동시에), 실패한 표만 MinerU 로 간다.
            jobs = [
                self._recognize(client, p, b)
                for p in pages
                for b in p.blocks
                if b["type"] not in _NOT_EXTRACTED and not (use_tf and b["type"] == "table")
            ]
            jobs += [
                self._recognize(client, p, b)
                for p in pages
                for b in p.blocks
                if b["type"] in _PICTURE_TYPES and self._picture_backend(b) in ("mineru", "paddle", "mineru_vlm")
            ]
            if use_tf:
                jobs += [
                    self._tableformer_page(client, pdf_path, p) for p in pages if any(b["type"] == "table" for b in p.blocks)
                ]
            await asyncio.gather(*jobs)
            # vlm 은 앞뒤 문맥(인식된 텍스트)을 프롬프트에 넣으므로 텍스트 인식 뒤에 돈다
            await asyncio.gather(
                *[
                    self._describe(client, p, i)
                    for p in pages
                    for i, b in enumerate(p.blocks)
                    if b["type"] in _PICTURE_TYPES and self._picture_backend(b) in ("vlm", "mineru_vlm")
                ]
            )
        _log.info("[parse] %s pages=%d blocks=%d", pdf_path, len(pages), sum(len(p.blocks) for p in pages))
        return pages

    async def _describe(self, client, page: ParsedPage, idx: int):
        """그림 크롭 + 같은 페이지 앞뒤 텍스트·캡션으로 외부 VLM 에 한국어 설명을 받는다."""
        block = page.blocks[idx]
        crop = crop_block(page.image, block)
        if crop is None:
            return

        def _texts(blocks):
            return [
                (b.get("content") or "").strip()
                for b in blocks
                if b["type"] not in _PICTURE_TYPES | _CAPTION_TYPES.union(_FURNITURE) and (b.get("content") or "").strip()
            ]

        n = self.image_context_chars
        near = page.blocks[max(0, idx - 2) : idx + 3]
        heading = next(
            (b.get("content") for b in reversed(page.blocks[:idx]) if b["type"] in _TITLE_TYPES and b.get("content")), ""
        )
        values = {
            "before_context": "\n".join(_texts(page.blocks[:idx]))[-n:],
            "after_context": "\n".join(_texts(page.blocks[idx + 1 :]))[:n],
            "caption": "\n".join((b.get("content") or "").strip() for b in near if b["type"] in _CAPTION_TYPES),
            "section_header": heading.strip(),
        }
        # 차트 프롬프트는 데이터를 표로 뽑게 한다. mineru_vlm 은 표를 MinerU 가 이미 냈으므로 설명 프롬프트만 쓴다(표 중복 방지).
        chart_prompt = block["type"] == "chart" and self._picture_backend(block) == "vlm"
        template = self.chart_prompt if chart_prompt else self.image_prompt
        prompt = _PROMPT_VAR_RE.sub(lambda m: values.get(m.group(1)) or "(없음)", template)
        # mineru_vlm: 앞서 받아 둔 MinerU 내용(표/mermaid/글자) 앞에 VLM 설명을 붙인다
        analysis = (block.get("content") or "").strip() if self._picture_backend(block) == "mineru_vlm" else ""
        try:
            description = (await self.image_vlm.ask(client, crop, prompt, {"temperature": 0.0})).strip()
        except Exception as exc:  # noqa: BLE001  설명 실패는 MinerU 내용만 남기고 문서는 계속
            _log.warning("[image_vlm] page=%d 그림 설명 실패: %s: %s", page.page_no, type(exc).__name__, exc)
            description = ""
        block["content"] = "\n".join(v for v in (description, analysis) if v)

    async def _ppocr_words(self, client, page: ParsedPage, block: dict) -> list[tuple]:
        """표 영역 크롭을 PP-OCR 단어 박스로 읽어 페이지 pt 좌표 [(x0, y0, x1, y1, text)] 로 돌려준다."""
        w, h = page.image.size
        x1, y1, x2, y2 = block["bbox"]
        left, top = x1 * w, y1 * h
        crop = page.image.crop((left, top, x2 * w, y2 * h))
        sx, sy = page.width / w, page.height / h
        return [
            ((left + bx0) * sx, (top + by0) * sy, (left + bx1) * sx, (top + by1) * sy, text)
            for text, (bx0, by0, bx1, by1) in await self.ppocr.ocr(client, crop, word_box=True)
        ]

    async def _tableformer_page(self, client, pdf_path: str, page: ParsedPage):
        idx = [i for i, b in enumerate(page.blocks) if b["type"] == "table"]
        boxes = [tuple(v * s for v, s in zip(page.blocks[i]["bbox"], (page.width, page.height) * 2)) for i in idx]
        failed = list(idx)
        try:
            png_b64, words = await asyncio.to_thread(tableformer_page_input, pdf_path, page.page_no, boxes)
            # 텍스트 레이어가 없는 표(스캔본, 이미지 표)는 PP-OCR 단어 박스를 토큰으로 쓴다. 회전 표는 MinerU 로 넘긴다.
            if self.ppocr is not None:
                for k, i in enumerate(idx):
                    if not words[k] and page.blocks[i].get("angle", 0) == 0:
                        words[k] = await self._ppocr_words(client, page, page.blocks[i])
            sem = self._tf_sems.setdefault(id(asyncio.get_running_loop()), asyncio.Semaphore(self.tf_max_concurrency))
            async with sem:
                tables = await tableformer_tables(
                    client,
                    self.tf_endpoint,
                    self.tf_timeout,
                    png_b64,
                    page.width,
                    page.height,
                    [w for ws in words for w in ws],
                    boxes,
                )
            failed = []
            for k, i in enumerate(idx):
                data = tables[k] if k < len(tables) else None
                if data is not None and any(c.text.strip() for c in data.table_cells):  # 토큰이 없으면 셀이 빈다
                    page.blocks[i]["table_data"] = data
                else:
                    failed.append(i)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[tableformer] page=%d 실패 -> MinerU 표 사용: %s: %s", page.page_no, type(exc).__name__, exc)
        if failed:
            await asyncio.gather(*[self._recognize(client, page, page.blocks[i], backend="mineru") for i in failed])


# ============================================================================ 블록 -> DoclingDocument
_FURNITURE = {
    "header": DocItemLabel.PAGE_HEADER,
    "footer": DocItemLabel.PAGE_FOOTER,
    "page_number": DocItemLabel.PAGE_FOOTER,
}
_TEXT_LABELS = {
    "page_footnote": DocItemLabel.FOOTNOTE,
    "footnote": DocItemLabel.FOOTNOTE,
    "table_footnote": DocItemLabel.FOOTNOTE,
    "image_footnote": DocItemLabel.FOOTNOTE,
    "caption": DocItemLabel.CAPTION,
    "table_caption": DocItemLabel.CAPTION,
    "image_caption": DocItemLabel.CAPTION,
    "code_caption": DocItemLabel.CAPTION,
}
_TITLE_TYPES = {"title", "doc_title", "paragraph_title"}
_PICTURE_TYPES = {"image", "chart"}


def _prov(block: dict, page: ParsedPage, text: str) -> ProvenanceItem:
    x0, y0, x1, y1 = block["bbox"]
    bbox = BoundingBox(
        l=x0 * page.width, t=y0 * page.height, r=x1 * page.width, b=y1 * page.height, coord_origin=CoordOrigin.TOPLEFT
    )
    # 예전 docling PDF 파이프라인 출력과 같은 좌하단 원점(GenOS chunk_bboxes 소비 측 규약)
    return ProvenanceItem(page_no=page.page_no, bbox=bbox.to_bottom_left_origin(page.height), charspan=(0, len(text)))


def _save_picture(page: ParsedPage, block: dict, image_dir: Path, index: int) -> Optional[ImageRef]:
    w, h = page.image.size
    x0, y0, x1, y1 = block["bbox"]
    crop = page.image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    if crop.width < 2 or crop.height < 2:
        return None
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"image_{index:06d}_{hashlib.sha256(crop.tobytes()).hexdigest()}.png"
    crop.save(path)
    ref = ImageRef.from_pil(crop, dpi=round(w / page.width * 72))
    ref.uri = path
    return ref


def build_document(
    pages: list[ParsedPage], *, name: str, pdf_path: str, image_dir: Optional[Path] = None
) -> DoclingDocument:
    with open(pdf_path, "rb") as f:
        binary_hash = int(hashlib.sha256(f.read()).hexdigest()[:16], 16)
    doc = DoclingDocument(
        name=name,
        origin=DocumentOrigin(mimetype="application/pdf", filename=Path(pdf_path).name, binary_hash=binary_hash),
    )
    title_done, picture_index = False, 0
    for page in pages:
        doc.add_page(page_no=page.page_no, size=Size(width=page.width, height=page.height))
        for block in page.blocks:
            btype, text = block["type"], (block.get("content") or "").strip()
            if btype in _PICTURE_TYPES:
                ref = _save_picture(page, block, image_dir, picture_index) if image_dir else None
                picture_index += 1
                picture = doc.add_picture(image=ref, prov=_prov(block, page, ""))
                if text:  # 청커가 annotation 텍스트를 청크 본문에 싣는다(PICTURE_ANNOTATION_TEXT)
                    picture.annotations.append(DescriptionAnnotation(text=text, provenance="image-description"))
            elif btype == "table":
                data = block.get("table_data") or table_output_to_data(text)
                if data.num_rows:
                    doc.add_table(data=data, prov=_prov(block, page, ""))
            elif not text or btype in _NOT_EXTRACTED:
                continue
            elif btype in _TITLE_TYPES:
                if not title_done:
                    doc.add_title(text=text, prov=_prov(block, page, text))
                    title_done = True
                else:
                    doc.add_heading(text=text, level=1, prov=_prov(block, page, text))
            elif btype in _FURNITURE:
                doc.add_text(
                    label=_FURNITURE[btype],
                    text=text,
                    prov=_prov(block, page, text),
                    content_layer=ContentLayer.FURNITURE,
                )
            elif btype in ("code", "algorithm"):
                doc.add_code(text=text, prov=_prov(block, page, text))
            elif btype == "equation":
                doc.add_formula(text=text, prov=_prov(block, page, text))
            else:
                doc.add_text(label=_TEXT_LABELS.get(btype, DocItemLabel.TEXT), text=text, prov=_prov(block, page, text))
    return doc


# ============================================================================ 청킹 / GenOS 벡터
class GenosSmartChunker(sc.SmartChunkerBase):
    PICTURE_ANNOTATION_TEXT = True
    TABLE_DESCRIPTION_MODE = "full"
    CHUNK_HEADER_SEP = _CHUNK_HEADER_SEP
    CHUNK_PATH_SEP = _CHUNK_PATH_SEP
    CHUNK_PATH_MAX_LEAVES = _CHUNK_PATH_MAX_LEAVES


class GenOSVectorMeta(BaseModel):
    class Config:
        extra = "allow"

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    e_page: int = None
    i_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
    title: str = None
    file_path: Optional[str] = None
    has_table: bool = False
    table_refs: Optional[str] = None
    table_split_index: Optional[int] = None
    table_split_total: Optional[int] = None


class GenOSVectorMetaBuilder(vm.VectorMetaBuilderBase):
    def __init__(self):
        super().__init__()
        self.title: Optional[str] = None
        self.file_path: Optional[str] = None

    def build(self) -> GenOSVectorMeta:
        return GenOSVectorMeta.model_validate(
            {**self.core_payload(), "title": self.title, "file_path": self.file_path, **self.extra_metadata}
        )


class DocumentProcessor:
    def __init__(self, config_path: Optional[str] = None):
        with open(config_path or _DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._parser = Parser(cfg.get("models", {}))
        chunking = cfg.get("chunking", {})
        self._use_pdf_sdk = bool(cfg.get("conversion", {}).get("use_pdf_sdk", False))
        self._chunk_size = cp.parse_optional_int(chunking.get("chunk_size"), "chunk_size") or 0
        self._chunk_mode = str(chunking.get("chunk_mode", "split_only")).strip().lower()
        self._include_chunk_header = bool(chunking.get("include_chunk_header", True))
        self._table_as_chunk = bool(chunking.get("table_as_chunk", True))
        self._tokenizer_type = str(chunking.get("tokenizer_type", "char")).strip().lower()
        # 경로/HF id 문자열. char 모드에서는 청커가 로드하지 않고 들고만 있다(필드 자체는 필수).
        self._tokenizer = cp.resolve_tokenizer(chunking)
        self.page_chunk_counts: dict = defaultdict(int)

    # 1. 변환
    def _to_pdf(self, file_path: str) -> tuple[str, Optional[str]]:
        bad = fp.detect_unsupported_file(file_path)
        if bad:
            raise GenosServiceException(
                "1",
                f"{bad} 입니다. 정상 문서로 다시 업로드하세요: {os.path.basename(file_path)}",
                stage="convert",
                error_type="permanent",
            )
        if fp.is_pdf(file_path):
            return file_path, None
        pdf_path = pc.convert_to_pdf(file_path, use_pdf_sdk=self._use_pdf_sdk)
        if not pdf_path:
            raise GenosServiceException(
                "1", f"PDF 변환 실패: {os.path.basename(file_path)}", stage="convert", error_type="permanent"
            )
        return pdf_path, pdf_path

    # 2~3. 파싱 -> DoclingDocument
    async def load_document(self, pdf_path: str) -> DoclingDocument:
        pages = await self._parser.parse(pdf_path)
        stem = Path(pdf_path).stem
        return build_document(pages, name=stem, pdf_path=pdf_path, image_dir=Path(pdf_path).parent / stem)

    # 4. 청킹
    def split_documents(self, document: DoclingDocument, **kwargs) -> list[DocChunk]:
        chunk_size = cp.parse_optional_int(kwargs.get("chunk_size"), "chunk_size")
        chunk_size = cp.clamp_chunk_size(self._chunk_size if chunk_size is None else chunk_size, _MIN_CHUNK_SIZE)
        chunker = GenosSmartChunker(
            max_tokens=chunk_size or 0,
            merge_peers=True,
            tokenizer=self._tokenizer,
            tokenizer_type=self._tokenizer_type,
            chunk_mode=cp.resolve_chunk_mode(kwargs, self._chunk_mode),
            include_chunk_header=cp.resolve_include_chunk_header(kwargs, self._include_chunk_header),
            table_as_chunk=cp.resolve_table_as_chunk(kwargs, self._table_as_chunk),
        )
        chunks = list(chunker.chunk(dl_doc=document, **kwargs))
        self._table_split_totals = getattr(chunker, "_table_split_totals", {})
        self.page_chunk_counts = defaultdict(int)
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    # 5. GenOS
    async def compose_vectors(
        self,
        document: DoclingDocument,
        chunks: list[DocChunk],
        request: Request,
        converted_pdf_path: Optional[str] = None,
        **kwargs,
    ) -> list[GenOSVectorMeta]:
        include_header = cp.resolve_include_chunk_header(kwargs, self._include_chunk_header)
        title = next(
            (
                item.text.strip()
                for item, _ in document.iterate_items()
                if getattr(item, "label", None) == DocItemLabel.TITLE and item.text
            ),
            "",
        )
        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec="seconds") + "Z",
            title=title,
        )
        if converted_pdf_path:
            global_metadata["file_path"] = converted_pdf_path

        vectors, upload_tasks, uploaded, table_piece_seen = [], [], set(), {}
        current_page, chunk_index_on_page = None, 0
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items[0].prov else 0
            if chunk_page != current_page:
                current_page, chunk_index_on_page = chunk_page, 0
            content = (
                hp.build_header_line(
                    chunk.meta.headings, include_header, _CHUNK_HEADER_SEP, _CHUNK_PATH_SEP, _CHUNK_PATH_MAX_LEAVES
                )
                + chunk.text
            )
            vectors.append(
                GenOSVectorMetaBuilder()
                .set_text(content)
                .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                .set_chunk_index(chunk_idx)
                .set_global_metadata(**global_metadata)
                .set_chunk_bboxes(chunk.meta.doc_items, document)
                .set_media_files(chunk.meta.doc_items)
                .set_table_info(chunk.meta.doc_items, getattr(self, "_table_split_totals", {}), table_piece_seen)
                .build()
            )
            chunk_index_on_page += 1
            # 같은 그림이 여러 청크에 걸치면 한 번만 올린다(먼저 끝난 업로드가 원본을 지운다).
            files = [
                {"path": str(item.image.uri), "name": str(item.image.uri).rsplit("/", 1)[-1]}
                for item in chunk.meta.doc_items
                if isinstance(item, PictureItem) and item.image and str(item.image.uri) not in uploaded
            ]
            if upload_files and files:
                uploaded.update(f["path"] for f in files)
                upload_tasks.append(asyncio.create_task(upload_files(files, request=request)))
        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        return vectors

    async def _upload_converted_pdf(self, converted_pdf_path: Optional[str], request: Request, **kwargs):
        # upload_files 는 올린 뒤 원본을 지운다. 변환 PDF 의 NFS 원본은 GenOS UI 미리보기가 직접 보므로 사본만 올린다.
        if not (converted_pdf_path and upload_files):
            return
        name = os.path.splitext(kwargs.get("file_name") or os.path.basename(converted_pdf_path))[0] + ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copy(converted_pdf_path, tmp.name)
        await upload_files([{"path": tmp.name, "name": name}], request=request)

    async def __call__(self, request: Request, file_path: str, **kwargs):
        _log.info("file_path: %s kwargs: %s", file_path, kwargs)
        pdf_path, converted_pdf_path = self._to_pdf(file_path)
        try:
            document = await self.load_document(pdf_path)
        except Exception as exc:  # noqa: BLE001
            raise GenosServiceException("1", f"파싱 실패: {exc}", stage="parse", error_type="transient") from exc
        chunks = self.split_documents(document, **kwargs)
        vectors = await self.compose_vectors(document, chunks, request, converted_pdf_path, **kwargs)
        await self._upload_converted_pdf(converted_pdf_path, request, **kwargs)
        return vectors
