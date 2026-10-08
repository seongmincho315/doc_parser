"""적재용 최소 전처리기 facade (test/model_only) -- 단일 파일, 기존 doc-parser 이미지 그대로 사용.

GenOS 는 이 파일을 /app/src/preprocessor.py 로 마운트해 `DocumentProcessor()` 를 무인자로 만들고
`await processor(request, file_path, **params)` 로 부른다(src/main.py). 이미지를 새로 빌드하지 않도록
이미지에 이미 있는 패키지(httpx, pymupdf, bs4, docling-core, genon.preprocessor.processing 의 청커)만 쓴다.
mineru-vl-utils 는 이미지에 없으므로 2단계 호출을 아래에 직접 구현했다.

파이프라인 (모델은 전부 HTTP 호출):
  1. 변환   : PDF 가 아니면 PDF 로 (hwp/hwpx -> rhwp, 그 외 -> LibreOffice)
  2. 레이아웃: MinerU2.5 VLM (페이지 -> 블록 bbox/type)
  3. 인식   : 블록 크롭을 type 별 백엔드로 -- 텍스트: mineru|paddle, 표: mineru|paddle|tableformer,
              수식: mineru (paddle 이면 paddle). 그림/차트/묶음 블록(list 등)은 인식하지 않는다.
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
from docling_core.types.doc.document import ContentLayer, DocumentOrigin
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
    """OpenAI 호환(vLLM) 서버 하나. 동시 요청 수를 세마포어로 묶는다."""

    def __init__(self, cfg: dict):
        self.endpoint = cfg["endpoint"].rstrip("/") + "/v1/chat/completions"
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
            r = await client.post(self.endpoint, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


# ============================================================================ MinerU 2단계 (mineru-vl-utils 이식)
_MINERU_SYSTEM_PROMPT = "You are a helpful assistant."
_MINERU_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
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


def tableformer_tables(
    endpoint: str, timeout: float, pdf_path: str, page_no: int, bboxes_pt: list[tuple]
) -> list[TableData]:
    """한 페이지 표들을 TableFormer 파드(`POST /table/structure`)로. 셀 텍스트는 PDF 텍스트 레이어 단어를 서버가 매칭."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        png = page.get_pixmap(matrix=pymupdf.Matrix(_TF_SCALE, _TF_SCALE)).tobytes("png")
        width, height = page.rect.width, page.rect.height
        words = page.get_text("words")
    tokens = []
    for idx, (x0, y0, x1, y1, text, *_rest) in enumerate(words):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if text.strip() and any(l <= cx <= r and t <= cy <= b for l, t, r, b in bboxes_pt):
            bbox = BoundingBox(
                l=x0 * _TF_SCALE, t=y0 * _TF_SCALE, r=x1 * _TF_SCALE, b=y1 * _TF_SCALE, coord_origin=CoordOrigin.TOPLEFT
            )
            tokens.append({"id": idx, "text": text, "bbox": bbox.model_dump()})
    payload = {
        "width": width * _TF_SCALE,
        "height": height * _TF_SCALE,
        "image_b64": base64.b64encode(png).decode("ascii"),
        "tokens": tokens,
        "table_bboxes": [[round(v) * _TF_SCALE for v in bb] for bb in bboxes_pt],
        "do_matching": True,
    }
    r = httpx.post(endpoint, json=payload, timeout=timeout)
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
        self.paddle = VlmServer(cfg["paddle"]) if cfg.get("paddle", {}).get("endpoint") else None
        self.text_backend = str(cfg.get("text_backend", "mineru")).lower()
        self.table_backend = str(cfg.get("table_backend", "mineru")).lower()
        tf = cfg.get("tableformer", {})
        self.tf_endpoint, self.tf_timeout = tf.get("endpoint", ""), float(tf.get("timeout", 60))
        if "paddle" in (self.text_backend, self.table_backend) and self.paddle is None:
            raise ValueError("text_backend/table_backend 가 paddle 인데 paddle.endpoint 가 없습니다")

    async def _recognize(self, client, page: ParsedPage, block: dict):
        crop = crop_block(page.image, block)
        if crop is None:
            return
        btype = block["type"]
        kind = "table" if btype == "table" else "equation" if btype == "equation" else "text"
        backend = self.table_backend if kind == "table" else self.text_backend
        if backend == "paddle":
            prompt = {"table": "Table Recognition:", "equation": "Formula Recognition:"}.get(kind, "OCR:")
            block["content"] = await self.paddle.ask(client, crop, prompt, {"temperature": 0.0})
        else:  # mineru (tableformer 표도 실패 시 폴백용으로 MinerU 결과를 받아 둔다)
            key = kind if kind in _MINERU_PROMPTS else "[default]"
            block["content"] = await self.mineru.ask(
                client, crop, _MINERU_PROMPTS[key], _MINERU_SAMPLING[key], _MINERU_SYSTEM_PROMPT
            )

    async def parse(self, pdf_path: str) -> list[ParsedPage]:
        pages = await asyncio.to_thread(render_pages, pdf_path, self.dpi)
        # vLLM(uvicorn) keep-alive 기본 5초 == httpx keepalive_expiry 기본 5초라, 쉬던 연결을 재사용하는 순간 서버가
        # 막 닫아 "Server disconnected without sending a response" 가 났다. 클라이언트가 먼저(2초) 버리게 한다.
        async with httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=2.0)) as client:
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
            await asyncio.gather(
                *[self._recognize(client, p, b) for p in pages for b in p.blocks if b["type"] not in _NOT_EXTRACTED]
            )
        if self.table_backend == "tableformer" and self.tf_endpoint:
            await asyncio.to_thread(self._apply_tableformer, pdf_path, pages)
        _log.info("[parse] %s pages=%d blocks=%d", pdf_path, len(pages), sum(len(p.blocks) for p in pages))
        return pages

    def _apply_tableformer(self, pdf_path: str, pages: list[ParsedPage]):
        for page in pages:
            idx = [i for i, b in enumerate(page.blocks) if b["type"] == "table"]
            if not idx:
                continue
            boxes = [tuple(v * s for v, s in zip(page.blocks[i]["bbox"], (page.width, page.height) * 2)) for i in idx]
            try:
                tables = tableformer_tables(self.tf_endpoint, self.tf_timeout, pdf_path, page.page_no, boxes)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[tableformer] page=%d 실패 -> MinerU 표 사용: %s", page.page_no, exc)
                continue
            for i, data in zip(idx, tables):
                if any(c.text.strip() for c in data.table_cells):  # 텍스트 레이어 없는 페이지면 셀이 빈다
                    page.blocks[i]["table_data"] = data


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
                doc.add_picture(image=ref, prov=_prov(block, page, ""))
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
