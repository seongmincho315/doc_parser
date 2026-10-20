"""LiteParse(LlamaIndex) 규칙 기반 전처리기 facade -- 모델 호출 없음.

test_processor 와 같은 후단(DoclingDocument -> GenosSmartChunker -> GenOSVectorMeta)을 쓰고,
레이아웃·인식 단계만 LiteParse(Rust/PDFium)의 `extract_blocks` 결과로 바꾼다.

  1. 변환   : PDF 가 아니면 PDF 로 (test_processor 와 동일)
  2. 파싱   : LiteParse.parse(extract_blocks=True) -> 페이지별 blocks (pt, 좌상단 원점, 읽기 순서)
  3. 블록   : heading/paragraph/list_item/code/table/figure -> test_processor 블록 dict (0~1 정규화)
  4. 청킹 이후는 test_processor.DocumentProcessor 그대로

설정: resource/liteparse_processor_config.yaml
의존: `pip install liteparse` (Python 3.10+). OCR 은 기본 꺼짐(디지털 PDF 전용, 스캔본은 빈 결과).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml
from docling_core.types.doc import TableCell, TableData

from genon.preprocessor.processing.common import config_parse as cp

from genon.preprocessor.facade import test_processor as tp

_log = logging.getLogger(__name__)
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "resource" / "liteparse_processor_config.yaml"

# LiteParse block.kind -> test_processor 블록 type. rule(구분선)·grid_fallback 은 버린다.
_KIND_TO_TYPE = {
    "heading": "paragraph_title",
    "paragraph": "text",
    "list_item": "text",
    "code": "code",
    "figure": "image",
}


def _cell_text(cell) -> str:
    return (getattr(cell, "text", None) if not isinstance(cell, dict) else cell.get("text")) or ""


def table_data_from_block(block) -> TableData:
    """LiteParse table block(header 한 줄 + rows) -> TableData. 병합 셀 정보가 없어 span 은 모두 1."""
    grid = []
    if block.header:
        grid.append((True, block.header))
    grid.extend((False, row) for row in (block.rows or []))
    cells, num_cols = [], 0
    for r, (is_header, row) in enumerate(grid):
        num_cols = max(num_cols, len(row))
        for c, cell in enumerate(row):
            cells.append(
                TableCell(
                    text=_cell_text(cell).strip(),
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    column_header=is_header,
                )
            )
    return TableData(table_cells=cells, num_rows=len(grid), num_cols=num_cols)


def _block_text(block) -> str:
    text = (block.text or "").strip()
    if block.kind == "list_item" and block.marker and not text.startswith(block.marker):
        text = f"{block.marker} {text}"
    return text


def convert_blocks(page_blocks, width: float, height: float) -> list[dict]:
    out = []
    for b in page_blocks or []:
        box = b.bbox
        if box is None:
            continue
        bbox = [
            max(0.0, box.x / width),
            max(0.0, box.y / height),
            min(1.0, (box.x + box.width) / width),
            min(1.0, (box.y + box.height) / height),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        if b.kind == "table":
            data = table_data_from_block(b)
            if data.num_rows:
                out.append({"type": "table", "bbox": bbox, "table_data": data})
        elif b.kind in _KIND_TO_TYPE:
            out.append({"type": _KIND_TO_TYPE[b.kind], "bbox": bbox, "content": _block_text(b)})
    return out


class Parser:
    def __init__(self, cfg: dict):
        self.dpi = int(cfg.get("dpi", 200))
        self.options = dict(cfg.get("liteparse", {}))
        self.options["extract_blocks"] = True
        self.options.setdefault("ocr_enabled", False)
        self.options.setdefault("quiet", True)

    def _run(self, pdf_path: str):
        from liteparse import LiteParse  # 지연 임포트: 이미지에 없을 때 import 오류를 파싱 시점에 낸다

        return LiteParse(**self.options).parse(pdf_path)

    async def parse(self, pdf_path: str) -> list[tp.ParsedPage]:
        result = await asyncio.to_thread(self._run, pdf_path)
        pages = await asyncio.to_thread(tp.render_pages, pdf_path, self.dpi)
        by_no = {p.page_num: p for p in result.pages}
        for page in pages:
            lp = by_no.get(page.page_no)
            if lp is None:
                continue
            # 좌표는 LiteParse 페이지 크기(72dpi viewport) 기준이라 그 크기로 정규화한다.
            page.blocks = convert_blocks(lp.blocks, lp.width, lp.height)
        return pages


class DocumentProcessor(tp.DocumentProcessor):
    def __init__(self, config_path: Optional[str] = None):
        # tp.DocumentProcessor.__init__ 은 모델 서버용 Parser 를 만들므로 쓰지 않고, 같은 필드를 직접 채운다.
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
        self._tokenizer = cp.resolve_tokenizer(chunking)
        self.page_chunk_counts: dict = defaultdict(int)
