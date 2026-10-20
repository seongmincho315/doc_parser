"""OpenDataLoader PDF 규칙 기반 전처리기 facade -- 모델 호출 없음.

test_processor 와 같은 후단(DoclingDocument -> GenosSmartChunker -> GenOSVectorMeta)을 쓰고,
레이아웃·인식 단계만 OpenDataLoader(Java) JSON 결과로 바꾼다.

  1. 변환   : PDF 가 아니면 PDF 로 (test_processor 와 동일)
  2. 파싱   : opendataloader_pdf.convert(format="json") -> kids (평탄 리스트, page number, bbox pt 좌하단 원점)
  3. 블록   : heading/paragraph/caption/list/table/image/header/footer/formula -> test_processor 블록 dict
  4. 청킹 이후는 test_processor.DocumentProcessor 그대로

설정: resource/opendataloader_processor_config.yaml
의존: `pip install opendataloader-pdf` + Java 11+. convert() 는 호출마다 JVM 을 띄우므로 문서당 1회만 부른다.
OCR 은 hybrid 서버에서만 지원되며 여기서는 쓰지 않는다(디지털 PDF 전용).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
import tempfile
from pathlib import Path
from typing import Optional

import yaml
from docling_core.types.doc import TableCell, TableData

from genon.preprocessor.processing.common import config_parse as cp

from genon.preprocessor.facade import test_processor as tp

_log = logging.getLogger(__name__)
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "resource" / "opendataloader_processor_config.yaml"

_TEXT_TYPES = {
    "paragraph": "text",
    "heading": "paragraph_title",
    "caption": "caption",
    "formula": "equation",
    "text block": "text",
    "line": "text",
}
_FURNITURE_TYPES = {"header": "header", "footer": "footer"}


def node_text(node: dict) -> str:
    """요소의 글자. content 가 없으면 자식(kids/list items)의 글자를 이어 붙인다."""
    if node.get("content"):
        return str(node["content"]).strip()
    parts = [node_text(k) for k in node.get("kids", []) + node.get("list items", [])]
    return " ".join(p for p in parts if p)


def table_data_from_node(node: dict) -> TableData:
    cells, num_rows, num_cols = [], 0, 0
    for row in node.get("rows", []):
        for cell in row.get("cells", []):
            r, c = int(cell["row number"]) - 1, int(cell["column number"]) - 1
            rs, cs = max(1, int(cell.get("row span", 1))), max(1, int(cell.get("column span", 1)))
            num_rows, num_cols = max(num_rows, r + rs), max(num_cols, c + cs)
            cells.append(
                TableCell(
                    text=node_text(cell),
                    row_span=rs,
                    col_span=cs,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + rs,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + cs,
                    column_header=bool(cell.get("is_header")),
                )
            )
    return TableData(table_cells=cells, num_rows=num_rows, num_cols=num_cols)


def _norm_bbox(box, width: float, height: float) -> Optional[list[float]]:
    """[left, bottom, right, top] pt, 좌하단 원점 -> [x0, y0, x1, y1] 0~1, 좌상단 원점."""
    if not box or len(box) != 4:
        return None
    left, bottom, right, top = box
    out = [
        max(0.0, left / width),
        max(0.0, (height - top) / height),
        min(1.0, right / width),
        min(1.0, (height - bottom) / height),
    ]
    return out if out[2] > out[0] and out[3] > out[1] else None


def convert_nodes(nodes: list[dict], width: float, height: float) -> list[dict]:
    out = []
    for n in nodes:
        kind, bbox = n.get("type"), _norm_bbox(n.get("bounding box"), width, height)
        if bbox is None:
            continue
        if kind == "table":
            data = table_data_from_node(n)
            if data.num_rows:
                out.append({"type": "table", "bbox": bbox, "table_data": data})
        elif kind == "list":
            # 목록은 항목마다 한 블록. 번호가 content 에 없으므로 항목 순서대로 둔다.
            for item in n.get("list items", []):
                ibox, text = _norm_bbox(item.get("bounding box"), width, height) or bbox, node_text(item)
                if text:
                    out.append({"type": "text", "bbox": ibox, "content": text})
        elif kind == "image":
            out.append({"type": "image", "bbox": bbox, "content": ""})
        elif kind in _FURNITURE_TYPES:
            text = node_text(n)
            if text:
                out.append({"type": _FURNITURE_TYPES[kind], "bbox": bbox, "content": text})
        elif kind in _TEXT_TYPES:
            text = node_text(n)
            if text:
                out.append({"type": _TEXT_TYPES[kind], "bbox": bbox, "content": text})
    return out


class Parser:
    def __init__(self, cfg: dict):
        self.dpi = int(cfg.get("dpi", 200))
        self.options = dict(cfg.get("opendataloader", {}))

    def _run(self, pdf_path: str) -> dict:
        import opendataloader_pdf  # 지연 임포트

        with tempfile.TemporaryDirectory() as out_dir:
            opendataloader_pdf.convert(input_path=[pdf_path], output_dir=out_dir, format="json", **self.options)
            return json.loads((Path(out_dir) / f"{Path(pdf_path).stem}.json").read_text(encoding="utf-8"))

    async def parse(self, pdf_path: str) -> list[tp.ParsedPage]:
        doc = await asyncio.to_thread(self._run, pdf_path)
        pages = await asyncio.to_thread(tp.render_pages, pdf_path, self.dpi)
        by_page: dict[int, list[dict]] = {}
        for node in doc.get("kids", []):
            by_page.setdefault(int(node.get("page number", 0)), []).append(node)
        for page in pages:
            page.blocks = convert_nodes(by_page.get(page.page_no, []), page.width, page.height)
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
