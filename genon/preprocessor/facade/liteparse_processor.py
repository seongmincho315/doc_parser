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
import hashlib
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pymupdf
import yaml
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
_NOT_EXTRACTED = {"list", "image_block", "equation_block", "image", "chart"}


# ============================================================================ test_processor 에서 옮긴 공통 후단 (파드에는 facade 패키지가 없어 단일 파일로 둔다)
@dataclass
class ParsedPage:
    page_no: int
    width: float  # PDF 포인트
    height: float
    image: Image.Image
    blocks: list[dict] = field(default_factory=list)
    # PDF 텍스트 레이어 블록 [(x0, y0, x1, y1, text)] (0~1 정규화). 그림 VLM 문맥용, vlm 백엔드일 때만 채운다
    text_blocks: list[tuple] = field(default_factory=list)


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
                data = block["table_data"]
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
        # created_date/appendix 는 베이스 빌더 버전마다 있고 없어서 쓰지 않는다.
        return GenOSVectorMeta.model_validate(
            {**self.core_payload(), "title": self.title, "file_path": self.file_path, **self.extra_metadata}
        )



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

    async def parse(self, pdf_path: str) -> list[ParsedPage]:
        result = await asyncio.to_thread(self._run, pdf_path)
        pages = await asyncio.to_thread(render_pages, pdf_path, self.dpi)
        by_no = {p.page_num: p for p in result.pages}
        for page in pages:
            lp = by_no.get(page.page_no)
            if lp is None:
                continue
            # 좌표는 LiteParse 페이지 크기(72dpi viewport) 기준이라 그 크기로 정규화한다.
            page.blocks = convert_blocks(lp.blocks, lp.width, lp.height)
        return pages




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