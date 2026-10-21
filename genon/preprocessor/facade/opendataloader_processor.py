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
import hashlib
import json
import logging
import os
import contextlib
import fcntl
import shutil
import subprocess
import tempfile
import time
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


class _JvmSlots:
    """동시에 뜨는 OpenDataLoader JVM 수를 제한한다.

    JVM 하나가 벡터 도형이 많은 페이지에서 힙을 수 GB 쓰는데, gunicorn 워커는 서로 다른 프로세스라
    스레드 세마포어로는 막을 수 없다. 슬롯 파일에 flock 을 걸어 프로세스 간에 제한한다.
    """

    def __init__(self, n: int, base: str = "/tmp/odl_jvm_slot"):
        self.n, self.base = max(1, n), base

    @contextlib.contextmanager
    def acquire(self):
        while True:
            for i in range(self.n):
                f = open(f"{self.base}.{i}", "a")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
                return
            time.sleep(0.2)


class Parser:
    def __init__(self, cfg: dict):
        self.dpi = int(cfg.get("dpi", 200))
        opts = dict(cfg.get("opendataloader", {}))
        # JVM 기본 힙은 컨테이너 메모리의 25%(16Gi 파드면 4GiB)라 벡터 도형이 많은 페이지에서 OOM 이 난다.
        # 실측(Document-136973, 벡터 도형 7만 개): 4g OOM, 8g 성공(약 7초, 최대 메모리 약 6GB), 12g 이득 없음.
        self.java_max_heap = str(opts.pop("java_max_heap", "8g"))
        self.timeout = float(opts.pop("timeout", 600))
        self._slots = _JvmSlots(int(opts.pop("max_concurrent_jvm", 2)))
        self.options = opts

    def _cli_extra_args(self) -> list[str]:
        args = []
        for key, value in self.options.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                args.append(flag)
            elif value not in (False, None):
                args += [flag, str(value)]
        return args

    def _convert(self, pdf_path: str) -> dict:
        import opendataloader_pdf  # 지연 임포트

        # opendataloader_pdf.convert() 는 java 힙 옵션을 받지 않아서 번들 JAR 를 직접 실행한다.
        jar = Path(opendataloader_pdf.__file__).parent / "jar" / "opendataloader-pdf-cli.jar"
        with tempfile.TemporaryDirectory() as out_dir, self._slots.acquire():
            subprocess.run(
                [
                    "java",
                    f"-Xmx{self.java_max_heap}",
                    "-Djava.awt.headless=true",
                    "-jar",
                    str(jar),
                    pdf_path,
                    "--format",
                    "json",
                    "-o",
                    out_dir,
                    *self._cli_extra_args(),
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
            return json.loads((Path(out_dir) / f"{Path(pdf_path).stem}.json").read_text(encoding="utf-8"))

    def _run(self, pdf_path: str) -> dict:
        try:
            return self._convert(pdf_path)
        except subprocess.CalledProcessError as exc:
            # 객체 구조가 깨진 PDF("Error while parsing object : N 0")는 OpenDataLoader(veraPDF)가 거부한다.
            # PyMuPDF 로 열어 다시 저장하면 구조가 복구돼 파싱된다(dp-500 실측: 원본 약 150개 실패).
            _log.warning(
                "opendataloader 파싱 실패, PyMuPDF 로 PDF 를 다시 저장해 재시도: %s (%s)",
                pdf_path,
                (exc.stderr or "").strip()[-300:],
            )
            with tempfile.TemporaryDirectory() as repair_dir:
                repaired = str(Path(repair_dir) / Path(pdf_path).name)
                with pymupdf.open(pdf_path) as src:
                    src.save(repaired, garbage=4, deflate=True, clean=True)
                return self._convert(repaired)

    async def parse(self, pdf_path: str) -> list[ParsedPage]:
        doc = await asyncio.to_thread(self._run, pdf_path)
        pages = await asyncio.to_thread(render_pages, pdf_path, self.dpi)
        by_page: dict[int, list[dict]] = {}
        for node in doc.get("kids", []):
            by_page.setdefault(int(node.get("page number", 0)), []).append(node)
        for page in pages:
            page.blocks = convert_nodes(by_page.get(page.page_no, []), page.width, page.height)
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