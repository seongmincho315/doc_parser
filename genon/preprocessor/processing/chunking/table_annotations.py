"""표에 붙은 설명 annotation 을 읽는 헬퍼 (청커 전용).

원래 enrichment/table_description.py 에 있던 읽기 전용 부분만 옮겼다. enrichment(LLM 표 설명)를
걷어낸 뒤에도 청커는 TableItem.annotations 를 같은 규약으로 읽으므로, 설명이 붙어 있지 않으면
모든 메서드가 빈 값을 돌려주고 청킹 결과는 annotation 이 없던 때와 같다.
"""

from __future__ import annotations

from typing import Any

from docling_core.types.doc import DescriptionAnnotation, TableItem
from docling_core.types.doc.document import MiscAnnotation

from genon.preprocessor.processing.chunking.table_html import sanitize_table_html

# 표 description 이 부착하는 annotation 의 provenance(무관 annotation 배제용).
TABLE_DESCRIPTION_PROVENANCE = "facade_table_description"
TABLE_TEXT_DESCRIPTION_PROVENANCE = "facade_table_text_description"

# 표 RAG 설명을 청크 본문에 실을 때 쓰는 라벨.
TABLE_RETRIEVAL_LABEL = "[표 검색 설명]"


def refined_html_to_format(refined_html: str, table_format: str, compact_tables: bool = True) -> str:
    """재구성 HTML 표를 청크 텍스트로 낸다. 격자 태그만 남긴 HTML 을 돌려준다.

    예전 구현은 table_format=markdown 이면 docling HTML 백엔드로 재파싱했는데, 재구성 HTML 은
    LLM 표 설명이 만들던 것이라 지금 파이프라인에서는 생기지 않는다. 형식 변환 없이 HTML 로 둔다.
    """
    if not refined_html:
        return refined_html
    return sanitize_table_html(refined_html)


class TableDescriptionExtractor:
    """TableItem 에 부착된 annotation 에서 요약/재구성 HTML/검색 설명을 꺼낸다."""

    @staticmethod
    def _has_provenance(annotation: Any, provenance: str) -> bool:
        if getattr(annotation, "provenance", "") == provenance:
            return True
        if isinstance(annotation, MiscAnnotation):
            return (getattr(annotation, "content", None) or {}).get("provenance") == provenance
        return False

    @classmethod
    def extract_summary(cls, item: TableItem) -> str:
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, DescriptionAnnotation):
                continue
            if not (
                cls._has_provenance(annotation, TABLE_DESCRIPTION_PROVENANCE)
                or cls._has_provenance(annotation, TABLE_TEXT_DESCRIPTION_PROVENANCE)
            ):
                continue
            text = str(getattr(annotation, "text", "") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def extract_refined_html(item: TableItem) -> str:
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, MiscAnnotation):
                continue
            content = getattr(annotation, "content", None) or {}
            html = str(content.get("refined_html", "") or "").strip()
            if html:
                return html
        return ""

    @classmethod
    def extract_retrieval(cls, item: TableItem) -> dict:
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, MiscAnnotation):
                continue
            if not cls._has_provenance(annotation, TABLE_TEXT_DESCRIPTION_PROVENANCE):
                continue
            retrieval = (getattr(annotation, "content", None) or {}).get("table_retrieval")
            if isinstance(retrieval, dict):
                return retrieval
        return {}

    @classmethod
    def strip_text_descriptions(cls, annotations: list) -> list:
        return [
            annotation
            for annotation in (annotations or [])
            if not cls._has_provenance(annotation, TABLE_TEXT_DESCRIPTION_PROVENANCE)
        ]

    @classmethod
    def clean_copy(cls, item: TableItem) -> TableItem:
        """텍스트 표 설명을 뗀 복사본(직렬화 중 같은 문장이 두 번 실리는 것 방지)."""
        copied = item.model_copy(deep=True)
        copied.annotations = cls.strip_text_descriptions(getattr(copied, "annotations", None))
        return copied

    @classmethod
    def retrieval_text(cls, item: TableItem, *, split_piece: bool = False) -> str:
        retrieval = cls.extract_retrieval(item)
        if not retrieval:
            return ""
        context = str(retrieval.get("retrieval_context") or "").strip()
        if split_piece:
            return context if retrieval.get("repeat_context_on_split", True) else ""
        lines = [context] if context else []
        facts = [str(v).strip() for v in retrieval.get("key_facts", []) if str(v).strip()]
        terms = [str(v).strip() for v in retrieval.get("search_terms", []) if str(v).strip()]
        if facts:
            lines.append("핵심 사실: " + " | ".join(facts))
        if retrieval.get("include_search_terms") and terms:
            lines.append("검색어: " + ", ".join(terms))
        return "\n".join(lines)

    @classmethod
    def retrieval_prefix(cls, item: TableItem, *, split_piece: bool = False) -> str:
        text = cls.retrieval_text(item, split_piece=split_piece)
        return f"{TABLE_RETRIEVAL_LABEL}\n{text}\n" if text else ""
