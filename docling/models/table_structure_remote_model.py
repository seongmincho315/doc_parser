"""TableFormer client that calls a separately-served pod instead of loading
TFPredictor in-process.

Replaces docling.models.table_structure_model.TableStructureModel inside this
repo's pipelines: the preprocessor pod no longer loads TableFormer itself
(resource waste per worker instance, CPU slowdown, GPU OOM risk when a page's
tables are all pushed through the model at once — see CLAUDE.md TODO #1).
Instead it POSTs to the dedicated tableformer pod
(genon/serving/tableformer/), batching every table on a page into a single
request/response round-trip.

Pre/post-processing (cluster collection, token building, scale, TableCell
assembly) is copied from TableStructureModel.__call__ so the pod's response
schema — {"results": [{"tf_responses":[...], "predict_details":{...}}, ...]}
— maps 1:1 onto what TFPredictor.multi_table_predict used to return locally.
"""

import copy
import logging
from collections.abc import Iterable

import numpy
import requests
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCellUnit

from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableStructureOptions
from docling.models.base_model import BasePageModel
from docling.utils.image_codec import numpy_to_png_bytes
from docling.utils.profiling import TimeRecorder

import base64

_log = logging.getLogger(__name__)


class TableStructureRemoteModel(BasePageModel):
    """Calls the TableFormer pod's ``POST /table/structure`` endpoint."""

    def __init__(
        self,
        enabled: bool,
        options: TableStructureOptions,
    ):
        self.options = options
        self.do_cell_matching = self.options.do_cell_matching
        self.enabled = enabled
        self.scale = 2.0  # Scale up table input images to 144 dpi (서버 쪽 가정과 일치해야 함)

        if self.enabled:
            remote = self.options.tableformer_remote_options
            if not remote.endpoint:
                raise ValueError(
                    "TableStructureOptions.tableformer_remote_options.endpoint must be "
                    "set when using TableStructureModelType.TABLEFORMER "
                    "(TableFormer는 별도 파드로만 서빙됩니다)."
                )
            self.endpoint = remote.endpoint
            self.timeout = remote.timeout
            self.headers = dict(remote.headers)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "table_structure"):
                    assert page.predictions.layout is not None
                    assert page.size is not None

                    page.predictions.tablestructure = TableStructurePrediction()

                    in_tables = [
                        (
                            cluster,
                            [
                                round(cluster.bbox.l) * self.scale,
                                round(cluster.bbox.t) * self.scale,
                                round(cluster.bbox.r) * self.scale,
                                round(cluster.bbox.b) * self.scale,
                            ],
                        )
                        for cluster in page.predictions.layout.clusters
                        if cluster.label
                        in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
                    ]
                    if not len(in_tables):
                        yield page
                        continue

                    # 페이지 내 모든 테이블의 토큰을 한 번만 모아 페이지당 1회 호출한다
                    # (기존 인프로세스 코드는 테이블마다 페이지 이미지를 통째로 재전송하며
                    # multi_table_predict 를 반복 호출했는데, 그 API 자체가 여러 bbox를
                    # 한 번에 받도록 설계돼 있어 원래도 불필요한 반복이었다).
                    tokens = []
                    seen_ids = set()
                    for table_cluster, _ in in_tables:
                        sp = page._backend.get_segmented_page()
                        if sp is not None:
                            tcells = sp.get_cells_in_bbox(
                                cell_unit=TextCellUnit.WORD,
                                bbox=table_cluster.bbox,
                            )
                            if len(tcells) == 0:
                                tcells = table_cluster.cells
                        else:
                            tcells = table_cluster.cells
                        for c in tcells:
                            if len(c.text.strip()) == 0:
                                continue
                            if c.index in seen_ids:
                                continue
                            seen_ids.add(c.index)
                            new_cell = copy.deepcopy(c)
                            new_cell.rect = BoundingRectangle.from_bounding_box(
                                new_cell.rect.to_bounding_box().scaled(scale=self.scale)
                            )
                            tokens.append(
                                {
                                    "id": new_cell.index,
                                    "text": new_cell.text,
                                    "bbox": new_cell.rect.to_bounding_box().model_dump(),
                                }
                            )

                    table_bboxes = [tbl_box for _, tbl_box in in_tables]
                    payload = {
                        "width": page.size.width * self.scale,
                        "height": page.size.height * self.scale,
                        "image_b64": base64.b64encode(
                            numpy_to_png_bytes(
                                numpy.asarray(page.get_image(scale=self.scale))
                            )
                        ).decode("ascii"),
                        "tokens": tokens,
                        "table_bboxes": table_bboxes,
                        "do_matching": self.do_cell_matching,
                    }

                    r = requests.post(
                        self.endpoint,
                        json=payload,
                        headers=self.headers or None,
                        timeout=self.timeout,
                    )
                    if not r.ok:
                        raise RuntimeError(
                            f"TableFormer 파드 HTTP {r.status_code}: {r.text[:500]}"
                        )
                    results = r.json()["results"]

                    for (table_cluster, _), table_out in zip(in_tables, results):
                        table_cells = []
                        for element in table_out["tf_responses"]:
                            if not self.do_cell_matching:
                                the_bbox = BoundingBox.model_validate(
                                    element["bbox"]
                                ).scaled(1 / self.scale)
                                text_piece = page._backend.get_text_in_rect(the_bbox)
                                element["bbox"]["token"] = text_piece

                            tc = TableCell.model_validate(element)
                            if tc.bbox is not None:
                                tc.bbox = tc.bbox.scaled(1 / self.scale)
                            table_cells.append(tc)

                        assert "predict_details" in table_out

                        num_rows = table_out["predict_details"].get("num_rows", 0)
                        num_cols = table_out["predict_details"].get("num_cols", 0)
                        otsl_seq = (
                            table_out["predict_details"]
                            .get("prediction", {})
                            .get("rs_seq", [])
                        )

                        tbl = Table(
                            otsl_seq=otsl_seq,
                            table_cells=table_cells,
                            num_rows=num_rows,
                            num_cols=num_cols,
                            id=table_cluster.id,
                            page_no=page.page_no,
                            cluster=table_cluster,
                            label=table_cluster.label,
                        )

                        page.predictions.tablestructure.table_map[
                            table_cluster.id
                        ] = tbl

                yield page
