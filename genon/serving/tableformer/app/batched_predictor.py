"""TableFormer 배치 추론.

docling_ibm_models 3.10.3 의 TableModel04_rs.predict 는 표 한 장(배치 1)을 토큰 하나씩 디코딩한다. GPU 는 거의 놀고
파이썬 오버헤드(토큰마다 .item() 동기화, 보정 규칙)가 시간을 지배하므로, 여러 표를 스텝마다 같이 돌려 그 오버헤드를 나눈다.

  - predict_batch      : TableModel04_rs.predict 의 배치판. 표 B 장을 같이 디코딩하고 B 개의 (seq, class, coord) 를 돌려준다.
  - bbox_inference_vectorized : BBoxDecoder.inference 가 셀마다 돌리던 루프를 셀 묶음(chunk) 단위로 한 번에 계산한다.
  - TFPredictorBatched : multi_table_predict 를 여러 요청(job)에 대해 한 번에 돌린다.
      원본의 크기 조정·크롭·셀 매칭·후처리·인덱스 정렬은 그대로 쓰고, 모델 호출(self._model.predict)만 미리 배치로 계산한
      결과로 바꿔 끼운다. 그래서 원본 로직(라이브러리 버전이 바뀌어도)을 복사해 두지 않는다.

원본과 같은 입력이면 같은 출력이어야 한다(verify_batched.py 가 확인). 부동소수점 연산 순서가 배치에 따라 달라 아주 드물게
argmax 가 갈릴 수 있으니, 검증은 구조·텍스트가 같은지와 bbox 차이가 작은지를 본다.
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Optional

import torch
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

_log = logging.getLogger("tableformer-batch")

# bbox 디코더 어텐션은 (셀 수, 픽셀 수 784, attention_dim) 크기의 텐서를 만든다. 셀이 많은 표에서 메모리가 튀지 않게 끊어서 돌린다.
BBOX_CHUNK = 64


class _SeqState:
    """표 한 장의 디코딩 상태. TableModel04_rs.predict 의 지역 변수들과 같다."""

    __slots__ = (
        "seq", "out", "tag_H", "merge", "done", "skip_next_tag", "prev_tag_ucel", "line_num",
        "first_lcel", "cur_bbox_ind", "bbox_ind",
    )

    def __init__(self, start_tag: int):
        self.seq = [start_tag]  # <start> + 예측 태그 (<end> 포함)
        self.out = []  # 예측 태그만(output_tags)
        self.tag_H = []  # 셀 bbox 를 뽑을 위치의 디코더 은닉 상태 [(1, D)]
        self.merge = {}  # bboxes_to_merge
        self.done = False
        self.skip_next_tag = True
        self.prev_tag_ucel = False
        self.line_num = 0
        self.first_lcel = True
        self.cur_bbox_ind = -1
        self.bbox_ind = 0


def bbox_inference_vectorized(dec, encoder_out: torch.Tensor, tag_H: list[torch.Tensor], chunk: int = BBOX_CHUNK):
    """BBoxDecoder.inference 와 같은 결과를 셀 묶음 단위로 계산한다.

    원본은 셀마다 (init_h → 어텐션 → 게이트 → bbox/class) 를 한 번씩 돌린다. 초기 은닉 상태 h 는 모든 셀이 같고
    CellAttention 은 (셀 수, 픽셀 수) 형태를 이미 받으므로 셀을 쌓아서 한 번에 넣으면 된다.
    """
    if hasattr(dec, "_input_filter"):
        encoder_out = dec._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    encoder_dim = encoder_out.size(3)
    encoder_out = encoder_out.reshape(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

    if len(tag_H) == 0:
        return torch.empty(0), torch.empty(0)

    h0 = dec._init_hidden_state(encoder_out, 1)  # (1, decoder_dim), 모든 셀 공통
    tag_all = torch.cat(tag_H, dim=0)  # (num_cells, tag_decoder_dim)
    classes, boxes = [], []
    for start in range(0, tag_all.size(0), chunk):
        cell_tag_H = tag_all[start : start + chunk]
        h = h0.expand(cell_tag_H.size(0), -1)
        awe, _ = dec._attention(encoder_out, cell_tag_H, h)
        gate = dec._sigmoid(dec._f_beta(h))
        awe = gate * awe
        h = awe * h
        boxes.append(dec._bbox_embed(h).sigmoid())
        classes.append(dec._class_embed(h))
    return torch.cat(classes, dim=0), torch.cat(boxes, dim=0)


def _merge_span_boxes(model, outputs_class, outputs_coord, bboxes_to_merge):
    """가로로 걸친 셀(lcel)의 첫·끝 bbox 를 합친다. TableModel04_rs.predict 의 마지막 병합 부분과 같다."""
    device = model._device
    outputs_class1, outputs_coord1, boxes_to_skip = [], [], []
    for box_ind in range(len(outputs_coord)):
        box1 = outputs_coord[box_ind].to(device)
        cls1 = outputs_class[box_ind].to(device)
        if box_ind in bboxes_to_merge:
            box2 = outputs_coord[bboxes_to_merge[box_ind]].to(device)
            boxes_to_skip.append(bboxes_to_merge[box_ind])
            outputs_coord1.append(model.mergebboxes(box1, box2).to(device))
            outputs_class1.append(cls1)
        elif box_ind not in boxes_to_skip:
            outputs_coord1.append(box1)
            outputs_class1.append(cls1)
    coord = torch.stack(outputs_coord1) if len(outputs_coord1) > 0 else torch.empty(0)
    cls = torch.stack(outputs_class1) if len(outputs_class1) > 0 else torch.empty(0)
    return cls, coord


@torch.no_grad()
def predict_batch(model, imgs: torch.Tensor, bbox_chunk: int = BBOX_CHUNK) -> list[tuple[list, torch.Tensor, torch.Tensor]]:
    """TableModel04_rs.predict 의 배치판.

    imgs: (B, 3, 448, 448). 반환: 표마다 (seq, outputs_class, outputs_coord) — 원본 predict 의 반환과 같은 형식.
    모든 표가 같은 크기(784 위치)라 인코더와 디코더를 그대로 배치로 돌릴 수 있다. 디코딩은 스텝 동기식이고, 먼저 끝난
    표는 <end> 를 계속 먹이되 결과는 무시한다(표끼리 어텐션이 섞이지 않는다).
    """
    tt = model._tag_transformer
    tt.eval()
    device = model._device
    word_map = model._init_data["word_map"]["word_map_tag"]
    n_heads = tt._n_heads
    max_pred_len = model._max_pred_len

    enc_out = model._encoder(imgs)  # (B, H, W, C)
    encoder_out = tt._input_filter(enc_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    batch = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)
    enc_inputs = encoder_out.reshape(batch, -1, encoder_dim).to(device).permute(1, 0, 2)  # (P, B, D)
    positions = enc_inputs.shape[0]
    encoder_mask = torch.zeros((batch * n_heads, positions, positions), device=device) == torch.ones(
        (batch * n_heads, positions, positions), device=device
    )
    memory = tt._encoder(enc_inputs, mask=encoder_mask)

    start_tag, end_tag = word_map["<start>"], word_map["<end>"]
    fcel, ecel, ched, rhed, srow = word_map["fcel"], word_map["ecel"], word_map["ched"], word_map["rhed"], word_map["srow"]
    nl, ucel, xcel, lcel = word_map["nl"], word_map["ucel"], word_map["xcel"], word_map["lcel"]
    cell_tags = (fcel, ecel, ched, rhed, srow, nl, ucel)

    states = [_SeqState(start_tag) for _ in range(batch)]
    decoded_tags = torch.full((1, batch), start_tag, dtype=torch.long, device=device)  # (T, B)
    cache = None
    active = list(range(batch))  # 지금 배치에 들어 있는 표(states 인덱스). 끝난 표는 빼서 계산을 낭비하지 않는다

    while active:
        emb = tt._positional_encoding(tt._embedding(decoded_tags))
        decoded, cache = tt._decoder(emb, memory, cache, memory_key_padding_mask=encoder_mask)
        last = decoded[-1].clone()  # (B', D). 뷰가 아니라 복사본이라야 스텝마다 큰 텐서를 붙들지 않는다
        new_tags = tt._fc(last).argmax(1).tolist()  # 스텝당 GPU->CPU 동기화 한 번(원본은 표마다 한 번)

        next_col, keep = [], []
        for b, si in enumerate(active):
            s = states[si]
            tag = new_tags[b]

            # STRUCTURE ERROR CORRECTION (원본과 같은 규칙. line_num 은 원본에서도 갱신되지 않는다)
            if s.line_num == 0 and tag == xcel:
                tag = lcel
            if s.prev_tag_ucel and tag == lcel:
                tag = fcel

            if tag == end_tag:
                s.out.append(tag)
                s.seq.append(tag)
                s.done = True
                next_col.append(tag)
                continue
            s.out.append(tag)

            # BBOX 위치 기록: 셀 수와 bbox 수를 맞춘다
            if not s.skip_next_tag and tag in cell_tags:
                s.tag_H.append(last[b : b + 1])
                if s.first_lcel is not True:
                    s.merge[s.cur_bbox_ind] = s.bbox_ind
                s.bbox_ind += 1
            if tag != lcel:
                s.first_lcel = True
            elif s.first_lcel:
                s.tag_H.append(last[b : b + 1])
                s.first_lcel = False
                s.cur_bbox_ind = s.bbox_ind
                s.merge[s.cur_bbox_ind] = -1
                s.bbox_ind += 1

            s.skip_next_tag = tag in (nl, ucel, xcel)
            s.prev_tag_ucel = tag == ucel
            s.seq.append(tag)
            next_col.append(tag)
            if len(s.out) >= max_pred_len:  # 원본: while len(output_tags) < max_pred_len
                s.done = True

        keep = [b for b, si in enumerate(active) if not states[si].done]
        if not keep:
            break
        decoded_tags = torch.cat([decoded_tags, torch.tensor(next_col, dtype=torch.long, device=device).unsqueeze(0)], dim=0)
        if len(keep) < len(active):  # 끝난 표를 배치에서 뺀다
            idx = torch.tensor(keep, dtype=torch.long, device=device)
            decoded_tags = decoded_tags.index_select(1, idx)
            cache = cache.index_select(2, idx)  # (layers, T, B, D)
            memory = memory.index_select(1, idx)  # (P, B, D)
            encoder_mask = torch.zeros((len(keep) * n_heads, positions, positions), device=device) == torch.ones(
                (len(keep) * n_heads, positions, positions), device=device
            )
            active = [active[b] for b in keep]

    results = []
    for b, s in enumerate(states):
        outputs_class, outputs_coord = bbox_inference_vectorized(
            model._bbox_decoder, enc_out[b : b + 1], s.tag_H, chunk=bbox_chunk
        )
        outputs_class, outputs_coord = _merge_span_boxes(model, outputs_class, outputs_coord, s.merge)
        results.append((s.seq, outputs_class, outputs_coord))
    return results


class TFPredictorBatched(TFPredictor):
    """여러 요청(job)의 표를 한꺼번에 모델에 넣는 TFPredictor."""

    def _table_crops(self, iocr_page: dict, table_bboxes: list) -> list[tuple[torch.Tensor, tuple[int, float]]]:
        """multi_table_predict 가 모델에 넣는 것과 같은 표 입력 텐서(1, 3, 448, 448)와 길이 짐작값. 입력은 건드리지 않는다.

        길이 짐작값 = (표 안 토큰 수, 면적). 디코딩 길이(셀 수)와 비슷하게 움직여서, 비슷한 길이끼리 묶는 데 쓴다.
        """
        page_image_resized, scale_factor = self.resize_img(iocr_page["image"], height=1024)
        tokens = iocr_page.get("tokens") or []
        out = []
        for bbox in table_bboxes:
            l, t, r, b = (v * scale_factor for v in bbox[:4])
            table_image = page_image_resized[round(t) : round(b), round(l) : round(r)]
            n_tokens = 0
            for tok in tokens:
                tb = tok["bbox"]
                cx, cy = (tb["l"] + tb["r"]) / 2, (tb["t"] + tb["b"]) / 2
                if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                    n_tokens += 1
            out.append((self._prepare_image(table_image), (n_tokens, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))))
        return out

    def predict_jobs(
        self, jobs: list[dict[str, Any]], max_batch: int = 8, bbox_chunk: int = BBOX_CHUNK
    ) -> list[Any]:
        """jobs: [{"iocr_page", "table_bboxes", "do_matching"(옵션)}]. 반환: job 마다 multi_table_predict 결과 또는 예외 객체.

        표를 job 경계와 상관없이 max_batch 장씩 묶어 모델을 돌린 뒤, job 별로 원본 multi_table_predict 를 호출하되 모델
        호출만 미리 계산한 결과로 바꿔 끼운다.
        """
        results: list[Any] = [None] * len(jobs)
        pending: list[tuple[int, int, torch.Tensor, tuple]] = []  # (job 번호, 그 job 안의 표 번호, 표 입력, 길이 짐작값)
        n_tables = [0] * len(jobs)
        for ji, job in enumerate(jobs):
            try:
                crops = self._table_crops(job["iocr_page"], job["table_bboxes"])
            except Exception as exc:  # noqa: BLE001  이 job 만 실패시킨다
                results[ji] = exc
                continue
            n_tables[ji] = len(crops)
            pending += [(ji, ti, tensor, hint) for ti, (tensor, hint) in enumerate(crops)]

        # 긴 표끼리, 짧은 표끼리 묶는다(먼저 끝난 표를 기다리는 낭비를 줄인다). 결과는 (job, 표 번호) 로 되찾는다.
        pending.sort(key=lambda x: x[3], reverse=True)
        table_out: dict[tuple[int, int], Any] = {}
        with torch.no_grad():
            for start in range(0, len(pending), max_batch):
                chunk = pending[start : start + max_batch]
                outs = predict_batch(self._model, torch.cat([c[2] for c in chunk], dim=0), bbox_chunk)
                for (ji, ti, _, _), out in zip(chunk, outs):
                    table_out[(ji, ti)] = out
        per_job = {ji: collections.deque(table_out[(ji, ti)] for ti in range(n_tables[ji])) for ji in range(len(jobs))
                   if not isinstance(results[ji], Exception)}

        model = self._model
        for ji, job in enumerate(jobs):
            if isinstance(results[ji], Exception):
                continue
            queue = per_job[ji]
            # 인스턴스 속성으로 predict 를 가려서 원본 multi_table_predict 가 미리 계산한 결과를 받게 한다
            model.predict = lambda image_batch, max_steps, k, return_attention=False, _q=queue: _q.popleft()
            try:
                results[ji] = self.multi_table_predict(
                    job["iocr_page"],
                    [list(b) for b in job["table_bboxes"]],
                    do_matching=job.get("do_matching", True),
                )
            except Exception as exc:  # noqa: BLE001
                results[ji] = exc
            finally:
                del model.__dict__["predict"]
        return results
