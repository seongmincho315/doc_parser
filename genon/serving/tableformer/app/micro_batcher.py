"""요청을 모아서 묶는 큐 (Triton dynamic batcher 와 같은 동작).

- 요청(job)이 들어오면 큐에 넣고, 워커 스레드 하나가 모델을 돌린다(GPU 는 한 번에 한 배치).
- 워커는 첫 요청을 받으면 max_delay_ms 동안 더 기다리며 요청을 모은다. 표 수가 max_batch 에 차면 기다리지 않고 바로 돌린다.
- max_delay_ms = 0 이면 기다리지 않는다. 대신 모델이 도는 동안 쌓인 요청을 다음 배치로 한꺼번에 가져가서(opportunistic),
  놀고 있을 때는 지연이 0 이고 몰릴 때만 자연스럽게 묶인다.
- 요청이 1개뿐이면 최대 max_delay_ms 만 기다리고 배치 1 로 처리한다. 짝이 올 때까지 무한정 기다리지 않는다.
- 배치 단위는 표 수다. 한 요청에 표가 여러 개면 그 표들이 같이 배치에 들어간다.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter
from concurrent.futures import Future
from typing import Any

_log = logging.getLogger("tableformer-batch")


class MicroBatcher:
    def __init__(self, predictor, max_batch: int = 4, max_delay_ms: float = 5.0, bbox_chunk: int = 64):
        self._predictor = predictor
        self.max_batch = max(1, int(max_batch))
        self.max_delay = max(0.0, float(max_delay_ms)) / 1000.0
        self.bbox_chunk = bbox_chunk
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stats = {"batches": 0, "jobs": 0, "tables": 0, "wait_s": 0.0, "run_s": 0.0}
        self._sizes: Counter = Counter()  # 배치 표 수 -> 횟수
        self._thread = threading.Thread(target=self._loop, name="tableformer-batcher", daemon=True)
        self._thread.start()

    def submit(self, job: dict[str, Any]) -> Future:
        fut: Future = Future()
        self._q.put((job, fut, time.monotonic()))
        return fut

    def stats(self) -> dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            sizes = dict(sorted(self._sizes.items()))
        b = s["batches"] or 1
        return {
            **s,
            "mean_tables_per_batch": round(s["tables"] / b, 2),
            "mean_wait_ms": round(1000 * s["wait_s"] / max(s["jobs"], 1), 1),
            "mean_run_ms": round(1000 * s["run_s"] / b, 1),
            "batch_size_histogram": sizes,
            "max_batch": self.max_batch,
            "max_delay_ms": self.max_delay * 1000,
        }

    def _collect(self) -> list[tuple]:
        first = self._q.get()  # 요청이 올 때까지 대기
        batch = [first]
        tables = len(first[0]["table_bboxes"])
        deadline = time.monotonic() + self.max_delay
        while tables < self.max_batch:
            remaining = deadline - time.monotonic()
            try:
                nxt = self._q.get(timeout=remaining) if remaining > 0 else self._q.get_nowait()
            except queue.Empty:
                break
            batch.append(nxt)
            tables += len(nxt[0]["table_bboxes"])
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._collect()
            started = time.monotonic()
            jobs = [b[0] for b in batch]
            n_tables = sum(len(j["table_bboxes"]) for j in jobs)
            try:
                results = self._predictor.predict_jobs(jobs, max_batch=self.max_batch, bbox_chunk=self.bbox_chunk)
            except Exception as exc:  # noqa: BLE001  모델 전체 실패(예: CUDA OOM) 는 배치의 모든 요청에 전달한다
                _log.exception("batch failed")
                results = [exc] * len(batch)
            run_s = time.monotonic() - started
            for (_, fut, _), res in zip(batch, results):
                if isinstance(res, Exception):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)
            wait_s = sum(started - b[2] for b in batch)
            with self._lock:
                self._stats["batches"] += 1
                self._stats["jobs"] += len(batch)
                self._stats["tables"] += n_tables
                self._stats["wait_s"] += wait_s
                self._stats["run_s"] += run_s
                self._sizes[n_tables] += 1
            _log.info(
                "[batch] jobs=%d tables=%d wait_ms(avg)=%.1f run_ms=%.1f",
                len(batch), n_tables, 1000 * wait_s / len(batch), 1000 * run_s,
            )
