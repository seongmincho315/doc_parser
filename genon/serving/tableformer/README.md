# doc-parser-tableformer

TableFormer(표 구조 인식)를 전처리기 파드와 분리된 별도 파드로 서빙한다. 예전에는
`docling/models/table_structure_model.py`가 전처리기 워커 프로세스 안에서 TFPredictor를
직접 로드했는데(인스턴스마다 중복 로드, CPU 서빙 시 느림, GPU 서빙 시 페이지의 모든 테이블을
한 번에 밀어넣어 OOM 위험 — CLAUDE.md TODO #1), 이제 전처리기는 이 파드를 HTTP로만 호출한다
(`docling/models/table_structure_remote_model.py`, `TableStructureRemoteModel`).

PaddleOCR 서빙(`genon/serving/paddle/`)과 같은 구조지만, PaddleX처럼 기성 CLI 서버가 없어서
`app/server.py`(FastAPI)를 직접 작성했다. `docling-ibm-models`는 PyPI에서 바로 설치되므로
paddle과 달리 별도 wheel 인덱스가 필요 없다.

## API

- `GET /health` → `{"status": "ok"|"loading", "mode": "accurate"|"fast"}`
- `GET /stats` → 워커별 배치 통계(`batches`, `mean_tables_per_batch`, `batch_size_histogram` 등). 워커가 여러 개면 호출할 때마다 다른 워커 값이 나온다
- `POST /table/structure`
  - 요청: `{width, height, image_b64(base64 PNG), tokens:[{id,text,bbox}], table_bboxes:[[l,t,r,b],...], do_matching}`
  - 이미지는 둘 중 하나로 보낸다.
    - `image_b64`: 페이지 전체(144dpi PNG). 예전 방식이고 그대로 받는다.
    - `crops`: `[{"bbox":[l,t,r,b], "image_b64":표 영역 PNG}, ...]` — 표 영역만 보낸다(`image_b64`는 비워도 된다).
      `bbox`는 `width`/`height`와 같은 좌표계(px)다. **표 둘레에 4pt 이상 여백을 붙여서 잘라야** 한다. 서버가 페이지를 높이
      1024 로 줄일 때 경계 픽셀이 주변 픽셀을 섞어 읽으므로, 여백이 없으면 셀 bbox 가 달라질 수 있다.
  - 응답: `{"results": [...]}` — `results[i]`는 `table_bboxes[i]`와 같은 순서이며
    `docling_ibm_models`의 `TFPredictor.multi_table_predict`가 원래 반환하던
    `{"tf_responses":[...], "predict_details":{...}}`와 완전히 동일한 스키마.
  - 페이지 하나에 테이블이 여러 개여도 **1회 호출**로 처리한다(`multi_table_predict`가 원래
    여러 bbox를 한 번에 받도록 설계돼 있음 — 예전 인프로세스 코드가 테이블마다 반복 호출했던 게
    오히려 이 API의 배치 능력을 안 쓴 비효율이었다).

## 로컬 빌드/실행

```shell
docker build -f genon/serving/tableformer/docker/Dockerfile -t doc-parser-tableformer:dev .
docker run --rm -p 8081:8080 \
  -e TABLEFORMER_MODE=accurate \
  -e TABLEFORMER_DEVICE=cpu \
  doc-parser-tableformer:dev

# 다른 터미널에서
TABLEFORMER_PORT=8081 bash genon/serving/tableformer/etc/smoke_test.sh
```

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `TABLEFORMER_MODE` | `accurate` | `accurate`\|`fast` — 빌드 시 이미지에 두 세트 모두 굽지만 파드는 하나만 로드 |
| `TABLEFORMER_DEVICE` | `auto` | `auto`\|`cpu`\|`cuda` |
| `TABLEFORMER_NUM_THREADS` | `8` | CPU 추론 시 스레드 수 |
| `TABLEFORMER_WORKERS` | `2` | uvicorn 워커 수. 워커마다 모델을 한 벌 올린다(GPU 워커당 약 1.5~2GB). `docker/Dockerfile`·supervisord 가 읽는다 |
| `TABLEFORMER_MAX_BATCH` | `8` | 워커당 한 번에 모델에 넣는 표 수. `1`이면 표를 하나씩 처리(예전 동작). GPU 메모리 상한 겸용 |
| `TABLEFORMER_BATCH_DELAY_MS` | `5` | 묶음을 기다리는 최대 시간(ms). 요청이 1개뿐이면 이만큼만 기다리고 배치 1 로 처리. `0`이면 기다리지 않는다 |
| `TABLEFORMER_BBOX_CHUNK` | `64` | bbox 디코더가 한 번에 계산하는 셀 수. 셀이 많은 표에서 메모리가 튀는 것을 막는다 |
| `TABLEFORMER_REQUEST_TIMEOUT_SEC` | `600` | 요청 하나가 배치 큐에서 결과를 기다리는 최대 시간 |
| `TABLEFORMER_MODEL_ROOT` | `/models/docling-models` | 빌드 시 구운 아티팩트 경로 |

## 배치와 워커

- **워커**(`TABLEFORMER_WORKERS`): uvicorn 프로세스마다 모델을 한 벌씩 올린다. 셀 매칭·후처리 같은 CPU 작업이 파이썬
  프로세스 하나(GIL)에 묶여 있어서, 워커를 늘리면 이 부분과 디코딩 오버헤드가 병렬로 돈다.
- **배치**(`TABLEFORMER_MAX_BATCH`, `TABLEFORMER_BATCH_DELAY_MS`): 워커 안의 `MicroBatcher`(`app/micro_batcher.py`)가
  요청을 모아서 표 여러 개를 한 번에 모델에 넣는다(`app/batched_predictor.py`). Triton 의 dynamic batcher 와 같은 동작이다.
  - 요청이 1개뿐이면 `BATCH_DELAY_MS` 만 기다리고 배치 1 로 처리한다. 짝을 무한정 기다리지 않는다.
  - 요청이 몰리면 `MAX_BATCH`(표 수) 까지 묶는다. 먼저 끝난 표는 배치에서 빼고, 비슷한 길이끼리 묶는다.
  - 결과는 원본 `multi_table_predict` 와 같다(표 24개에서 태그 시퀀스·셀 구조·텍스트 일치, bbox 차이는 최대 0.18px).
- **효과(측정, GPU 1장·표 24개)**: 원본 21.2초 → 배치 16 하나 약 15.8초 → **워커 2개 x 배치 8 약 12.1초**.
  CPU 에서는 배치·워커 이득이 없다(연산이 병목). 시간 대부분은 (1) 토큰이 많은 표의 셀 매칭(CPU, 파이썬)과
  (2) 끝나지 않는 디코딩(최대 길이 1024 까지 도는 비정상 표)이라 배치로 줄일 수 없다.
- **GPU 메모리**: 워커 1개 약 1.7GB(CUDA 컨텍스트 포함). 2 워커 x 배치 8 이 합쳐 약 3.3GB.
- 검증·측정 스크립트는 `tools/`: `verify_batched.py`(원본과 결과 비교·처리량), `test_micro_batcher.py`(묶음 큐),
  `profile_split.py`(모델 vs 매칭 시간 분해), `measure_vram.py`(배치별 GPU 메모리). 모델 아티팩트 경로를
  `--models-root`로 준다(이미지 안에서는 `/models/docling-models`).

## k8s 배포

- GPU: `k8s-manifest/doc-parser-tableformer-deployment.yaml`
- CPU 전용: `k8s-manifest/doc-parser-tableformer-deployment-cpu.yaml`
- 둘 다 `Service: doc-parser-tableformer-service`(namespace `llmops`, port 8080)를 만든다 —
  사이트마다 하나만 적용.
- 전처리기 쪽 설정: `pdf_pipeline.tableformer_remote.endpoint`에
  `http://doc-parser-tableformer-service:8080/table/structure` 기입
  (자세한 건 `genon/preprocessor/facade/gitbook_doc/installation.md` 참고).

## 이미지 레지스트리 등록

paddle과 같은 트랙(사내 레지스트리에 아직 없음 — 직접 빌드 후 사이트로 운반):

```shell
docker tag doc-parser-tableformer:dev mncregistry:30500/doc-parser-tableformer:0.0.0
docker push mncregistry:30500/doc-parser-tableformer:0.0.0
```
