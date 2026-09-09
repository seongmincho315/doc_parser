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
- `POST /table/structure`
  - 요청: `{width, height, image_b64(base64 PNG), tokens:[{id,text,bbox}], table_bboxes:[[l,t,r,b],...], do_matching}`
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
| `TABLEFORMER_MAX_CONCURRENCY` | `1` | 동시 추론 개수(세마포어) — GPU OOM 방지가 목적이라 함부로 올리지 말 것 |
| `TABLEFORMER_MODEL_ROOT` | `/models/docling-models` | 빌드 시 구운 아티팩트 경로 |

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
