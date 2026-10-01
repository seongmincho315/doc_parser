# doc parser

-  Fork 과정: docling -> https://github.com/genonai/doc_parser -> https://github.com/seongmincho315/doc_parser


## 레포 구조


## 이미지 빌드 방법


## 배포 방법

<details>
<summary>테이블포머 배포 방법</summary>

<details>
<summary>쿠버네티스 방법</summary>

- GPU: `genon/serving/tableformer/k8s-manifest/doc-parser-tableformer-deployment.yaml`
- CPU 전용: `genon/serving/tableformer/k8s-manifest/doc-parser-tableformer-deployment-cpu.yaml`
- 둘 다 `Service: doc-parser-tableformer-service`(namespace `llmops`, port 8080)를 만든다 —
  사이트마다 하나만 적용.
- 전처리기 쪽 설정: `pdf_pipeline.tableformer_remote.endpoint`에
  `http://doc-parser-tableformer-service:8080/table/structure` 기입
  (자세한 건 `genon/preprocessor/facade/gitbook_doc/installation.md` 참고).

</details>

<details>
<summary>도커 방법</summary>

쿠버네티스가 없는 서버는 `docker run`으로 직접 띄운다. k8s 매니페스트의 env를 그대로 옮긴 것.

GPU:

```bash
docker load -i doc-parser-tableformer-dev.tar.gz   # 이미지 먼저 로드

docker run -d \
  --name doc-parser-tableformer \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  -e TZ=Asia/Seoul \
  -e PROFILE=prod \
  -e TABLEFORMER_MODE=accurate \
  -e TABLEFORMER_DEVICE=cuda \
  -e TABLEFORMER_MAX_CONCURRENCY=1 \
  doc-parser-tableformer:dev
```

CPU 전용:

```bash
docker run -d \
  --name doc-parser-tableformer \
  --restart unless-stopped \
  -p 8080:8080 \
  -e TZ=Asia/Seoul \
  -e PROFILE=prod \
  -e TABLEFORMER_MODE=accurate \
  -e TABLEFORMER_DEVICE=cpu \
  -e TABLEFORMER_NUM_THREADS=8 \
  -e TABLEFORMER_MAX_CONCURRENCY=1 \
  doc-parser-tableformer:dev
```

헬스체크: `curl http://localhost:8080/health`

전처리기 쪽 설정: k8s Service DNS가 없으니 `pdf_pipeline.tableformer_remote.endpoint`에
`http://<서버IP>:8080/table/structure` 처럼 실제 접근 가능한 주소를 기입.

</details>

</details>

<details>
<summary>hi_res 배포 방법</summary>

<details>
<summary>쿠버네티스 방법</summary>

- `genon/serving/unstructured_hires/k8s-manifest/doc-parser-unstructured-hires-deployment.yaml`
  — CPU 전용(YOLOX는 onnxruntime CPU로 충분히 빠르고 Table Transformer도 이미지당 표 개수만큼만
  가볍게 돎). GPU가 필요하면 `resources.limits.nvidia.com/gpu`를 추가.
- `Service: doc-parser-unstructured-hires-service`(namespace `llmops`, port 8080).
- 전처리기 쪽 설정: yaml의 `unstructured_hires.endpoint`에
  `http://doc-parser-unstructured-hires-service:8080/partition` 기입
  (`genon/preprocessor/facade/gitbook_doc/installation.md` 참고). **이 파드는 옵션이 아니라 필수** —
  로컬 폴백이 없으므로 endpoint 미설정 시 이미지/미지 확장자 처리가 `ValueError`로 즉시 실패한다.

</details>

<details>
<summary>도커 방법</summary>

쿠버네티스가 없는 서버는 `docker run`으로 직접 띄운다. k8s 매니페스트의 env를 그대로 옮긴 것.

CPU 전용(기본, k8s 매니페스트와 동일 설정 — YOLOX/Table Transformer 둘 다 CPU로 충분):

```bash
docker load -i doc-parser-unstructured-hires-dev.tar.gz   # 이미지 먼저 로드

docker run -d \
  --name doc-parser-unstructured-hires \
  --restart unless-stopped \
  -p 8081:8080 \
  -e TZ=Asia/Seoul \
  -e PROFILE=prod \
  -e HIRES_MAX_CONCURRENCY=4 \
  doc-parser-unstructured-hires:dev
```

GPU가 필요하면(Table Transformer는 torch가 CUDA 빌드라 GPU 노출 시 자동으로 씀. YOLOX는
`onnxruntime`(CPU 전용 패키지)이라 GPU를 줘도 그쪽은 CPU로 돈다 — 부분 가속):

```bash
docker run -d \
  --name doc-parser-unstructured-hires \
  --restart unless-stopped \
  --gpus all \
  -p 8081:8080 \
  -e TZ=Asia/Seoul \
  -e PROFILE=prod \
  -e HIRES_MAX_CONCURRENCY=4 \
  doc-parser-unstructured-hires:dev
```

헬스체크: `curl http://localhost:8081/health`

전처리기 쪽 설정: k8s Service DNS가 없으니 `unstructured_hires.endpoint`에
`http://<서버IP>:8081/partition` 처럼 실제 접근 가능한 주소를 기입. **이 파드는 옵션이 아니라 필수**
— 로컬 폴백이 없으므로 endpoint 미설정 시 이미지/미지 확장자 처리가 `ValueError`로 즉시 실패한다.

</details>

</details>


## 사용 방법