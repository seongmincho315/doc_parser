# doc-parser-unstructured-hires

`unstructured`의 hi_res 전략(YOLOX 레이아웃 검출 + Microsoft Table Transformer 표 구조 인식)을
전처리기 파드와 분리된 별도 파드로 서빙한다. 예전에는 이미지(JPG/PNG)·미지 확장자 첨부파일을
`UnstructuredImageLoader`/`UnstructuredFileLoader`(langchain) 또는 `partition_image`를 통해
전처리기 프로세스 안에서 직접 처리했는데, 그러면 `unstructured-inference`(torch 하드 의존)가
doc-parser 본체에 항상 깔려 있어야 했다. 이제 doc-parser는 이 파드를 HTTP로만 호출한다
(`genon/preprocessor/facade/common/loaders.py`, `RemoteHiResLoader`).

DOC(.doc)/PPT(.ppt,.pptx)/MD는 이 파드로 옮기지 않았다 — `unstructured` 라이브러리 코드 레벨에서
strategy가 죽은 값이거나(docx) 기본이 `fast`(pptx)이거나 strategy 개념이 아예 없어서(md)
hi_res/torch를 안 타기 때문. **이미지와 미지 확장자 fallback만** 이 파드로 간다.

## API

- `GET /health` → `{"status": "ok"|"loading"}`
- `POST /partition` (multipart/form-data)
  - `file`: 업로드 파일
  - `languages`: 콤마 구분 언어 목록(기본 `"kor,eng"`)
  - `strategy`: 기본 `"hi_res"`
  - 응답: `{"elements": [{"text", "category", "element_id", "metadata"}, ...]}` —
    `unstructured.documents.elements.Element.to_dict()`와 동일한 필드 구성.

## 로컬 빌드/실행

```shell
docker build -f genon/serving/unstructured_hires/docker/Dockerfile -t doc-parser-unstructured-hires:dev .
docker run --rm -p 8082:8080 doc-parser-unstructured-hires:dev

# 다른 터미널에서
HIRES_PORT=8082 bash genon/serving/unstructured_hires/etc/smoke_test.sh
```

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `HIRES_MAX_CONCURRENCY` | `4` | 동시 추론 개수(세마포어) — TableFormer 파드와 같은 이유의 안전장치 |

## k8s 배포

- `k8s-manifest/doc-parser-unstructured-hires-deployment.yaml` — CPU 전용(YOLOX는 onnxruntime CPU로
  충분히 빠르고 Table Transformer도 이미지당 표 개수만큼만 가볍게 돎). GPU가 필요하면
  `resources.limits.nvidia.com/gpu`를 추가.
- `Service: doc-parser-unstructured-hires-service`(namespace `llmops`, port 8080).
- 전처리기 쪽 설정: yaml의 `unstructured_hires.endpoint`에
  `http://doc-parser-unstructured-hires-service:8080/partition` 기입
  (`genon/preprocessor/facade/gitbook_doc/installation.md` 참고). **이 파드는 옵션이 아니라 필수** —
  로컬 폴백이 없으므로 endpoint 미설정 시 이미지/미지 확장자 처리가 `ValueError`로 즉시 실패한다.

## 이미지 레지스트리 등록

paddle/tableformer와 같은 트랙(사내 레지스트리에 아직 없음 — 직접 빌드 후 사이트로 운반):

```shell
docker tag doc-parser-unstructured-hires:dev mncregistry:30500/doc-parser-unstructured-hires:0.0.0
docker push mncregistry:30500/doc-parser-unstructured-hires:0.0.0
```
