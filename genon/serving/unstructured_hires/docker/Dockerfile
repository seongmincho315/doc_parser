# syntax=docker/dockerfile:1.7
#
# genon/serving/tableformer/docker/Dockerfile 과 동일한 4-stage 패턴.
# OS 패키지(tesseract-ocr*, poppler-utils, libgl1)는 genon/preprocessor/docker/Dockerfile.standard
# 에서 unstructured hi_res 용으로 이미 쓰던 것을 그대로 가져왔다 — 이 파드로 옮겨왔으니
# doc-parser 본체 이미지에서는 이제 빼도 된다(별도 확인 후 정리).

ARG PY_VER=3.11
ARG DEBIAN_DIST=bookworm
ARG APP_DIR=/app

########################################
# 1) Base (OS + uv + supervisor)
########################################
FROM python:${PY_VER}-slim-${DEBIAN_DIST} AS base

ARG APP_DIR
ENV APP_DIR=${APP_DIR}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tini \
        supervisor \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir -p ${APP_DIR}
WORKDIR ${APP_DIR}

########################################
# 2) deps (pyproject → venv)
########################################
FROM base AS deps

ARG APP_DIR
WORKDIR ${APP_DIR}

COPY genon/serving/unstructured_hires/pyproject.toml ${APP_DIR}/pyproject.toml
COPY genon/serving/unstructured_hires/uv.lock ${APP_DIR}/uv.lock

RUN uv venv ${APP_DIR}/.venv
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python ${VIRTUAL_ENV}/bin/python

########################################
# 3) models (YOLOX/Table Transformer 체크포인트 미리 굽기)
########################################
FROM deps AS models

ARG APP_DIR
COPY genon/serving/unstructured_hires/app/warm_models.py ${APP_DIR}/warm_models.py
RUN ${APP_DIR}/.venv/bin/python ${APP_DIR}/warm_models.py

########################################
# 4) runtime
########################################
FROM base AS runtime

ARG APP_DIR
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV HIRES_MAX_CONCURRENCY=4
# 체크포인트는 models 스테이지에서 이미 이미지에 구워졌다 — 런타임에 HF Hub로 캐시 최신여부를
# 확인하러 나가지 않게 강제 오프라인(폐쇄망 배포 전제, paddle/tableformer와 동일 기조).
ENV HF_HUB_OFFLINE=1

WORKDIR ${APP_DIR}

COPY genon/serving/unstructured_hires/app ${APP_DIR}/app
COPY genon/serving/unstructured_hires/config ${APP_DIR}/config
COPY genon/serving/unstructured_hires/etc ${APP_DIR}/etc
# models 스테이지에서 워밍업하며 채운 HF/onnx 캐시(기본 ~/.cache)와 NLTK 데이터를 그대로 가져온다.
COPY --from=models /root/.cache /root/.cache
COPY --from=models /root/nltk_data /root/nltk_data
COPY --from=deps ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN ln -sf ${APP_DIR}/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8080

CMD ["supervisord", "-n"]
