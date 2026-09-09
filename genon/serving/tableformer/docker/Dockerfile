# syntax=docker/dockerfile:1.7
#
# genon/serving/paddle/docker/Dockerfile 과 동일한 4-stage 패턴.
# paddle과 다른 점: 기성 CLI 서버가 없어 app/server.py(FastAPI)를 직접 만들었고,
# docling-ibm-models 는 PyPI에서 바로 설치되므로 별도 wheel 인덱스가 필요 없다.

ARG PY_VER=3.10
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

COPY genon/serving/tableformer/pyproject.toml ${APP_DIR}/pyproject.toml
COPY genon/serving/tableformer/uv.lock ${APP_DIR}/uv.lock

RUN uv venv ${APP_DIR}/.venv
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python ${VIRTUAL_ENV}/bin/python

########################################
# 3) models (TableFormer accurate/fast 아티팩트를 이미지에 굽기)
########################################
FROM deps AS models

ARG APP_DIR
ENV MODEL_ROOT=/models/docling-models
COPY genon/serving/tableformer/app/download_models.py ${APP_DIR}/download_models.py
RUN ${APP_DIR}/.venv/bin/python ${APP_DIR}/download_models.py

########################################
# 4) runtime
########################################
FROM base AS runtime

ARG APP_DIR
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV TABLEFORMER_MODEL_ROOT=/models/docling-models
# 기본값: accurate 모드 / auto 디바이스(GPU 있으면 사용) / 동시추론 1개(OOM 방지)
ENV TABLEFORMER_MODE=accurate
ENV TABLEFORMER_DEVICE=auto
ENV TABLEFORMER_NUM_THREADS=8
ENV TABLEFORMER_MAX_CONCURRENCY=1

WORKDIR ${APP_DIR}

COPY genon/serving/tableformer/app ${APP_DIR}/app
COPY genon/serving/tableformer/config ${APP_DIR}/config
COPY genon/serving/tableformer/etc ${APP_DIR}/etc
COPY --from=models /models /models
COPY --from=deps ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN ln -sf ${APP_DIR}/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8080

CMD ["supervisord", "-n"]
