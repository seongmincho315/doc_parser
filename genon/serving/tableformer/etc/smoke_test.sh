#!/usr/bin/env bash
# TableFormer 서빙 파드 smoke test entrypoint. genon/serving/paddle/etc/smoke_test.sh 와 동일 패턴.
#
# 사용법:
#   bash etc/smoke_test.sh              # 전체 (health → inference) 순차 수행
#   bash etc/smoke_test.sh health       # health 엔드포인트 응답만 확인
#   bash etc/smoke_test.sh inference    # /table/structure 1회 호출 + 응답 스키마 확인
#
# 환경변수:
#   TABLEFORMER_PORT    기본 8080
#   HEALTH_TIMEOUT_SEC  헬스 응답 대기 최대 초 (기본 120)
#   SMOKE_OUT_DIR       결과 JSON 저장 디렉토리 (기본 /tmp/tableformer_smoke)
set -euo pipefail

PORT="${TABLEFORMER_PORT:-8080}"
TIMEOUT="${HEALTH_TIMEOUT_SEC:-120}"
OUT_DIR="${SMOKE_OUT_DIR:-/tmp/tableformer_smoke}"
APP_DIR="${APP_DIR:-/app}"
PY="${APP_DIR}/.venv/bin/python"

mkdir -p "${OUT_DIR}"

cmd_health() {
  echo "[smoke] waiting for http://127.0.0.1:${PORT}/health (timeout ${TIMEOUT}s)"
  local elapsed=0
  while ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "${elapsed}" -ge "${TIMEOUT}" ]; then
      echo "[smoke] FAIL: health endpoint did not respond within ${TIMEOUT}s"
      exit 1
    fi
  done
  echo "[smoke] PASS: health endpoint OK"
}

cmd_inference() {
  echo "[smoke] calling /table/structure with a synthetic image"
  "${PY}" "${APP_DIR}/etc/smoke_test_inference.py" \
      --port "${PORT}" --out "${OUT_DIR}/result.json"
}

case "${1:-all}" in
  health)    cmd_health ;;
  inference) cmd_inference ;;
  all|"")    cmd_health; cmd_inference ;;
  *)
    echo "unknown subcommand: $1"
    echo "usage: smoke_test.sh [health|inference|all]"
    exit 2
    ;;
esac
