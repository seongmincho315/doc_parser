#!/usr/bin/env bash
# Supervisor EventListener 프로토콜: READY/RESULT 로 통신
# 참고: https://supervisord.org/events.html
# genon/serving/paddle/etc/health_checker.sh 와 동일 패턴(재시작 대상 프로그램명만 다름).

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${TABLEFORMER_PORT:-8080}/health}"
TIMEOUT="${HEALTH_TIMEOUT:-3}"
RETRIES="${HEALTH_RETRIES:-3}"

while true; do
  echo "READY"
  read -r line || exit 1
  headers=""
  while read -r h && [ "$h" != "" ]; do
    headers+="$h"$'\n'
  done

  ok=0
  for i in $(seq 1 "$RETRIES"); do
    if curl -fsS --max-time "$TIMEOUT" "$HEALTH_URL" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 1
  done

  if [ $ok -eq 1 ]; then
    echo -ne "RESULT 2\nOK"
  else
    /usr/bin/supervisorctl restart tableformer >/dev/null 2>&1 || true
    echo -ne "RESULT 13\nRESTARTED"
  fi
done
