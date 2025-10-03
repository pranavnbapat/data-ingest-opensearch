#!/bin/sh
set -eu

: "${TARGET_URL:=http://data-ingest-opensearch:8000/run}"
: "${ENV_MODE:=PRD}"
: "${TZ:=UTC}"    # add a default so set -u doesn’t error if TZ isn’t provided
: "${SPLAY:=0}"   # optional random delay in seconds to avoid thundering herd

# optional small random start delay
sleep "$SPLAY" || true

echo "Scheduler started. Posting to: $TARGET_URL  (env_mode=$ENV_MODE)  TZ=$TZ"

while true; do
  # sleep until the next 23:00 in the container's TZ
  # (TZ is controlled by the TZ env var; see docker-compose.yml)
  h=$(date +%H)
  m=$(date +%M)
  s=$(date +%S)

  # seconds since local midnight; use 10# to avoid octal interpretation on leading zeros
  since_midnight=$(( 10#$h*3600 + 10#$m*60 + 10#$s ))

  target=0            # 00:00 → seconds since midnight
  day=86400

  if [ "$since_midnight" -lt "$target" ]; then
    sleep_seconds=$(( target - since_midnight ))
  else
    sleep_seconds=$(( day - since_midnight + target ))
  fi

  sleep "$sleep_seconds"

  # fire the job
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") posting job…"
  curl -sS -H "Content-Type: application/json" -X POST \
    -d "{\"background\": true, \"env_mode\": \"${ENV_MODE}\"}" \
    "${TARGET_URL}" || echo "POST failed (will try again tomorrow)"
done
