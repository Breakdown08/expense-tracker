#!/usr/bin/env bash
# Restore the database from a dump: `latest`, a name in the pod's /backups, or a file on the host.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER="${CLUSTER:-expense-tracker}"
NAMESPACE="${NAMESPACE:-expense-tracker}"
CONTEXT="k3d-$CLUSTER"

usage() {
  echo "usage: restore.sh latest | tracker-<timestamp>.dump | /path/to/file.dump" >&2
  exit 2
}

k() {
  kubectl --context "$CONTEXT" -n "$NAMESPACE" "$@"
}

[[ $# -eq 1 ]] || usage
dump="$1"
host_file=""
case "$dump" in
latest) ;;
*/*)
  if [[ ! -f "$dump" ]]; then
    echo "file not found: $dump" >&2
    exit 1
  fi
  host_file="$dump"
  ;;
tracker-*.dump) ;;
*) usage ;;
esac

if [[ -z "$host_file" ]]; then
  if [[ "$dump" == latest ]]; then
    dump="$(k exec postgres-0 -- sh -c 'ls -1 /backups/tracker-*.dump 2>/dev/null | sort | tail -n 1' || true)"
    if [[ -z "$dump" ]]; then
      echo "no dumps in /backups" >&2
      exit 1
    fi
  else
    dump="/backups/$dump"
  fi
fi

trap 'k scale deployment/bot --replicas=1' EXIT
k scale deployment/bot --replicas=0
k wait --for=delete pod -l app.kubernetes.io/name=bot --timeout=120s

restore=(pg_restore --clean --if-exists --single-transaction --no-owner -U tracker -d tracker)
if [[ -n "$host_file" ]]; then
  k exec -i postgres-0 -- "${restore[@]}" <"$host_file"
else
  k exec postgres-0 -- "${restore[@]}" "$dump"
fi

count="$(k exec postgres-0 -- psql -U tracker -d tracker -tAc 'select count(*) from expenses')"
echo "expenses: $count"
