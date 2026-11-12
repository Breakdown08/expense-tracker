#!/usr/bin/env bash
# Run the nightly backup now: a Job made from the CronJob.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER="${CLUSTER:-expense-tracker}"
NAMESPACE="${NAMESPACE:-expense-tracker}"
CONTEXT="k3d-$CLUSTER"

job="pg-backup-manual-$(date -u +%Y%m%d%H%M%S)"
kubectl --context "$CONTEXT" -n "$NAMESPACE" create job --from=cronjob/pg-backup "$job"
if ! kubectl --context "$CONTEXT" -n "$NAMESPACE" wait --for=condition=complete --timeout=300s "job/$job"; then
  kubectl --context "$CONTEXT" -n "$NAMESPACE" logs "job/$job" || true
  echo "backup job $job did not complete" >&2
  exit 1
fi
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs "job/$job"
kubectl --context "$CONTEXT" -n "$NAMESPACE" exec postgres-0 -- ls -1 /backups
