#!/usr/bin/env bash
# Build the image, load it into the cluster, create the Secret from ENV_FILE and apply the manifests.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER="${CLUSTER:-expense-tracker}"
NAMESPACE="${NAMESPACE:-expense-tracker}"
ENV_FILE="${ENV_FILE:-.env}"
WAIT_BOT="${WAIT_BOT:-1}"
ALLOW_COMPOSE_BOT="${ALLOW_COMPOSE_BOT:-0}"
CONTEXT="k3d-$CLUSTER"

if [[ ! -e "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ "$ALLOW_COMPOSE_BOT" != 1 ]]; then
  running="$(docker ps -q \
    --filter label=com.docker.compose.project=expense-tracker \
    --filter label=com.docker.compose.service=bot)"
  if [[ -n "$running" ]]; then
    echo "compose bot is running: docker compose stop bot" >&2
    exit 1
  fi
fi

docker build -t expense-tracker:local .
k3d image import expense-tracker:local -c "$CLUSTER"

kubectl --context "$CONTEXT" create namespace "$NAMESPACE" --dry-run=client -o yaml |
  kubectl --context "$CONTEXT" apply -f -

secret_args=(--from-env-file="$ENV_FILE")
if ! grep -q '^POSTGRES_PASSWORD=' "$ENV_FILE"; then
  secret_args+=(--from-literal=POSTGRES_PASSWORD=tracker)
fi
kubectl --context "$CONTEXT" -n "$NAMESPACE" create secret generic expense-tracker-env \
  "${secret_args[@]}" --dry-run=client -o yaml |
  kubectl --context "$CONTEXT" apply -f -

kubectl --context "$CONTEXT" apply -k k8s/overlays/k3d
kubectl --context "$CONTEXT" -n "$NAMESPACE" rollout restart deployment/bot
kubectl --context "$CONTEXT" -n "$NAMESPACE" rollout status statefulset/postgres --timeout=180s
if [[ "$WAIT_BOT" == 1 ]]; then
  kubectl --context "$CONTEXT" -n "$NAMESPACE" rollout status deployment/bot --timeout=300s
fi
