#!/usr/bin/env bash
# Start the k3d cluster, creating it on the first run. Storage lives on the host.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER="${CLUSTER:-expense-tracker}"
EXPENSE_TRACKER_DATA="${EXPENSE_TRACKER_DATA:-$HOME/.local/share/expense-tracker}"

if k3d cluster list --no-headers 2>/dev/null | awk '{print $1}' | grep -qx "$CLUSTER"; then
  k3d cluster start "$CLUSTER"
  exit 0
fi

mkdir -p "$EXPENSE_TRACKER_DATA/k3d-storage"
k3d cluster create "$CLUSTER" \
  --image rancher/k3s:v1.32.5-k3s1 \
  --servers 1 --agents 0 --no-lb \
  --k3s-arg "--disable=traefik@server:0" \
  --volume "$EXPENSE_TRACKER_DATA/k3d-storage:/var/lib/rancher/k3s/storage@server:0"
