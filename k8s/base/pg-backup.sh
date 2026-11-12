#!/usr/bin/env bash
# Nightly pg_dump into BACKUP_DIR, checked with pg_restore --list, then rotation.
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/backups}"
BACKUP_KEEP="${BACKUP_KEEP:-14}"

if [[ ! "$BACKUP_KEEP" =~ ^[0-9]+$ ]] || ((10#$BACKUP_KEEP < 1)); then
  echo "BACKUP_KEEP must be an integer >= 1, got: $BACKUP_KEEP" >&2
  exit 2
fi
BACKUP_KEEP=$((10#$BACKUP_KEEP))

name="tracker-$(date -u +%Y%m%dT%H%M%SZ).dump"
partial="$BACKUP_DIR/$name.partial"
trap 'rm -f "$partial"' EXIT

pg_dump --format=custom --file="$partial"
pg_restore --list "$partial" >/dev/null
mv "$partial" "$BACKUP_DIR/$name"
echo "created $name"

mapfile -t dumps < <(
  cd "$BACKUP_DIR"
  printf '%s\n' tracker-*.dump | grep -E '^tracker-[0-9]{8}T[0-9]{6}Z\.dump$' | sort || true
)
excess=$((${#dumps[@]} - BACKUP_KEEP))
for ((i = 0; i < excess; i++)); do
  rm -f "$BACKUP_DIR/${dumps[i]}"
  echo "deleted ${dumps[i]}"
done
