#!/usr/bin/env bash
# Wrapper around the official Comet Opik local install (needs Docker).
# Usage: scripts/opik.sh [up|down|status|logs|reset]   UI: http://localhost:5173
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/.opik"   # clone of comet-ml/opik, git-ignored

ensure_repo() {
  [ -d "$DIR" ] || git clone --depth 1 https://github.com/comet-ml/opik.git "$DIR"
}

case "${1:-status}" in
  up)     ensure_repo; (cd "$DIR" && ./opik.sh) ;;
  down)   (cd "$DIR" && ./opik.sh --stop) ;;
  status) docker ps --filter "name=opik" --format "table {{.Names}}\t{{.Status}}" ;;
  logs)   docker logs -f "$(docker ps --filter 'name=opik-backend' --format '{{.Names}}' | head -1)" ;;
  reset)  (cd "$DIR" && ./opik.sh --stop) || true; docker volume ls -q --filter "name=opik" | xargs -r docker volume rm ;;
  *)      echo "usage: $0 [up|down|status|logs|reset]"; exit 1 ;;
esac
