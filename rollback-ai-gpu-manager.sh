#!/usr/bin/env bash
set -euo pipefail
log(){ printf '[rollback-ai-gpu-manager] %s\n' "$*"; }
log 'stopping candidate unit if present'
sudo systemctl stop ai-gpu-manager-candidate.service ai-gpu-manager.service 2>/dev/null || true
log 'starting legacy retrieval services'
sudo systemctl start qwen3-embedding.service qwen3-reranker.service
ok=1
for u in qwen3-embedding.service qwen3-reranker.service; do
  if ! systemctl is-active --quiet "$u"; then ok=0; fi
done
for url in http://127.0.0.1:8081/health http://127.0.0.1:8082/health; do
  if ! curl -fsS --max-time 10 "$url" >/dev/null; then ok=0; fi
done
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
if [ "$ok" -eq 1 ]; then log PASS; else log FAIL; exit 1; fi
