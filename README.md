RTX3050 PRODUCTION CANDIDATE SHADOW STACK

This directory is disposable and was not cut over to production.

Run (localhost only, disposable candidate):
CUDA_VISIBLE_DEVICES=GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1 GPU_UUID=GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1 CHATTERBOX_PYTHON=/home/rugent/chatterbox-sk/.venv/bin/python TTS_IDLE_TIMEOUT_SECONDS=60 python -m uvicorn manager_app:app --host 127.0.0.1 --port 18100

Endpoints:
- GET /health
- POST /v1/embeddings
- POST /v1/rerank
- POST /v1/audio/transcriptions
- POST /v1/audio/speech

Components:
- retrieval_worker.py: Qwen Embedding + Reranker, one persistent CPU process, direct PyTorch offload.
- chatterbox_worker.py: Slovak Chatterbox, clean Rudo WAV conditioning prepared once, CPU-resident process with direct component/conditioning offload.
- stt_worker.py: one-use NeMo GPU worker, process retirement, CPU standby replacement.
- adapters.py: framework-independent worker adapters and RPC.
- manager_app.py: FIFO single-owner manager and localhost shadow API.
- SpeechT5 worker retired; no active production endpoint or cutover.
- TTS idle timeout defaults to 60 s; GPU transfer RPC timeouts default to 60 s. TTS quality warnings are recorded and do not fail manager transitions.
- Candidate TTS retries stay disabled until a measured tuning run demonstrates the retry acceptance criteria.

No production systemd unit, port, client, ai-mode, LiteLLM, OpenViking, Qdrant, or existing service was modified.
