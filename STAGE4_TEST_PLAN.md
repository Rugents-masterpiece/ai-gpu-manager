Stage 4 disposable acceptance plan

1. Start candidate unit only after stopping legacy Qwen services in a controlled maintenance window.
2. Verify /health, /ready, /status, /events and CPU_READY workers.
3. Run FP16 real mixed stress, including retrieval, TTS and STT.
4. Run 20 graceful restart cycles and read back worker/GPU/port state.
5. Inject manager SIGKILL and verify external supervisor restart plus reconciliation.
6. Inject retrieval/TTS/STT worker failure and verify lease release/recovery.
7. Compare production and candidate schemas/numerical outputs.
8. Obtain explicit speaker licence and human TTS approval.
9. Only then prepare a separate approved production endpoint migration.
