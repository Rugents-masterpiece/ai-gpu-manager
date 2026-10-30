Stage 4 cutover gate

The production cutover was intentionally NOT performed.

Reasons:
- candidate API is a combined shadow API and is not yet a transparent replacement for separate production ports 8081, 8082, and 8005;
- TTS speaker embedding is not licence-verified;
- human TTS audio review is pending;
- final FP16 mixed stress and restart/crash acceptance were still pending at the start of Stage 4;
- legacy rollback remains the safer live state.

Candidate unit:
- systemd/ai-gpu-manager-candidate.service
- localhost port 18080
- not installed, not enabled, not started by this task

Rollback script:
- rollback-ai-gpu-manager.sh
