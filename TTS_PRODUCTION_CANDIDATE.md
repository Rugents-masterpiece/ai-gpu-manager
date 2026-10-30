# TTS production candidate status

The candidate manager's active TTS worker is Chatterbox Slovak, defined in `chatterbox_worker.py` and wired by `adapters.TTSWorkerAdapter`. The former SpeechT5 worker implementation was retired and preserved in `snapshots/chatterbox-migration-before/production-candidate/tts_worker.py` for rollback/audit only; it is not imported or started by the candidate manager.

Candidate behavior:
- Chatterbox Slovak weights load strict in the isolated `/home/rugent/chatterbox-sk/.venv` runtime.
- CPU-resident idle state; clean Rudo WAV conditioning is prepared once and retained across component offloads.
- GPU binding uses RTX 3050 UUID `GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1`.
- `POST /v1/audio/speech` defaults to `normalize=true`; long text uses sentence/chunk generation with deterministic pauses.
- Quality warnings are surfaced in manager events and do not fail a transition. Automatic retry remains disabled until the tuning benchmark confirms its acceptance criteria.
- This is a candidate/shadow implementation only. GPU lifecycle, live API, human audio review, mixed-profile transitions, and the 100-transition stress run remain pending; no production cutover is authorized or performed.

Historical SpeechT5 benchmark results and samples remain in the surrounding prototype reports as audit records; they are not runtime assets or worker configuration.
