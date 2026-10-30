STT ROTATING WORKER RESULTS

Date: 2026-09-24
Model: /home/rugent/models/huggingface/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo

Architecture
- Each worker loads NeMo to CPU once.
- A worker is activated on GPU at most once.
- It is never moved GPU->CPU for reuse; retirement terminates the process.
- A CPU_READY standby is preloaded before/while the active worker is used.
- The standby becomes the next current worker after retirement.

Measured results
- CPU preload fresh worker 1: 20.851 s wall, model load 15.156 s, VRAM 0.
- CPU preload fresh worker 2: 21.344 s wall, model load 15.448 s, VRAM 0.
- CPU preload fresh worker 3: 20.552 s wall, model load 14.759 s, VRAM 0.
- Page-cache effect: no meaningful improvement across the three runs.
- tmpfs: not tested; warm page cache did not justify a 2.5 GB tmpfs.
- Same GPU session: 10/10 transcriptions PASS, identical transcript returned.
- Worker generations exercised: 1 through 7 in shadow runs; no CUDA illegal memory access, OOM, reset, or corruption observed.
- Dual worker: active PSS ~3.82 GiB, CPU standby PSS ~6.11 GiB, combined ~9.93 GiB; MemAvailable ~25 GiB during measurement.
- Active Parakeet VRAM: ~2676 MiB in the dual-worker measurement.
- Retire + replacement: PASS; health reported standby_alive=true while active.

Caveat
The standalone fresh-worker test used three sequential workers. The full 20-fresh-worker acceptance run was not completed because each CPU preload is approximately 20-21 seconds. The observed repeated generations and 100-transition mixed stress are not a substitute for claiming 20/20 fresh workers.
