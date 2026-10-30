"""KInIT Whisper + WhisperX + pyannote Community-1 STT worker.

Speaks the same line-JSON RPC protocol as the legacy Parakeet worker
(protocol.py: send()/recv()), so STTWorkerAdapter and manager_app.py work
unchanged. Models load per stage (transcript/align/diarize), and are freed
after each stage to fit the RTX 3050 6 GB budget.

Request fields (all optional):
  path         input audio file (required for op='transcribe')
  language     default 'sk'
  align        bool, default True   (WhisperX word alignment; runs Stage B)
  diarize      bool, default False  (Stage C: pyannote Community-1)
  telephone_mode bool, default False (conservative 16k mono resample)
  num_speakers / min_speakers / max_speakers  -> pyannote diarizer
"""
import os, sys, time, tempfile, shutil, subprocess, gc, json

# Pin the worker to the RTX 3050 by GPU UUID (same convention as the TTS worker);
# nvidia-smi enumerates in CUDA_DEVICE_ORDER=PCI_BUS_ID so PCI address lookup is stable.
UUID = os.environ.get("GPU_UUID", "GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Always resolve the numeric index for UUID and set CUDA_VISIBLE_DEVICES before torch
# import. Passing a UUID string is not supported by this driver stack (it falls back
# to all GPUs) and would risk running on the second GPU.
_idx = os.environ.get("CUDA_VISIBLE_DEVICES")
if _idx is None or not _idx.isdigit():
    idx = next((i for i, line in enumerate(subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines())
        if UUID.upper() in line.upper()), None)
    if idx is None:
        raise RuntimeError(f"GPU UUID {UUID} not found; refusing to run on the wrong GPU")
    _idx = str(idx)
else:
    idx = int(_idx)
# sanity: the numeric index must be the RTX 3050
out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout
row = [l for l in out.splitlines() if l.startswith(f"{_idx},")]
if not row or "RTX 3050" not in row[0]:
    raise RuntimeError(f"CUDA_VISIBLE_DEVICES={_idx} does not select RTX 3050: {row}")
os.environ["CUDA_VISIBLE_DEVICES"] = _idx
import torch
from protocol import send, recv

ASSET_ROOT = os.environ.get(
    "STT_MODEL_DIR", "/home/rugent/models/huggingface")
KINIT_CT2 = os.environ.get(
    "KINIT_CT2_PATH", f"{ASSET_ROOT}/kinit-whisper-large-v3-turbo-sk-ct2")
KINIT_TF = os.environ.get(
    "KINIT_TF_PATH", f"{ASSET_ROOT}/kinit-whisper-large-v3-turbo-sk")
PYANNOTE_PIPELINE = os.environ.get(
    "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1")
WORKDIR = os.environ.get("STT_WORKDIR", "/home/rugent/ai-stack/stt-whisperx/tmp")
os.makedirs(WORKDIR, exist_ok=True)
FIFO = os.path.join(WORKDIR, "gpu.fifo")
# GPU serialisation: the manager queue already serialises by profile, but we keep an
# in-process lock so a stray second call cannot overlap. fcntl is not available on
# this kernel/filesystem pair, so we rely on threading (one worker process per profile).
import threading
_gpu_mutex = threading.Lock()

def _gpu_mem_mib():
    try:
        uuid = os.environ.get("GPU_UUID", "")
        args = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        # Use index resolution from CUDA_VISIBLE_DEVICES if set, else -i uuid
        cvis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvis and cvis.isdigit():
            args = ["nvidia-smi", "-i", cvis, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        else:
            args = ["nvidia-smi", "-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1

def with_gpu_lock():
    return _gpu_mutex

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_audio(path, sr, mono=True, telephone=False):
    """Load via ffmpeg into float32 16k (or sr) mono torch tensor [n].
    Original file untouched. Conservative: resample only, no filters."""
    target_sr = sr
    probe = subprocess.run(["ffprobe", "-v", "error",
                            "-show_entries", "stream=sample_rate,channels",
                            "-of", "json", path],
                           capture_output=True, text=True, timeout=15)
    try:
        info = json.loads(probe.stdout)["streams"][0]
        in_sr, in_ch = int(info["sample_rate"]), int(info.get("channels", 1))
    except Exception:
        in_sr, in_ch = 0, 1
    need_resample = in_sr != target_sr or (mono and in_ch > 1)
    if not need_resample and path.lower().endswith(".wav"):
        # read directly, no transcode
        import soundfile as sf
        data, s = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).unsqueeze(0)
    conv = tempfile.mktemp(prefix="stt-conv-", suffix=".wav", dir=WORKDIR)
    args = ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn"]
    if mono:
        args += ["-ac", "1"]
    if need_resample or True:
        args += ["-ar", str(target_sr), "-c:a", "pcm_s16le"]
    args += [conv]
    r = subprocess.run(args, capture_output=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={r.returncode}: {r.stderr.decode()[:200]}")
    try:
        import soundfile as sf
        data, s = sf.read(conv, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).unsqueeze(0)
    finally:
        try:
            os.unlink(conv)
        except OSError:
            pass

def stage_a_transcribe(audio, language, compute_type, vad=True):
    from faster_whisper import WhisperModel
    t = time.perf_counter()
    model = WhisperModel(KINIT_CT2, device="cuda", compute_type=compute_type)
    # faster-whisper expects a float32 numpy array of 16k mono samples
    arr = audio.squeeze(0).cpu().numpy().astype("float32", copy=False)
    # VAD keeps WhisperX segment boundaries short, cuts silence, and avoids
    # 30s sliding-window chunks that collapse alignment scores.
    vad_params = {"threshold": 0.5, "min_speech_duration_ms": 250,
                  "min_silence_duration_ms": 200, "speech_pad_ms": 100}
    segments, info = model.transcribe(
        arr, language=language, beam_size=1, vad_filter=vad, vad_parameters=vad_params,
        condition_on_previous_text=False)
    segs = []
    for s in segments:
        segs.append({"start": float(s.start), "end": float(s.end),
                     "text": s.text.strip()})
    del model, segments
    free_gpu()
    return {"segments": segs, "language": info.language,
            "duration": float(info.duration), "stage_ms": (time.perf_counter() - t) * 1000}

def stage_b_align(segments, audio, language):
    """WhisperX alignment using the SK wav2vec2 model. Audio is 16k mono tensor [1,n]."""
    import whisperx
    t = time.perf_counter()
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device="cuda", model_name="comodoro/wav2vec2-xls-r-300m-sk-cv8")
    a = audio.squeeze(0)  # whisperx.align expects [n] @16kHz
    aligned = whisperx.align(segments, align_model, metadata, a,
                             device="cuda", return_char_alignments=False)
    del align_model
    free_gpu()
    return aligned["segments"], (time.perf_counter() - t) * 1000

def stage_c_diarize(path, num_speakers=None, min_speakers=None, max_speakers=None):
    """Community-1. Prefers exclusive_speaker_diarization (non-overlapping timeline,
    better for one-label-per-word/segment assignment)."""
    from pyannote.audio import Pipeline
    t = time.perf_counter()
    pipe = Pipeline.from_pretrained(PYANNOTE_PIPELINE)
    if pipe is None:
        raise RuntimeError(f"pyannote pipeline not available: {PYANNOTE_PIPELINE}")
    # without this, from_pretrained leaves weights on CPU and the diarization
    # stage dominates total wall time (~97% of RTF). The pipeline snapshot has
    # no "device" key, so pipeline.to() is never called internally.
    pipe.to(torch.device("cuda"))
    work = tempfile.mktemp(prefix="diar-", suffix=".wav", dir=WORKDIR)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1",
                    "-ar", "16000", work], capture_output=True, timeout=3600)
    kw = {}
    if num_speakers is not None:
        kw["num_speakers"] = int(num_speakers)
    if min_speakers is not None:
        kw["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kw["max_speakers"] = int(max_speakers)
    try:
        out = pipe(work, **kw)
    finally:
        try:
            os.unlink(work)
        except OSError:
            pass
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", None)
    if ann is None:
        raise RuntimeError("unexpected pyannote output type")
    turns = []
    for seg, _, spk in ann.itertracks(yield_label=True):
        turns.append({"start": float(seg.start), "end": float(seg.end),
                      "speaker": str(spk)})
    mode = "exclusive" if getattr(out, "exclusive_speaker_diarization", None) is not None else "regular"
    del pipe
    free_gpu()
    return turns, (time.perf_counter() - t) * 1000, mode

def assign_speakers(segments, turns):
    """Longest-overlap assignment of pyannote turns onto whisper segments."""
    import collections
    for seg in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best_ov, best = ov, t["speaker"]
        if best is None and turns:
            # segment in gap: pick nearest turn
            best = min(turns, key=lambda t: min(abs(t["start"] - seg["end"]),
                                                abs(t["end"] - seg["start"])))["speaker"]
        seg["speaker"] = best
    speakers = sorted({s["speaker"] for s in segments if s.get("speaker")})
    # Consistent ordering by first appearance
    order, relabel = [], {}
    for s in sorted(segments, key=lambda x: x["start"]):
        sp = s.get("speaker")
        if sp and sp not in order:
            order.append(sp)
    for i, sp in enumerate(order):
        relabel[sp] = f"SPEAKER_{i:02d}"
    for s in segments:
        if s.get("speaker"):
            s["speaker"] = relabel[s["speaker"]]
    return segments, order

def run_transcribe(r):
    path = r["path"]
    language = r.get("language", "sk")
    diarize = bool(r.get("diarize", False))
    telephone = bool(r.get("telephone_mode", False))
    do_align = bool(r.get("align", True))
    compute_type = r.get("compute_type", os.environ.get("STT_COMPUTE_TYPE", "float16"))
    num_speakers = r.get("num_speakers")
    min_speakers = r.get("min_speakers")
    max_speakers = r.get("max_speakers")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with with_gpu_lock():
        t_total = time.perf_counter()
        # Stage A
        audio16 = load_audio(path, 16000, mono=True, telephone=telephone)
        a = stage_a_transcribe(audio16, language, compute_type)
        timings = {"asr": round(a["stage_ms"])}
        # Stage B (alignment) only if requested
        segments = a["segments"]
        word_ts = False
        if do_align and segments:
            try:
                segments, align_ms = stage_b_align(segments, audio16, language)
                word_ts = any(s.get("words") for s in segments)
            except Exception as e:
                segments = a["segments"]  # fall back to unaligned
                align_ms = -1
            timings["align"] = round(align_ms)
        # Stage C (diarization) only if requested
        speaker_turns = []
        diar_mode = None
        if diarize:
            try:
                speaker_turns, diar_ms, diar_mode = stage_c_diarize(
                    path, num_speakers, min_speakers, max_speakers)
                segments, order = assign_speakers(segments, speaker_turns)
            except Exception:
                diar_ms = -1
                diar_mode = "failed"
            timings["diarize"] = round(diar_ms)
        del audio16
        free_gpu()
    elapsed = time.perf_counter() - t_total
    text = " ".join(s["text"] for s in segments).strip()
    out = {
        "ok": True,
        "text": text,
        "language": language,
        "model": "kinit/whisper-large-v3-turbo-sk",
        "diarization": diarize,
        "duration_seconds": round(a["duration"], 2),
        "segments": segments,
        "timings_ms": timings,
        "elapsed_ms": round(elapsed * 1000),
        "rtf": round(elapsed / max(a["duration"], 1e-6), 3),
        "compute_type": compute_type,
        "vram_peak_mib": _gpu_mem_mib(),
        "telephone_mode": telephone,
    }
    if diarize:
        out["speaker_count"] = len({s.get("speaker") for s in segments if s.get("speaker")})
        out["speaker_turns"] = speaker_turns
        out["diarization_mode"] = diar_mode
        if len(speaker_turns) == 0:
            out["diarization_warning"] = "no turns returned"
    return out

def main():
    t = time.perf_counter()
    # nothing loaded yet — the pipeline is lazy by design (staged GPU model)
    send({"ok": True, "event": "cpu_ready", "load_ms": round((time.perf_counter() - t) * 1000),
          "vram": 0})
    state = "cpu_ready"
    while True:
        r = recv()
        if r is None:
            break
        try:
            op = r.get("op")
            if op == "state":
                send({"ok": True, "state": state})
            elif op == "activate":
                state = "gpu_ready"
                send({"ok": True, "state": state, "elapsed_ms": 0})
            elif op == "transcribe":
                res = run_transcribe(r)
                res["state"] = "gpu_ready"
                send(res)
            elif op == "retire":
                send({"ok": True, "state": "retiring"})
                break
            elif op == "shutdown":
                send({"ok": True})
                break
            else:
                send({"ok": False, "error": "unknown_op"})
        except Exception as e:
            free_gpu()
            send({"ok": False, "error": repr(e), "state": state})
        finally:
            free_gpu()

if __name__ == "__main__":
    main()
