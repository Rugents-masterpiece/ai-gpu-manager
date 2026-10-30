"""CPU-resident Chatterbox Slovak worker for the disposable GPU manager candidate."""
from __future__ import annotations

import hashlib
import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path

APP = Path(os.environ.get("CHATTERBOX_APP", "/home/rugent/chatterbox-sk/app"))
PROJECT = Path(os.environ.get("CHATTERBOX_PROJECT", "/home/rugent/chatterbox-sk"))
sys.path.insert(0, str(APP / "vendor"))
sys.path.insert(0, str(APP))

import numpy as np
import soundfile as sf
import torch
from load_slovak import load_model
from slovak_text_normalizer import normalize_text
from protocol import send, recv

REFERENCE = PROJECT / "reference/rudo_vasky_clean_5s.wav"
UUID = os.environ.get("GPU_UUID", "GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1")
BASELINE = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.6,
            "top_p": 0.92, "repetition_penalty": 1.35, "min_p": 0.05}
MAX_CHARS = 120


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    # A sentence is split at every ". " boundary; abbreviations (t. j., i. e., e. g., 01. 01.)
    # become isolated one- or two-token fragments. Tiny fragments make the alignment analyser
    # crash with an empty reduction dim, and are not speakable on their own, so glue them back.
    merged: list[str] = []
    for sentence in sentences:
        if merged and (len(sentence.split()) <= 2 or len(sentence) < 10):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    sentences = merged
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        current = ""
        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def move_model(model, device: str) -> float:
    start = time.perf_counter()
    model.t3.to(device)
    model.s3gen.to(device)
    model.ve.to(device)
    if model.conds is not None:
        model.conds.to(device)
    model.device = str(device)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(0)
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    return (time.perf_counter() - start) * 1000


def verify_target_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for TTS GPU activation")
    if "RTX 3050" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES did not select RTX 3050: {torch.cuda.get_device_name(0)}")
    result = subprocess.run(["nvidia-smi", "-i", UUID, "--query-gpu=uuid,name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5, check=True)
    if UUID not in result.stdout or "RTX 3050" not in result.stdout:
        raise RuntimeError(f"configured GPU UUID does not identify the expected RTX 3050: {result.stdout.strip()}")


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "-i", UUID, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


class WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def process_pss_rss_kb() -> dict:
    values = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith(("Pss:", "Rss:")):
                key, amount, *_ = line.split()
                values[key[:-1].lower() + "_kb"] = int(amount)
    except OSError as exc:
        values["error"] = str(exc)
    return values


class Worker:
    def __init__(self):
        references = {"rudo": REFERENCE, "melisko": PROJECT / "voices/melisko/reference_175.wav",
                      "fico": PROJECT / "voices/fico/reference.wav"}
        self.reference_hashes = {}
        for voice, reference in references.items():
            if not reference.is_file():
                raise FileNotFoundError(reference)
            self.reference_hashes[voice] = hashlib.sha256(reference.read_bytes()).hexdigest()
        if self.reference_hashes["melisko"] != "16db210800a88b5dfbc32b1fa54701eafd91d85c4c6a2078d0638505bc4e88dc":
            raise ValueError("Melisko original reference SHA256 mismatch")
        if self.reference_hashes["fico"] != "010a6b216d1f4ac7479bb9f51f34be7c8bb3af290069ff27661e9557d5518efe":
            raise ValueError("Fico reference SHA256 mismatch")
        self.model, self.load_info = load_model("cpu")
        self.conditionals = {}
        for voice, reference in references.items():
            self.model.prepare_conditionals(str(reference))  # once per original reference, CPU only
            self.conditionals[voice] = self.model.conds
            self.conditionals[voice].to("cpu")
        self.current_voice = "rudo"
        self.model.conds = self.conditionals["rudo"]
        self.state = "cpu_ready"
        self.activations = 0
        self.deactivations = 0
        self.offload_vram: list[int] = []
        self.leak_suspected = False
        self.last_metrics: dict = {}
        self.current_diagnostics: dict = {}

    def select_voice(self, voice: str) -> None:
        if voice not in self.conditionals:
            raise ValueError(f"unsupported voice: {voice!r}")
        if voice == self.current_voice:
            return
        # Release the previous GPU tensors BEFORE allocating the other voice's cache.
        self.model.conds.to("cpu")
        selected = self.conditionals[voice]
        try:
            selected.to(self.model.device)
        except Exception as exc:
            if is_fatal_cuda_error(exc):
                self.state = "error"
                raise  # The context is unsafe; never try CUDA rollback after a fatal error.
            selected.to("cpu")
            self.model.conds.to(self.model.device)
            raise
        self.model.conds = selected
        self.current_voice = voice

    def snapshot(self) -> dict:
        return {"state": self.state, "can_accept_request": self.state in {"cpu_ready", "gpu_ready"},
                "gpu_ready": self.state == "gpu_ready", "engine": "chatterbox-slovak",
                "normalizer_default": True, "conditionals_prepared": self.model.conds is not None,
                "conditionals_preparations": 3, "voice": self.current_voice,
                "voices": ["rudo", "melisko", "fico"], "reference_sha256": self.reference_hashes,
                "conditionals_preparations_by_voice": {"rudo": 1, "melisko": 1, "fico": 1}, "activations": self.activations,
                "deactivations": self.deactivations, "post_offload_vram_mib": self.offload_vram[-1] if self.offload_vram else None,
                "vram_leak_suspected": self.leak_suspected, "last_metrics": self.last_metrics,
                "cpu_resident_memory": process_pss_rss_kb()}

    def activate(self) -> dict:
        if self.state == "gpu_ready":
            return {"ok": True, **self.snapshot(), "elapsed_ms": 0}
        verify_target_gpu()
        self.state = "gpu_loading"
        try:
            elapsed = move_model(self.model, "cuda:0")
            self.activations += 1
            self.state = "gpu_ready"
            return {"ok": True, **self.snapshot(), "elapsed_ms": elapsed}
        except Exception as exc:
            if is_fatal_cuda_error(exc):
                self.state = "error"
                raise
            try:
                move_model(self.model, "cpu")
                self.state = "cpu_ready"
            except Exception:
                self.state = "error"
            raise

    def deactivate(self) -> dict:
        if self.state == "cpu_ready":
            return {"ok": True, **self.snapshot(), "elapsed_ms": 0}
        self.state = "offloading"
        try:
            elapsed = move_model(self.model, "cpu")
            self.deactivations += 1
            used = gpu_used_mib()
            if used is not None:
                self.offload_vram.append(used)
                if len(self.offload_vram) >= 3 and self.offload_vram[-1] - self.offload_vram[-3] > 128:
                    self.leak_suspected = True
            self.state = "cpu_ready"
            if self.leak_suspected:
                logging.error("TTS residual VRAM trend indicates possible leak: %s", self.offload_vram[-3:])
            return {"ok": True, **self.snapshot(), "elapsed_ms": elapsed}
        except Exception:
            self.state = "error"
            raise

    def generate(self, request: dict) -> dict:
        voice = request.get("voice", "rudo")
        if voice not in ("rudo", "melisko", "fico"):
            raise ValueError(f"unsupported voice: {voice!r}")
        if self.state != "gpu_ready":
            raise RuntimeError(f"TTS GPU worker not ready: {self.state}")
        raw = request.get("input", "")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("input must be a non-empty string")
        text = normalize_text(raw) if request.get("normalize", True) else raw.strip()
        seed = int(request["seed"]) if request.get("seed") is not None else (42 if voice == "rudo" else 0)
        if voice == "melisko":
            from melisko_postprocessing import split_text as melisko_chunks
            chunks = melisko_chunks(text)
        else:
            chunks = split_text(text)
        if not chunks:
            raise ValueError("input contains no speakable text")
        self.select_voice(voice)
        wav_parts = []
        records = []
        handler = WarningCapture()
        root = logging.getLogger()
        root.addHandler(handler)
        started = time.perf_counter()
        if voice in ("melisko", "fico"):
            import random
            rng_state = (random.getstate(), np.random.get_state())
        try:
            for index, chunk in enumerate(chunks):
                chunk_seed = seed if voice in ("melisko", "fico") else seed + index
                self.current_diagnostics.update({"stage": "model_generate", "voice": voice,
                    "seed": chunk_seed, "chunk": index + 1, "chunk_count": len(chunks),
                    "chunk_chars": len(chunk), "chunk_sha256": hashlib.sha256(chunk.encode()).hexdigest()})
                if voice in ("melisko", "fico"):
                    random.seed(chunk_seed)
                    np.random.seed(chunk_seed % (2**32))
                torch.manual_seed(chunk_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(chunk_seed)
                message_start = len(handler.messages)
                with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    wav = self.model.generate(text=chunk, language_id="sk", **BASELINE)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(0)
                emitted = handler.messages[message_start:]
                warning_text = emitted or [str(w.message) for w in caught]
                if torch.cuda.is_available():
                    torch.cuda.synchronize(0)
                audio = wav.detach().float().reshape(-1).cpu().numpy()
                if audio.size == 0 or not np.isfinite(audio).all() or not np.any(audio):
                    raise ValueError(f"technical audio validation failed for chunk {index + 1}")
                if voice == "rudo" and np.max(np.abs(audio)) > 1.00001:
                    raise ValueError(f"audio peak exceeds full scale for chunk {index + 1}")
                records.append({"chunk": index + 1, "seed": chunk_seed, "input_chars": len(chunk),
                                "duration_s": len(audio) / self.model.sr,
                                "forced_eos_count": sum(bool(re.search(r"forcing EOS token", message, re.I)) for message in warning_text),
                                "repetition_count": sum(bool(re.search(r"Detected \d+x repetition", message, re.I)) for message in warning_text)})
                wav_parts.append(audio)
                if index < len(chunks) - 1:
                    pause_ms = 100 if voice in ("melisko", "fico") else (210 if re.search(r"[.!?]$", chunk) else 125)
                    wav_parts.append(np.zeros(round(self.model.sr * pause_ms / 1000), dtype=np.float32))
        finally:
            root.removeHandler(handler)
            if voice in ("melisko", "fico"):
                random.setstate(rng_state[0])
                np.random.set_state(rng_state[1])
        audio = np.concatenate(wav_parts).astype(np.float32, copy=False)
        if audio.size == 0 or not np.isfinite(audio).all() or not np.any(audio):
            raise ValueError("technical validation failed for concatenated WAV")
        if voice == "melisko":
            from melisko_postprocessing import process_audio
            blob = process_audio(audio, self.model.sr)
            audio, processed_sr = sf.read(io.BytesIO(blob), dtype="float32")
            if processed_sr != self.model.sr:
                raise ValueError("postprocessing changed sample rate")
        if voice == "fico":
            from fico_postprocessing import process_audio as fico_process_audio
            blob = fico_process_audio(audio, self.model.sr)
            audio, processed_sr = sf.read(io.BytesIO(blob), dtype="float32")
            if processed_sr != self.model.sr:
                raise ValueError("postprocessing changed sample rate")
        duration = len(audio) / self.model.sr
        latency = time.perf_counter() - started
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        peak = float(np.max(np.abs(audio)))
        if voice == "rudo":
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.model.sr, format="WAV", subtype="PCM_16")
            blob = buffer.getvalue()
        info = sf.info(io.BytesIO(blob))
        if info.samplerate != self.model.sr or info.channels != 1 or info.frames <= 0:
            raise ValueError("encoded WAV validation failed")
        request_id = str(request.get("request_id") or uuid.uuid4())
        eos = sum(x["forced_eos_count"] for x in records)
        rep = sum(x["repetition_count"] for x in records)
        cfg = {**BASELINE, "language_id": "sk"}
        short = duration < 1.5 and len(text.split()) >= 4
        self.last_metrics = {"request_id": request_id, "seed": seed, "config": cfg,
                             "voice": voice, "reference_sha256": self.reference_hashes[voice],
                             "forced_eos_count": eos, "repetition_count": rep,
                             "duration_s": duration, "latency_s": latency,
                             "rtf": latency / duration, "rms_dbfs": 20 * np.log10(max(rms, 1e-12)),
                             "peak_dbfs": 20 * np.log10(max(peak, 1e-12)),
                             "valid_wav": True, "short_output_warning": bool(short),
                             "normalized_chars": len(text), "chunk_count": len(chunks)}
        logging.info("TTS quality %s", json.dumps(self.last_metrics, ensure_ascii=False))
        return {"ok": True, "state": self.state, "request_id": request_id,
                "voice": voice, "reference_sha256": self.reference_hashes[voice],
                "audio_b64": base64.b64encode(blob).decode("ascii"), "elapsed_ms": latency * 1000,
                "seconds": duration, "metrics": self.last_metrics, "chunks": records}


def cuda_error_type(exc: Exception) -> str | None:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "cuda_oom"
    if not isinstance(exc, RuntimeError):
        return None
    message = str(exc).lower()
    if "cuda" not in message:
        return None
    if "device-side assert" in message:
        return "cuda_device_assert"
    if "illegal memory access" in message or "misaligned address" in message:
        return "cuda_illegal_memory_access"
    if "unspecified launch failure" in message:
        return "cuda_launch_failure"
    return None


def is_fatal_cuda_error(exc: Exception) -> bool:
    return cuda_error_type(exc) is not None


def request_diagnostics(request: dict) -> dict:
    text = request.get("input", "")
    text = text if isinstance(text, str) else ""
    return {"request_id": request.get("request_id"), "operation": request.get("op"),
            "voice": request.get("voice", "rudo"), "seed": request.get("seed"),
            "input_chars": len(text), "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "stage": "dispatch"}


def main():
    worker = Worker()
    send({"ok": True, "event": "cpu_ready", "load": worker.load_info, **worker.snapshot()})
    while True:
        request = recv()
        if request is None:
            return
        request.setdefault("request_id", uuid.uuid4().hex)
        worker.current_diagnostics = request_diagnostics(request)
        op = request.get("op")
        try:
            if op == "state":
                send({"ok": True, **worker.snapshot()})
            elif op == "activate":
                send(worker.activate())
            elif op == "deactivate":
                send(worker.deactivate())
            elif op == "synth":
                response = worker.generate(request)
                response["request_id"] = request["request_id"]
                send(response)
            elif op == "shutdown":
                if worker.state == "gpu_ready":
                    worker.deactivate()
                send({"ok": True, "state": worker.state})
                return
            else:
                send({"ok": False, "error": f"unknown op {op}"})
        except Exception as exc:
            fatal = is_fatal_cuda_error(exc)
            if fatal:
                worker.state = "error"
            diagnostics = dict(worker.current_diagnostics)
            logging.exception("TTS request failed diagnostics=%s fatal=%s",
                              json.dumps(diagnostics, ensure_ascii=False), fatal)
            send({"ok": False, "error": repr(exc), "state": worker.state,
                  "fatal_error": fatal, "error_type": cuda_error_type(exc) or type(exc).__name__,
                  "operation": op, "request_id": request["request_id"], "diagnostics": diagnostics})
            if fatal:
                # send() flushes the failure before process exit. No offload, retry,
                # synchronize or empty_cache is safe on a poisoned CUDA context.
                return


if __name__ == "__main__":
    main()
