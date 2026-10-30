import adapters


class FakeRPC:
    def __init__(self, script, env=None, python=None):
        self.script = script
        self.env = env or {}
        self.python = python
        self.calls = []
        self.p = self.Process()

    class Process:
        def poll(self):
            return None

    def call(self, payload, timeout=900):
        self.calls.append((payload, timeout))
        op = payload["op"]
        if op == "activate":
            return {"ok": True, "state": "gpu_ready", "elapsed_ms": 650}
        if op == "deactivate":
            return {"ok": True, "state": "cpu_ready", "elapsed_ms": 880}
        if op == "synth":
            return {"ok": True, "audio_b64": "", "state": "gpu_ready"}
        return {"ok": True, "state": "cpu_ready"}

    def terminate(self):
        pass


def test_tts_adapter_uses_chatterbox_and_explicit_runtime(monkeypatch):
    made = []

    def rpc(script, env=None, python=None):
        obj = FakeRPC(script, env, python)
        made.append(obj)
        return obj

    monkeypatch.setattr(adapters, "RPC", rpc)
    monkeypatch.setenv("CHATTERBOX_PYTHON", "/isolated/chatterbox/.venv/bin/python")
    worker = adapters.TTSWorkerAdapter()

    assert made[0].script == "chatterbox_worker.py"
    assert made[0].python == "/isolated/chatterbox/.venv/bin/python"
    assert worker.health()["state"] == "cpu_ready"


def test_tts_adapter_forwards_normalization_and_lifecycle(monkeypatch):
    monkeypatch.setattr(adapters, "RPC", lambda *args, **kwargs: FakeRPC(*args, **kwargs))
    worker = adapters.TTSWorkerAdapter()

    activated = worker.activate_gpu()
    assert activated["state"] == "gpu_ready"
    result = worker.synth("Suma 84,53 €", normalize=True)
    payload = worker.w.calls[-1][0]
    assert payload == {"op": "synth", "input": "Suma 84,53 €", "normalize": True, "seed": None}
    assert result["ok"] is True
    deactivated = worker.deactivate_gpu()
    assert deactivated["state"] == "cpu_ready"


def test_manager_ready_exposes_cpu_ready_tts(monkeypatch):
    import manager_app
    class FakeManager:
        def snapshot(self):
            return {"status":"healthy","accepting":True,"active_profile":None,"queue_depth":0,"workers":{"tts":{"state":"cpu_ready","alive":True,"gpu_ready":False}}}
    monkeypatch.setattr(manager_app, "M", FakeManager())
    state = manager_app.ready()
    assert state["state"] == "cpu_ready" and state["can_accept_request"] is True and state["gpu_ready"] is False


def test_speech_endpoint_defaults_to_normalization(monkeypatch):
    import base64, manager_app
    class FakeWorker:
        def synth(self, text, normalize=True, seed=None):
            assert text == "Suma 84,53 €" and normalize is True and seed is None
            return {"ok":True,"audio_b64":base64.b64encode(b"RIFF").decode(),"elapsed_ms":12.5,"request_id":"r-1","metrics":{"request_id":"r-1","seed":42,"config":{"temperature":0.6},"forced_eos_count":0,"repetition_count":0}}
    class FakeManager:
        workers = {"tts":FakeWorker()}
        def submit(self, profile, fn):
            assert profile == "tts"
            return fn()
        def _event(self, name, **data):
            assert name == "tts_quality"
    monkeypatch.setattr(manager_app, "M", FakeManager())
    response = manager_app.tts(manager_app.SpeechReq(input="Suma 84,53 €"))
    assert response.status_code == 200 and response.body == b"RIFF"
    assert response.media_type == "audio/wav" and response.headers["x-request-id"] == "r-1"
