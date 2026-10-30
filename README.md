# AI GPU Manager

**RTX 3050 model orchestration service** — manages dynamic loading/offloading of ML models (Chatterbox TTS, WhisperX STT, retrieval embeddings) with FastAPI on `:18080`. Designed for memory-constrained GPUs (8GB VRAM) with intelligent model lifecycle management.

## Features

- **Dynamic Model Loading**: Load models on-demand, offload after idle timeout
- **Multi-Model Support**: Chatterbox TTS, WhisperX STT, retrieval embeddings
- **Worker Pool**: Rotating workers for STT, persistent TTS worker
- **Health Monitoring**: `/health`, `/models`, `/workers` endpoints
- **GPU Memory Management**: VRAM tracking, automatic offloading to RAM
- **Systemd Integration**: Production-ready service definition

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  AI GPU Manager (FastAPI/Uvicorn on :18080)                │
│  ├── manager_app.py       → Main API, model lifecycle      │
│  ├── adapters.py          → Model adapters (TTS/STT/embed) │
│  ├── chatterbox_worker.py → TTS worker (persistent)        │
│  ├── whisperx_worker.py   → STT worker (rotating pool)     │
│  ├── retrieval_worker.py  → Embedding worker               │
│  ├── stt_worker.py        → STT orchestration              │
│  └── protocol.py          → Internal messaging             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   ┌───────────────┐
   │ RTX 3050      │
   │ (8GB VRAM)    │
   └───────────────┘
```

## Prerequisites

- Python 3.10+
- RTX 3050 (8GB VRAM) or compatible
- CUDA 12.x
- PyTorch with CUDA

## Installation

```bash
git clone https://github.com/Rugents-masterpiece/ai-gpu-manager.git
cd ai-gpu-manager

# Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Create from adapters.py imports

# Configure
cp .env.example .env  # If exists, or create manually
# AI_GPU_MANAGER_URL=http://127.0.0.1:18080
# TTS_IDLE_TIMEOUT_SECONDS=60
# GPU_IDLE_TIMEOUT_SECONDS=300

# Run
uvicorn manager_app:app --host 127.0.0.1 --port 18080

# Or with systemd
sudo cp systemd/ai-gpu-manager.service /etc/systemd/system/
sudo systemctl enable --now ai-gpu-manager
```

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_IDLE_TIMEOUT_SECONDS` | 60 | TTS worker idle timeout |
| `GPU_IDLE_TIMEOUT_SECONDS` | 300 | GPU model idle timeout |
| `RETRIEVAL_ACTIVATE_TIMEOUT_SECONDS` | 180 | Retrieval model activation timeout |
| `RETRIEVAL_DEACTIVATE_TIMEOUT_SECONDS` | 180 | Retrieval model deactivation timeout |
| `TTS_ACTIVATE_TIMEOUT_SECONDS` | 180 | TTS model activation timeout |
| `TTS_DEACTIVATE_TIMEOUT_SECONDS` | 180 | TTS model deactivation timeout |
| `STT_ACTIVATE_TIMEOUT_SECONDS` | 180 | STT model activation timeout |
| `STT_RETIRE_TIMEOUT_SECONDS` | 180 | STT worker retirement timeout |
| `CUDA_VISIBLE_DEVICES` | GPU UUID | Specific GPU to use |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + GPU status |
| `/models` | GET | List loaded/available models |
| `/workers` | GET | Worker pool status |
| `/tts` | POST | Text-to-Speech (via Chatterbox) |
| `/stt` | POST | Speech-to-Text (via WhisperX) |
| `/embeddings` | POST | Text embeddings |
| `/activate/{model}` | POST | Pre-load model |
| `/deactivate/{model}` | POST | Offload model |

## Model Adapters (`adapters.py`)

- **ChatterboxTTSAdapter**: Voice cloning, multi-speaker
- **WhisperXSTTAdapter**: Transcription + diarization
- **RetrievalAdapter**: Embedding generation

## Workers

| Worker | Type | Lifecycle |
|--------|------|-----------|
| `chatterbox_worker.py` | TTS | Persistent (kept warm) |
| `whisperx_worker.py` | STT | Rotating pool (on-demand) |
| `retrieval_worker.py` | Embeddings | On-demand |
| `stt_worker.py` | STT orchestration | Pool manager |

## Systemd Service

```ini
# /etc/systemd/system/ai-gpu-manager.service
[Unit]
Description=RTX3050 AI GPU manager production stack
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rugent
WorkingDirectory=/home/rugent/rtx3050-profile-prototype/production-candidate
EnvironmentFile=-/etc/ai-stack/secrets.env
Environment=CUDA_VISIBLE_DEVICES=GPU-<uuid>
ExecStart=/home/rugent/ai-stack/qwen-runtime/.venv/bin/python -m uvicorn manager_app:app --host 127.0.0.1 --port 18080
Restart=on-failure
RestartSec=10
TimeoutStartSec=1800
TimeoutStopSec=240

[Install]
WantedBy=multi-user.target
```

## Development

```bash
# Run tests
pytest test_candidate_manager.py test_chatterbox_adapter.py

# Manual test
python3 test_candidate_manager.py

# Check GPU
nvidia-smi
```

## Rollback

```bash
./rollback-ai-gpu-manager.sh
```

## Security

- No credentials in repository
- Secrets via `/etc/ai-stack/secrets.env` (not in repo)
- Runs on localhost (127.0.0.1) only
- GPU access via CUDA_VISIBLE_DEVICES


## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new features, documentation improvements, or testing — every contribution helps.

### Ways to Contribute

| Type | Description |
|------|-------------|
| 🐛 **Bug Reports** | Found an issue? Open a GitHub Issue with steps to reproduce, expected vs actual behavior, and environment details |
| ✨ **Feature Requests** | Have an idea? Open a GitHub Issue with the "enhancement" label describing the use case |
| 📝 **Documentation** | Fix typos, clarify setup steps, add examples — PRs to `README.md` or `/docs` are always appreciated |
| 🔧 **Code** | Bug fixes, new features, refactoring, tests |
| 🧪 **Testing** | Test on different hardware/OS, report compatibility |
| 💡 **Ideas & Feedback** | Discussions, RFCs, architecture suggestions |

### Quick Start for Contributors

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/REPO-NAME.git
cd REPO-NAME

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make changes, test locally
# (see README for local development setup)

# 5. Commit with conventional commits
git commit -m "feat: add amazing feature"

# 6. Push and open a Pull Request
git push origin feature/your-feature-name
```

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat:     new feature
fix:      bug fix
docs:     documentation only
style:    formatting, no code change
refactor: code restructuring
test:     adding tests
chore:    maintenance, deps, build
```

Examples:
- `feat: add voice selection UI for TTS`
- `fix: handle GPU OOM in model offloading`
- `docs: clarify Docker setup for ARM64`

### Pull Request Guidelines

- ✅ **One feature/fix per PR** — keeps reviews focused
- ✅ **Tests pass** — run local tests before pushing
- ✅ **Update docs** — if behavior changes, update README/comments
- ✅ **No credentials** — never commit `.env`, keys, tokens
- ✅ **Small, focused commits** — easier to review

### Development Setup

See the **Installation** section above for local setup. Most projects use:

```bash
# Python projects
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or pip install -e ".[dev]" for dev dependencies

# .NET projects
dotnet restore
dotnet build

# Docker projects
docker compose up -d --build
```

### Code Style

| Language | Formatter | Linter |
|----------|-----------|--------|
| Python | `black` | `ruff` |
| C# | `dotnet format` | Built-in |
| JavaScript/TypeScript | `prettier` | `eslint` |
| Blazor | `dotnet format` | Built-in |

Run formatters before committing:
```bash
# Python
black . && ruff check .

# .NET
dotnet format

# JS/TS (if applicable)
prettier --write .
```

### Reporting Security Issues

**Do not open public issues for security vulnerabilities.** Email directly or use GitHub's private vulnerability reporting.

### Community

- 💬 **Discussions**: Use GitHub Discussions for questions, ideas, RFCs
- 🐛 **Issues**: Bug reports, feature requests
- 📧 **Contact**: Open an issue for maintainer contact

### Recognition

Contributors are recognized in:
- GitHub Contributors graph
- Release notes (for significant contributions)
- `AUTHORS.md` or `CONTRIBUTORS.md` (if maintained)

---

**Thank you for contributing!** 🎉

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

MIT License allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ No liability
- ❌ No warranty

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps with visibility and motivation.

## License

MIT
