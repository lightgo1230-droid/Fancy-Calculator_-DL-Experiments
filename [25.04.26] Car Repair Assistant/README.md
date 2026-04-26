# Car Repair Assistant

> AI-powered car diagnostic & repair guidance desktop application  
> Rust (egui) + Python (PyTorch · FAISS · Ollama RAG)

---

## Overview

Car Repair Assistant is a fully offline, privacy-first desktop application that helps drivers diagnose vehicle problems and find repair guidance instantly. It combines a high-performance semantic search engine with optional local LLM generation (Ollama) to provide expert-level answers without sending any data to the cloud.

---

## Features

| Feature | Detail |
|---|---|
| Semantic Search | FAISS IndexFlatIP with L2-normalised embeddings (cosine similarity) |
| Local LLM (RAG) | Ollama integration — llama3.2 / mistral / phi3 / gemma2 and more |
| 60+ Q&A Entries | Curated car repair knowledge base across 14 categories |
| Category Detection | Auto-classifies queries: Brakes, Engine, Battery, AC, Transmission … |
| Multi-turn Context | Conversation history kept for follow-up questions |
| Favorites | Save answers with one click, persisted across sessions |
| Chat History | Previous sessions saved and restorable |
| Cost Guide | 53 typical US repair cost ranges with category filter |
| Dark / Light Theme | Toggle in header, preference saved automatically |
| Copy Button | Copy any answer to clipboard instantly |
| Confidence Badge | High / Medium / Low confidence indicator per answer |
| Ollama Manager | Built-in Options panel: install Ollama + pull models with one click |
| Korean / English Font | Malgun Gothic + Segoe UI — both languages render correctly |

---

## Architecture

```
┌────────────────────────────────────────────┐
│           Rust GUI  (egui / eframe)         │
│  Left Panel   Chat Area    Right Panel      │
│  Topics       Messages     Analysis         │
│  Favorites    Input Bar    Cost Guide       │
│  History      Header       Options Window   │
└──────────────┬─────────────────────────────┘
               │  stdin / stdout  (JSON lines)
┌──────────────▼─────────────────────────────┐
│        Python Backend  (engine_server.py)   │
│                                             │
│  SentenceTransformer  ──►  FAISS Index      │
│  (paraphrase-multilingual-MiniLM-L12-v2)    │
│                                             │
│  Ollama HTTP API  ──►  LLM Answer (RAG)     │
│  (localhost:11434)                          │
└─────────────────────────────────────────────┘
```

### Component Breakdown

| File | Role |
|---|---|
| `src/main.rs` | Rust GUI — all UI panels, theming, persistence, install manager |
| `backend/engine_server.py` | JSON bridge: receives queries via stdin, sends results via stdout |
| `backend/engine.py` | Core engine: embeddings, FAISS search, Ollama RAG, category detection |
| `backend/car_repair_en.jsonl` | English Q&A knowledge base (60 entries) |
| `backend/_cache/` | FAISS index disk cache (auto-invalidated on data change) |

---

## Requirements

### Runtime
- **Windows 10 / 11** (x64)
- **Python 3.9+** — must be in system PATH
- Python packages: `torch`, `sentence-transformers`, `faiss-cpu`, `numpy`

### Optional (for AI-generated answers)
- **Ollama** — [ollama.com](https://ollama.com) (can be installed from within the app)
- At least one pulled model (e.g. `llama3.2`, `mistral`, `phi3`)

### For Building from Source
- **Rust 1.75+** — [rustup.rs](https://rustup.rs)
- `cargo build --release`

---

## Installation

### Quick Start (pre-built binary)
1. Copy `Car_Repair_Assistant_Pro.exe` and the `backend/` folder to the same directory
2. Install Python dependencies:
   ```
   pip install torch sentence-transformers faiss-cpu numpy
   ```
3. Double-click `Car_Repair_Assistant_Pro.exe`

### First Launch
- The app builds the FAISS embedding index on first run (~30 seconds)
- Subsequent launches use the disk cache and start in ~5 seconds

### Enabling AI Answers (Ollama)
1. Click **⚙️ Options** in the header
2. Click **📥 Install / Update Ollama** — downloads and installs automatically
3. Click **⬇ Pull** next to any model (e.g. `llama3.2`)
4. Restart the app — AI answers will be enabled automatically

---

## Build from Source

```bash
# Clone or copy the project
cd car_pro_gui

# Build optimised release binary
cargo build --release

# Binary is at:
# target/release/car_pro_gui.exe
```

---

## Supported Categories

Brakes · Battery · Engine · AC/Heating · Steering · Suspension · Fluids · Maintenance · Tires · Transmission · Cooling · Electrical · Windshield · Fuel

---

## License

MIT License

---

## Credits

- UI framework: [egui](https://github.com/emilk/egui) / [eframe](https://github.com/emilk/egui)
- Embeddings: [sentence-transformers](https://www.sbert.net) — `paraphrase-multilingual-MiniLM-L12-v2`
- Vector search: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI)
- Local LLM: [Ollama](https://ollama.com)
- Copyright © 2026 lightgo · lgithgo1230@gmail.com
