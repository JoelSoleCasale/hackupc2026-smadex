# CLAUDE.md — Smadex Creative Intelligence Challenge

## Project Overview

HackUPC 2026 hackathon project. Build a creative intelligence tool for mobile advertisers using a synthetic ad-tech dataset (1,080 creatives, 192k daily rows). Goal: help advertisers understand which creatives work, why, and when they fatigue. The final deliverable is a **Streamlit dashboard** launched via `uv run streamlit run main.py`.

## Environment

- **Python package manager: `uv`** — always use `uv` for all dependency and environment operations.
  - Install deps: `uv sync`
  - Add a package: `uv add <package>`
  - Run scripts: `uv run python <script>`
  - Run notebooks: `uv run jupyter notebook`
  - Run dashboard: `uv run streamlit run main.py`
- Python ≥ 3.12 required (`requires-python = ">=3.12"` in `pyproject.toml`).
- CUDA backend is `cu126` (CUDA 12.6). PyTorch and torchvision are pinned to the PyTorch cu126 index via explicit `[[tool.uv.index]]` and `[tool.uv.sources]` in `pyproject.toml`. Do **not** use `torch-backend` — it silently resolves to the CPU wheel when the index lacks the requested version.

## Logging

- **Use `loguru` for all logging** — never use `print` or the stdlib `logging` module in scripts.
- Standard import: `from loguru import logger`
- Always log whether the model/compute is running on CPU or GPU at startup (see `src/embeddings/generate_embeddings.py` for the pattern).

## GPU / CUDA Notes

- The embedding generation script (`src/embeddings/generate_embeddings.py`) requires a CUDA GPU for practical throughput.
- Without a GPU, inference takes ~8 minutes per image on CPU. Always check `torch.cuda.is_available()` and warn loudly if running on CPU.
- 4-bit quantization via `bitsandbytes` is only applied when CUDA is available — skip `BitsAndBytesConfig` on CPU.

## Project Structure

```
smadex/
├── main.py                  # Streamlit entry point
├── pyproject.toml
├── uv.lock
│
├── src/                     # All application source code
│   ├── data/                # Data loading and preprocessing utilities
│   ├── embeddings/          # Embedding generation and loading
│   │   └── generate_embeddings.py   # Offline Gemma-4 embedding script
│   ├── analysis/            # Fatigue detection and creative performance analysis
│   ├── features/            # Feature engineering
│   ├── models/              # ML classifiers and rankers
│   └── dashboard/           # Streamlit pages and components
│
├── data/                    # Dataset (CSVs + assets)
│   ├── assets/              # 1,080 synthetic ad creative PNGs
│   └── embeddings/          # Generated outputs (creative_embeddings.npz) — gitignored
│
├── notebooks/               # Exploratory analysis notebooks (01–07) and report builders
└── docs/                    # Project documentation, insights, and challenge brief
```

## Key Data Files

| File | Description |
|------|-------------|
| `data/creative_summary.csv` | One row per creative with aggregated KPIs and `creative_status` label |
| `data/creative_daily_country_os_stats.csv` | Granular daily data for fatigue detection |
| `data/creatives.csv` | Creative metadata + `asset_file` path to PNG |
| `data/data_dictionary.csv` | Column definitions for all files |

Join keys: `advertiser_id → campaign_id → creative_id`

## Dataset Quirks

- `fatigue_day` is only populated for creatives with `creative_status == "fatigued"`.
- Portfolio is perfectly uniform: 36 advertisers × 5 campaigns × 6 creatives = 1,080 creatives. "Most active advertiser" analyses will always tie — focus on performance metrics.
- All data is synthetic (no real PII or real ad performance).

## Model

- **Embedding model:** `google/gemma-4-E4B-it` via HuggingFace `transformers`
- Embeddings are mean-pooled last hidden states, L2-normalised (cosine similarity = dot product downstream).
- Output shape: `(1080, hidden_dim)` stored in `data/embeddings/creative_embeddings.npz`.

## Dependencies (key ones)

- `torch`, `torchvision` — deep learning, CUDA 12.6 backend
- `transformers`, `accelerate`, `bitsandbytes` — Gemma-4 model loading and quantization
- `loguru` — logging (use this everywhere)
- `pandas`, `numpy`, `scikit-learn`, `scipy` — data processing
- `umap-learn` — dimensionality reduction for embedding visualization
- `plotly`, `seaborn`, `matplotlib` — visualization
- `pillow` — image loading
- `tqdm` — progress bars
- `streamlit` — dashboard UI
