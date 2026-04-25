# Smadex Creative Intelligence

HackUPC 2026 — Ad creative performance & fatigue analysis dashboard.

Built for the [Smadex Creative Intelligence Challenge](docs/hackathon-brief.md): help mobile advertisers understand which creatives work, why, and when they fatigue.

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- NVIDIA GPU with CUDA 12.6 — required only for embedding generation; the dashboard runs on CPU

## Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Install pre-commit hooks
uv run pre-commit install
```

## Running the Dashboard

```bash
uv run streamlit run main.py
```

## Generating Embeddings (GPU required)

The embedding generation script processes all 1,080 creative PNGs through Gemma 4 and saves the result to `data/embeddings/creative_embeddings.npz`. Run it once on a machine with an NVIDIA GPU:

```bash
uv run python src/embeddings/generate_embeddings.py

# Resume an interrupted run:
uv run python src/embeddings/generate_embeddings.py --resume
```

## Project Structure

```
smadex/
├── main.py                  # Streamlit entry point — run with: uv run streamlit run main.py
├── pyproject.toml           # Dependencies and tool config (uv)
│
├── src/                     # All application source code
│   ├── data/                # Data loading and preprocessing utilities
│   ├── embeddings/          # Embedding generation (generate_embeddings.py) and loading
│   ├── analysis/            # Fatigue detection and creative performance analysis
│   ├── features/            # Feature engineering
│   ├── models/              # ML classifiers and rankers
│   └── dashboard/           # Streamlit pages and components
│
├── data/                    # Dataset (not committed — provided by Smadex)
│   ├── advertisers.csv
│   ├── campaigns.csv
│   ├── creatives.csv
│   ├── creative_summary.csv
│   ├── creative_daily_country_os_stats.csv
│   ├── campaign_summary.csv
│   ├── data_dictionary.csv
│   ├── assets/              # 1,080 synthetic creative PNGs
│   └── embeddings/          # Generated embeddings (creative_embeddings.npz)
│
├── notebooks/               # Exploratory analysis notebooks (01–07)
└── docs/                    # Project documentation and insights
```

## Key Data Files

| File | Description |
|------|-------------|
| `data/creative_summary.csv` | One row per creative — aggregated KPIs + `creative_status` label |
| `data/creative_daily_country_os_stats.csv` | Granular daily data — use for fatigue detection |
| `data/creatives.csv` | Creative metadata + `asset_file` path to PNG |
| `data/data_dictionary.csv` | Column definitions for all files |

Join keys: `advertiser_id → campaign_id → creative_id`

## Dataset Overview

| Entity | Count |
|--------|-------|
| Advertisers | 36 |
| Campaigns | 180 (5 per advertiser) |
| Creatives | 1,080 (6 per campaign) |
| Daily rows | 192,315 |

All data is synthetic. See [docs/insights.md](docs/insights.md) for EDA findings and [docs/hackathon-brief.md](docs/hackathon-brief.md) for the challenge brief.
