# EDA — Smadex Creative Intelligence Challenge

Exploratory data analysis for the Smadex hackathon dataset. Run each notebook in order; they build on each other's findings.

## Quick Start

```bash
pip install pandas numpy matplotlib seaborn scikit-learn Pillow
jupyter notebook
```

All notebooks use `../` relative paths to load CSVs from the parent `smadex/` directory.

---

## Notebooks

| # | Notebook | Business Question |
|---|----------|-------------------|
| 01 | `01_dataset_overview.ipynb` | What does the data look like? Are joins clean? Any quirks? |
| 02 | `02_creative_performance.ipynb` | Which creatives win, and across which dimensions? |
| 03 | `03_creative_fatigue.ipynb` | When does fatigue hit, and what does the decay curve look like? |
| 04 | `04_feature_importance.ipynb` | Which design attributes drive performance? |
| 05 | `05_geo_os_analysis.ipynb` | Where and on what device do creatives perform best? |
| 06 | `06_creative_assets.ipynb` | What do the creative images actually look like? |

---

## Dataset at a Glance

| File | Rows | Grain |
|------|------|-------|
| `advertisers.csv` | 36 | 1 row per advertiser |
| `campaigns.csv` | 180 | 1 row per campaign (5 per advertiser) |
| `creatives.csv` | 1,080 | 1 row per creative (6 per campaign) |
| `creative_daily_country_os_stats.csv` | 192,315 | 1 row per date × creative × country × OS |
| `creative_summary.csv` | 1,080 | 1 row per creative, pre-aggregated |
| `assets/` | 1,080 PNGs | synthetic creative images |

Join path: `advertisers → campaigns → creatives → daily stats`

---

## Key Findings (Summary)

See **[insights.md](insights.md)** for full details. Headline takeaways:

- Creative fatigue typically manifests around day 20–35 of a campaign.
- Design features (novelty, motion, has_gameplay) correlate more strongly with CTR than with CVR.
- iOS systematically outperforms Android on CVR across most verticals.
- Top performers are disproportionately concentrated in a few themes and formats per vertical.
- The `perf_score` column in `creative_summary.csv` is a reliable composite signal for ranking.
