# Smadex Creative Intelligence

## Inspiration

Creative fatigue is expensive and usually detected too late. We wanted a tool that gives non-technical advertisers quick, practical answers: what is working, what is fading, and what to do next.

---

## What it does

Smadex Creative Intelligence is a 3-level dashboard powered by two ML models.

### Advertiser overview
- Benchmark KPIs (CTR, CVR, ROAS) vs peers in the same vertical
- Peer ranking chart
- Country performance map
- Clickable campaign cards

### Campaign view
- Creative ranking by status (top performer, stable, fatigued, underperformer)
- Trend charts
- Quick visual comparison strip

### Ad detail view
- Creative image + design attributes
- Audience-specific effect tags (green positive, red negative)
- Three audience-fit scores
- Predicted fatigue/profitability-loss day with uncertainty
- Recommendation: scale, watch, or pause

---

## How we built it

**Data pipeline**
Loaded the core campaign and creative tables, mapped internal IDs to readable names, and cached results to keep navigation fast.

**Audience-specific correlation analysis**
Computed trait-to-performance relationships by device, age group, and country. We used Pearson correlation for numeric fields and signed Random Forest importance for categorical fields.

**Creative fitness scoring**
Combined correlation signals into three scoring styles: linear total alignment, consistency-weighted (Sharpe-style), and a focused Top-10 strongest-signals score.

**Feature engineering**
Built model features from first-7-day CTR/CVR/ROAS behavior and static creative attributes.

**Profitability event definition**
Defined profitability loss as the first day in a 3-day ROAS < 1 streak, then used that day as the regression target.

**ML modeling**
Trained two offline LightGBM regressors: one predicts profitability-loss day, the other predicts fatigue day. Model quality metrics are surfaced in the UI.

**Visual embeddings**
Generated Gemma-4 embeddings for all 1,080 creatives to support visual similarity features.

**Dashboard implementation**
Built the product in Streamlit with Plotly and session-state-driven navigation.

---

## Challenges we ran into

- Correlations needed to be audience-specific; global scores were misleading.
- The dataset is perfectly uniform, so volume-based activity comparisons always tie.
- `fatigue_day` null means "not fatigued," not missing data.
- Profitability-loss and fatigue are different prediction targets.
- Streamlit reruns required careful state handling to preserve navigation.

---

## What we're proud of

- End-to-end flow from advertiser view to creative recommendation
- Explainable, audience-specific attribute tags
- Predictions shown with uncertainty bands
- Fully offline execution (no API keys or external calls)

---

## What we learned

- Fatigue is multi-signal: CTR often drops before CVR.
- iOS often showed stronger CVR; broad platform targeting could underperform focused targeting.
- Portrait format consistently beat landscape.
- Novelty was the strongest durability signal (about +15 days to fatigue for high-novelty creatives).

---

## What's next

- Add a 2D embedding map of visually similar creatives, colored by performance
- Recommend lookalike creatives from other campaigns in the same category
- Add alerts when live CTR crosses predicted fatigue thresholds

---

## Built with

Python 3.12, uv, Streamlit, Plotly, LightGBM, scikit-learn, SciPy, Pandas, NumPy, PyTorch, HuggingFace Transformers, bitsandbytes, UMAP-learn, Loguru, Pillow
