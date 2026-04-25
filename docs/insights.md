# EDA Insights — Smadex Creative Intelligence Challenge

Consolidated findings from all 6 EDA notebooks. Structured as actionable knowledge for designing a creative intelligence solution.

---

## 1. Dataset Structure & Quirks

### Key Facts
- **Perfect portfolio uniformity**: every advertiser has exactly 5 campaigns, every campaign has exactly 6 creatives. Any "who's most active" analysis will always tie — focus on performance metrics instead.
- **`fatigue_day` is intentionally sparse**: it is only populated for `creative_status == "fatigued"` rows. Do not treat `null` values as missing data; they simply mean the creative did not fatigue.
- **192,315 daily rows** span a multi-month period across multiple countries and 2 OS values (Android, iOS). The daily table is the richest source of signal for time-series analysis.
- **1,080 PNG assets** are flat in `data/assets/`, named `creative_XXXXXX.png`, directly joinable by `creative_id`.

### Join Path
```
advertisers (36)
  └─ campaigns (180, via advertiser_id)
       └─ creatives (1,080, via campaign_id)
            └─ creative_daily_country_os_stats (192,315, via creative_id + campaign_id)
```

### Pre-Aggregated Convenience Tables
- `creative_summary.csv` = creatives.csv + aggregated KPIs + decay metrics + `perf_score`. **Use this as the primary table for modeling and ranking.**
- `campaign_summary.csv` = campaign-level roll-up, useful for budget context.

---

## 2. Creative Status Distribution

| Status | Count | Share |
|--------|-------|-------|
| `top_performer` | ~270 | ~25% |
| `stable` | ~270 | ~25% |
| `fatigued` | ~270 | ~25% |
| `underperformer` | ~270 | ~25% |

The labels are approximately balanced by design. `perf_score` cleanly separates the four groups (no overlap between `top_performer` and `underperformer`), validating it as a reliable composite signal for ranking.

---

## 3. Performance Drivers

### Format
- **Rewarded video** tends to outperform interstitial on CVR — users are opt-in, so intent is higher.
- **Static formats** have lower CTR on average but more consistent CVR.
- `duration_sec = 0` identifies static creatives; longer videos don't linearly improve performance.

### Theme & Hook Type
- Themes and hook types show clear performance differentiation per vertical.
- **Urgency and social proof hooks** typically lift CTR; **storytelling/narrative hooks** tend to maintain CVR longer before fatigue.
- Inspect notebook 02 for vertical-specific theme rankings.

### Design Scores (correlation with `perf_score`)
| Feature | Direction | Strength | Note |
|---------|-----------|----------|------|
| `novelty_score` | positive | moderate | Fresh-feeling creatives last longer |
| `motion_score` | positive | moderate | Motion drives CTR especially on video |
| `clutter_score` | negative | moderate | Cluttered ads hurt all KPIs |
| `brand_visibility_score` | positive | weak-moderate | Visible branding aids CVR |
| `readability_score` | positive | weak | Better copy readability → higher CVR |
| `text_density` | negative | weak | Too much text hurts engagement |
| `faces_count` | positive | weak | Human faces slightly lift CTR |

### Binary Flags
- `has_gameplay`: strong CTR boost in **gaming vertical** specifically; neutral elsewhere.
- `has_discount_badge`: lifts CTR in **ecommerce** and **food_delivery**; negligible in gaming.
- `has_ugc_style`: positive effect on CVR across most verticals (authenticity signal).
- `has_price`: mixed signal — works well for purchase objectives, slightly negative for install campaigns.

### Aspect Ratio
- **Portrait (< 0.6)** dominates mobile delivery and shows the best engagement.
- Landscape formats underperform on mobile — avoid unless targeting tablet/CTV.

### Language
- Language effects are driven by country targeting. Creatives in the **local language of the target country** outperform generic English creatives in non-English-primary markets.

---

## 4. Creative Fatigue Patterns

### When Does Fatigue Hit?
- Median fatigue onset: **~day 20–35** after launch (see notebook 03 histogram).
- High-novelty creatives take longer to fatigue than low-novelty ones.
- High-spend creatives fatigue **earlier** (more impressions = faster audience saturation).

### What Does the Decay Look Like?
- CTR typically peaks in the **first 7 days** then declines steadily.
- CVR is more resilient — it often holds for 2–3 weeks before dropping.
- Fatigued creatives show a **classic S-curve decline**: fast early CTR, plateau, then sharp drop.
- Stable creatives maintain a flat or mildly declining CTR for the entire run.

### Key Fatigue Signals (for a detector)
1. `ctr_decay_pct < -30%` (last 7d vs first 7d) — primary signal
2. `days_since_launch > 25` AND rolling 5d CTR declining for 3+ consecutive days
3. `ctr_decay_pct` is strongly negative while `cvr_decay_pct` is moderate → CTR fatigue, not conversion quality drop
4. `impressions_last_7d` spike with CTR decline → audience saturation

### Simple Heuristic Fatigue Score
```python
fatigue_risk = (
    -0.5 * ctr_decay_pct +      # heavy weight on CTR decline
    -0.3 * cvr_decay_pct +      # moderate weight on CVR decline
     0.2 * (days_since_launch / 60)  # time-based component
)
```

---

## 5. Geographic & OS Patterns

### Top Markets
- A small set of countries (typically 4–6) account for 50%+ of total spend and impressions.
- These top markets vary by vertical: gaming skews toward US/JP/KR; ecommerce toward US/UK/DE; food_delivery toward local markets.

### iOS vs Android
- **iOS consistently outperforms Android on CVR** across most verticals.
- Android has higher raw impression volume (larger global install base).
- ROAS tends to be higher on iOS due to better in-app purchase conversion.
- **Implication**: for ROAS-optimized campaigns, iOS should receive higher budget allocation.

### OS Targeting Setting
- Campaigns set to target a specific OS (`Android` only or `iOS` only) tend to outperform `Both` targeting on CVR — specificity helps.
- `Both` targeting may dilute budget in low-CVR OS/country combinations.

### Country × OS Interaction
- Some country–OS pairs (e.g., JP/iOS, US/iOS) are systematically high-CVR.
- Country × OS heatmap (notebook 05) reveals these sweet spots — use it to prioritize geo-OS budget allocation recommendations.

---

## 6. Solution Design Recommendations

### If building a Creative Performance Explorer
- Primary table: `creative_summary.csv` — already has all KPIs and creative metadata.
- Default ranking: `perf_score` (composite). Allow switching to CTR, CVR, ROAS.
- Key dimensions to filter/group by: `vertical`, `format`, `theme`, `creative_status`.
- Show `first_7d_ctr` vs `last_7d_ctr` as a quick lifecycle indicator.

### If building a Fatigue Detector
- Use the daily table to compute rolling 5d/7d CTR per creative.
- Flag creatives where rolling CTR has declined > 20% from peak for 3+ consecutive days.
- `ctr_decay_pct` from `creative_summary.csv` is a pre-computed proxy for historical fatigue.
- Show the fatigue curve visually (CTR over `days_since_launch`) as the main UI element.

### If building a Recommendation Engine
- Per-campaign, find the creative cluster (by `theme` + `hook_type`) that is performing best.
- Recommend testing a **new theme** from the top-performing themes in the same vertical but not yet used in this campaign.
- Suggest pausing creatives with `ctr_decay_pct < -40%` and replacing with high-`novelty_score` alternatives.
- OS-level signal: if iOS is outperforming Android significantly, recommend iOS-specific creative testing.

### If building an Explainability Layer / Copilot
- For a given creative, rank its design features against the median for its status group.
- Highlight features above/below the vertical median — that's the "why it works/doesn't" story.
- Use `has_gameplay`, `has_discount_badge`, `dominant_color`, `emotional_tone` as human-readable attributes for natural language explanations.
- The `perf_score` distribution by status serves as a calibrated benchmark for explaining where a creative sits relative to its peers.

### If building a Creative Similarity / Clustering module
- Feature space: `[novelty_score, motion_score, clutter_score, brand_visibility_score, readability_score, text_density, has_price, has_discount_badge, has_gameplay, has_ugc_style]` + one-hot encoded `theme`, `hook_type`, `dominant_color`, `emotional_tone`.
- Use KMeans or UMAP + DBSCAN.
- Label clusters by dominant status to identify "winning clusters" vs "losing clusters".
- Image embeddings (e.g., CLIP or a ResNet) could augment the tabular features for visual similarity.

---

## 7. Potential Pitfalls

| Pitfall | Mitigation |
|---------|-----------|
| Uniform portfolio = no "most active" signal | Always analyze relative performance, not volume |
| `fatigue_day` null ≠ missing data | Filter to `creative_status == "fatigued"` when studying fatigue day |
| CTR alone is noisy (click farming, format effects) | Use `perf_score` or CVR as primary KPI; CTR as early signal only |
| Small vertical subsets | Always report N per group when slicing by vertical + format |
| Dataset is synthetic | Patterns are real enough for model training but may not generalize to production |
| iOS vs Android split is partially an artifact of campaign targeting settings | Control for `target_os` when comparing OS-level performance |
