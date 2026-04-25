# Gemma 4 Embedding Analysis — Key Insights

Findings from `07_gemma_embeddings_analysis.ipynb` using `google/gemma-4-E4B-it` embeddings
on 1,080 ad creative PNGs.

> **Note:** Fill in `[TBD]` values after running `src/embeddings/generate_embeddings.py` and the analysis notebook.

---

## 1. Embedding Space Overview

| Property | Value |
|----------|-------|
| Model | `google/gemma-4-E4B-it` (4-bit quantised, ~4–5 GB VRAM) |
| Embedding method | Mean-pool of last hidden state over full token sequence |
| Embedding dimension | [TBD — typically 2560 for 4B] |
| Normalisation | L2 (cosine similarity = dot product) |
| PCA variance (PC1 + PC2) | [TBD]% |
| Best KMeans k (silhouette) | [TBD] |
| Linear probing accuracy | [TBD]% (vs 25% random baseline) |

---

## 2. Visual Cluster Structure

After running KMeans with the best-k selected by silhouette score:

| Cluster | Size | Avg perf_score | Top-Performer % | Underperformer % | Dominant Vertical | Dominant Theme |
|---------|------|---------------|-----------------|-----------------|-------------------|---------------|
| 0 | [TBD] | [TBD] | [TBD]% | [TBD]% | [TBD] | [TBD] |
| 1 | [TBD] | [TBD] | [TBD]% | [TBD]% | [TBD] | [TBD] |
| 2 | [TBD] | [TBD] | [TBD]% | [TBD]% | [TBD] | [TBD] |
| 3 | [TBD] | [TBD] | [TBD]% | [TBD]% | [TBD] | [TBD] |

**Key question to answer after running:** Does any cluster have a disproportionate share of top_performers?
If yes → creatives in that visual neighbourhood are inherently higher-performing, and new creatives
should be designed to match that cluster's visual style.

---

## 3. Performance–Embedding Alignment

### PCA / UMAP Separation

- **Expected finding:** PCA and UMAP projections should show *partial* separation between
  `top_performer` and `underperformer` groups. Perfect separation is unlikely (visual style
  alone does not determine performance), but clustering above chance confirms visual signal.

- **What to look for:** Do top_performers cluster in a specific region of the embedding space,
  or are they distributed throughout? Concentrated regions suggest consistent visual templates
  for success.

### Linear Probing

A logistic regression on raw embeddings → `creative_status` label measures
how much performance signal Gemma 4 encodes purely from image understanding:

- **> 60% accuracy** → strong visual signal; embeddings are meaningful features for performance prediction.
- **40–60% accuracy** → moderate signal; embeddings complement but don't replace tabular features.
- **~25% accuracy** → visual style alone is not predictive; performance is driven by non-visual factors.

---

## 4. Multimodal Alignment

When image embeddings and text embeddings (headline + CTA) are projected into a shared PCA space:

- **Same creative:** image and text embeddings should land near each other (multimodal coherence).
  Large distances suggest the visual and verbal messages are misaligned.
- **Cluster consistency:** if creatives with the same `theme` cluster together in *both*
  the image and text space, the dataset's labelling is consistent with the model's understanding.

---

## 5. Nearest-Neighbour Analysis

For each **fatigued** creative, the most visually similar **top-performer** is identified via
cosine similarity in embedding space. This enables:

1. **Replacement recommendations:** swap a fatigued creative with its nearest top-performer
   that shares the same vertical and format.
2. **A/B hypothesis generation:** similar visuals, different performance → the difference
   is explained by copy, targeting, or timing rather than creative design.

---

## 6. Surprising Findings (fill in after running)

- **Hard negatives:** [TBD — creatives with cosine similarity > 0.9 but opposite status labels]
  These are the most actionable for explainability: nearly identical visuals, very different outcomes.

- **Cross-vertical clusters:** [TBD — any gaming and ecommerce creatives sharing the same cluster?]
  If yes, visual motifs transfer across verticals.

- **DBSCAN outliers:** [TBD — fraction marked as noise (-1)]
  Outlier creatives may be uniquely experimental; worth manual review.

---

## 7. Actionable Rules from Embeddings

After filling in the cluster profiles, derive 3–5 rules like:

1. "Creatives in Cluster X (high novelty_score, portrait, gaming) achieve avg ROAS > [TBD]x — prioritise this visual template."
2. "Fatigued creatives in Cluster Y can be replaced by top-performers in Cluster X with cosine similarity > 0.8."
3. "The linear probe at [TBD]% accuracy justifies using the first [TBD] PCA components as features in a fatigue-prediction model."
