# Next Steps — Gemma 4 Embedding Layer

Prioritised roadmap for turning the pre-computed embeddings into production features and products.

---

## 1. Production Embedding Pipeline

### Real-Time Similarity Search

Index the 1,080 pre-computed embeddings with **FAISS** or **Qdrant** for sub-millisecond lookup:

```python
# faiss index (exact L2, upgrade to IVF for >10K creatives)
import faiss
import numpy as np

data = np.load('data/embeddings/creative_embeddings.npz')
X = data['embeddings'].astype('float32')

index = faiss.IndexFlatIP(X.shape[1])  # inner product = cosine similarity (L2-normalised)
index.add(X)
faiss.write_index(index, 'data/embeddings/faiss.index')

# Query: given a new image embedding, find 5 most similar creatives
D, I = index.search(query_vec.reshape(1, -1), k=5)
```

For new creatives (not in the 1,080), embed on-demand using `src/embeddings/generate_embeddings.py`
in single-image mode — takes ~2 s on GPU.

### Incremental Updates

- Keep a running `data/embeddings/creative_embeddings.npz` that is appended when new creatives are uploaded.
- Re-index FAISS after each batch upload.

---

## 2. Chatbot / RAG Integration

### Pattern: Retrieval-Augmented Recommendation

```
User query: "What creative should I run for a gaming campaign in Japan?"

1. Embed the query text with Gemma 4 (text-only mode).
2. Search FAISS for top-K creatives with high cosine similarity AND
   filtered by (vertical=gaming, target_country=JP, creative_status=top_performer).
3. Format retrieved creatives as context:
   creative_id, headline, perf_score, key design scores.
4. Pass to Gemma 4 / any LLM for final recommendation generation.
```

**Example system prompt template:**

```
You are a creative strategy assistant. Based on the following top-performing ad creatives,
recommend the best option for the user's campaign:

{retrieved_creatives_context}

User request: {user_query}

Provide a concise recommendation with the rationale based on the creative's design features
and historical performance metrics.
```

### Key advantages of using Gemma 4 embeddings

- Joint image-text space: a text query like "bright colours, gameplay footage" retrieves
  visually matching creatives without any keyword tagging.
- No manual feature engineering: the model captures visual semantics automatically.

---

## 3. Feature Engineering for ML Models

The embedding PCA components are directly usable as features in downstream models:

| Task | Suggested approach |
|------|--------------------|
| **Fatigue prediction** | Logistic regression / XGBoost on `[tabular features] + [first 50 PCA components]` |
| **Performance scoring** | Replace or augment `perf_score` with embedding-based ranking |
| **Creative clustering** | Use UMAP 2D coordinates as a visual interface for campaign managers |
| **Anomaly detection** | Flag new creatives with low max-cosine-similarity to any existing cluster |

To extract PCA features for training:

```python
from sklearn.decomposition import PCA
import numpy as np, pandas as pd

data = np.load('data/embeddings/creative_embeddings.npz')
X = data['embeddings']

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

emb_feat = pd.DataFrame(
    X_pca,
    columns=[f'emb_pc{i}' for i in range(50)]
)
emb_feat['creative_id'] = data['creative_ids']
# merge with creative_summary.csv before training
```

---

## 4. Fine-Tuning Opportunities

### Adapter Fine-Tuning for Performance Prediction

Fine-tune a LoRA adapter on top of Gemma 4 E4B, using `creative_status` or `perf_score`
as the supervision signal:

- **Training data:** 1,080 labelled (image, label) pairs — small; use heavy augmentation.
- **Label:** binary top_performer / other (easier than 4-way) or regression on `perf_score`.
- **Architecture:** add a linear head on the mean-pooled last hidden state.
- **Expected benefit:** the fine-tuned model encodes *ad-specific* performance signals,
  not just general visual aesthetics.

> This is a stretch goal — the pre-trained embeddings alone provide substantial value
> and the dataset is small enough that fine-tuning risks overfitting.

---

## 5. What to Build Next — Ranked Priority

| # | Feature | Value | Effort | Notes |
|---|---------|-------|--------|-------|
| 1 | **Creative similarity search UI** | High | Low | FAISS already planned; add a simple Streamlit front-end |
| 2 | **Fatigue early-warning + replacement suggestion** | High | Low | Combine decay signals with nearest-neighbour recommendation from embeddings |
| 3 | **RAG-powered creative copilot** | High | Medium | Gemma 4 + FAISS retrieval + prompt template |
| 4 | **Embedding-augmented perf predictor** | Medium | Medium | PCA components + tabular features → XGBoost |
| 5 | **Interactive UMAP explorer** | Medium | Low | Ship the UMAP scatter from the notebook as a standalone Streamlit page |

---

## 6. Infrastructure Checklist

- [ ] FAISS index built and saved to `data/embeddings/faiss.index`
- [ ] Metadata JSON for each creative saved alongside the index (for display in UI)
- [ ] API endpoint wrapping single-image embedding + similarity search
- [ ] Monitoring: track embedding drift as new creatives are added
- [ ] GPU or CPU serving strategy (CPU inference is feasible with quantised model at ~5 s/image)
