"""
Offline embedding generation using Gemma 4 E4B (google/gemma-4-E4B-it).

Processes all 1,080 ad creative PNGs and saves multimodal embeddings to disk.
Run once on GPU; the output is consumed by the analysis notebook and report builder.

Usage:
    uv run python src/embeddings/generate_embeddings.py [--resume]

    --resume    Continue from the last checkpoint (safe to interrupt and restart)

Output:
    data/embeddings/creative_embeddings.npz
        'embeddings'   float32  (N, hidden_dim)   — L2-normalised at save time
        'creative_ids' int32    (N,)               — matches creative_id column in CSVs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "embeddings"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.npz"
OUTPUT_PATH = OUT_DIR / "creative_embeddings.npz"

MODEL_ID = "google/gemma-4-E2B-it"
CHECKPOINT_EVERY = 100

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)


def _log_device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Device: GPU — {name} ({vram:.1f} GB VRAM)")
        return "cuda"
    else:
        import platform

        cpu = platform.processor() or "unknown CPU"
        logger.warning(
            f"Device: CPU — {cpu}. No CUDA-capable GPU detected. "
            "Embedding generation will be extremely slow (minutes per image). "
            "Run on a machine with an NVIDIA GPU for practical throughput."
        )
        return "cpu"


def load_model() -> tuple[AutoModelForCausalLM, AutoProcessor]:
    device = _log_device_info()
    logger.info(f"Loading {MODEL_ID} …")

    if device == "cuda":
        logger.info("Applying 4-bit quantization (bitsandbytes NF4)")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map="cuda:0",
            dtype=torch.float16,
            attn_implementation="eager",
        )
    else:
        # bitsandbytes 4-bit quantization requires CUDA; skip it on CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="eager",
        )

    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory allocated after load: {used:.1f} GB")
    else:
        logger.info("Model loaded on CPU")

    return model, processor


def embed_image(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image_path: Path,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this ad creative."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        images=[image],
        text=text,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Mean-pool the last hidden state over the full sequence.
    # Shape: (1, seq_len, hidden_dim) → (hidden_dim,)
    last_hidden = outputs.hidden_states[-1]
    embedding = last_hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
    return embedding


def load_checkpoint() -> tuple[list[int], list[np.ndarray]]:
    if CHECKPOINT_PATH.exists():
        data = np.load(CHECKPOINT_PATH)
        return data["creative_ids"].tolist(), list(data["embeddings"])
    return [], []


def save_checkpoint(creative_ids: list[int], embeddings: list[np.ndarray]) -> None:
    np.savez(
        CHECKPOINT_PATH,
        creative_ids=np.array(creative_ids, dtype=np.int32),
        embeddings=np.array(embeddings, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already-processed creatives)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        ROOT / "data" / "creative_summary.csv",
        usecols=["creative_id", "asset_file"],
    )
    records: list[tuple[int, str]] = list(
        zip(df["creative_id"].tolist(), df["asset_file"].tolist())
    )

    done_ids: list[int] = []
    done_embeddings: list[np.ndarray] = []

    if args.resume and CHECKPOINT_PATH.exists():
        done_ids, done_embeddings = load_checkpoint()
        done_set = set(done_ids)
        records = [(cid, af) for cid, af in records if cid not in done_set]
        logger.info(f"Resuming: {len(done_ids)} already done, {len(records)} remaining")

    model, processor = load_model()

    t0 = time.time()
    for i, (creative_id, asset_file) in enumerate(
        tqdm(records, desc="Embedding creatives", unit="img")
    ):
        # Handle case where asset_file might or might not include 'data/' prefix
        asset_rel = asset_file[5:] if asset_file.startswith("data/") else asset_file
        image_path = ROOT / "data" / asset_rel
        try:
            embedding = embed_image(model, processor, image_path)
        except Exception as exc:
            logger.warning(f"Skipping {image_path.name} — {exc}")
            continue

        done_ids.append(creative_id)
        done_embeddings.append(embedding)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(done_ids, done_embeddings)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(records) - i - 1) / rate / 60
            logger.info(
                f"Checkpoint {len(done_ids)} saved | {rate:.1f} img/s | ETA ~{eta_min:.0f} min"
            )

    embeddings_arr = np.array(done_embeddings, dtype=np.float32)
    ids_arr = np.array(done_ids, dtype=np.int32)

    # L2-normalise so downstream cosine similarity = dot product
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_arr = (embeddings_arr / norms).astype(np.float32)

    np.savez(OUTPUT_PATH, embeddings=embeddings_arr, creative_ids=ids_arr)

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    logger.success(f"Done in {elapsed / 60:.1f} min")
    logger.info(f"Saved:     {OUTPUT_PATH}")
    logger.info(f"Shape:     {embeddings_arr.shape}")
    if torch.cuda.is_available():
        logger.info(f"Peak VRAM: {peak_mem:.1f} GB")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()
