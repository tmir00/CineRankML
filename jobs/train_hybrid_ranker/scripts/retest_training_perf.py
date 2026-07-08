"""One-off hybrid training performance retest (stdout JSON summary)."""

from __future__ import annotations

import json
import sys
import time

import torch
from torch.utils.data import DataLoader

from common.config.settings import get_hybrid_training_settings
from common.storage.hybrid_ranker_dataset_reader import (
    load_hybrid_ranker_dataset_manifest,
    resolve_hybrid_ranker_dataset_version,
)
from common.storage.s3 import create_s3_client, download_file
from train_hybrid_ranker.dataset import (
    HybridParquetIterableDataset,
    _features_and_ratings_from_table,
    _row_order,
)
from train_hybrid_ranker.evaluate import _collate_features_batch
from train_hybrid_ranker.model import HybridRankerMLP
import pyarrow.parquet as pq
import tempfile
from pathlib import Path


def main() -> None:
    settings = get_hybrid_training_settings()
    client = create_s3_client(settings)
    dataset_version = resolve_hybrid_ranker_dataset_version(
        client, settings.s3_bucket, settings.dataset_version
    )
    manifest = load_hybrid_ranker_dataset_manifest(client, settings.s3_bucket, dataset_version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    part = manifest.train_parts[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "part-00000.parquet"
        t0 = time.perf_counter()
        download_file(client, settings.s3_bucket, part.object_key, local_path)
        download_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        table = pq.read_table(local_path, columns=["features", "rating"])
        read_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        features, ratings = _features_and_ratings_from_table(table)
        decode_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        _row_order(len(ratings), shuffle_within_part=True, seed=42)
        shuffle_s = time.perf_counter() - t0

    # Stream 3 train parts through DataLoader + optional GPU steps.
    parts = manifest.train_parts[:3]
    dataset = HybridParquetIterableDataset(
        client, settings.s3_bucket, parts, shuffle_within_part=True, seed=42
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        num_workers=0,
        collate_fn=_collate_features_batch,
    )
    model = HybridRankerMLP(input_dim=manifest.input_dim).to(device)
    model.train()
    loss_fn = torch.nn.MSELoss()

    rows = 0
    batches = 0
    gpu_s = 0.0
    stream_start = time.perf_counter()
    max_batches = 30

    for feature_batch, rating_batch in loader:
        t0 = time.perf_counter()
        feature_batch = feature_batch.to(device)
        rating_batch = rating_batch.to(device)
        preds = model(feature_batch)
        loss = loss_fn(preds, rating_batch)
        loss.backward()
        gpu_s += time.perf_counter() - t0
        rows += int(rating_batch.numel())
        batches += 1
        if batches >= max_batches:
            break

    stream_s = time.perf_counter() - stream_start
    batches_per_part = settings.batch_size and (100_000 / settings.batch_size)
    est_train_parts_s = (stream_s / batches) * (manifest.train_row_count / settings.batch_size)
    est_val_parts_s = (stream_s / batches) * (manifest.validation_row_count / settings.batch_size)

    summary = {
        "dataset_version": dataset_version,
        "device": str(device),
        "train_row_count": manifest.train_row_count,
        "validation_row_count": manifest.validation_row_count,
        "train_parts": len(manifest.train_parts),
        "validation_parts": len(manifest.validation_parts),
        "batch_size": settings.batch_size,
        "part0_download_s": round(download_s, 3),
        "part0_read_s": round(read_s, 3),
        "part0_decode_s": round(decode_s, 3),
        "part0_shuffle_s": round(shuffle_s, 3),
        "stream_3parts_batches": batches,
        "stream_3parts_rows": rows,
        "stream_3parts_total_s": round(stream_s, 3),
        "stream_avg_batch_s": round(stream_s / max(1, batches), 4),
        "stream_gpu_s": round(gpu_s, 3),
        "stream_data_s": round(stream_s - gpu_s, 3),
        "est_full_epoch_train_s": round(est_train_parts_s, 1),
        "est_full_epoch_val_s": round(est_val_parts_s, 1),
        "est_full_epoch_total_s": round(est_train_parts_s + est_val_parts_s, 1),
        "batches_per_100k_part": round(batches_per_part, 1) if batches_per_part else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
