"""Opt-in measurements for replay evaluation, separate from claim statistics."""

import contextlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from jhu_ta1.magnet._helm_pair_data import file_hash


@contextlib.contextmanager
def measure(stages, name):
    """Record exclusive stage wall/process CPU time when a mapping is supplied."""
    if stages is None:
        yield
        return
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        row = stages.setdefault(name, {"wall_seconds": 0.0, "process_cpu_seconds": 0.0})
        row["wall_seconds"] += time.perf_counter() - wall
        row["process_cpu_seconds"] += time.process_time() - cpu


def runtime_provenance():
    packages = {}
    for name in ("jhu_ta1", "dkps", "numpy", "scipy", "scikit-learn", "sentence-transformers"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"version": None}
            continue
        row = {"version": distribution.version}
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        # Retain commit identity without serializing potentially credentialed URLs.
        row["vcs_commit"] = direct.get("vcs_info", {}).get("commit_id")
        packages[name] = row
    modules = {}
    for name in ("jhu_ta1.magnet.pair_coverage", "jhu_ta1.magnet._measurements",
                 "jhu_ta1.magnet._helm_pair_data", "jhu_ta1.magnet._embedding_cache",
                 "dkps.dkps", "dkps.helm"):
        path = getattr(sys.modules.get(name), "__file__", None)
        if path and Path(path).is_file():
            modules[name] = {"file": path, "sha256": file_hash(path)}
    # `python -m ...pair_coverage` names the entry module __main__ at runtime.
    entry = Path(__file__).with_name("pair_coverage.py")
    modules["jhu_ta1.magnet.pair_coverage"] = {
        "file": str(entry), "sha256": file_hash(entry),
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "sources": modules,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


class ReplayMeasurements:
    def __init__(self, batch_stream=None):
        self.wall_start, self.cpu_start = time.perf_counter(), time.process_time()
        self.stages = {}
        self.batch_stream = batch_stream
        self.batch_count = 0
        self.requested_rows = 0
        self.unique_items = {}
        self.panel_models = {}

    def batch(self, panel, indices, embeddings, representation, stages, targets, budget, replicate):
        batch_id = f"batch-{self.batch_count}"
        self.batch_count += 1
        self.requested_rows += len(indices) * len(embeddings)
        self.unique_items.setdefault(panel.dataset, set()).update(map(int, indices))
        self.panel_models[panel.dataset] = len(embeddings)
        row = {
            "batch_id": batch_id, "dataset": panel.dataset, "n_eval": budget,
            "replicate": replicate, "query_ids": [panel.item_ids[i] for i in indices],
            "representation": representation,
            "models": list(embeddings), "num_target_fits": len(targets),
            "requested_response_rows": len(indices) * len(embeddings),
            "array_bytes": sum(array.nbytes for array in embeddings.values()),
            "array_shapes": {name: list(array.shape) for name, array in embeddings.items()},
            "stages": stages,
        }
        self.add_stages(stages)
        if self.batch_stream is not None:
            self.batch_stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.batch_stream.flush()
        return batch_id

    def add_stages(self, stages):
        for name, values in stages.items():
            row = self.stages.setdefault(name, dict.fromkeys(values, 0.0))
            for key, value in values.items():
                row[key] += value

    def summary(self):
        return {
            "schema_version": 1,
            "scope": "cached-response replay; all budgets, targets and repeated query draws",
            "elapsed_wall_seconds": time.perf_counter() - self.wall_start,
            "process_cpu_seconds": time.process_time() - self.cpu_start,
            "stages": self.stages,
            "representation_batches": self.batch_count,
            "representation_requested_rows": self.requested_rows,
            "representation_unique_response_rows": sum(
                len(items) * self.panel_models[dataset]
                for dataset, items in self.unique_items.items()),
            "acquisition": {
                "target_generation": "precomputed HELM responses",
                "target_grading": "precomputed HELM item scores",
                "original_generation_seconds": None, "original_grading_seconds": None,
                "reference_bank_acquisition_seconds": None,
                "embedding_precompute_seconds": None,
                "encoder_api_calls": None, "physical_disk_read_bytes": None,
            },
            "runtime": runtime_provenance(),
        }
