"""Wearable activity datasets as irregular forecasting records.

The original public datasets are activity-recognition datasets.  For the
HumanActivity-like forecasting direction we use their continuous sensor streams
and a fixed sensor-group asynchronous observation protocol.  This turns each
subject/session into records of the same shape used by the HumanActivity
provider: ``(record_id, tt, vals, mask)``.
"""

from __future__ import annotations

import io
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch


@dataclass(frozen=True)
class WearableSpec:
    name: str
    url: str
    zip_name: str
    feature_dim: int
    downsample: int
    group_slices: tuple[tuple[int, ...], ...]


UCI_BASE = "https://archive.ics.uci.edu/static/public"


SPECS: dict[str, WearableSpec] = {
    "MHEALTH": WearableSpec(
        name="MHEALTH",
        url=f"{UCI_BASE}/319/mhealth+dataset.zip",
        zip_name="mhealth+dataset.zip",
        feature_dim=23,
        downsample=2,
        group_slices=(
            tuple(range(0, 5)),      # chest acceleration + ECG
            tuple(range(5, 14)),     # left ankle IMU
            tuple(range(14, 23)),    # right wrist IMU
        ),
    ),
    "PAMAP2": WearableSpec(
        name="PAMAP2",
        url=f"{UCI_BASE}/231/pamap2+physical+activity+monitoring.zip",
        zip_name="pamap2+physical+activity+monitoring.zip",
        feature_dim=37,
        downsample=5,
        group_slices=(
            (0,),                    # heart rate, naturally lower-rate
            tuple(range(1, 13)),     # hand IMU
            tuple(range(13, 25)),    # chest IMU
            tuple(range(25, 37)),    # ankle IMU
        ),
    ),
    "OPPORTUNITY": WearableSpec(
        name="OPPORTUNITY",
        url=f"{UCI_BASE}/226/opportunity+activity+recognition.zip",
        zip_name="opportunity+activity+recognition.zip",
        feature_dim=64,
        downsample=3,
        group_slices=(
            tuple(range(0, 16)),
            tuple(range(16, 32)),
            tuple(range(32, 48)),
            tuple(range(48, 64)),
        ),
    ),
}


class WearableActivity:
    def __init__(self, root: str | os.PathLike[str], dataset: str, download: bool = True):
        self.root = Path(root)
        self.dataset = dataset
        if dataset not in SPECS:
            raise ValueError(f"unknown wearable dataset: {dataset}")
        self.spec = SPECS[dataset]
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(f"{dataset} not found under {self.root}")
        self.data = torch.load(self.processed_folder / "data.pt", map_location="cpu")

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> Path:
        return self.root / "raw"

    @property
    def processed_folder(self) -> Path:
        return self.root / "processed"

    def _check_exists(self) -> bool:
        return (self.processed_folder / "data.pt").exists()

    def download(self) -> None:
        if self._check_exists():
            return
        self.raw_folder.mkdir(parents=True, exist_ok=True)
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        zip_path = self.raw_folder / self.spec.zip_name
        if zip_path.exists() and not self._zip_is_valid(zip_path):
            zip_path.unlink()
        if not zip_path.exists():
            response = requests.get(self.spec.url, stream=True, timeout=120, verify=False)
            response.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        extract_dir = self.raw_folder / "extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        for nested_zip in extract_dir.rglob("*.zip"):
            nested_dir = nested_zip.with_suffix("")
            if not nested_dir.exists():
                nested_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(nested_zip, "r") as zf:
                    zf.extractall(nested_dir)
        records = self._build_records(extract_dir)
        torch.save(records, self.processed_folder / "data.pt")

    def _zip_is_valid(self, zip_path: Path) -> bool:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return zf.testzip() is None
        except zipfile.BadZipFile:
            return False

    def _build_records(self, extract_dir: Path):
        if self.dataset == "MHEALTH":
            arrays = self._load_mhealth(extract_dir)
        elif self.dataset == "PAMAP2":
            arrays = self._load_pamap2(extract_dir)
        elif self.dataset == "OPPORTUNITY":
            arrays = self._load_opportunity(extract_dir)
        else:
            raise ValueError(self.dataset)
        arrays = [(rid, arr[:: self.spec.downsample].astype(np.float32)) for rid, arr in arrays if len(arr) > 0]
        mean, std = self._global_standardizer([arr for _, arr in arrays])
        records = []
        for rid, arr in arrays:
            finite = np.isfinite(arr)
            vals = (np.where(finite, arr, mean) - mean) / std
            mask = finite.astype(np.float32)
            mask = self._apply_group_async_mask(mask)
            vals = np.where(mask > 0, vals, 0.0).astype(np.float32)
            tt = np.arange(vals.shape[0], dtype=np.float32)
            records.append((rid, torch.from_numpy(tt), torch.from_numpy(vals), torch.from_numpy(mask)))
        return records

    def _global_standardizer(self, arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros(self.spec.feature_dim, dtype=np.float64)
        sqs = np.zeros(self.spec.feature_dim, dtype=np.float64)
        counts = np.zeros(self.spec.feature_dim, dtype=np.float64)
        for arr in arrays:
            finite = np.isfinite(arr)
            safe = np.where(finite, arr, 0.0)
            sums += safe.sum(axis=0)
            sqs += (safe * safe).sum(axis=0)
            counts += finite.sum(axis=0)
        counts = np.maximum(counts, 1.0)
        mean = sums / counts
        var = np.maximum(sqs / counts - mean * mean, 1e-6)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    def _apply_group_async_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask
        async_mask = np.zeros_like(mask, dtype=np.float32)
        n_groups = len(self.spec.group_slices)
        for t in range(mask.shape[0]):
            group = t % n_groups
            cols = self.spec.group_slices[group]
            async_mask[t, list(cols)] = 1.0
        return mask.astype(np.float32) * async_mask

    def _load_mhealth(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        files = sorted(extract_dir.rglob("mHealth_subject*.log"))
        arrays = []
        for path in files:
            raw = np.loadtxt(path, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 24:
                continue
            # Last column is activity label; the first 23 are continuous sensors.
            arrays.append((path.stem, raw[:, :23]))
        return arrays

    def _load_pamap2(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        files = sorted(extract_dir.rglob("subject*.dat"))
        arrays = []
        feature_idx = [
            2,
            *range(4, 16),
            *range(21, 33),
            *range(38, 50),
        ]
        for path in files:
            raw = np.loadtxt(path, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 54:
                continue
            # Remove transient activity id 0, following common PAMAP2 practice.
            raw = raw[raw[:, 1] > 0]
            arrays.append((path.stem, raw[:, feature_idx]))
        return arrays

    def _load_opportunity(self, extract_dir: Path) -> list[tuple[str, np.ndarray]]:
        files = sorted(extract_dir.rglob("*.dat"))
        arrays = []
        for path in files:
            try:
                raw = np.loadtxt(path, dtype=np.float32)
            except Exception:
                continue
            if raw.ndim != 2 or raw.shape[1] < self.spec.feature_dim + 8:
                continue
            # OPPORTUNITY has labels in trailing columns.  Keep the first stable
            # sensor block after the timestamp-like first column.
            sensor = raw[:, 1 : 1 + self.spec.feature_dim]
            valid_rate = np.isfinite(sensor).mean(axis=0)
            keep = np.argsort(-valid_rate)[: self.spec.feature_dim]
            sensor = sensor[:, np.sort(keep)]
            arrays.append((path.stem, sensor))
        return arrays


def Wearable_time_chunk(data, configs):
    chunk_data = []
    history = int(configs.seq_len)
    pred_window = int(configs.pred_len)
    stride = max(history + pred_window, 1)
    sample_id = 0
    for record_id, tt, vals, mask in data:
        if len(tt) < history + pred_window + 1:
            continue
        t_max = int(tt.max().item())
        for st in range(0, max(t_max - history - pred_window, 1), stride):
            et_x = st + history
            et_y = st + history + pred_window
            idx_x = torch.where((tt >= st) & (tt < et_x))[0]
            idx_y = torch.where((tt >= et_x) & (tt < et_y))[0]
            if len(idx_x) < 2 or len(idx_y) < 1:
                continue
            if mask[idx_y].sum() <= 0:
                continue
            t_start = tt[idx_x][0]
            t_end = tt[idx_y][-1] + 1
            denom = max(float((t_end - t_start).item()), 1.0)
            chunk_data.append({
                "sample_ID": sample_id,
                "x_mark": (tt[idx_x] - t_start) / denom,
                "y_mark": (tt[idx_y] - t_start) / denom,
                "x": vals[idx_x],
                "y": vals[idx_y],
                "x_mask": mask[idx_x],
                "y_mask": mask[idx_y],
            })
            sample_id += 1
    return chunk_data
