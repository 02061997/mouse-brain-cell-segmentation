from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def output_dir(smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path("artifacts") / f"{stamp}-{'smoke' if smoke else 'full'}"
    path.mkdir(parents=True)
    return path


def save(path: Path, value: object) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [clean(item) for item in obj]
        if isinstance(obj, tuple):
            return [clean(item) for item in obj]
        if isinstance(obj, (np.integer, np.floating)):
            obj = obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        return obj

    def default(obj):
        raise TypeError(type(obj).__name__)

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, default=default, allow_nan=False) + "\n"
    )


def publish_latest(out: Path) -> None:
    latest = Path("reports/latest")
    latest.mkdir(parents=True, exist_ok=True)
    for existing in latest.iterdir():
        if existing.is_file():
            existing.unlink()
    for source in out.iterdir():
        if source.is_file():
            shutil.copy2(source, latest / source.name)
    Path("reports/SOURCE_RUN.txt").write_text(f"{out}\n")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
