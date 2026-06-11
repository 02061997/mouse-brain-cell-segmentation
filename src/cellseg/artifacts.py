from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def output_dir(smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path("artifacts") / f"{stamp}-{'smoke' if smoke else 'full'}"
    path.mkdir(parents=True)
    return path


def save(path: Path, value: object) -> None:
    def default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
