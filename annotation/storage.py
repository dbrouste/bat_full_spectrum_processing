from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class AnnotationStore:
    """Persistent JSON store for manual chirp ground truth."""

    def __init__(self, path: Path | str, root: Path | str, seed: int):
        self.path = Path(path)
        self.root = Path(root)
        self.seed = int(seed)
        self.data: Dict[str, Any] = {
            "schema_version": 1,
            "root": str(self.root.resolve()),
            "units": {"time": "ms", "frequency": "kHz"},
            "seed": self.seed,
            "files": {},
        }

        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update(loaded)
                self.data.setdefault("files", {})
                self.data.setdefault("units", {"time": "ms", "frequency": "kHz"})
                self.data.setdefault("schema_version", 1)

        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def get(self, relative_path: str) -> Dict[str, Any]:
        return self.data["files"].get(relative_path, {})

    def set(self, relative_path: str, record: Dict[str, Any]) -> None:
        self.data["files"][relative_path] = record

        # Creating a new chirp should be instantaneous.  The controller stores an
        # in-progress chirp before it contains its first control point; there is
        # no useful ground truth to persist yet, so defer the disk write until a
        # point exists (or until the status becomes something other than
        # in_progress).  This also prevents a filesystem write from blocking the
        # New chirp UI transition.
        chirps = record.get("chirps", [])
        empty_in_progress = (
            record.get("status") == "in_progress"
            and bool(chirps)
            and all(not chirp.get("points") for chirp in chirps)
        )
        if not empty_in_progress:
            self.save()

    def status(self, relative_path: str) -> Optional[str]:
        return self.get(relative_path).get("status")
