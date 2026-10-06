from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Iterable


VALID_STATUSES = {"annotated", "no_chirp"}


@dataclass
class ExportStats:
    expected: int = 0
    copied: int = 0
    missing: int = 0
    duplicates: int = 0
    skipped_status: int = 0


def load_annotation_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_valid_records(annotation_data: dict) -> Iterable[tuple[str, dict]]:
    files = annotation_data.get("files", {}) or {}
    for relative_path, record in files.items():
        if record.get("status") in VALID_STATUSES:
            yield relative_path, record


def build_wav_index(source_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for wav_path in source_root.rglob("*.wav"):
        index.setdefault(wav_path.name.lower(), []).append(wav_path)
    return index


def resolve_source_wav(
    source_root: Path,
    relative_path: str,
    wav_index: dict[str, list[Path]],
) -> tuple[Path | None, bool]:
    rel = Path(relative_path)

    direct = source_root / rel
    if direct.exists() and direct.is_file():
        return direct, False

    candidates = wav_index.get(rel.name.lower(), [])
    if len(candidates) == 1:
        return candidates[0], False
    if len(candidates) > 1:
        return None, True

    return None, False


def safe_relative_destination(relative_path: str) -> Path:
    rel = Path(relative_path)
    parts = [p for p in rel.parts if p not in ("..", ".", rel.anchor)]
    if not parts:
        return Path(rel.name)
    return Path(*parts)


def choose_path_dialogs() -> tuple[Path | None, Path | None, Path | None]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    annotation_json = filedialog.askopenfilename(
        title="Sélectionner bat_chirp_annotations.json",
        filetypes=[("JSON", "*.json"), ("Tous les fichiers", "*.*")],
    )
    if not annotation_json:
        root.destroy()
        return None, None, None

    source_root = filedialog.askdirectory(
        title="Sélectionner le dossier contenant les WAV",
    )
    if not source_root:
        root.destroy()
        return None, None, None

    destination_root = filedialog.askdirectory(
        title="Sélectionner le dossier de destination",
    )
    root.destroy()

    if not destination_root:
        return None, None, None

    return Path(annotation_json), Path(source_root), Path(destination_root)


def export_dataset(
    annotation_json: Path,
    source_root: Path,
    destination_root: Path,
) -> ExportStats:
    annotation_json = annotation_json.resolve()
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()

    annotation_data = load_annotation_json(annotation_json)

    dataset_root = destination_root / "bat_benchmark_dataset"
    wav_destination_root = dataset_root / "wav"
    wav_destination_root.mkdir(parents=True, exist_ok=True)

    wav_index = build_wav_index(source_root)

    stats = ExportStats()
    report_lines: list[str] = []

    valid_records = list(iter_valid_records(annotation_data))
    stats.expected = len(valid_records)

    report_lines.append(f"Annotation JSON: {annotation_json}")
    report_lines.append(f"Source WAV root: {source_root}")
    report_lines.append(f"Destination: {dataset_root}")
    report_lines.append("")
    report_lines.append(f"Valid statuses: {sorted(VALID_STATUSES)}")
    report_lines.append(f"Expected WAV files: {stats.expected}")
    report_lines.append("")

    for relative_path, record in valid_records:
        src, duplicate = resolve_source_wav(
            source_root,
            relative_path,
            wav_index,
        )

        if duplicate:
            stats.duplicates += 1
            matches = wav_index.get(Path(relative_path).name.lower(), [])
            report_lines.append(f"DUPLICATE: {relative_path}")
            for match in matches:
                report_lines.append(f"  - {match}")
            continue

        if src is None:
            stats.missing += 1
            report_lines.append(f"MISSING: {relative_path}")
            continue

        rel_dest = safe_relative_destination(relative_path)
        dst = wav_destination_root / rel_dest
        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)
        stats.copied += 1

        report_lines.append(
            f"COPIED: {relative_path} -> {dst.relative_to(dataset_root)}"
        )

    json_destination = dataset_root / "bat_chirp_annotations.json"
    shutil.copy2(annotation_json, json_destination)

    report_lines.append("")
    report_lines.append("=== SUMMARY ===")
    report_lines.append(f"Expected:   {stats.expected}")
    report_lines.append(f"Copied:     {stats.copied}")
    report_lines.append(f"Missing:    {stats.missing}")
    report_lines.append(f"Duplicates: {stats.duplicates}")
    report_lines.append("")
    report_lines.append(f"Annotation JSON copied to: {json_destination.name}")

    report_path = dataset_root / "export_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export validated annotated/no_chirp WAV files referenced by "
            "bat_chirp_annotations.json into a self-contained benchmark dataset."
        )
    )
    parser.add_argument("--json", dest="annotation_json", type=Path)
    parser.add_argument("--source", dest="source_root", type=Path)
    parser.add_argument("--destination", dest="destination_root", type=Path)
    args = parser.parse_args()

    if args.annotation_json and args.source_root and args.destination_root:
        annotation_json = args.annotation_json
        source_root = args.source_root
        destination_root = args.destination_root
    else:
        annotation_json, source_root, destination_root = choose_path_dialogs()
        if not annotation_json or not source_root or not destination_root:
            print("Export annulé.")
            return 1

    stats = export_dataset(
        annotation_json=annotation_json,
        source_root=source_root,
        destination_root=destination_root,
    )

    print("")
    print("Export terminé")
    print(f"  attendus   : {stats.expected}")
    print(f"  copiés     : {stats.copied}")
    print(f"  manquants  : {stats.missing}")
    print(f"  doublons   : {stats.duplicates}")

    output_path = destination_root.resolve() / "bat_benchmark_dataset"
    print(f"  dossier    : {output_path}")

    return 0 if stats.missing == 0 and stats.duplicates == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
