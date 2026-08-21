from __future__ import annotations

from pathlib import Path
from typing import Optional

from dash import Dash

from .app_v2 import run_annotator


def choose_wav_folder(initial_dir: Optional[str] = None) -> Optional[str]:
    """Open a native Windows folder chooser and return the selected folder.

    Returns None if the user cancels the dialog.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            title="Select folder containing WAV recordings",
            initialdir=initial_dir or str(Path.home()),
            mustexist=True,
        )
    finally:
        root.destroy()

    return selected or None


def run_annotator_dialog(
    initial_dir: Optional[str] = None,
    annotation_filename: str = "bat_chirp_annotations.json",
    **kwargs,
) -> Optional[Dash]:
    """Select a WAV root folder with a native dialog, then launch the annotator.

    The annotation JSON is stored in the selected WAV root folder by default.
    Extra keyword arguments are forwarded to ``run_annotator``.
    """
    folder = choose_wav_folder(initial_dir=initial_dir)
    if folder is None:
        print("Folder selection cancelled.")
        return None

    annotation_path = Path(folder) / annotation_filename
    print(f"WAV root: {folder}")
    print(f"Annotations: {annotation_path}")

    return run_annotator(
        folder=folder,
        annotations=str(annotation_path),
        **kwargs,
    )
