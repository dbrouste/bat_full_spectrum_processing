# Bat Full Spectrum Processing

Python tools for full-spectrum bat call processing, annotation, modelling, feature extraction, and benchmarking.

## Architecture

```text
bat_analysis/
    detection.py
    ridge.py
    modelling.py
    features.py
    spectrogram.py

annotation/
    app.py
    snap.py
    storage.py

benchmark/
    detection_metrics.py
    curve_metrics.py

notebooks/
    annotation.ipynb
```

The current development focus is the manual annotation tool used to build a ground-truth dataset for chirp detection and curve reconstruction.

## Install

```bash
pip install -r requirements.txt
```

## Run from Jupyter

```python
%load_ext autoreload
%autoreload 2

from annotation import run_annotator

WAV_FOLDER = r"D:\Bat\WAV"
ANNOTATIONS_FILE = r"D:\Bat\bat_chirp_annotations.json"

app = run_annotator(
    folder=WAV_FOLDER,
    annotations=ANNOTATIONS_FILE,
    fmin_khz=20,
    fmax_khz=180,
    db_floor=-90,
    jupyter_mode="external",
)
```

The WAV folder is searched recursively. Files are presented in a deterministic random order. Annotation state is saved after every edit so sessions can be resumed.

### Annotation workflow

1. Click **New chirp**.
2. Click the chirp start point and end point.
3. A PCHIP curve is displayed between the points.
4. Use **Add point** only where the PCHIP curve departs from the visible ridge.
5. Optional **Snap +45°** moves a click to the strongest spectrogram value along a +45° screen-space line.
6. Click **Finish chirp**.
7. Add more chirps if necessary, then **Validate & next**.
8. Use **No chirp** for true negative files and **Ignore / unusable** for files that should not enter the benchmark.
