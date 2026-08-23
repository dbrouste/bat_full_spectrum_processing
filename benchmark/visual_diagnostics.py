from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
from scipy.signal import stft


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return y, int(sr)


def _load_annotations(annotation_json: str | Path) -> tuple[dict[str, Any], Path]:
    annotation_json = Path(annotation_json).expanduser().resolve()
    with annotation_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    root = Path(data.get("root", annotation_json.parent)).expanduser().resolve()
    return data, root


def build_error_cases(result) -> pd.DataFrame:
    """Return one row per false negative or false positive.

    FN rows come from the manual chirp table. FP rows come from unmatched detector
    candidates. ``case_id`` is stable within one benchmark result and can be passed
    directly to :func:`plot_error_case`.
    """
    rows: list[dict[str, Any]] = []

    chirps = result.chirps.copy()
    if not chirps.empty:
        fn = chirps[chirps["detected"] == False].copy()
        for _, r in fn.iterrows():
            center = (float(r["manual_start_ms"]) + float(r["manual_end_ms"])) / 2.0
            rows.append({
                "case_type": "FN",
                "relative_path": r["relative_path"],
                "chirp_id": r.get("chirp_id"),
                "detected_index": np.nan,
                "center_ms": center,
                "start_ms": float(r["manual_start_ms"]),
                "end_ms": float(r["manual_end_ms"]),
                "duration_ms": float(r["manual_duration_ms"]),
                "peak_freq_khz": np.nan,
                "peak_db": np.nan,
            })

    detections = result.detections.copy()
    if not detections.empty:
        fp = detections[detections["matched"] == False].copy()
        for _, r in fp.iterrows():
            rows.append({
                "case_type": "FP",
                "relative_path": r["relative_path"],
                "chirp_id": np.nan,
                "detected_index": int(r["detected_index"]),
                "center_ms": float(r["time_mid_ms"]),
                "start_ms": float(r["t_start_ms"]),
                "end_ms": float(r["t_end_ms"]),
                "duration_ms": float(r["duration_ms"]),
                "peak_freq_khz": float(r.get("peak_freq_khz", np.nan)),
                "peak_db": float(r.get("peak_db", np.nan)),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["case_type", "relative_path", "center_ms"]).reset_index(drop=True)
    out.insert(0, "case_id", [f"{t}{i+1:03d}" for i, t in enumerate(out["case_type"])])
    return out


def _manual_points(annotation_data: dict[str, Any], relative_path: str, chirp_id: Any) -> tuple[np.ndarray, np.ndarray]:
    record = (annotation_data.get("files", {}) or {}).get(relative_path, {})
    for chirp in record.get("chirps", []) or []:
        if str(chirp.get("chirp_id")) == str(chirp_id):
            pts = chirp.get("points", []) or []
            if not pts:
                return np.array([]), np.array([])
            pts = sorted(pts, key=lambda p: float(p["t_ms"]))
            return (
                np.asarray([float(p["t_ms"]) for p in pts]),
                np.asarray([float(p["f_khz"]) for p in pts]),
            )
    return np.array([]), np.array([])


def _spectrogram_window(
    wav_path: Path,
    center_ms: float,
    window_ms: float,
    *,
    n_fft: int = 512,
    hop: int = 64,
    fmin_khz: float = 20.0,
    fmax_khz: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, sr = _read_mono(wav_path)
    half_s = window_ms / 2000.0
    center_s = center_ms / 1000.0
    start_s = max(0.0, center_s - half_s)
    end_s = min(len(y) / sr, center_s + half_s)
    i0 = int(start_s * sr)
    i1 = int(end_s * sr)
    chunk = y[i0:i1]
    if len(chunk) < n_fft:
        chunk = np.pad(chunk, (0, max(0, n_fft - len(chunk))))

    noverlap = max(0, n_fft - hop)
    freqs, times, Z = stft(
        chunk,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(Z) ** 2
    ref = max(float(np.max(power)), np.finfo(float).tiny)
    db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / ref)
    mask = (freqs >= fmin_khz * 1000.0) & (freqs <= fmax_khz * 1000.0)
    return (start_s * 1000.0 + times * 1000.0, freqs[mask] / 1000.0, db[mask])


def plot_error_case(
    result,
    annotation_json: str | Path,
    case: int | str | pd.Series,
    *,
    cases: pd.DataFrame | None = None,
    window_ms: float = 40.0,
    fmin_khz: float = 20.0,
    fmax_khz: float = 150.0,
    db_floor: float = -90.0,
    show_nearby_manual: bool = True,
    show_nearby_detections: bool = True,
) -> go.Figure:
    """Plot one FP/FN with nearby manual chirps and detector candidates.

    Parameters
    ----------
    case:
        Integer row position, ``case_id`` string (for example ``FN001``), or a
        row returned by :func:`build_error_cases`.
    window_ms:
        Total horizontal time window around the error center.
    """
    if cases is None:
        cases = build_error_cases(result)
    if cases.empty:
        raise ValueError("No FP/FN cases in this benchmark result")

    if isinstance(case, pd.Series):
        row = case
    elif isinstance(case, str):
        m = cases[cases["case_id"] == case]
        if m.empty:
            raise KeyError(f"Unknown case_id: {case}")
        row = m.iloc[0]
    else:
        row = cases.iloc[int(case)]

    annotation_data, wav_root = _load_annotations(annotation_json)
    rel = str(row["relative_path"])
    wav_path = wav_root / rel
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    center_ms = float(row["center_ms"])
    x, y, z = _spectrogram_window(
        wav_path,
        center_ms,
        window_ms,
        fmin_khz=fmin_khz,
        fmax_khz=fmax_khz,
    )

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x,
        y=y,
        z=z,
        zmin=db_floor,
        zmax=0,
        colorscale="Viridis",
        colorbar=dict(title="dB rel."),
        hovertemplate="t=%{x:.3f} ms<br>f=%{y:.2f} kHz<br>%{z:.1f} dB<extra></extra>",
    ))

    half = window_ms / 2.0
    x0, x1 = center_ms - half, center_ms + half

    # Manual annotations in/near this window.
    if show_nearby_manual:
        record = (annotation_data.get("files", {}) or {}).get(rel, {})
        for chirp in record.get("chirps", []) or []:
            pts = chirp.get("points", []) or []
            if len(pts) < 2:
                continue
            pts = sorted(pts, key=lambda p: float(p["t_ms"]))
            t = np.asarray([float(p["t_ms"]) for p in pts])
            f = np.asarray([float(p["f_khz"]) for p in pts])
            if t[-1] < x0 or t[0] > x1:
                continue
            target = row["case_type"] == "FN" and str(chirp.get("chirp_id")) == str(row["chirp_id"])
            fig.add_trace(go.Scatter(
                x=t,
                y=f,
                mode="lines+markers",
                name=(f"FN annotation chirp {chirp.get('chirp_id')}" if target else f"manual chirp {chirp.get('chirp_id')}"),
                line=dict(width=4 if target else 2, color="red" if target else "orange"),
                marker=dict(size=7 if target else 5, color="red" if target else "orange"),
                hovertemplate="manual chirp " + str(chirp.get("chirp_id")) + "<br>t=%{x:.3f} ms<br>f=%{y:.2f} kHz<extra></extra>",
            ))

    # All detector candidates in/near the window. FP target is emphasized.
    if show_nearby_detections:
        dets = result.detections[result.detections["relative_path"] == rel].copy()
        dets = dets[(dets["time_mid_ms"] >= x0) & (dets["time_mid_ms"] <= x1)]
        for _, det in dets.iterrows():
            target = (
                row["case_type"] == "FP"
                and int(det["detected_index"]) == int(row["detected_index"])
            )
            color = "cyan" if target else "white"
            width = 3 if target else 1
            fig.add_vrect(
                x0=float(det["t_start_ms"]),
                x1=float(det["t_end_ms"]),
                line_width=width,
                line_color=color,
                fillcolor=color,
                opacity=0.12 if target else 0.04,
                layer="above",
            )
            pf = float(det.get("peak_freq_khz", np.nan))
            if np.isfinite(pf):
                fig.add_trace(go.Scatter(
                    x=[float(det["time_mid_ms"])],
                    y=[pf],
                    mode="markers",
                    name=(f"FP detection {int(det['detected_index'])}" if target else f"detection {int(det['detected_index'])}"),
                    marker=dict(symbol="x", size=13 if target else 9, color=color, line=dict(width=2, color=color)),
                    hovertemplate=(
                        f"detection {int(det['detected_index'])}<br>"
                        + "t=%{x:.3f} ms<br>f=%{y:.2f} kHz"
                        + (f"<br>peak={float(det.get('peak_db', np.nan)):.1f} dB" if np.isfinite(float(det.get('peak_db', np.nan))) else "")
                        + "<extra></extra>"
                    ),
                ))

    case_type = str(row["case_type"])
    descriptor = (
        f"manual chirp {row['chirp_id']}"
        if case_type == "FN"
        else f"detection {int(row['detected_index'])}"
    )
    fig.update_layout(
        title=f"{row['case_id']} — {case_type} — {descriptor}<br><sup>{rel}</sup>",
        xaxis_title="Time (ms)",
        yaxis_title="Frequency (kHz)",
        xaxis=dict(range=[x0, x1]),
        yaxis=dict(range=[fmin_khz, fmax_khz]),
        height=650,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=30, t=100, b=60),
    )
    return fig


def error_case_summary(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(columns=["case_type", "count"])
    return cases.groupby("case_type").size().rename("count").reset_index()
