from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _bilinear_sample(z: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    x1 = np.clip(x0 + 1, 0, z.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, z.shape[0] - 1)
    x0 = np.clip(x0, 0, z.shape[1] - 1)
    y0 = np.clip(y0, 0, z.shape[0] - 1)
    wx = xi - x0
    wy = yi - y0
    return (
        z[y0, x0] * (1 - wx) * (1 - wy)
        + z[y0, x1] * wx * (1 - wy)
        + z[y1, x0] * (1 - wx) * wy
        + z[y1, x1] * wx * wy
    )


def snap_plus45_display(
    click_t_ms: float,
    click_f_khz: float,
    spec: Dict[str, Any],
    x_range: Optional[List[float]],
    y_range: Optional[List[float]],
    half_length_px: int = 22,
    samples: int = 81,
    plot_width_px: int = 950,
    plot_height_px: int = 620,
) -> Tuple[float, float, float]:
    """Snap a click to the spectrogram maximum along a +45° screen-space line.

    The line is +45° in DISPLAY coordinates, not in kHz/ms. This makes the snap
    geometrically consistent after zooming because the current axis ranges are
    converted to pixel-to-data scale factors.
    """
    times = np.asarray(spec["times_ms"])
    freqs = np.asarray(spec["freqs_khz"])
    db = np.asarray(spec["db"])

    if x_range is None:
        x_range = [float(times[0]), float(times[-1])]
    if y_range is None:
        y_range = [float(freqs[0]), float(freqs[-1])]

    x_span = max(abs(float(x_range[1]) - float(x_range[0])), 1e-12)
    y_span = max(abs(float(y_range[1]) - float(y_range[0])), 1e-12)
    dx_per_px = x_span / max(plot_width_px, 1)
    dy_per_px = y_span / max(plot_height_px, 1)

    u = np.linspace(-half_length_px, half_length_px, int(samples))
    cand_t = click_t_ms + u * dx_per_px
    cand_f = click_f_khz + u * dy_per_px

    valid = (
        (cand_t >= times[0]) & (cand_t <= times[-1])
        & (cand_f >= freqs[0]) & (cand_f <= freqs[-1])
        & (cand_t >= min(x_range)) & (cand_t <= max(x_range))
        & (cand_f >= min(y_range)) & (cand_f <= max(y_range))
    )
    if not np.any(valid):
        return float(click_t_ms), float(click_f_khz), float("nan")

    cand_t = cand_t[valid]
    cand_f = cand_f[valid]
    xi = np.interp(cand_t, times, np.arange(len(times), dtype=float))
    yi = np.interp(cand_f, freqs, np.arange(len(freqs), dtype=float))
    vals = _bilinear_sample(db, xi, yi)
    k = int(np.nanargmax(vals))
    return float(cand_t[k]), float(cand_f[k]), float(vals[k])
