from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from scipy.interpolate import PchipInterpolator

from bat_analysis.spectrogram import compute_spectrogram
from annotation.snap import snap_plus45_display
from annotation.storage import AnnotationStore


@dataclass
class AppConfig:
    root: str
    annotations: str
    seed: int = 12345
    fmin_khz: float = 20.0
    fmax_khz: float = 180.0
    nperseg: int = 1024
    noverlap: int = 896
    db_floor: float = -90.0
    db_ceiling: float = 0.0
    snap_enabled: bool = True
    snap_half_length_px: int = 22
    snap_samples: int = 81


def discover_wavs(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")


def pchip_curve(points: List[Dict[str, float]], samples: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        return np.array([]), np.array([])

    pts = sorted(points, key=lambda p: p["t_ms"])
    x = np.array([p["t_ms"] for p in pts], dtype=float)
    y = np.array([p["f_khz"] for p in pts], dtype=float)

    # Keep the last point when several points share exactly the same time.
    last_for_time = {float(xv): i for i, xv in enumerate(x)}
    inds = np.array([last_for_time[v] for v in sorted(last_for_time)], dtype=int)
    x = x[inds]
    y = y[inds]
    if len(x) < 2:
        return np.array([]), np.array([])

    xx = np.linspace(x.min(), x.max(), samples)
    yy = PchipInterpolator(x, y, extrapolate=False)(xx)
    return xx, yy


def make_figure(
    spec: Dict[str, Any],
    chirps: List[Dict[str, Any]],
    active_chirp_id: Optional[int],
    db_floor: float,
    db_ceiling: float,
    uirevision: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=spec["times_ms"],
            y=spec["freqs_khz"],
            z=spec["db"],
            zmin=db_floor,
            zmax=db_ceiling,
            colorscale="Viridis",
            colorbar=dict(title="dB rel."),
            hovertemplate="t=%{x:.3f} ms<br>f=%{y:.2f} kHz<br>%{z:.1f} dB<extra></extra>",
        )
    )

    for chirp in chirps:
        pts = chirp.get("points", [])
        if not pts:
            continue
        is_active = chirp.get("chirp_id") == active_chirp_id
        xs = [p["t_ms"] for p in pts]
        ys = [p["f_khz"] for p in pts]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=10 if is_active else 8, symbol="circle-open" if is_active else "circle"),
                name=f"Chirp {chirp['chirp_id']} points",
                hovertemplate="t=%{x:.3f} ms<br>f=%{y:.2f} kHz<extra></extra>",
            )
        )
        if len(pts) >= 2:
            xx, yy = pchip_curve(pts)
            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yy,
                    mode="lines",
                    line=dict(width=3 if is_active else 2),
                    name=f"Chirp {chirp['chirp_id']} PCHIP",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        margin=dict(l=60, r=20, t=35, b=55),
        xaxis_title="Temps (ms)",
        yaxis_title="Fréquence (kHz)",
        clickmode="event+select",
        dragmode="zoom",
        uirevision=uirevision,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig


def build_app(config: AppConfig) -> Dash:
    root = Path(config.root).expanduser().resolve()
    annotation_path = Path(config.annotations).expanduser().resolve()
    wav_paths = discover_wavs(root)
    if not wav_paths:
        raise ValueError(f"No WAV files found recursively in: {root}")

    rel_paths = [str(p.relative_to(root)) for p in wav_paths]
    rng = random.Random(config.seed)
    order = rel_paths[:]
    rng.shuffle(order)

    store = AnnotationStore(annotation_path, root, config.seed)
    pending = [r for r in order if store.status(r) is None]
    processed = [r for r in order if store.status(r) is not None]
    queue = pending + processed
    spec_cache: Dict[str, Dict[str, Any]] = {}

    def load_spec(rel: str) -> Dict[str, Any]:
        if rel not in spec_cache:
            spec_cache[rel] = compute_spectrogram(
                root / rel,
                config.fmin_khz,
                config.fmax_khz,
                config.nperseg,
                config.noverlap,
            )
        return spec_cache[rel]

    def initial_file_state(index: int) -> Dict[str, Any]:
        rel = queue[index]
        rec = store.get(rel)
        chirps = rec.get("chirps", []) if rec else []
        next_id = max([c.get("chirp_id", 0) for c in chirps], default=0) + 1
        return {
            "index": index,
            "relative_path": rel,
            "chirps": chirps,
            "active_chirp_id": None,
            "next_chirp_id": next_id,
            "mode": "navigate",
            "message": "Choose 'New chirp' then click start and end points.",
            "last_view": {},
        }

    app = Dash(__name__)
    app.title = "Bat Chirp Annotator"

    app.layout = html.Div(
        [
            dcc.Store(id="session-state", data=initial_file_state(0)),
            html.Div(
                [html.Div(id="file-label", style={"fontWeight": 600}), html.Div(id="progress-label")],
                style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Button("New chirp", id="new-chirp", n_clicks=0),
                    html.Button("Add point", id="add-point", n_clicks=0),
                    html.Button("Move point", id="move-point", n_clicks=0),
                    html.Button("Delete point", id="delete-point", n_clicks=0),
                    html.Button("Undo", id="undo", n_clicks=0),
                    html.Button("Finish chirp", id="finish-chirp", n_clicks=0),
                    html.Button("Delete chirp", id="delete-chirp", n_clicks=0),
                ],
                style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "margin": "10px 0"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("dB floor"),
                            dcc.Input(id="db-floor", type="number", value=config.db_floor, step=1, debounce=True),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                    ),
                    dcc.Checklist(
                        id="snap-enabled",
                        options=[{"label": "Snap +45°", "value": "on"}],
                        value=["on"] if config.snap_enabled else [],
                        inline=True,
                    ),
                    html.Div(id="mode-label", style={"fontWeight": 600}),
                ],
                style={"display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap"},
            ),
            dcc.Graph(
                id="spectrogram",
                style={"height": "72vh"},
                config={"displaylogo": False, "scrollZoom": True},
            ),
            html.Div(id="status-message", style={"minHeight": "28px", "margin": "6px 0", "fontFamily": "monospace"}),
            html.Div(
                [
                    html.Button("Previous WAV", id="prev-wav", n_clicks=0),
                    html.Button("No chirp", id="no-chirp", n_clicks=0),
                    html.Button("Ignore / unusable", id="ignore", n_clicks=0),
                    html.Button("Validate & next", id="validate-next", n_clicks=0),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "8px"},
            ),
            html.Hr(),
            html.Div(
                [
                    html.Div(f"Root: {root}"),
                    html.Div(f"Annotations: {annotation_path}"),
                    html.Div(f"Seed: {config.seed}"),
                    html.Div("Use the Plotly modebar for zoom, pan, autoscale and reset axes."),
                ],
                style={"fontSize": "0.9rem", "opacity": 0.75},
            ),
        ],
        style={"maxWidth": "1500px", "margin": "0 auto", "padding": "12px"},
    )

    @app.callback(
        Output("spectrogram", "figure"),
        Output("file-label", "children"),
        Output("progress-label", "children"),
        Output("mode-label", "children"),
        Output("status-message", "children"),
        Input("session-state", "data"),
        Input("db-floor", "value"),
    )
    def render(state: Dict[str, Any], db_floor: float):
        rel = state["relative_path"]
        try:
            spec = load_spec(rel)
            floor = float(db_floor) if db_floor is not None else config.db_floor
            fig = make_figure(
                spec,
                state.get("chirps", []),
                state.get("active_chirp_id"),
                floor,
                config.db_ceiling,
                uirevision=rel,
            )
            info = f"{rel}  |  Fs={spec['sr']/1000:.1f} kHz  |  duration={spec['duration_ms']:.1f} ms"
        except Exception as exc:
            fig = go.Figure().update_layout(title=f"Error loading {rel}: {exc}")
            info = rel
        return (
            fig,
            info,
            f"{state['index'] + 1} / {len(queue)}",
            f"Mode: {state.get('mode', 'navigate')}",
            state.get("message", ""),
        )

    @app.callback(
        Output("session-state", "data", allow_duplicate=True),
        Input("spectrogram", "relayoutData"),
        State("session-state", "data"),
        prevent_initial_call=True,
    )
    def remember_view(relayout: Optional[Dict[str, Any]], state: Dict[str, Any]):
        if not relayout:
            return no_update
        state = dict(state)
        view = dict(state.get("last_view", {}))
        for key in ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]:
            if key in relayout:
                view[key] = relayout[key]
        state["last_view"] = view
        return state

    @app.callback(
        Output("session-state", "data"),
        Input("new-chirp", "n_clicks"),
        Input("add-point", "n_clicks"),
        Input("move-point", "n_clicks"),
        Input("delete-point", "n_clicks"),
        Input("undo", "n_clicks"),
        Input("finish-chirp", "n_clicks"),
        Input("delete-chirp", "n_clicks"),
        Input("spectrogram", "clickData"),
        Input("prev-wav", "n_clicks"),
        Input("no-chirp", "n_clicks"),
        Input("ignore", "n_clicks"),
        Input("validate-next", "n_clicks"),
        State("snap-enabled", "value"),
        State("db-floor", "value"),
        State("session-state", "data"),
        prevent_initial_call=True,
    )
    def controller(
        _new, _add, _move, _delete_point, _undo, _finish, _delete_chirp,
        click_data, _prev, _none, _ignore, _validate,
        snap_values, db_floor, state,
    ):
        trigger = ctx.triggered_id
        state = dict(state)
        chirps = [dict(c) for c in state.get("chirps", [])]
        state["chirps"] = chirps

        def active_index() -> Optional[int]:
            aid = state.get("active_chirp_id")
            for i, chirp in enumerate(chirps):
                if chirp.get("chirp_id") == aid:
                    return i
            return None

        def save_current(status: Optional[str] = None) -> None:
            rel = state["relative_path"]
            old = store.get(rel)
            store.set(
                rel,
                {
                    "status": status or old.get("status") or "in_progress",
                    "chirps": chirps,
                    "display": {
                        "db_floor": float(db_floor) if db_floor is not None else config.db_floor,
                        "fmin_khz": config.fmin_khz,
                        "fmax_khz": config.fmax_khz,
                    },
                },
            )

        def goto(index: int) -> Dict[str, Any]:
            return initial_file_state(max(0, min(len(queue) - 1, index)))

        if trigger == "new-chirp":
            cid = state.get("next_chirp_id", 1)
            chirps.append({"chirp_id": cid, "points": []})
            state.update(active_chirp_id=cid, next_chirp_id=cid + 1, mode="add_start_end")
            state["message"] = f"Chirp {cid}: click START, then END."
            save_current("in_progress")
            return state

        if trigger == "add-point":
            state["mode"] = "add_point" if active_index() is not None else "navigate"
            state["message"] = "Click where PCHIP deviates from the chirp." if active_index() is not None else "No active chirp."
            return state

        if trigger == "move-point":
            if active_index() is None:
                state["message"] = "No active chirp."
            else:
                state["mode"] = "move_select"
                state.pop("selected_point_index", None)
                state["message"] = "Click the point to move, then click its new position."
            return state

        if trigger == "delete-point":
            if active_index() is None:
                state["message"] = "No active chirp."
            else:
                state["mode"] = "delete_point"
                state["message"] = "Click the point to delete."
            return state

        if trigger == "undo":
            ai = active_index()
            if ai is not None and chirps[ai].get("points"):
                removed = chirps[ai]["points"].pop()
                state["message"] = f"Removed last point at {removed['t_ms']:.3f} ms / {removed['f_khz']:.2f} kHz."
                save_current("in_progress")
            else:
                state["message"] = "Nothing to undo."
            return state

        if trigger == "finish-chirp":
            ai = active_index()
            if ai is None:
                state["message"] = "No active chirp."
                return state
            if len(chirps[ai].get("points", [])) < 2:
                state["message"] = "A chirp needs at least START and END points."
                return state
            chirps[ai]["points"] = sorted(chirps[ai]["points"], key=lambda p: p["t_ms"])
            state["active_chirp_id"] = None
            state["mode"] = "navigate"
            state["message"] = f"Chirp {chirps[ai]['chirp_id']} finished."
            save_current("in_progress")
            return state

        if trigger == "delete-chirp":
            ai = active_index()
            if ai is not None:
                cid = chirps[ai]["chirp_id"]
                chirps.pop(ai)
                state["active_chirp_id"] = None
                state["mode"] = "navigate"
                state["message"] = f"Deleted chirp {cid}."
                save_current("in_progress")
            else:
                state["message"] = "No active chirp to delete."
            return state

        if trigger == "spectrogram":
            if not click_data or not click_data.get("points"):
                return no_update
            pt = click_data["points"][0]
            click_t = float(pt["x"])
            click_f = float(pt["y"])
            mode = state.get("mode", "navigate")
            ai = active_index()
            if ai is None or mode == "navigate":
                state["message"] = "Use 'New chirp' or an edit button before clicking the graph."
                return state

            if mode in {"move_select", "delete_point"}:
                points = chirps[ai].get("points", [])
                if not points:
                    state["message"] = "This chirp has no points."
                    return state
                spec = load_spec(state["relative_path"])
                t_span = max(spec["times_ms"][-1] - spec["times_ms"][0], 1e-9)
                f_span = max(spec["freqs_khz"][-1] - spec["freqs_khz"][0], 1e-9)
                d2 = [((p["t_ms"] - click_t) / t_span) ** 2 + ((p["f_khz"] - click_f) / f_span) ** 2 for p in points]
                pi = int(np.argmin(d2))
                if mode == "delete_point":
                    removed = points.pop(pi)
                    state["mode"] = "add_point"
                    state["message"] = f"Deleted point {removed['t_ms']:.3f} ms / {removed['f_khz']:.2f} kHz."
                    save_current("in_progress")
                    return state
                state["selected_point_index"] = pi
                state["mode"] = "move_place"
                state["message"] = f"Selected point #{pi + 1}. Click its new position."
                return state

            t_new, f_new, snap_db = click_t, click_f, float("nan")
            if "on" in (snap_values or []):
                view = state.get("last_view", {})
                xr = [float(view["xaxis.range[0]"]), float(view["xaxis.range[1]"])] if "xaxis.range[0]" in view and "xaxis.range[1]" in view else None
                yr = [float(view["yaxis.range[0]"]), float(view["yaxis.range[1]"])] if "yaxis.range[0]" in view and "yaxis.range[1]" in view else None
                t_new, f_new, snap_db = snap_plus45_display(
                    click_t,
                    click_f,
                    load_spec(state["relative_path"]),
                    xr,
                    yr,
                    config.snap_half_length_px,
                    config.snap_samples,
                )

            new_point = {
                "t_ms": round(t_new, 6),
                "f_khz": round(f_new, 6),
                "source": "snap+45" if "on" in (snap_values or []) else "manual",
            }
            if np.isfinite(snap_db):
                new_point["snap_db"] = round(float(snap_db), 3)

            if mode == "move_place":
                pi = state.pop("selected_point_index", None)
                if pi is not None and 0 <= pi < len(chirps[ai]["points"]):
                    chirps[ai]["points"][pi] = new_point
                    chirps[ai]["points"] = sorted(chirps[ai]["points"], key=lambda p: p["t_ms"])
                    state["mode"] = "add_point"
                    state["message"] = f"Point moved to {t_new:.3f} ms / {f_new:.2f} kHz."
                    save_current("in_progress")
                return state

            chirps[ai].setdefault("points", []).append(new_point)
            chirps[ai]["points"] = sorted(chirps[ai]["points"], key=lambda p: p["t_ms"])
            npts = len(chirps[ai]["points"])
            if mode == "add_start_end":
                if npts == 1:
                    state["message"] = f"START set at {t_new:.3f} ms / {f_new:.2f} kHz. Now click END."
                else:
                    state["mode"] = "add_point"
                    state["message"] = "START/END set. Add intermediate points only where PCHIP deviates, then Finish chirp."
            else:
                state["message"] = f"Point added: {t_new:.3f} ms / {f_new:.2f} kHz."
            save_current("in_progress")
            return state

        if trigger == "prev-wav":
            save_current()
            return goto(state["index"] - 1)

        if trigger == "no-chirp":
            state["chirps"] = []
            chirps = state["chirps"]
            save_current("no_chirp")
            return goto(state["index"] + 1) if state["index"] < len(queue) - 1 else state

        if trigger == "ignore":
            save_current("ignored")
            return goto(state["index"] + 1) if state["index"] < len(queue) - 1 else state

        if trigger == "validate-next":
            unfinished = [c for c in chirps if len(c.get("points", [])) < 2]
            if unfinished:
                state["message"] = "Cannot validate: at least one chirp has fewer than 2 points."
                return state
            if not chirps:
                state["message"] = "No chirp annotated. Use 'No chirp' for a true negative, or 'Ignore'."
                return state
            state["active_chirp_id"] = None
            state["mode"] = "navigate"
            save_current("annotated")
            return goto(state["index"] + 1) if state["index"] < len(queue) - 1 else state

        return state

    return app


def run_annotator(
    folder: str,
    annotations: str = "bat_chirp_annotations.json",
    seed: int = 12345,
    fmin_khz: float = 20.0,
    fmax_khz: float = 180.0,
    nperseg: int = 1024,
    noverlap: int = 896,
    db_floor: float = -90.0,
    port: int = 8050,
    debug: bool = False,
    jupyter_mode: str = "external",
) -> Dash:
    """Build and run the annotation app from a notebook or Python session."""
    cfg = AppConfig(
        root=folder,
        annotations=annotations,
        seed=seed,
        fmin_khz=fmin_khz,
        fmax_khz=fmax_khz,
        nperseg=nperseg,
        noverlap=noverlap,
        db_floor=db_floor,
    )
    app = build_app(cfg)
    app.run(debug=debug, port=port, jupyter_mode=jupyter_mode)
    return app
