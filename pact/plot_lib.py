#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:05:27 2026

@author: avicenna
"""

import json

import numpy as np
import pandas as pd
try:
  import plotly.colors as pc
  import plotly.graph_objects as go
  _PLOTLY_EXISTS = True
except ModuleNotFoundError:
  _PLOTLY_EXISTS = False
  print("Warning: plotly not found — plotting functionalities will not be available.")
try:
  from rpy2.robjects.packages import PackageNotInstalledError
  import PyRacmacs as pr
  _PR_EXISTS=True
except (ModuleNotFoundError, PackageNotInstalledError):
  _PR_EXISTS=False
  print("Warning PyRacmacs not found will not be able to use its plotting functionalities.")
from copy import deepcopy


_SHAPE_MAP = {
    "circle":   "circle",
    "box":      "square",
    "square":   "square",
    "triangle": "triangle-up",
    "diamond":  "diamond",
}

# Diverging blue-white-red scale matching _avidity_to_hex
_AV_SCALE = [
    [0.0, "rgb(0,85,170)"],
    [0.5, "rgb(255,255,255)"],
    [1.0, "rgb(204,0,0)"],
]

# JavaScript templates for click-to-show-connections.
# The figure has exactly three traces: [0] antigens, [1] sera, [2] connection.
# On click, segment data is swapped into trace 2 via Plotly.restyle — no
# per-point hidden traces, so render cost is O(1) regardless of map size.
# Placeholders PYEDM_DIV_ID, PYEDM_AG_SEGS, PYEDM_SR_SEGS are replaced at
# call time (string .replace, not .format, to avoid JS brace conflicts).
_CLICK_JS_2D = """\
(function() {
    var gd = document.getElementById('PYEDM_DIV_ID');
    if (!gd) return;
    var lastClicked = null;
    var agSegs = PYEDM_AG_SEGS;
    var srSegs = PYEDM_SR_SEGS;

    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var ti = pt.curveNumber;
        var pi = pt.pointIndex;
        var segs;
        if      (ti === 0) { segs = agSegs[pi]; }
        else if (ti === 1) { segs = srSegs[pi]; }
        else { return; }

        var key = ti + ',' + pi;
        if (key === lastClicked) {
            Plotly.restyle(gd, {x: [[]], y: [[]], visible: false}, [2]);
            Plotly.restyle(gd, {x: [[]], y: [[]], text: [[]], visible: false}, [3]);
            lastClicked = null;
        } else {
            Plotly.restyle(gd, {x: [segs.x], y: [segs.y], visible: true}, [2]);
            Plotly.restyle(gd, {x: [segs.mx], y: [segs.my], text: [segs.text], visible: true}, [3]);
            lastClicked = key;
        }
    });
})();
"""

_CLICK_JS_3D = """\
(function() {
    var gd = document.getElementById('PYEDM_DIV_ID');
    if (!gd) return;
    var lastClicked = null;
    var agSegs = PYEDM_AG_SEGS;
    var srSegs = PYEDM_SR_SEGS;

    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var ti = pt.curveNumber;
        var pi = pt.pointIndex;
        var segs;
        if      (ti === 0) { segs = agSegs[pi]; }
        else if (ti === 1) { segs = srSegs[pi]; }
        else { return; }

        var key = ti + ',' + pi;
        if (key === lastClicked) {
            Plotly.restyle(gd, {x: [[]], y: [[]], z: [[]], visible: false}, [2]);
            Plotly.restyle(gd, {x: [[]], y: [[]], z: [[]], text: [[]], visible: false}, [3]);
            lastClicked = null;
        } else {
            Plotly.restyle(gd, {x: [segs.x], y: [segs.y], z: [segs.z], visible: true}, [2]);
            Plotly.restyle(gd, {x: [segs.mx], y: [segs.my], z: [segs.mz], text: [segs.text], visible: true}, [3]);
            lastClicked = key;
        }
    });
})();
"""

# Removes body margins and resizes the Plotly div to fill the browser viewport.
# Also re-fires on window resize so the plot stays full-page after resizing.
_FULL_PAGE_JS = """\
(function() {
    var gd = document.getElementById('PYEDM_DIV_ID');
    var s = document.createElement('style');
    s.textContent = 'body,html{margin:0;padding:0;overflow:hidden;}';
    document.head.appendChild(s);
    function resize() {
        Plotly.relayout(gd, {width: window.innerWidth * 0.95, height: window.innerHeight});
    }
    resize();
    window.addEventListener('resize', resize);
})();
"""


def _avidity_to_hex(values):
    """
    Map floats to a blue-white-red hex palette centred at 0.
    NaN → grey (#808080).
    """
    vals = np.asarray(values, dtype=float)
    vmax = np.nanmax(np.abs(vals))
    if vmax == 0 or np.isnan(vmax):
        return ["#808080"] * len(vals)
    colors = []
    for v in vals:
        if np.isnan(v):
            colors.append("#808080")
            continue
        t = float(np.clip(v / vmax, -1.0, 1.0))
        if t <= 0:
            f = -t
            r = int(255 * (1 - f))
            g = int(255 * (1 - f) + 85  * f)
            b = int(255 * (1 - f) + 170 * f)
        else:
            f = t
            r = int(255 * (1 - f) + 204 * f)
            g = int(255 * (1 - f))
            b = int(255 * (1 - f))
        colors.append(f"#{r:02X}{g:02X}{b:02X}")
    return colors


def _year_to_hex(years, yr_min, yr_max):
    """Map a list of year values to Viridis hex colors; transparent for None/NaN."""
    scale = yr_max - yr_min if yr_max > yr_min else 1.0
    colors = []
    for year in years:
        if year is None or year != year:   # year != year catches NaN
            colors.append("rgba(0,0,0,0)")
            continue
        t = max(0.0, min(1.0, (year - yr_min) / scale))
        rgb = pc.sample_colorscale('Viridis', [t])[0]   # 'rgb(r, g, b)'
        r, g, b = [int(x) for x in rgb[4:-1].split(',')]
        colors.append(f"#{r:02X}{g:02X}{b:02X}")
    return colors


def _scalar_avidities(av_dict, names):
    """
    For each name, extract one representative avidity scalar:
    the value with the largest absolute value across all IDs.
    Returns a list of floats (NaN when no data).
    """
    result = []
    for name in names:
        vals = [v for v in av_dict.get(name, {}).values()
                if v is not None and not np.isnan(v)]
        result.append(max(vals, key=abs) if vals else np.nan)
    return result


def _connection_data(m, ag_c, sr_c, is_3d):
    """
    Build per-point segment lookup for JS injection.

    Returns (ag_segs, sr_segs): lists of dicts with keys 'x', 'y' (and 'z'
    for 3D) — NaN-separated flat coordinate lists — plus 'mx'/'my'/'mz'
    midpoint lists and 'text' rounded-mean-titer labels per connection.
    NaN is encoded as None so json.dumps produces valid JSON;
    Plotly.js treats null as a line break identical to NaN.
    """
    table  = m.processed_table
    ag_pos = {name: i for i, name in enumerate(m.ag_names)}
    sr_pos = {name: i for i, name in enumerate(m.sr_names)}

    # Mean titer and mean distance per (antigen, serum) across non-unmeasured rows
    t_all = table.loc[table['distance_type'] != 1,
                      ['antigen', 'serum', 'titer', 'target_distances']].copy()
    t_all['titer_num'] = pd.to_numeric(
        t_all['titer'].astype(str).str.lstrip('<>'), errors='coerce')
    grouped = t_all.groupby(['antigen', 'serum'])[['titer_num', 'target_distances']].mean()
    mean_titer_map = {}
    for (ag, sr), row in grouped.iterrows():
        t_val, d_val = row['titer_num'], row['target_distances']
        if t_val == t_val and d_val == d_val:
            mean_titer_map[(ag, sr)] = f"{round(t_val)} ({round(d_val, 1)})"
        elif t_val == t_val:
            mean_titer_map[(ag, sr)] = str(round(t_val))
        else:
            mean_titer_map[(ag, sr)] = ""

    t = (table.loc[table['distance_type'] != 1, ['antigen', 'serum']]
         .drop_duplicates().copy())
    t['ai'] = t['antigen'].map(ag_pos)
    t['si'] = t['serum'].map(sr_pos)
    t = t.dropna(subset=['ai', 'si'])
    ai = t['ai'].to_numpy(dtype=int)
    si = t['si'].to_numpy(dtype=int)
    ag_names_arr = t['antigen'].to_numpy()
    sr_names_arr = t['serum'].to_numpy()

    nans = np.full(len(ai), np.nan)
    all_x = np.column_stack([ag_c[ai, 0], sr_c[si, 0], nans]).ravel()
    all_y = np.column_stack([ag_c[ai, 1], sr_c[si, 1], nans]).ravel()
    all_z = (np.column_stack([ag_c[ai, 2], sr_c[si, 2], nans]).ravel()
             if is_3d else None)

    mid_x = (ag_c[ai, 0] + sr_c[si, 0]) / 2
    mid_y = (ag_c[ai, 1] + sr_c[si, 1]) / 2
    mid_z = ((ag_c[ai, 2] + sr_c[si, 2]) / 2 if is_3d else None)

    titer_labels = np.array(
        [mean_titer_map.get((ag, sr), "") for ag, sr in zip(ag_names_arr, sr_names_arr)]
    )

    offsets = np.array([0, 1, 2])

    def _to_js(arr):
        # NaN is not valid JSON; null is accepted by Plotly.js as a line break
        return [None if v != v else v for v in arr.tolist()]

    def _build_segs(point_indices, n_points):
        segs = []
        for k in range(n_points):
            row_hits = np.where(point_indices == k)[0]
            if len(row_hits):
                seg_idx = (row_hits[:, None] * 3 + offsets).ravel()
                seg = {
                    'x': _to_js(all_x[seg_idx]),
                    'y': _to_js(all_y[seg_idx]),
                    'mx': mid_x[row_hits].tolist(),
                    'my': mid_y[row_hits].tolist(),
                    'text': titer_labels[row_hits].tolist(),
                }
                if is_3d:
                    seg['z'] = _to_js(all_z[seg_idx])
                    seg['mz'] = mid_z[row_hits].tolist()
            else:
                seg = {'x': [], 'y': [], 'mx': [], 'my': [], 'text': [],
                       **({'z': [], 'mz': []} if is_3d else {})}
            segs.append(seg)
        return segs

    return _build_segs(ai, len(m.ag_names)), _build_segs(si, len(m.sr_names))


def _build_figure(m, optim=0):
    if not _PLOTLY_EXISTS:
        raise RuntimeError("Can't use plotting functions without the plotly package.")
    if m.n_optimizations == 0:
        raise ValueError("Map has no optimizations to plot.")
    if m.dim not in (2, 3):
        raise ValueError(f"Map.dim must be 2 or 3; got {m.dim}.")

    is_3d = (m.dim == 3)
    ag_c  = m.ag_coords(optim)
    sr_c  = m.sr_coords(optim)

    def _scatter(coords, names, fill, outline, size, shape, zorder=0):
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2] if is_3d else None
        kw = dict(
            mode="markers",
            showlegend=False,
            text=list(names),
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                color=list(fill),
                size=[s * 3 for s in size],
                symbol=[_SHAPE_MAP.get(sh, "circle") for sh in shape],
                line=dict(color=list(outline), width=1.5),
            ),
        )
        if is_3d:
            return go.Scatter3d(x=x, y=y, z=z, **kw)
        return go.Scatter(x=x, y=y, zorder=zorder, **kw)

    ag_trace = _scatter(ag_c, m.ag_names, m.ag_fill, m.ag_outline,
                        m.ag_size, m.ag_shape, zorder=m.ag_zorder)
    sr_trace = _scatter(sr_c, m.sr_names, m.sr_fill, m.sr_outline,
                        m.sr_size, m.sr_shape, zorder=m.sr_zorder)

    # Single empty connection trace; segment data is injected by write_html()
    _conn_kw = dict(mode="lines", line=dict(color="#888888", width=1),
                    visible=False, showlegend=False, hoverinfo="skip")
    conn = (go.Scatter3d(x=[], y=[], z=[], **_conn_kw) if is_3d
            else go.Scatter(x=[], y=[], **_conn_kw))

    # Empty text label trace for titer values at connection midpoints
    _txt_kw = dict(mode="text", text=[], textposition="middle center",
                   visible=False, showlegend=False, hoverinfo="skip",
                   textfont=dict(size=10, color="#000000"))
    txt_trace = (go.Scatter3d(x=[], y=[], z=[], **_txt_kw) if is_3d
                 else go.Scatter(x=[], y=[], **_txt_kw))

    # --- Coloring dropdown ---
    # cb_options: list of (colorbar_trace, button_label, ag_colors, sr_colors)
    # Colorbar traces are appended after [ag, sr, conn, txt] at indices 4, 5, ...
    cb_options = []
    _cb_kw = dict(mode='markers', showlegend=False, hoverinfo='skip', visible=False)

    has_row = len(m.row_avidities) > 0
    has_col = len(m.col_avidities) > 0

    if has_row or has_col:
        raw_ag = (_scalar_avidities(m.ag_avidities(optim=optim), m.ag_names)
                  if has_row else None)
        raw_sr = (_scalar_avidities(m.sr_avidities(optim=optim), m.sr_names)
                  if has_col else None)
        ag_av = _avidity_to_hex(raw_ag) if raw_ag is not None else ["#FFFFFF"] * m.n_antigens
        sr_av = _avidity_to_hex(raw_sr) if raw_sr is not None else ["#FFFFFF"] * m.n_sera
        all_raw = (raw_ag or []) + (raw_sr or [])
        finite  = [v for v in all_raw if v is not None and not np.isnan(v)]
        vmax    = max(abs(v) for v in finite) if finite else 1.0
        _av_marker = dict(color=[0], colorscale=_AV_SCALE, cmin=-vmax, cmax=vmax,
                          showscale=False, colorbar=dict(title="Avidity", thickness=15, len=0.75,
                                                         x=1.02, xanchor="left"))
        av_cb = (go.Scatter3d(x=[None], y=[None], z=[None], marker=_av_marker, **_cb_kw)
                 if is_3d else go.Scatter(x=[None], y=[None], marker=_av_marker, **_cb_kw))
        cb_options.append((av_cb, "Avidity", ag_av, sr_av))

    all_coloring_keys = list(dict.fromkeys(list(m.ag_coloring) + list(m.sr_coloring)))
    for key in all_coloring_keys:
        ag_vals = m.ag_coloring.get(key, [])
        sr_vals = m.sr_coloring.get(key, [])
        all_vals = [v for v in ag_vals + sr_vals if v is not None and v == v]
        if not all_vals:
            continue
        v_min, v_max = min(all_vals), max(all_vals)
        ag_col = (_year_to_hex(ag_vals, v_min, v_max) if ag_vals
                  else ["rgba(0,0,0,0)"] * m.n_antigens)
        sr_col = (_year_to_hex(sr_vals, v_min, v_max) if sr_vals
                  else ["rgba(0,0,0,0)"] * m.n_sera)
        _col_marker = dict(color=[(v_min + v_max) / 2], colorscale='Viridis',
                           cmin=v_min, cmax=v_max, showscale=False,
                           colorbar=dict(title=key, thickness=15, len=0.75,
                                         x=1.02, xanchor="left"))
        col_cb = (go.Scatter3d(x=[None], y=[None], z=[None], marker=_col_marker, **_cb_kw)
                  if is_3d else go.Scatter(x=[None], y=[None], marker=_col_marker, **_cb_kw))
        cb_options.append((col_cb, key, ag_col, sr_col))

    n_opt     = len(cb_options)
    opt_idx   = list(range(4, 4 + n_opt))

    _orig = {
        "marker.color":     [list(m.ag_fill), list(m.sr_fill)] + [[0]] * n_opt,
        "marker.showscale": [False, False] + [False] * n_opt,
        "visible":          [True, True]   + [False] * n_opt,
    }
    buttons = [dict(label="Original", method="restyle",
                    args=[_orig, [0, 1] + opt_idx])]

    for k, (_, label, ag_c, sr_c) in enumerate(cb_options):
        upd = {
            "marker.color":     [ag_c, sr_c] + [[0]] * n_opt,
            "marker.showscale": [False, False] + [i == k for i in range(n_opt)],
            "visible":          [True, True]   + [i == k for i in range(n_opt)],
        }
        buttons.append(dict(label=label, method="restyle",
                            args=[upd, [0, 1] + opt_idx]))

    updatemenus = []
    if len(buttons) > 1:
        updatemenus = [dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.08, yanchor="top",
            bgcolor="white",
            bordercolor="#cccccc",
        )]

    _axis_kw = dict(showticklabels=False, zeroline=False,
                    dtick=1, gridcolor="rgba(0,0,0,0.18)")
    # r=80 is always reserved so the colorbar never displaces the plot area.
    _margin = dict(t=60, l=0, r=100, b=0)
    if is_3d:
        layout = go.Layout(
            autosize=True,
            showlegend=False, hovermode="closest",
            paper_bgcolor="white",
            updatemenus=updatemenus,
            margin=_margin,
            scene=dict(
                aspectmode="data",
                bgcolor="white",
                xaxis=dict(**_axis_kw),
                yaxis=dict(**_axis_kw),
                zaxis=dict(**_axis_kw),
            ),
        )
    else:
        layout = go.Layout(
            autosize=True,
            showlegend=False, hovermode="closest",
            paper_bgcolor="white",
            plot_bgcolor="white",
            updatemenus=updatemenus,
            margin=_margin,
            xaxis=dict(scaleanchor="y", scaleratio=1, **_axis_kw),
            yaxis=dict(constrain="domain", **_axis_kw),
        )

    data = [ag_trace, sr_trace, conn, txt_trace] + [t for t, *_ in cb_options]
    return go.Figure(data=data, layout=layout)


def write_html(m, filename, optim=0, **kwargs):
    '''
    Write a standalone interactive HTML file for the given Map.

    The page fills the entire browser window and resizes with it.
    Clicking an antigen or serum draws lines to all titrated partners;
    clicking the same point again hides them.  An "Avidity" coloring
    option appears in the dropdown when avidities were fitted.

    Parameters
    ----------
    m : Map
    filename : str or path-like
    optim : int
    **kwargs
        Forwarded to plotly Figure.write_html().
        Use include_plotlyjs=True (default) for a fully offline file or
        include_plotlyjs="cdn" for a smaller file that needs internet access.
    '''
    div_id = "pyedm_map"
    is_3d  = (m.dim == 3)
    fig = _build_figure(m, optim=optim)
    ag_c = m.ag_coords(optim)
    sr_c = m.sr_coords(optim)
    ag_segs, sr_segs = _connection_data(m, ag_c, sr_c, is_3d)
    click_js = (((_CLICK_JS_3D if is_3d else _CLICK_JS_2D)
                 .replace("PYEDM_DIV_ID", div_id)
                 .replace("PYEDM_AG_SEGS", json.dumps(ag_segs))
                 .replace("PYEDM_SR_SEGS", json.dumps(sr_segs))))
    page_js = _FULL_PAGE_JS.replace("PYEDM_DIV_ID", div_id)
    config = kwargs.pop('config', {})
    config.setdefault('scrollZoom', True)
    fig.write_html(filename, div_id=div_id,
                   post_script=page_js + "\n" + click_js,
                   config=config, **kwargs)


def view(m, optim=0, filename=None, display=True, **kwargs):
    '''
    Render a Map to an interactive HTML page.

    Writes to filename (or a temporary file when filename is None) and
    opens the result in the default browser when display=True.

    Parameters
    ----------
    m : Map
    optim : int
    filename : str, path-like, or None
        Destination path.  If None a temporary .html file is created.
    display : bool
        Open the file in the default browser after writing.
    **kwargs
        Forwarded to write_html() / plotly Figure.write_html().
    '''
    import os, tempfile, webbrowser
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix='.html')
        os.close(fd)
    write_html(m, filename, optim=optim, **kwargs)
    if display:
        webbrowser.open('file://' + os.path.abspath(str(filename)))


# ---------------------------------------------------------------------------
# PyRacmacs-based view (legacy)
# ---------------------------------------------------------------------------

def view_racmacs(base_racmap: pr.RacMap, results: dict,
                 optim_number: int = 0, view_args=None):
  
    if not _PR_EXISTS:
      raise RuntimeError("Can't use view_racmacs without PyRacmacs package.")
  
    if view_args is None:
        view_args = {}
    new_map = make_racmap_frombase(base_racmap, results)
    pr.view(new_map, **view_args)
    return new_map


def make_racmap_frombase(base_map: pr.RacMap, results: dict, optim_number: int = 0):

    table = results["processed_table"]

    coordinates = results["coordinates"]
    nag = len(set(table['antigen']))
    nsr = len(set(table['serum']))

    if not base_map.num_antigens == nag:
        raise ValueError()

    if not base_map.num_sera == nsr:
        raise ValueError()

    table_ag = results["level_sets"]["ag_name"]
    table_sr = results["level_sets"]["sr_name"]

    if set(base_map.ag_names) == set(table_ag) and set(base_map.sr_names) == set(table_sr):
        base_map = base_map.order_antigens(list(table_ag))
        base_map = base_map.order_sera(list(table_sr))

        new_map = deepcopy(base_map.keep_optimizations(0))

        new_map.ag_coordinates = coordinates[optim_number][:nag, :]
        new_map.sr_coordinates = coordinates[optim_number][nag:, :]

        new_map = pr.realign(new_map, base_map)
    else:
        new_map = base_map.keep_optimizations(0)

        new_map.ag_coordinates = coordinates[optim_number][:nag, :]
        new_map.sr_coordinates = coordinates[optim_number][nag:, :]

        new_map = pr.apply_plotspec(new_map, base_map)

    return new_map
