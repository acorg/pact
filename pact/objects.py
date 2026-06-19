#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:58:19 2026

@author: avicenna
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


_PALETTE = [
    "#CC0000", "#0055CC", "#007700", "#FF8800", "#8800AA",
    "#008888", "#AA4400", "#666600", "#004488", "#AA0055",
]


class Map:
    '''
    User-facing container for a pyedm optimization result.

    Stores all optimizations returned by gradient_MDS, basin_MDS, or
    perturb_MDS together with per-point plot styling.  Coordinates are
    kept in the same order as level_sets["ag_name"] / ["sr_name"]
    (alphabetical, matching the internal np.unique ordering).

    Construct via Map.from_result() or Map.from_basin_result() rather
    than calling __init__ directly.
    '''

    def __init__(self, ag_names, sr_names, coordinates, stresses, dim,
                 processed_table, row_avidities=None, col_avidities=None,
                 table_biases=None, uncoordinated=None, poorly_coordinated=None,
                 level_sets=None):
        self.ag_names        = list(ag_names)
        self.sr_names        = list(sr_names)
        self.coordinates     = coordinates       # list of (n_total, dim) arrays
        self.stresses        = list(stresses)
        self.dim             = dim
        self.processed_table = processed_table
        self.row_avidities   = row_avidities if row_avidities is not None else []
        self.col_avidities   = col_avidities if col_avidities is not None else []
        self.table_biases    = table_biases  if table_biases  is not None else []
        self.uncoordinated      = uncoordinated      or {"ag": set(), "sr": set()}
        self.poorly_coordinated = poorly_coordinated or {"ag": set(), "sr": set()}
        self.level_sets      = level_sets

        nag = len(self.ag_names)
        nsr = len(self.sr_names)
        self.ag_fill    = ["rgba(0,0,0,0)"] * nag
        self.ag_outline = ["#444444"]      * nag
        self.ag_size    = [5.0]            * nag
        self.ag_shape   = ["circle"]       * nag
        self.ag_zorder  = 0
        self.sr_fill    = ["rgba(0,0,0,0)"] * nsr
        self.sr_outline = ["#444444"]       * nsr
        self.sr_size    = [5.0]             * nsr
        self.sr_shape   = ["box"]           * nsr
        self.sr_zorder  = 0
        self._ag_coloring = {}
        self._sr_coloring = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_result(cls, result):
        '''Build a Map from a gradient_MDS or perturb_MDS result dict.'''
        ls = result["level_sets"]
        return cls(
            ag_names           = ls["ag_name"],
            sr_names           = ls["sr_name"],
            coordinates        = result["coordinates"],
            stresses           = result["stresses"],
            dim                = result["args"]["dim"],
            processed_table    = result["processed_table"],
            row_avidities      = result.get("row_avidities"),
            col_avidities      = result.get("col_avidities"),
            table_biases       = result.get("table_biases"),
            uncoordinated      = result.get("uncoordinated"),
            poorly_coordinated = result.get("poorly_coordinated"),
            level_sets         = ls,
        )

    @classmethod
    def from_basin_result(cls, result, stage='last'):
        '''
        Build a Map from a basin_MDS result dict (keyed by stage number).

        Parameters
        ----------
        result : dict
            Return value of basin_MDS: {1: result_dict, 2: result_dict, ...}
        stage : int or 'last'
            Which stage to use.  'last' picks the final (tightest-ftol) stage.
        '''
        stages = sorted(result.keys())
        if stage == 'last':
            stage = stages[-1]
        elif stage not in result:
            raise ValueError(f"stage={stage!r} not in result; available: {stages}")
        return cls.from_result(result[stage])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_antigens(self):
        return len(self.ag_names)

    @property
    def n_sera(self):
        return len(self.sr_names)

    @property
    def n_optimizations(self):
        return len(self.coordinates)

    @property
    def best_stress(self):
        return self.stresses[0] if self.stresses else None

    @property
    def ag_coloring(self):
        return self._ag_coloring

    @ag_coloring.setter
    def ag_coloring(self, d):
        if not d:
            self._ag_coloring = {}
            return
        d = {k: list(v) for k, v in d.items()}
        for k, v in d.items():
            if len(v) != self.n_antigens:
                raise ValueError(
                    f"ag_coloring[{k!r}] has {len(v)} entries but map has "
                    f"{self.n_antigens} antigens.")
        self._ag_coloring = d

    @property
    def sr_coloring(self):
        return self._sr_coloring

    @sr_coloring.setter
    def sr_coloring(self, d):
        if not d:
            self._sr_coloring = {}
            return
        d = {k: list(v) for k, v in d.items()}
        for k, v in d.items():
            if len(v) != self.n_sera:
                raise ValueError(
                    f"sr_coloring[{k!r}] has {len(v)} entries but map has "
                    f"{self.n_sera} sera.")
        self._sr_coloring = d

    # ------------------------------------------------------------------
    # Coordinate accessors
    # ------------------------------------------------------------------

    def ag_coords(self, optim=0):
        '''Antigen coordinates for the given optimization index, shape (n_antigens, dim).'''
        return self.coordinates[optim][:self.n_antigens]

    def sr_coords(self, optim=0):
        '''Serum coordinates for the given optimization index, shape (n_sera, dim).'''
        return self.coordinates[optim][self.n_antigens:]

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def rotate(self, angle, axis=None):
        '''
        Rotate all optimizations in-place by angle (radians).

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : 'x', 'y', 'z', or array-like of length 3, optional
            Rotation axis.  Used only when dim=3; ignored for dim=2.
            Defaults to 'z' when dim=3 and axis is not supplied.

        Returns
        -------
        self  (for method chaining)
        '''
        if self.dim == 2:
            R = _rotation_2d(angle)
        elif self.dim == 3:
            R = _rotation_3d(angle, 'z' if axis is None else axis)
        else:
            raise ValueError(f"rotate() supports dim=2 or dim=3; got {self.dim}.")
        return self.transform(R)

    def reflect(self, axis):
        '''
        Reflect all optimizations through a coordinate hyperplane in-place.

        Parameters
        ----------
        axis : 'x', 'y', or 'z'
            The axis whose coordinates are negated.  'z' is only valid for dim=3.

        Returns
        -------
        self  (for method chaining)
        '''
        _idx = {'x': 0, 'y': 1, 'z': 2}
        axis = axis.lower()
        if axis not in _idx:
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}.")
        idx = _idx[axis]
        if idx >= self.dim:
            raise ValueError(f"axis {axis!r} is not valid for dim={self.dim}.")
        R = np.eye(self.dim)
        R[idx, idx] = -1.0
        return self.transform(R)

    def transform(self, R):
        '''
        Apply an orthogonal transformation to all optimizations in-place.

        Parameters
        ----------
        R : array-like, shape (dim, dim)
            Orthogonal matrix (|det| = 1).  det = +1 is a pure rotation;
            det = -1 additionally reflects the map through a hyperplane.

        Returns
        -------
        self  (for method chaining)
        '''
        R = np.asarray(R, dtype=float)
        if R.shape != (self.dim, self.dim):
            raise ValueError(
                f"R must be ({self.dim}, {self.dim}); got {R.shape}.")
        det = np.linalg.det(R)
        if not np.isclose(abs(det), 1.0, atol=1e-6):
            raise ValueError(
                f"R must be orthogonal (|det| = 1); got det = {det:.6g}.")
        self.coordinates = [coords @ R.T for coords in self.coordinates]
        return self

    # ------------------------------------------------------------------
    # Avidity accessors
    # ------------------------------------------------------------------

    def ag_avidities(self, antigen_name=None, optim=0):
        '''
        Avidity values for antigens.

        Because multiple antigen_ids can map to the same antigen name
        (shared coordinate, distinct reactivity), results are returned as
        nested dicts keyed by id.

        Parameters
        ----------
        antigen_name : str or None
            If given, returns {antigen_id: value} for all ids that map to
            that name.  If None, returns {antigen_name: {antigen_id: value}}
            for every antigen.
        optim : int

        Returns
        -------
        dict, or None if row avidities were not fitted.
        '''
        if len(self.row_avidities) == 0:
            return None
        avs = self.row_avidities[optim]
        id_to_idx = {ag_id: i for i, ag_id in enumerate(self.level_sets["ag_id"])}
        pairs = (self.processed_table[['antigen', 'antigen_id']]
                 .drop_duplicates()
                 .groupby('antigen')['antigen_id']
                 .agg(list)
                 .to_dict())

        def _lookup(name):
            return {ag_id: float(avs[id_to_idx[ag_id]])
                    for ag_id in pairs.get(name, []) if ag_id in id_to_idx}

        if antigen_name is not None:
            return _lookup(antigen_name)
        return {name: _lookup(name) for name in self.ag_names}

    def sr_avidities(self, serum_name=None, optim=0):
        '''
        Avidity values for sera.

        Because multiple serum_ids can map to the same serum name
        (shared coordinate, distinct reactivity), results are returned as
        nested dicts keyed by id.

        Parameters
        ----------
        serum_name : str or None
            If given, returns {serum_id: value} for all ids that map to
            that name.  If None, returns {serum_name: {serum_id: value}}
            for every serum.
        optim : int

        Returns
        -------
        dict, or None if column avidities were not fitted.
        '''
        if len(self.col_avidities) == 0:
            return None
        avs = self.col_avidities[optim]
        id_to_idx = {sr_id: i for i, sr_id in enumerate(self.level_sets["sr_id"])}
        pairs = (self.processed_table[['serum', 'serum_id']]
                 .drop_duplicates()
                 .groupby('serum')['serum_id']
                 .agg(list)
                 .to_dict())

        def _lookup(name):
            return {sr_id: float(avs[id_to_idx[sr_id]])
                    for sr_id in pairs.get(name, []) if sr_id in id_to_idx}

        if serum_name is not None:
            return _lookup(serum_name)
        return {name: _lookup(name) for name in self.sr_names}

    def table_avidities(self, optim=0):
        '''
        Table bias values keyed by table_id.

        Returns
        -------
        dict {table_id: value}, or None if table biases were not fitted.
        '''
        if len(self.table_biases) == 0:
            return None
        biases = self.table_biases[optim]
        table_ids = self.level_sets.get("table_id") if self.level_sets else None
        if table_ids is None:
            return None
        return {tid: float(biases[i]) for i, tid in enumerate(table_ids)}

    # ------------------------------------------------------------------
    # Plot styling
    # ------------------------------------------------------------------

    def color_by(self, groups, group_colors=None):
        '''
        Assign colors from group membership.

        Follows the Racmacs convention: antigens get their fill color,
        sera get their outline color (sera are open shapes).  Points not
        present in the groups dict are left at their current color.

        Parameters
        ----------
        groups : dict
            {"ag": {ag_name: group_label}, "sr": {sr_name: group_label}}
            Either key may be absent or map to an empty dict.
        group_colors : dict, optional
            {group_label: hex_color_string}.  If None, colors are
            auto-assigned from the built-in 10-color palette in sorted
            group-label order; labels beyond 10 cycle through the palette.

        Returns
        -------
        self  (for method chaining)
        '''
        ag_groups = groups.get("ag", {})
        sr_groups = groups.get("sr", {})
        all_labels = sorted(set(ag_groups.values()) | set(sr_groups.values()))

        if group_colors is None:
            group_colors = {label: _PALETTE[i % len(_PALETTE)]
                            for i, label in enumerate(all_labels)}

        for i, name in enumerate(self.ag_names):
            if name in ag_groups:
                label = ag_groups[name]
                if label in group_colors:
                    self.ag_fill[i] = group_colors[label]

        for i, name in enumerate(self.sr_names):
            if name in sr_groups:
                label = sr_groups[name]
                if label in group_colors:
                    self.sr_outline[i] = group_colors[label]

        return self

    # ------------------------------------------------------------------
    # Subsetting and copying
    # ------------------------------------------------------------------

    def subset(self, antigens=None, sera=None):
        '''
        Return a new Map restricted to the given antigens and/or sera.

        Parameters
        ----------
        antigens : list of str or None
            Names to keep.  None keeps all antigens.
        sera : list of str or None
            Names to keep.  None keeps all sera.

        Returns
        -------
        Map
            New Map.  Plot spec is preserved for kept points.  Stresses
            reflect the original full-dataset optimizations.
        '''
        keep_ag = set(antigens) if antigens is not None else set(self.ag_names)
        keep_sr = set(sera)     if sera     is not None else set(self.sr_names)

        unknown_ag = keep_ag - set(self.ag_names)
        unknown_sr = keep_sr - set(self.sr_names)
        if unknown_ag:
            raise ValueError(f"Unknown antigen names: {sorted(unknown_ag)}")
        if unknown_sr:
            raise ValueError(f"Unknown serum names: {sorted(unknown_sr)}")

        new_ag_names = [n for n in self.ag_names if n in keep_ag]
        new_sr_names = [n for n in self.sr_names if n in keep_sr]

        old_ag_pos = {name: i for i, name in enumerate(self.ag_names)}
        old_sr_pos = {name: i for i, name in enumerate(self.sr_names)}

        ag_rows = [old_ag_pos[n] for n in new_ag_names]
        sr_rows = [self.n_antigens + old_sr_pos[n] for n in new_sr_names]
        keep_rows = ag_rows + sr_rows
        new_coords = [coords[keep_rows] for coords in self.coordinates]

        new_table = self.processed_table[
            self.processed_table['antigen'].isin(keep_ag) &
            self.processed_table['serum'].isin(keep_sr)
        ].copy()

        new_ls = dict(self.level_sets) if self.level_sets else {}
        new_ls["ag_name"] = np.array(new_ag_names)
        new_ls["sr_name"] = np.array(new_sr_names)
        new_ls["ag_id"] = pd.unique(new_table['antigen_id'].values)
        new_ls["sr_id"] = pd.unique(new_table['serum_id'].values)



        new_row_av = None
        if len(self.row_avidities) > 0:
            old_id_pos = {ag_id: i for i, ag_id in enumerate(self.level_sets["ag_id"])}
            new_row_av = [
                np.array([avs[old_id_pos[ag_id]] for ag_id in new_ls["ag_id"]])
                for avs in self.row_avidities
            ]

        new_col_av = None
        if len(self.col_avidities) > 0:
            old_id_pos = {sr_id: i for i, sr_id in enumerate(self.level_sets["sr_id"])}
            new_col_av = [
                np.array([avs[old_id_pos[sr_id]] for sr_id in new_ls["sr_id"]])
                for avs in self.col_avidities
            ]

        new_map = Map(
            ag_names           = new_ag_names,
            sr_names           = new_sr_names,
            coordinates        = new_coords,
            stresses           = self.stresses,
            dim                = self.dim,
            processed_table    = new_table,
            row_avidities      = new_row_av,
            col_avidities      = new_col_av,
            table_biases       = self.table_biases,
            uncoordinated      = {"ag": self.uncoordinated["ag"] & keep_ag,
                                  "sr": self.uncoordinated["sr"] & keep_sr},
            poorly_coordinated = {"ag": self.poorly_coordinated["ag"] & keep_ag,
                                  "sr": self.poorly_coordinated["sr"] & keep_sr},
            level_sets         = new_ls,
        )
        new_map._copy_plot_spec_from(self, ag_rows, old_sr_pos, new_sr_names)
        new_map.ag_zorder = self.ag_zorder
        new_map.sr_zorder = self.sr_zorder
        if self.ag_coloring:
            new_map.ag_coloring = {k: [v[old_ag_pos[n]] for n in new_ag_names]
                                   for k, v in self.ag_coloring.items()}
        if self.sr_coloring:
            new_map.sr_coloring = {k: [v[old_sr_pos[n]] for n in new_sr_names]
                                   for k, v in self.sr_coloring.items()}
        return new_map

    def copy(self):
        '''
        Shallow copy with independent plot spec lists.

        Coordinate arrays and the processed_table are shared (not
        deep-copied) since they are treated as read-only after construction.
        '''
        new_map = Map(
            ag_names           = list(self.ag_names),
            sr_names           = list(self.sr_names),
            coordinates        = self.coordinates,
            stresses           = list(self.stresses),
            dim                = self.dim,
            processed_table    = self.processed_table,
            row_avidities      = self.row_avidities,
            col_avidities      = self.col_avidities,
            table_biases       = self.table_biases,
            uncoordinated      = self.uncoordinated,
            poorly_coordinated = self.poorly_coordinated,
            level_sets         = self.level_sets,
        )
        ag_idx = list(range(self.n_antigens))
        sr_idx = {name: i for i, name in enumerate(self.sr_names)}
        new_map._copy_plot_spec_from(self, ag_idx, sr_idx, self.sr_names)
        new_map.ag_zorder = self.ag_zorder
        new_map.sr_zorder = self.sr_zorder
        new_map.ag_coloring = {k: list(v) for k, v in self.ag_coloring.items()}
        new_map.sr_coloring = {k: list(v) for k, v in self.sr_coloring.items()}
        return new_map

    def _copy_plot_spec_from(self, source, ag_indices, old_sr_pos, new_sr_names):
        sr_indices = [old_sr_pos[n] for n in new_sr_names]
        self.ag_fill    = [source.ag_fill[i]    for i in ag_indices]
        self.ag_outline = [source.ag_outline[i] for i in ag_indices]
        self.ag_size    = [source.ag_size[i]    for i in ag_indices]
        self.ag_shape   = [source.ag_shape[i]   for i in ag_indices]
        self.sr_fill    = [source.sr_fill[i]    for i in sr_indices]
        self.sr_outline = [source.sr_outline[i] for i in sr_indices]
        self.sr_size    = [source.sr_size[i]    for i in sr_indices]
        self.sr_shape   = [source.sr_shape[i]   for i in sr_indices]

    # ------------------------------------------------------------------

    def __repr__(self):
        stress = f"{self.best_stress:.4f}" if self.best_stress is not None else "n/a"
        return (f"Map({self.n_antigens} antigens, {self.n_sera} sera, "
                f"dim={self.dim}, {self.n_optimizations} optimization(s), "
                f"best stress={stress})")


@dataclass(frozen=True)
class _AvidityParams:
    row_indices: np.ndarray
    col_indices: np.ndarray
    row_coord_indices: np.ndarray
    col_coord_indices: np.ndarray
    row_av_penalties: np.ndarray
    row_av_means: np.ndarray
    col_av_penalties: np.ndarray
    col_av_means: np.ndarray
    table_bias_penalties: np.ndarray
    table_bias_means: np.ndarray
    off_avidities: np.ndarray
    off_table_biases: np.ndarray
    table_indices: np.ndarray
    nrows: int
    ncols: int
    nrow_names: int
    ncol_names: int
    n_tables: int


# ---------------------------------------------------------------------------
# Rotation matrix helpers
# ---------------------------------------------------------------------------

def _rotation_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, -s],
                     [ s,  c]], dtype=float)


def _rotation_3d(angle, axis='z'):
    '''
    3×3 rotation matrix for counter-clockwise rotation by angle (radians).

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    axis : 'x', 'y', 'z', or array-like of length 3
        Rotation axis.  A string selects one of the three coordinate axes;
        an array-like is interpreted as an arbitrary unit vector (it will be
        normalised automatically).

    Returns
    -------
    np.ndarray, shape (3, 3)
    '''
    c, s = np.cos(angle), np.sin(angle)
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == 'z':
            return np.array([[ c, -s,  0],
                             [ s,  c,  0],
                             [ 0,  0,  1]], dtype=float)
        if axis == 'y':
            return np.array([[ c,  0,  s],
                             [ 0,  1,  0],
                             [-s,  0,  c]], dtype=float)
        if axis == 'x':
            return np.array([[ 1,  0,  0],
                             [ 0,  c, -s],
                             [ 0,  s,  c]], dtype=float)
        raise ValueError(f"axis string must be 'x', 'y', or 'z'; got {axis!r}.")
    # Rodrigues' formula for an arbitrary unit vector
    u = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("axis vector must be non-zero.")
    u = u / norm
    ux, uy, uz = u
    return np.array([
        [c + ux*ux*(1-c),       ux*uy*(1-c) - uz*s,  ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,    c + uy*uy*(1-c),      uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,    uz*uy*(1-c) + ux*s,   c + uz*uz*(1-c)   ],
    ], dtype=float)