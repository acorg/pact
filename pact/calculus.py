#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:13:52 2022

@author: Sina Tureli

Facade for core C/Cython functionality. Module-level wrapper ensures
the function is picklable for use in multiprocessing.
"""


import numpy as np
from .lib.calculus import stress_and_jacobian_buffered as _sjb
from .objects import _AvidityParams

def compute_stress(coordinates, target_distances, distance_types,
                   row_indices, col_indices, nrows, ncols, dim, is_discrete):
    '''
    Compute stress for a coordinate set with no avidity or bias terms.

    Parameters
    ----------
    coordinates : array-like, shape (nrows+ncols, dim) or ((nrows+ncols)*dim,)
    target_distances : array-like, shape (n_measurements,)
    distance_types : array-like of int, shape (n_measurements,)
        0=measured, 1=unmeasured, 2=upper-bound (<), 3=lower-bound (>)
    row_indices : array-like of int, shape (n_measurements,)
    col_indices : array-like of int, shape (n_measurements,)
    nrows : int
    ncols : int
    dim : int
    is_discrete : bool

    Returns
    -------
    float
    '''
    z = np.asarray(coordinates, dtype=np.double).flatten()
    expected_len = (nrows + ncols) * dim
    if len(z) != expected_len:
        raise ValueError(
            f"coordinates has {len(z)} elements but expected "
            f"(nrows+ncols)*dim = ({nrows}+{ncols})*{dim} = {expected_len}. "
            f"Ensure coordinates contains both antigen and serum positions."
        )
    grad_buf = np.empty(expected_len)
    map_buf  = np.empty(nrows * ncols)

    _E  = np.empty(0, dtype=np.double)
    _Ei = np.empty(0, dtype='i')
    row_indices_arr = np.asarray(row_indices, dtype='i')
    col_indices_arr = np.asarray(col_indices, dtype='i')
    av = _AvidityParams(
        row_indices=row_indices_arr,
        col_indices=col_indices_arr,
        row_coord_indices=row_indices_arr,
        col_coord_indices=col_indices_arr,
        row_av_penalties=_E, row_av_means=_E,
        col_av_penalties=_E, col_av_means=_E,
        table_bias_penalties=_E, table_bias_means=_E,
        off_avidities=_Ei, off_table_biases=_Ei, table_indices=_Ei,
        nrows=nrows, ncols=ncols, nrow_names=nrows, ncol_names=ncols,
        n_tables=0,
    )

    return stress_and_jacobian_buffered(
        z, grad_buf, map_buf,
        np.asarray(target_distances, dtype=np.double),
        np.asarray(distance_types, dtype='i'),
        av, float(is_discrete), dim,
    )[0]


def stress_and_jacobian_buffered(z, grad_buf, map_buf,
                                 target_distances, distance_types,
                                 av: _AvidityParams,
                                 step, dims):

    return _sjb(z, grad_buf, map_buf,
                target_distances, distance_types,
                av.row_indices, av.col_indices,
                av.row_coord_indices, av.col_coord_indices,
                av.row_av_penalties, av.row_av_means,
                av.col_av_penalties, av.col_av_means,
                av.table_bias_penalties, av.table_bias_means,
                av.off_avidities, av.off_table_biases, av.table_indices,
                step, dims, av.nrows, av.ncols,
                av.nrow_names, av.ncol_names, av.n_tables)
