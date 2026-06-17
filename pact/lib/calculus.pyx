#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:56:55 2022

@author: avicenna
"""
import cython
cimport cython
cimport numpy as np
import numpy as np

cdef extern from "ctools.h":
    double _stress_and_jacobian_buffered(double*, double*, int*, int*, int*, int*, int*, int,
                                         double*, double*, double*, double*, double*, double*,
                                         int*, int*, int*,
                                         double, int, int, int, int, int, int,
                                         int, int, int, double*, double*)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def stress_and_jacobian_buffered(
        double[::1] z,
        double[::1] grad_buf,
        double[::1] map_buf,
        double[::1] target_distances,
        int[::1]    distance_types,
        int[::1]    row_indices,
        int[::1]    col_indices,
        int[::1]    row_coord_indices,
        int[::1]    col_coord_indices,
        double[::1] row_av_penalties,
        double[::1] row_av_means,
        double[::1] col_av_penalties,
        double[::1] col_av_means,
        double[::1] table_bias_penalties,
        double[::1] table_bias_means,
        int[::1]    off_avidities,
        int[::1]    off_table_biases,
        int[::1]    table_indices,
        double step, int dims, int nrows, int ncols,
        int nrow_names, int ncol_names, int n_tables):

    cdef int n_obs     = target_distances.shape[0]
    cdef int row_avidity_on = 1 if row_av_penalties.shape[0] > 0 else 0
    cdef int col_avidity_on = 1 if col_av_penalties.shape[0] > 0 else 0
    cdef int table_bias_on  = 1 if table_bias_penalties.shape[0] > 0 else 0

    cdef double* row_av_penalties_C = NULL
    cdef double* row_av_means_C     = NULL
    cdef double* col_av_penalties_C = NULL
    cdef double* col_av_means_C     = NULL
    cdef double* table_bias_penalties_C = NULL
    cdef double* table_bias_means_C     = NULL
    cdef int*    off_avidities_C    = NULL
    cdef int*    off_table_biases_C = NULL
    cdef int*    table_indices_C    = NULL

    if row_avidity_on:
        row_av_penalties_C = &row_av_penalties[0]
        row_av_means_C     = &row_av_means[0]
    if col_avidity_on:
        col_av_penalties_C = &col_av_penalties[0]
        col_av_means_C     = &col_av_means[0]
    if row_avidity_on or col_avidity_on:
        off_avidities_C    = &off_avidities[0]
    if table_bias_on:
        table_bias_penalties_C = &table_bias_penalties[0]
        table_bias_means_C     = &table_bias_means[0]
        off_table_biases_C     = &off_table_biases[0]
        table_indices_C        = &table_indices[0]

    stress_val = _stress_and_jacobian_buffered(
        &z[0], &target_distances[0], &distance_types[0],
        &row_indices[0], &col_indices[0],
        &row_coord_indices[0], &col_coord_indices[0],
        n_obs,
        row_av_penalties_C, row_av_means_C,
        col_av_penalties_C, col_av_means_C,
        table_bias_penalties_C, table_bias_means_C,
        off_avidities_C, off_table_biases_C, table_indices_C,
        step, nrows, ncols, nrow_names, ncol_names, dims, n_tables,
        row_avidity_on, col_avidity_on, table_bias_on,
        &map_buf[0], &grad_buf[0])

    return stress_val, grad_buf
