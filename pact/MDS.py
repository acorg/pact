#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 12:26:11 2022

@author: Sina Tureli
"""

import tqdm
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial

from . import calculus
from .preprocess import prep_inputs, prep_table, supress_params_for_uncoordinated
from .objects import _AvidityParams
from .messages import warn_once

def gradient_MDS(table, dim, initial_configurations, is_discrete,
                 method='L-BFGS-B', minimize_options=None, offset_penalty_scales=1,
                 row_avidity_on=False, col_avidity_on=False, table_bias_on=False, 
                 avidity_options=None, coordinate_bounds=None, supress_uncoordinated=True, 
                 col_bases=None, min_col_basis=None, num_cpus=1, 
                 hide_progress_bar=False, verbose=True, seed=0):

    '''
    Run gradient-based MDS on each initial configuration and return all
    converged results sorted by ascending stress.

    For each configuration in initial_configurations, scipy.optimize.minimize
    is called with the given method (default L-BFGS-B) using the exact
    gradient from the C stress kernel.  Results that fail to converge are
    discarded.  Antigens/sera with fewer unique measured titrations than dim
    are optionally set to NaN and excluded from the stress calculation
    (supress_uncoordinated=True).

    Parameters
    ----------
    table : DataFrame
        Wide format (antigens as index, sera as columns) or long format with
        columns 'antigen_id', 'serum_id', 'titer'.  Long format requires a
        'table_id' column when table_bias_on=True.
    dim : int
        Number of spatial dimensions.
    initial_configurations : list of array-like or int
        Starting coordinate vectors.  Each element is a flat
        (nrows+ncols)*dim array (coordinates only). If equal to integer=N,
        generates N random configurations uniformly distributed in a box
        determined by generate_initial_conditions.  When optimized_box_size=True
        (the default), bounds are inferred from a preliminary single-run
        optimization; otherwise each dimension spans [-max_dist, max_dist] where
        max_dist is the largest reliable measured distance.
    is_discrete : bool
        Dilution step size for the threshold-titer penalty (True → 1.0,
        False → 0.0).  Passed directly to the C stress kernel as the sigmoid
        offset for < titers.
    method : str
        Optimisation method passed to scipy.optimize.minimize.
    minimize_options : dict, optional
        Options forwarded to scipy.optimize.minimize.
        Defaults: {'disp': False, 'ftol': 1e-6}.  User-supplied values
        override these defaults.
    offset_penalty_scales: penalty scales to be applied to row_avidity,
        col_avidity and table_bias terms. Can be a scalar or dict.
    row_avidity_on : bool
        Fit a per-antigen avidity offset.
    col_avidity_on : bool
        Fit a per-serum avidity offset.
    table_bias_on : bool
        Fit a per-table bias term.
    avidity_options : dict, optional
        Initial values and regularisation parameters for avidity/bias terms.
        
        Recognised keys (defaults applied for any missing key):
          row_avidities (nrows,), row_av_penalties (nrows,),
          row_av_means (nrows,), row_av_bounds list[(lo,hi)],
          col_avidities (ncols,), col_av_penalties (ncols,),
          col_av_means (ncols,), col_av_bounds list[(lo,hi)],
          off_avidities (nrows+ncols,) int mask,
          table_biases (n_tables,), table_bias_penalties (n_tables,),
          table_bias_means (n_tables,), table_bias_bounds list[(lo,hi)],
          off_table_biases (n_tables,) int mask.
          
        Terms starting with off_ are used to turn on or off parameters
        by setting the gradient of that term to 0. They will still influence
        the stress even when their gradient is frozen. Other terms like row_avidities are initial
        values, or terms like row_av_penalties, row_av_means, row_av_bounds, 
        determine the prior constraints.
        
    coordinate_bounds : list of (float, float), optional
        Bounds on the flat coordinate vector, length (nrows+ncols)*dim.
        Defaults to unbounded.
    supress_uncoordinated : bool
        Iteratively set to NaN the coordinates of points with fewer unique
        measured titrations than dim, and remove their observations from the
        stress calculation.
    col_bases : dict, optional
        {serum_name: float} overriding auto-computed column bases.
    min_col_basis : float, optional
        Floor applied to every column basis after computation.
    num_cpus : int
        Number of worker processes.  1 runs serially.
    hide_progress_bar : bool
        Suppress the tqdm progress bar.
    verbose : bool
        Show or suppress warning messages.
    seed: int 
        Used for random generation of initial configurations if initial_configurations
        is an int.

    Returns
    -------
    dict with keys:
      'args'                 : dict of the original call arguments.
      'processed_table'      : prepped long-format DataFrame used internally.
      'coordinates'          : list of (nrows+ncols, dim) arrays, ascending stress.
      'stresses'             : list of float total stresses, ascending.
      'optim_results'        : list of scipy OptimizeResult, ascending stress.
      'row_avidities'        : list of (nrows,) arrays  [if row_avidity_on].
      'col_avidities'        : list of (ncols,) arrays  [if col_avidity_on].
      'table_biases'         : list of (n_tables,) arrays  [if table_bias_on].
      'map_stresses'         : coordinate-only stresses without avidity/bias
                               penalty terms  [if any avidity/bias is active].
    '''
    
    mds_result = {
      "args":{"method":method, 
              "minimize_options":minimize_options,
              "avidity_options":avidity_options, 
              "offset_penalty_scales": offset_penalty_scales,
              "coordinate_bounds":coordinate_bounds, 
              "supress_uncoordinated":supress_uncoordinated,
              "col_bases":col_bases,
              "min_col_basis":min_col_basis,
              "row_avidity_on":row_avidity_on,
              "col_avidity_on":col_avidity_on,
              "table_bias_on":table_bias_on,
              "is_discrete":is_discrete,
              "initial_configurations":initial_configurations,
              "dim":dim,
              "table":table,
              "verbose":verbose,
              "seed":seed,
              "num_cpus":num_cpus,
              }
      }
    

    if isinstance(initial_configurations, int):
      initial_configurations=\
        generate_initial_conditions(table, dim, initial_configurations, 
                                    is_discrete=is_discrete)
    elif not isinstance(initial_configurations, list):
      raise ValueError("initial_configurations can only be an integer or a list.")
      

    if minimize_options is None:
        minimize_options = {}
    minimize_options = dict({'disp': False, 'ftol': 1e-6}, **minimize_options)

    if isinstance(offset_penalty_scales, (int,float)):
      offset_penalty_scales = {"row": offset_penalty_scales, 
                               "col": offset_penalty_scales,
                               "table": offset_penalty_scales}

    table, n, indices, coordinate_bounds, mds_result =\
      prep_inputs(table, mds_result, col_bases=col_bases, min_col_basis=min_col_basis,
                  verbose=verbose)
      
    target_distances_flat = table['target_distances'].values.astype(np.double)
    distance_types_flat = table['distance_type'].values.astype('i')

    if np.count_nonzero(np.isinf(target_distances_flat)) > 0:
        raise ValueError('target_distances can not contain infinite values.')
    

    offsets, bounds, av_params =\
      _build_avidity_defaults(coordinate_bounds, row_avidity_on, col_avidity_on, 
                              table_bias_on, indices, n, avidity_options, 
                              offset_penalty_scales)

    mds_result.update({"av_params":av_params})

    partial_parfun = partial(
        _parfun,
        target_distances_flat, distance_types_flat,
        float(is_discrete), method, minimize_options, dim,
        row_avidity_on, offsets["row"],
        col_avidity_on, offsets["col"],
        table_bias_on, offsets["table"],
        av_params, bounds)

    if num_cpus > 1:
        with Pool(num_cpus, initializer=_pool_init) as p:
            optim_results = list(tqdm.tqdm(p.imap(partial_parfun, initial_configurations),
                                 total=len(initial_configurations),
                                 disable=hide_progress_bar))
    else:
        optim_results = []
        for configuration in tqdm.tqdm(initial_configurations, disable=hide_progress_bar):
            optim_results.append(partial_parfun(configuration))
                        
    mds_result.update({"optim_results": optim_results})
    
    return _build_result(mds_result, n, verbose)
    


def tolerance_annealing(table, dim, initial_configurations, is_discrete, 
                        ftols=None, refine_fractions=None, method='L-BFGS-B', 
                        minimize_options=None, offset_penalty_scales=1, row_avidity_on=False, 
                        col_avidity_on=False, table_bias_on=False, avidity_options=None, 
                        coordinate_bounds=None, supress_uncoordinated=True, col_bases=None, 
                        min_col_basis=None,  num_cpus=1, hide_progress_bar=False, 
                        seed=0, verbose=True):
    '''
    Multi-stage coarse-to-fine MDS optimisation that anneals on ftolerance parameter.

    Runs gradient_MDS in sequence with progressively tighter convergence
    tolerances.  After each stage except the last, the top-performing
    fraction of results (by stress) is selected and passed as initial
    configurations for the next stage.

    Parameters
    ----------
    table : DataFrame
        Same format as gradient_MDS.
    dim : int
    initial_configurations : list of array-like
        Starting configurations for the first stage. If equal to integer=N,
        N conditions generated randomly using generate_initial_conditions.
    is_discrete : bool
    ftols : list of float, optional
        Per-stage ftol values passed to minimize_options, from loosest to
        tightest.  Default: [1e-3, 1e-6].
    refine_fractions : list of float, optional
        Fraction of results carried forward between consecutive stages.
        Length must equal len(ftols) - 1.  Default: [0.25].
    method, minimize_options, offset_penalty_scales, row_avidity_on, col_avidity_on, 
    table_bias_on, avidity_options, coordinate_bounds, supress_uncoordinated, col_bases,
    min_col_basis, num_cpus, hide_progress_bar, seed verbose are forwarded to 
    gradient_MDS unchanged.

    Returns
    -------
    dict mapping stage number (1-indexed int) to the gradient_MDS result dict
    for that stage.  The final stage contains the best configurations found.
    '''
    
    if isinstance(initial_configurations, int):
      initial_configurations=\
        generate_initial_conditions(table, dim, initial_configurations, 
                                    is_discrete=is_discrete, seed=seed)
    elif not isinstance(initial_configurations, list):
      raise ValueError("initial_configurations can only be an integer or a list.")

    if ftols is None:
        ftols = [1e-3, 1e-6]
    if refine_fractions is None:
        refine_fractions = [0.25]

    if len(refine_fractions) != len(ftols) - 1:
        raise ValueError(
            f"refine_fractions must have length len(ftols)-1 "
            f"({len(ftols)-1}), got {len(refine_fractions)}"
        )
        
    if isinstance(offset_penalty_scales, (int,float)):
      offset_penalty_scales = {"row": offset_penalty_scales, 
                               "col": offset_penalty_scales,
                               "table": offset_penalty_scales}

    base_opts = dict(minimize_options) if minimize_options is not None else {}

    stage_results = []
    current_inits = initial_configurations
    
    #pre prep the table so it does not get prepped each time
    # col_bases and min_col_basis no longer need to be passed to gradient_MDS
    table = prep_table(table, col_bases=col_bases, min_col_basis=min_col_basis)

    for stage, ftol in enumerate(ftols, 1):
        opts = {**base_opts, 'ftol': ftol}
        result = gradient_MDS(table, dim, current_inits, is_discrete,
                              num_cpus=num_cpus, method=method,
                              minimize_options=opts,
                              offset_penalty_scales=offset_penalty_scales,
                              hide_progress_bar=hide_progress_bar,
                              row_avidity_on=row_avidity_on, 
                              col_avidity_on=col_avidity_on,
                              avidity_options=avidity_options,
                              coordinate_bounds=coordinate_bounds,
                              supress_uncoordinated=supress_uncoordinated,
                              table_bias_on=table_bias_on,
                              verbose=verbose)
        stage_results.append(result)

        if stage < len(ftols):
            if len(result['optim_results']) == 0:
                raise RuntimeError(f"No initial conditions converged in stage {stage}.")
            frac = refine_fractions[stage - 1]
            n_refine = max(1, int(len(result['optim_results']) * frac))
            current_inits = [r['x'] for r in result['optim_results'][:n_refine]]

    return stage_results


def perturb_search(result, chunk_size=5, N=20, radius=10, num_cpus=1, 
                   hide_progress_bar=False):
    '''
    One greedy pass of block-coordinate perturbation descent.

    Points (antigens then sera) are processed in fixed sequential chunks.  For
    each chunk, N perturbed starting configurations are generated (only that
    chunk's points are displaced; all others are frozen at the current best
    coordinates) and gradient_MDS is run on them.  If the best result from the
    chunk improves the overall stress, the current best is updated before
    moving to the next chunk.

    The input `result` is expected to be in the format returned by gradient_MDS
    or a prior call to perturb_MDS.  Coordinates, dimension, and avidity flags
    are all inferred from it.

    Parameters
    ----------
    result : dict
        Best result so far, in gradient_MDS output format.  Must contain
        'coordinates' and 'stresses'.  The presence of 'row_avidities',
        'col_avidities', and 'table_biases' keys enables those modes; their
        values are used to warm-start each chunk's optimization.
    chunk_size : int
        Number of points (antigens + sera combined) displaced per batch.
    N : int
        Number of perturbations generated per point per chunk.
    radius : float
        Maximum perturbation displacement.
    num_cpus, hide_progress_bar, 
        Passed through to gradient_MDS unchanged.

    Returns
    -------
    dict in gradient_MDS output format with a single entry in each list,
    representing the best configuration found after the full pass.  Can be
    passed back into perturb_MDS for additional passes.
    '''
    
    table = result["processed_table"]
    is_discrete = result["args"]["is_discrete"]
    avidity_options = result["args"]["avidity_options"]
    nrow_names = len(set(table['antigen']))
    ncol_names = len(set(table['serum']))

    current_coords = np.array(result['coordinates'][0], dtype=float)
    if current_coords.ndim == 1:
        current_coords = current_coords.reshape(nrow_names + ncol_names, -1)
    dim = current_coords.shape[1]
    current_stress = float(result['stresses'][0])

    row_avidity_on = 'row_avidities' in result
    col_avidity_on = 'col_avidities' in result
    table_bias_on  = 'table_biases'  in result
    any_av = row_avidity_on or col_avidity_on or table_bias_on

    if any_av:
        # Start from user's structural params; override VALUES with result's best
        merged_av = dict(avidity_options) if avidity_options is not None else {}
        if row_avidity_on:
            merged_av['row_avidities'] = np.array(result['row_avidities'][0])
        if col_avidity_on:
            merged_av['col_avidities'] = np.array(result['col_avidities'][0])
        if table_bias_on:
            merged_av['table_biases']  = np.array(result['table_biases'][0])
    else:
        merged_av = None


    all_indices = list(range(nrow_names + ncol_names))
    chunks = [all_indices[i:i + chunk_size]
              for i in range(0, len(all_indices), chunk_size)]

    initial_stress = current_stress
    pbar = tqdm.tqdm(chunks, disable=hide_progress_bar)
    for chunk in pbar:
        antigen_indices = [i for i in chunk if i < nrow_names]
        serum_indices   = [i - nrow_names for i in chunk if i >= nrow_names]

        initial_configs = generate_perturbed_conditions(
            current_coords, nrow_names, ncol_names, dim, N, radius,
            antigen_indices=antigen_indices,
            serum_indices=serum_indices,
        )

        chunk_result = gradient_MDS(
            table, dim, initial_configs, is_discrete,
            method=result["args"]["method"], 
            minimize_options=result["args"]["minimize_options"],
            offset_penalty_scales=result["args"]["offset_penalty_scales"],
            row_avidity_on=row_avidity_on,
            col_avidity_on=col_avidity_on,
            table_bias_on=table_bias_on,
            avidity_options=merged_av,
            coordinate_bounds=result["args"]["coordinate_bounds"],
            supress_uncoordinated=result["args"]["supress_uncoordinated"],
            num_cpus=num_cpus,
            hide_progress_bar=True,
        )

        if not chunk_result.get("coordinates"):
            continue

        chunk_stress = float(chunk_result["stresses"][0])
        if chunk_stress < current_stress:
            current_stress = chunk_stress
            current_coords = np.array(chunk_result["coordinates"][0])
            if row_avidity_on:
                merged_av["row_avidities"] = np.array(chunk_result["row_avidities"][0])
            if col_avidity_on:
                merged_av["col_avidities"] = np.array(chunk_result["col_avidities"][0])
            if table_bias_on:
                merged_av["table_biases"]  = np.array(chunk_result["table_biases"][0])
           

        reduction = (initial_stress - current_stress) / initial_stress
        pbar.set_postfix({"stress_reduction": f"{reduction:.4%}"})

    final_result = {
        "args":        result["args"],
        "processed_table":       result["processed_table"],
        "coordinates": [current_coords],
        "stresses":    [current_stress],
        "level_sets": result["level_sets"]
    }
    if row_avidity_on:
        final_result["row_avidities"] = [merged_av["row_avidities"]]
    if col_avidity_on:
        final_result["col_avidities"] = [merged_av["col_avidities"]]
    if table_bias_on:
        final_result["table_biases"]  = [merged_av["table_biases"]]
   

    return final_result


def classicalMDS(D, n=2):
    '''
    Classical (spectral) MDS.

    Parameters
    ----------
    D : array-like, shape (s, s)
        Square distance matrix.
    n : int
        Number of output dimensions (default 2).

    Returns
    -------
    X : ndarray, shape (s, n)
        Embedding coordinates.  Contains NaN along any dimension whose
        corresponding eigenvalue is negative (non-Euclidean distance matrix).
    '''

    s1,s2 = D.shape

    if not s1==s2:
      raise ValueError("distance matrix must be square")

    D2 = D*D

    I = np.eye(s1)
    J = 1/s1*np.ones((s1,s2))

    C = I - J

    B = -1/2*C@(D2)@C
    evals, evecs = eigsh(B, n, which='LM')
    I = np.argsort(evals)[::-1]
    evals = evals[I]
    evecs = evecs[:,I]

    no_negative_eigenvalues =  np.count_nonzero((evals<0)&(np.abs(evals)>1e-8))
    ratio_of_negative_eigenvalues = no_negative_eigenvalues/evals.size

    if ratio_of_negative_eigenvalues>0:
        print(f"ratio of negative eigenvalues is {ratio_of_negative_eigenvalues}")

    X = evecs@np.sqrt(np.diag(evals))

    return X


def generate_perturbed_conditions(coordinates, nrows, ncols, dim, N, radius,
                                  antigen_indices=None, serum_indices=None,
                                  seed=None):
    '''
    Given an optimized coordinate set, generate perturbed initial conditions by
    displacing each antigen and/or serum one at a time.

    For each selected point, N configurations are produced where that point is
    moved to one of N positions around its current location while all other
    points remain fixed.  In 2D the positions follow a golden-angle spiral
    (uniform area density, outward); in higher dimensions they are drawn
    uniformly from a ball of the given radius.

    Parameters
    ----------
    coordinates : array-like, shape (nrows+ncols, dim) or ((nrows+ncols)*dim,)
        Base configuration to perturb.
    nrows : int
        Number of antigens.
    ncols : int
        Number of sera.
    dim : int
        Spatial dimension.
    N : int
        Number of perturbations per point.
    radius : float
        Maximum displacement from each point's current position.
    antigen_indices : list of int or None
        Antigen indices (0-based) to perturb.  None = all antigens; [] = none.
    serum_indices : list of int or None
        Serum indices (0-based) to perturb.  None = all sera; [] = none.
    seed : optional
        Seed for the RNG (only used when dim > 2).

    Returns
    -------
    list of flat (nrows+ncols)*dim arrays, one per (point, perturbation) pair.
    Total length = N * len(antigen_indices) + N * len(serum_indices)
                   (with None expanding to all antigens/sera respectively).
    '''
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(nrows + ncols, dim)

    rng = np.random.default_rng(seed)

    if antigen_indices is None:
        antigen_indices = list(range(nrows))
    if serum_indices is None:
        serum_indices = list(range(ncols))

    perturb_indices = list(antigen_indices) + [nrows + j for j in serum_indices]

    results = []
    for idx in perturb_indices:
        offsets = _perturb_offsets(N, dim, radius, rng)
        for offset in offsets:
            config = coordinates.copy()
            config[idx] += offset
            results.append(config.flatten())

    return results


def generate_initial_conditions(table, dim, N, is_discrete, method='uniform',
                                optimized_box_size=True, method_settings=None,
                                col_bases=None, min_col_basis=None, seed=None):
    '''
    Generate N random starting configurations for gradient_MDS.

    Parameters
    ----------
    table : DataFrame
        Wide format (antigens as index, sera as columns) or long format with
        'antigen_id', 'serum_id', 'titer' columns.
    dim : int
        Number of spatial dimensions.
    N : int
        Number of starting configurations to generate.
    is_discrete : bool
        Passed to gradient_MDS when optimized_box_size=True.
    method : str
        Sampling strategy.  Currently only 'uniform' is supported.
    optimized_box_size : bool
        If True (default), run a single gradient_MDS pass to find a
        preliminary embedding and use its coordinate range (scaled by
        method_settings["scale"], default 1.5) as the sampling box.
        If False, each dimension spans [-max_dist, max_dist] * scale where
        max_dist is the largest reliable measured distance.
    method_settings : dict, optional
        Recognised keys:
          scale (float, default 1.5) — multiplicative factor applied to the
          box bounds before sampling.
    col_bases, min_col_basis
        Forwarded to prep_table / titers_to_distances.
    seed : optional
        Seed for the RNG.

    Returns
    -------
    list of flat (nrows+ncols)*dim arrays, length N.
    '''

    table = prep_table(table, col_bases=col_bases, min_col_basis=min_col_basis)

    nrow_names = len(set(table['antigen']))
    ncol_names = len(set(table['serum']))
    reliable = table['distance_type'] == 0
    max_dist = np.nanmax(table.loc[reliable, 'target_distances'].values)

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]
    rng = np.random.default_rng(seed)

    if optimized_box_size:
        box_starting_configuration = rng.uniform(-2*max_dist, 2*max_dist,
                                                 (nrow_names+ncol_names, dim)).flatten()
        result = gradient_MDS(table, dim, [box_starting_configuration], is_discrete,
                              hide_progress_bar=True, supress_uncoordinated=False)

        if not result.get('coordinates'):
          raise RuntimeError("optimized_box_size fit failed to converge; try a different seed or disable optimized_box_size")

        coordinates = result['coordinates'][0]
        box_bounds = [[np.nanmin(coordinates[:,i]) for i in range(dim)],
                      [np.nanmax(coordinates[:,i]) for i in range(dim)]]

        if all(np.isnan(x) for x in np.array(box_bounds).flatten()):
          raise RuntimeError("box_bounds is nan; try a different seed or disable optimized_box_size")

    else:
        box_bounds = [[-max_dist for _ in range(dim)],
                      [max_dist for _ in range(dim)]]


    if method.lower() == 'uniform':
        if method_settings is None:
            method_settings = {}
        method_settings = dict({"scale":1.5}, **method_settings)
        configs = rng.uniform(box_bounds[0], box_bounds[1],
                      size=(N, nrow_names + ncol_names, dim)) * method_settings["scale"]
        starting_configurations = [configs[i].flatten() for i in range(N)]
    else:
        raise ValueError("method can only be uniform.")

    return [x.flatten() for x in starting_configurations]


def _perturb_offsets(N, dim, radius, rng):
    if dim == 2:
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        k = np.arange(N)
        theta = k * golden_angle
        r = radius * np.sqrt((k + 1.0) / N)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        directions = rng.standard_normal((N, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        r = radius * rng.uniform(0.0, 1.0, N) ** (1.0 / dim)
        return directions * r[:, np.newaxis]


def _build_result(mds_result, n, verbose):
  
  optim_results = [r for r in mds_result["optim_results"] if r['success']]
  row_avidity_on = mds_result["args"]["row_avidity_on"]
  col_avidity_on = mds_result["args"]["col_avidity_on"]
  table_bias_on = mds_result["args"]["table_bias_on"]
  
  if len(optim_results)==0:
    print("Warning: none of the optim_results converged."*verbose)
    mds_result.update({"coordinates": [], "stresses": [], "optim_results": []})

    if row_avidity_on:
        mds_result['row_avidities'] = []
    if col_avidity_on:
        mds_result['col_avidities'] = []
    if table_bias_on:
        mds_result['table_biases']  = []
        
    return mds_result

  optimization_coordinates = [r['x'] for r in optim_results]
  stresses                  = [r['fun'] for r in optim_results]

  I = np.argsort(stresses)
  optimization_coordinates = [optimization_coordinates[i] for i in I]
  stresses                 = [stresses[i] for i in I]
  optim_results                  = [optim_results[i] for i in I]
  p_coord  = n["total"] * n["dim"]
  p_row_av = p_coord  + (n["rows"] if row_avidity_on else 0)
  p_col_av = p_row_av + (n["cols"] if col_avidity_on else 0)

  coordinates = [np.reshape(x[:p_coord], (n["total"], n["dim"])) for x in optimization_coordinates]
  ag_names = mds_result["level_sets"]["ag_name"].tolist()
  sr_names = mds_result["level_sets"]["sr_name"].tolist()
  ndim = mds_result["args"]["dim"]
  columns = ['x','y','z'][:ndim] if ndim<3 else [f"z_{i}" for i in range(ndim)]
  
  coordinate_dfs = [pd.DataFrame(c, index=ag_names+sr_names, columns=columns) 
                    for c in coordinates]
  ag_coordinate_dfs = [c.iloc[:len(ag_names),:] for c in coordinate_dfs]
  sr_coordinate_dfs = [c.iloc[len(ag_names):,:] for c in coordinate_dfs]
  
  if len(set(ag_names)) != len(ag_names):
    nonunique = [x for x in ag_names if list(ag_names).count(x)>1]
    warn_once("Set of antigen names is not unique, ag_coordinates will have "
              f"non-unique indices. Non-unique names: {nonunique}")
  
  if len(set(sr_names)) != len(sr_names):
    nonunique = [x for x in sr_names if list(sr_names).count(x)>1]
    warn_once("Set of antigen names is not unique, sr_coordinates will have "
              f"non-unique indices. Non-unique names: {nonunique}")

  mds_result.update({"coordinates":          coordinates,
                     "ag_coordinates":       ag_coordinate_dfs,
                     "sr_coordinates":       sr_coordinate_dfs,
                     "stresses":             stresses,
                     "optim_results":        optim_results
                     })

  if row_avidity_on:
      mds_result['row_avidities'] = [x[p_coord:p_row_av] for x in optimization_coordinates]
  if col_avidity_on:
      mds_result['col_avidities'] = [x[p_row_av:p_col_av] for x in optimization_coordinates]
  if table_bias_on:
      mds_result['table_biases']  = [x[p_col_av:] for x in optimization_coordinates]

  if mds_result["args"]["supress_uncoordinated"]:
    mds_result=\
      supress_params_for_uncoordinated(mds_result)

  return mds_result


def _build_avidity_defaults(bounds, row_avidity_on, col_avidity_on, table_bias_on,
                            indices, n, avidity_options, offset_penalty_scales):

  _E  = np.empty(0, dtype=np.double)
  _Ei = np.empty(0, dtype='i')
  any_av = row_avidity_on or col_avidity_on or table_bias_on

  if any_av:
      default_av = {}
      if row_avidity_on:
          default_av.update({
              'row_avidities':    np.zeros(n["rows"]),
              'row_av_penalties': np.ones(n["rows"]) * offset_penalty_scales["row"],
              'row_av_means':     np.zeros(n["rows"]),
              'row_av_bounds':    [(-np.inf, np.inf)] * n["rows"],
          })
      if col_avidity_on:
          default_av.update({
              'col_avidities':    np.zeros(n["cols"]),
              'col_av_penalties': np.ones(n["cols"]) * offset_penalty_scales["col"],
              'col_av_means':     np.zeros(n["cols"]),
              'col_av_bounds':    [(-np.inf, np.inf)] * n["cols"],
          })
      if row_avidity_on or col_avidity_on:
          default_av['off_avidities'] = np.zeros(n["rows"] + n["cols"], dtype='i')
      if table_bias_on:
          default_av.update({
              'table_biases':         np.zeros(n["tables"]),
              'table_bias_penalties': np.ones(n["tables"]) * offset_penalty_scales["table"],
              'table_bias_means':     np.zeros(n["tables"]),
              'table_bias_bounds':    [(-np.inf, np.inf)] * n["tables"],
              'off_table_biases':     np.array([1] + [0 for _ in range(n["tables"]-1)], dtype='i')

          })

      if avidity_options is None:
          avidity_options = {}
      
      avidity_options = dict(default_av, **avidity_options)
      
      _validate_avidity_sizes(avidity_options, n["rows"], n["cols"], n["tables"])

  row_avidities    = avidity_options['row_avidities'].astype(np.double)    if row_avidity_on else _E
  row_av_penalties = avidity_options['row_av_penalties'].astype(np.double) if row_avidity_on else _E
  row_av_means     = avidity_options['row_av_means'].astype(np.double)     if row_avidity_on else _E
  row_av_bounds    = avidity_options['row_av_bounds']                      if row_avidity_on else []

  col_avidities    = avidity_options['col_avidities'].astype(np.double)    if col_avidity_on else _E
  col_av_penalties = avidity_options['col_av_penalties'].astype(np.double) if col_avidity_on else _E
  col_av_means     = avidity_options['col_av_means'].astype(np.double)     if col_avidity_on else _E
  col_av_bounds    = avidity_options['col_av_bounds']                      if col_avidity_on else []

  off_avidities = avidity_options['off_avidities'] if (row_avidity_on or col_avidity_on) else _Ei

  table_biases         = avidity_options['table_biases'].astype(np.double)         if table_bias_on else _E
  table_bias_penalties = avidity_options['table_bias_penalties'].astype(np.double) if table_bias_on else _E
  table_bias_means     = avidity_options['table_bias_means'].astype(np.double)     if table_bias_on else _E
  table_bias_bounds    = avidity_options['table_bias_bounds']                      if table_bias_on else []
  off_table_biases     = avidity_options['off_table_biases']                       if table_bias_on else _Ei

  bounds += row_av_bounds + col_av_bounds + table_bias_bounds

  av_params = _AvidityParams(
      row_indices=indices["row"],
      col_indices=indices["col"],
      row_coord_indices=indices["row_coord"],
      col_coord_indices=indices["col_coord"],
      row_av_penalties=row_av_penalties,
      row_av_means=row_av_means,
      col_av_penalties=col_av_penalties,
      col_av_means=col_av_means,
      table_bias_penalties=table_bias_penalties,
      table_bias_means=table_bias_means,
      off_avidities=off_avidities,
      off_table_biases=off_table_biases,
      table_indices=indices["table"],
      nrows=n["rows"],
      ncols=n["cols"],
      nrow_names=n["row_names"],
      ncol_names=n["col_names"],
      n_tables=n["tables"],
  )
  
  offsets = {"col":col_avidities, "row":row_avidities, "table":table_biases}
  
  return offsets, bounds, av_params


def _validate_avidity_sizes(avidity_options, nrows, ncols, n_tables=0):
    expected = {
        'row_avidities':        nrows,
        'row_av_penalties':     nrows,
        'row_av_means':         nrows,
        'row_av_bounds':        nrows,
        'col_avidities':        ncols,
        'col_av_penalties':     ncols,
        'col_av_means':         ncols,
        'col_av_bounds':        ncols,
        'off_avidities':        nrows + ncols,
        'table_biases':         n_tables,
        'table_bias_penalties': n_tables,
        'table_bias_means':     n_tables,
        'table_bias_bounds':    n_tables,
        'off_table_biases':     n_tables,
    }
    for key, size in expected.items():
        if key in avidity_options:
            actual = len(avidity_options[key])
            if actual != size:
                raise ValueError(f'{key} has length {actual} but expected {size}')


def _parfun(target_distances, distance_types,
            is_discrete, method, minimize_options, dim,
            row_avidity_on, row_avidities,
            col_avidity_on, col_avidities,
            table_bias_on, table_biases,
            av, bounds, initial_configuration):

    nrows      = av.nrows
    ncols      = av.ncols
    nrow_names = av.nrow_names
    ncol_names = av.ncol_names
    n_tables   = av.n_tables

    coord_len = (nrow_names + ncol_names) * dim
    full_len  = (coord_len
                 + (nrows    if row_avidity_on else 0)
                 + (ncols    if col_avidity_on else 0)
                 + (n_tables if table_bias_on  else 0))

    x0 = list(initial_configuration.flatten())
    if len(x0) == coord_len:
        if row_avidity_on: x0 += row_avidities.flatten().tolist()
        if col_avidity_on: x0 += col_avidities.flatten().tolist()
        if table_bias_on:  x0 += table_biases.flatten().tolist()
    elif len(x0) != full_len:
        raise ValueError(
            f"initial_configuration has length {len(x0)}, expected "
            f"{coord_len} (coords only) or {full_len} (coords + avidities)"
        )

    # passing them as buffers so they don't need to be recreated
    # and destroyed at every iteration of cost and gradient
    grad_buf = np.empty(full_len)
    map_buf  = np.empty(nrow_names * ncol_names)

    fun = lambda x: calculus.stress_and_jacobian_buffered(
        x, grad_buf, map_buf,
        target_distances, distance_types,
        av, is_discrete, dim)

    return minimize(fun, np.array(x0), jac=True, method=method,
                    bounds=bounds, options=minimize_options)



def _pool_init():
      from threadpoolctl import threadpool_limits
      threadpool_limits(limits=1, user_api='blas')

