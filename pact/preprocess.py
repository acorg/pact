#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 2026

@author: Sina Tureli
"""

import numpy as np
import pandas as pd
from .messages import print_coordination_problems

def prep_table(table, col_bases=None, min_col_basis=None):

  if isinstance(table, pd.DataFrame) and 'titer' not in table.columns:
    table = melt_table(table) # will satisfy condition below automatically

  if isinstance(table, pd.DataFrame) and all(y in table.columns for y in
                                             ["titer","antigen_id","serum_id","antigen","serum"]):

      if 'target_distances' in table.columns and 'distance_type' in table.columns:
          return table

      return titers_to_distances(table, col_bases=col_bases, min_col_basis=min_col_basis)

  raise ValueError("Must be a pandas DataFrame which either has titer, "
                   "antigen_id, serum_id, antigen, serum in columns or a table "
                   "with antigens as index and sera as columns")


def prep_inputs(table, mds_result, col_bases=None, min_col_basis=None,
                verbose=True):

  table = prep_table(table, col_bases, min_col_basis)

  if mds_result["args"]["supress_uncoordinated"]:
    poor, uncoordinated, table=\
      identify_problematic_coordinations(table, mds_result["args"]["dim"])
      
    mds_result["uncoordinated"] = uncoordinated
    mds_result["poorly_coordinated"] = poor
    print_coordination_problems(poor, uncoordinated, mds_result["args"]["dim"],
                                verbose)
  else:
    mds_result["uncoordinated"] = {"ag":[], "sr":[]}
    mds_result["poorly_coordinated"] = {"ag":[], "sr":[]}

  
  #checking that same ID must always map to the same name
  ag_id_name = table[['antigen_id', 'antigen']].drop_duplicates()
  if ag_id_name['antigen_id'].duplicated().any():
      bad = ag_id_name[ag_id_name['antigen_id'].duplicated(keep=False)].values.tolist()
      raise ValueError(f"Same antigen_id maps to multiple antigen names: {bad}")
  sr_id_name = table[['serum_id', 'serum']].drop_duplicates()
  if sr_id_name['serum_id'].duplicated().any():
      bad = sr_id_name[sr_id_name['serum_id'].duplicated(keep=False)].values.tolist()
      raise ValueError(f"Same serum_id maps to multiple serum names: {bad}")

  indices = {}
  n = {}
  level_sets = {}
  
  # Name-based coordinate indices, name order is kept so if the input was
  # a wide table it will be the same serum (column) and antigen (row) order
  # as the original table.
  row_coord_inv, unique_ag_names = pd.factorize(table['antigen'].values, sort=False)
  col_coord_inv, unique_sr_names = pd.factorize(table['serum'].values, sort=False)
  indices["row_coord"] = row_coord_inv.astype('i')
  indices["col_coord"] = col_coord_inv.astype('i')
  
  n["row_names"] = len(unique_ag_names)
  n["col_names"] = len(unique_sr_names)
  n["ag_names"] = unique_ag_names
  n["sr_names"] = unique_sr_names
  n["total"] = n["row_names"] + n["col_names"]

  # ID-based avidity indices 
  unique_ags, row_inv = np.unique(table['antigen_id'].values, return_inverse=True)
  unique_srs, col_inv = np.unique(table['serum_id'].values, return_inverse=True)
  indices["row"] = row_inv.astype('i')
  indices["col"] = col_inv.astype('i')
  n["rows"] = len(unique_ags)
  n["cols"] = len(unique_srs)
  
  n["dim"] = mds_result["args"]["dim"]

  mds_result["processed_table"] = table

  if mds_result["args"]["table_bias_on"]:
      if 'table_id' not in table.columns:
          raise ValueError("table_bias_on=True requires a 'table_id' column in the table.")
      unique_table_ids, table_indices = np.unique(
          table['table_id'].values, return_inverse=True)
      indices["table"] = table_indices.astype('i')
      n["tables"] = len(unique_table_ids)
  else:
      indices["table"] = np.empty(0, dtype='i')
      n["tables"] = 0
      unique_table_ids = None

  if mds_result["args"]["coordinate_bounds"] is None:
      coordinate_bounds = [(-np.inf, np.inf) for _ in range(n["total"]*n["dim"])]
  else:
      coordinate_bounds = mds_result["args"]["coordinate_bounds"].copy()
      
  level_sets["ag_name"] = unique_ag_names
  level_sets["sr_name"] = unique_sr_names
  level_sets["ag_id"] = unique_ags
  level_sets["sr_id"] = unique_srs
  level_sets["table_id"] = unique_table_ids
  mds_result["level_sets"] = level_sets

  return table, n, indices, coordinate_bounds, mds_result


def supress_params_for_uncoordinated(mds_result):
  
  uncoordinated = mds_result["uncoordinated"]
  level_sets = mds_result["level_sets"]
  table_flat = mds_result["processed_table"]
  
  final_configs_arr = np.array(mds_result["coordinates"])
  row_avidities = np.array(mds_result.get("row_avidities",[]))
  col_avidities = np.array(mds_result.get("col_avidities",[]))
  ag_id_ind = []
  sr_id_ind = []
  ag_id_to_name = dict(zip(table_flat.antigen_id, table_flat.antigen))
  sr_id_to_name = dict(zip(table_flat.serum_id, table_flat.serum))
  uncoordinated_ag_ids = [aid for aid in set(table_flat.antigen_id) if 
                          ag_id_to_name[aid] in uncoordinated["ag"]]
  
  uncoordinated_sr_ids = [sid for sid in set(table_flat.serum_id) if 
                          sr_id_to_name[sid] in uncoordinated["sr"]]
  nag = len(level_sets["ag_name"])
  
  
  for ag in uncoordinated["ag"]:
    inda = list(level_sets["ag_name"]).index(ag)
    final_configs_arr[:, inda, :] = np.nan
  
  for aid in uncoordinated_ag_ids:
    ag_id_ind.append(list(level_sets["ag_id"]).index(aid))
    
  for sr in uncoordinated["sr"]:
    inds = list(level_sets["sr_name"]).index(sr)
    final_configs_arr[:, nag+inds, :] = np.nan
  
  for aid in uncoordinated_sr_ids:
    sr_id_ind.append(list(level_sets["sr_id"]).index(aid))
  
  if row_avidities.size>0:
    row_avidities[:, ag_id_ind] = np.nan
    mds_result["row_avidities"] = list(row_avidities)

  if col_avidities.size>0:
    col_avidities[:, sr_id_ind] = np.nan
    mds_result["col_avidities"] = list(col_avidities)
    
  mds_result["coordinates"] = list(final_configs_arr)
  
  return mds_result


def melt_table(titer_table):

  titer_table = titer_table.copy()
  titer_table.index.name="antigen"
  titer_table.columns.name="serum"
  
  antigens = list(titer_table.index)     
  sera     = list(titer_table.columns)  

  
  titer_table = titer_table.reset_index()
  table_flat = pd.melt(titer_table, id_vars="antigen",
                       var_name="serum", value_name="titer")

  table_flat = table_flat.assign(table_id=0)

 
  antigen_idx = {ag: i for i, ag in enumerate(antigens)}
  serum_idx  = {sr: j for j, sr in enumerate(sera)}
  table_flat["antigen_id"] = table_flat["antigen"].map(antigen_idx).astype("i")
  table_flat["serum_id"]    = table_flat["serum"].map(serum_idx).astype("i")

  return table_flat


def titers_to_distances(table, col_bases=None, min_col_basis=None):
    '''
    Convert a long-format titer table to distances suitable for gradient_MDS.

    table must have columns 'antigen', 'serum', 'titer' (and 'antigen_id',
    'serum_id') where titer values are strings: plain numbers, '<x', '>x',
    or '*' (unmeasured).

    Adds columns to the table (leaving existing columns unchanged):
      - 'col_bases':        per-serum-id reference log2-titer
      - 'target_distances': float distances (NaN for type-1 entries)
      - 'distance_type':    0=measured, 1=unmeasured, 2=upper bound, 3=lower bound

    Parameters
    ----------
    col_bases : dict, optional
        {serum_name: float} supplying a pre-computed col_bases per serum name.
        Overrides any existing 'col_bases' column and auto-computation.
        Raises ValueError if any serum in the table has no entry in the dict.
    min_col_basis : float, optional
        Floor applied after col_bases is determined:
        col_bases = max(col_bases, min_col_basis) for every row.

    Returns
    -------
    table : DataFrame
        Copy of input with 'col_bases', 'target_distances', 'distance_type' added.
    '''

    distance_types = np.zeros(len(table), dtype='i')

    s = table['titer'].astype(str).str.strip()
    is_unmeasured = s.isin(['*', 'nan', ''])
    is_lld = s.str.startswith('<').to_numpy()
    is_uld = s.str.startswith('>').to_numpy()
    numeric = pd.to_numeric(s.str.lstrip('<>'), errors='coerce')
    unrecognized = numeric.isna() & ~is_unmeasured

    if unrecognized.any():
        bad = s[unrecognized].tolist()
        raise ValueError(f"Unrecognized titers: {bad}")
    log2_values = np.log2(numeric / 10)

    distance_types[is_lld] = 2
    distance_types[is_uld] = 3
    distance_types[is_unmeasured] = 1

    log2_series = pd.Series(log2_values, index=table.index)
    dtype_series = pd.Series(distance_types, index=table.index)

    table = table.copy()
    table['distance_type'] = distance_types

    if col_bases is not None:
        table['col_bases'] = table['serum'].map(col_bases)
        if table['col_bases'].isna().any():
            raise ValueError("col_bases is missing values for some sera.")
    elif 'col_bases' in table.columns:
        if table['col_bases'].isna().any():
            raise ValueError("col_bases column contains NaN for some rows.")
    else:
        computed = {}
        for serum_id, grp in table.groupby('serum_id'):
            reliable = dtype_series[grp.index].isin([0, 3])
            candidates = log2_series[grp.index][reliable]
            if len(candidates) > 0:
                computed[serum_id] = candidates.max()
            else:
                upper = log2_series[grp.index][dtype_series[grp.index] == 2]
                serum_name = grp['serum'].iloc[0]
                if len(upper) == 0:
                    raise ValueError(
                        f"serum {serum_name!r} (id={serum_id}) has no measurements, "
                        f"it should be removed.")
                computed[serum_id] = upper.max()
        table['col_bases'] = table['serum_id'].map(computed)

    if min_col_basis is not None:
        table['col_bases'] = np.maximum(table['col_bases'], min_col_basis)

    table['target_distances'] = table['col_bases'] - log2_series

    return table


def identify_problematic_coordinations(table_flat, dimensions,
                                     verbose=False):

    uncoordinated_antigens = set([])
    poorly_coordinated_antigens = set([])
    uncoordinated_sera = set([])
    poorly_coordinated_sera = set([])
    additional_uncoordinated = None
    
    skip_ag = set([])
    skip_sr = set([])
    antigen_set = set(table_flat.antigen)
    serum_set = set(table_flat.serum)
    
    while additional_uncoordinated is None or len(additional_uncoordinated)>0:
      
      additional_uncoordinated = set()
      
      for indag,ag in enumerate(antigen_set.difference(skip_ag)):
          obs_mask = (table_flat.antigen.values == ag) & (table_flat.distance_type.values == 0)
          unique_measured = len(np.unique(table_flat.serum.values[obs_mask]))
          if unique_measured == dimensions and ag not in poorly_coordinated_antigens:
              poorly_coordinated_antigens.add(ag)
          elif unique_measured < dimensions:
              if ag in poorly_coordinated_antigens:
                poorly_coordinated_antigens.remove(ag)
              skip_ag.add(ag)
              uncoordinated_antigens.add(ag)
              table_flat.loc[table_flat.antigen == ag, "distance_type"] = 1
              additional_uncoordinated.add(ag)
          
      for indsr,sr in enumerate(serum_set.difference(skip_sr)):
          obs_mask = (table_flat.serum.values == sr) & (table_flat.distance_type.values == 0)
          unique_measured = len(np.unique(table_flat.antigen.values[obs_mask]))

          if unique_measured == dimensions and sr not in poorly_coordinated_sera:
              poorly_coordinated_sera.add(sr)
          elif unique_measured < dimensions:
              if sr in poorly_coordinated_sera:
                poorly_coordinated_sera.remove(sr)
              skip_sr.add(sr)
              uncoordinated_sera.add(sr)
              table_flat.loc[table_flat.serum == sr, "distance_type"] = 1
              additional_uncoordinated.add(sr)
          
    return {"sr": poorly_coordinated_sera, "ag": poorly_coordinated_antigens}, \
              {"sr": uncoordinated_sera, "ag": uncoordinated_antigens},\
                table_flat
