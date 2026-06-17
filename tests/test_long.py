#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:48:57 2025

@author: avicenna
"""
import pandas as pd
import sys
try:
  from rpy2.robjects.packages import PackageNotInstalledError
  import PyRacmacs as pr
  _PR_EXISTS=True
except (ModuleNotFoundError, PackageNotInstalledError):
  _PR_EXISTS=False
  print("Warning PyRacmacs not found will skip optimization with Racmacs.")
  
import numpy as np
sys.path.insert(0, "../../")
import pact
from pact.MDS import generate_initial_conditions, tolerance_annealing, gradient_MDS


def _get_year(x):
  if x=='NIB-8':
    return np.nan
  if x=='ZZ/0/00':
    return np.nan
  
  if '/' not in x:
    return np.nan
  
  try:
    year = x.split('/')[-1]
    if len(year)==2 and int(year)<20:
      return int('20'+year)
    if len(year)==2 and int(year)>20:
      return int('19'+year)
    return int(year)
  except:
    return np.nan

if __name__ == "__main__":
  titer_table = pd.read_csv("./data/test_long.csv", index_col=0)
  
  N = 10000
  dims = 2
  
  inits=\
    generate_initial_conditions(titer_table, 2,  N, is_discrete=True)
    
  res1 = gradient_MDS(titer_table, dims, inits, True, num_cpus=24,
                      supress_uncoordinated=True, col_avidity_on=True,
                      table_bias_on=True, offset_penalty_scales=0.8)
  
  res2 = tolerance_annealing(titer_table, 2, inits, num_cpus=24, is_discrete=True,
                             ftols=[1e-3, 1e-6], refine_fractions=[0.25])
  
  
  print(res1['stresses'][0])
  print(res2['stresses'][0])
  
  ag_id_to_name = {
                ag_id:ag_name for ag_id,ag_name in zip(res2["processed_table"]["antigen_id"].values, 
                                                       res2["processed_table"]["antigen"].values)
                }
  
  sr_id_to_name = {
                ag_id:ag_name for ag_id,ag_name in zip(res2["processed_table"]["serum_id"].values, 
                                                       res2["processed_table"]["serum"].values)
                }
  
  
  if _PR_EXISTS:
    acmap = pr.read_racmap("./data/opt_full_styled.ace")

    acmap.ag_names = [ag_id_to_name[x] for x in acmap.ag_ids]
    acmap.sr_names = [sr_id_to_name[x] for x in acmap.sr_ids]
    
    # just showing how to create a racmap object from PyRacmacs, but is not necessary for plotting
    # the object m below is sufficient
    new_map = pact.plot_lib.make_racmap(acmap, res2)
    
    ag_years = [_get_year(x) for x in new_map.ag_names]
    sr_years = [_get_year(x) for x in new_map.sr_names]
    
    m = pact.Map.from_result(res2)
    m.ag_coloring = {"Year":ag_years}
    m.sr_coloring = {"Year":sr_years}
    m.ag_fill = ["rgba(0,0,0,0)" if x=="white" else x for x in new_map.ag_fills]
    m.rotate(80)
    m.reflect('x')
    m.sr_zorder = [-1]*m.n_sera
    pact.view(m)
  else:
    m = pact.Map.from_result(res2)
    pact.view(m)