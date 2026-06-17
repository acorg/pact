#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:48:57 2025

@author: avicenna
"""
import pandas as pd
import time 
import sys
import numpy as np
try:
  from rpy2.robjects.packages import PackageNotInstalledError
  import PyRacmacs as pr
  _PR_EXISTS=True
except (ModuleNotFoundError, PackageNotInstalledError):
  _PR_EXISTS=False
  print("Warning PyRacmacs not found will skip optimization with Racmacs.")
  
sys.path.insert(0, "../../")
from pact.MDS import generate_initial_conditions, tolerance_annealing, perturb_search,\
  gradient_MDS
from pact.preprocess import prep_table


if __name__ == "__main__":
  '''
  The minima in the map hosted in the Racmacs website is not an easy one
  to find. First of all its column basis is different from max in col, 
  I do not know if it was a mistake or an optimziation but for the tests
  below I set it back to max in col.
  Even with this fixed, Racmacs won't even find it with 10000 optimiziations 
  and it is likely that one needs to also do a grid search by moving antigens 
  and sera around the minima. That is what we do here with pact. Original map 
  stress is: 3536.24 (after fixing the col basis and relaxing). 
  Reoptimization from scratch with Racmacs itself with more than 10000 will 
  often land in a stress around ~3550-3580. 
  '''
  
  titer_table = pd.read_csv("./data/titer_table.csv", index_col=0)
  titer_table.index.name="antigen"
  titer_table.columns.name="serum"
  f = 0.25
  
  N = 30000
  
  if _PR_EXISTS:
    # for some reason map in the website does not have
    # col_basis = max_in_col and it is not a min_col_basis difference
    # either. Possibly a column basis optimization? Anyways, setting
    # it back to max_in_col for fair comparison.
    acmap = pr.read_racmap("./data/h3map2004.ace")
    acmap.fixed_col_bases = np.nanmax(acmap.log_titer_table.values, axis=0)
    acmap = pr.relax_map(acmap)
    print(f"racmacs stress: {acmap.get_stress(0):.2f}")
    
  
  t0 = time.time()
  
  titer_table = prep_table(titer_table)
  res1 = tolerance_annealing(titer_table, 2, N, num_cpus=24, is_discrete=True,
                             ftols=[1e-3, 1e-6], refine_fractions=[f],
                             seed=0)
  
  print(f"\npact tolerance_annealing stress: {res1[-1]['stresses'][0]:.2f}")
  time.sleep(1)
  res2 = perturb_search(res1[-1], num_cpus=24)
  
  t1 = time.time()
  # basin MDS returns a dictionary results where each result is optimizations
  # obtained with successive ftols. Last one will be the one we care about
  # unless we are debugging sth.
  print(f"\npact final stress: {res2['stresses'][0]:.2f}, time={t1-t0:.2f}")
  
  # example run
  # racmacs stress: 3536.24
  # 100%|██████████| 30000/30000 [00:30<00:00, 971.67it/s] 
  # 100%|██████████| 7500/7500 [00:08<00:00, 906.27it/s] 

  # pact tolerance_annealing stress: 3555.23
  # 100%|██████████| 71/71 [00:25<00:00,  2.73it/s, stress_reduction=0.5377%]
  # pact final stress: 3536.11, time=68.87
  
  


  