#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:48:57 2025

@author: avicenna
"""
import pandas as pd
import time 
import sys

try:
  from rpy2.robjects.packages import PackageNotInstalledError
  import PyRacmacs as pr
  _PR_EXISTS=True
except Exception:  # rpy2/PyRacmacs optional; PackageNotInstalledError may be undefined here
  _PR_EXISTS=False
  print("Warning PyRacmacs not found will skip optimization with Racmacs.")
  
sys.path.insert(0, "../../")
from pact.MDS import tolerance_annealing, gradient_MDS, generate_initial_conditions
from pact.preprocess import prep_table


if __name__ == "__main__":
  
  titer_table = pd.read_csv("./data/titer_table.csv", index_col=0)
  titer_table.index.name="antigen"
  titer_table.columns.name="serum"
  f = 0.25
  
  N1 = 10000 
  N2 = 15000 # if N=10000 for tolerance annealing, then it actually runs 12500
             # optimziations with the settings below so set N=12500 for others
             # for fair benchmarking
  
  if _PR_EXISTS:
    t0 = time.time()
    acmap = pr.make_map_from_table(titer_table, dilution_stepsize=1,
                                   number_of_dimensions=2, 
                                   number_of_optimizations=N2)
    # note that pact will perform N + N*f optimizations hence why the number above

    t1 = time.time()
    print(f"racmacs stress: {acmap.get_stress(0):.2f}, time={t1-t0:.2f}")
    
  
  
  # even though you can just supply number of initial conditions to the optimizers
  # for consistency I am pre-generating them
  inits=\
    generate_initial_conditions(titer_table, 2,  N2, is_discrete=True)
    
  
  t0 = time.time()
  titer_table = prep_table(titer_table)
  res = tolerance_annealing(titer_table, 2, inits[:N1], num_cpus=24, is_discrete=True,
                            ftols=[1e-3, 1e-6], refine_fractions=[f])
  t1 = time.time()
  # basin MDS returns a dictionary results where each result is optimizations
  # obtained with successive ftols. Last one will be the one we care about
  # unless we are debugging sth.
  print(f"\npact tolerance annealing stress: {res[-1]['stresses'][0]:.2f}, time={t1-t0:.2f}")
  
  
  t0 = time.time()
  titer_table = prep_table(titer_table)
  res = gradient_MDS(titer_table, 2, inits, num_cpus=24, is_discrete=True)
  t1 = time.time()
  print(f"\npact gradient MDS stress: {res['stresses'][0]:.2f}, time={t1-t0:.2f}")

  # example run:
  # racmacs stress: 3569.69, time=76.40

  # 100%|██████████| 10000/10000 [00:15<00:00, 632.98it/s]
  # 100%|██████████| 2500/2500 [00:03<00:00, 629.85it/s]
  # pact tolerance annealing stress: 3563.53, time=20.57
  
  # 100%|██████████| 12500/12500 [00:25<00:00, 493.26it/s]
  # pact gradient MDS stress: 3563.16, time=25.86
  
  # Note that this is a quite complicated map; lots of local optima due sparsity
  # of titration patterns. So it makes a good dataset for robustness of "cutting the
  # corner" methods like tolerance_annealing. tolerance_annealing generally returns
  # similar stress to gradient_MDS (with +-1) but can occasionally be somewhat
  # higher (within +-10), most likely due to single antigens/sera trapped in local
  # minima, so use your best judgement when using it.
  
  
  