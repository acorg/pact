#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:48:57 2025

@author: Sina Tureli
"""
import pandas as pd
import time 
import sys
import numpy as np

try:
  #from rpy2.robjects.packages import PackageNotInstalledError
  import PyRacmacs as pr
  _PR_EXISTS=True
except (ModuleNotFoundError):
  _PR_EXISTS=False
  print("Warning PyRacmacs not found will skip optimization with Racmacs.")
  
sys.path.insert(0, "../../")
from pact.MDS import tolerance_annealing, gradient_MDS, generate_initial_conditions
from pact.preprocess import prep_table


if __name__ == "__main__":
  
  titer_table = pd.read_csv("./data/titer_table.csv", index_col=0)
  titer_table = titer_table.iloc[:40,:10]
  titer_table.index.name="antigen"
  titer_table.columns.name="serum"
  
  ags = titer_table.index
  srs = titer_table.columns

  N1 = 100
  t0 = time.time()
 
  t0 = time.time()
  titer_table_f = prep_table(titer_table)
  res = gradient_MDS(titer_table_f, 2, N1, num_cpus=20, is_discrete=True)
  
  for i in range(N1):
    assert np.all(res["ag_coordinates"][i].index == ags)
    assert np.all(res["sr_coordinates"][i].index == srs)
    c = np.concatenate([res["ag_coordinates"][i].values, 
                        res["sr_coordinates"][i].values])
    np.testing.assert_allclose(c, res["coordinates"][i])
  t1 = time.time()


  t0 = time.time()
  res = gradient_MDS(titer_table, 2, N1, num_cpus=20, is_discrete=True)
  
  for i in range(N1):
    assert np.all(res["ag_coordinates"][i].index == ags)
    assert np.all(res["sr_coordinates"][i].index == srs)
    c = np.concatenate([res["ag_coordinates"][i].values, 
                        res["sr_coordinates"][i].values])
    np.testing.assert_allclose(c, res["coordinates"][i])
  t1 = time.time()

