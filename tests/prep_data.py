#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:03:03 2026

@author: avicenna
"""

import PyRacmacs as pr
import pandas as pd
acmap = pr.read_racmap("h3map2004.ace")

ag_ind = [ind for ind,gr in enumerate(acmap.ag_groups) if gr in ["SI87","BE89","BE92"]]
sr_ind = [ind for ind,gr in enumerate(acmap.sr_groups) if gr in ["SI87","BE89","BE92"]]

#acmap = acmap.subset_map(sr_ind, ag_ind)
titer_table = acmap.titer_table
titer_table.to_csv("titer_table.csv", index=True)
table = pd.DataFrame(acmap.table_distances(),
                     index=acmap.ag_names,
                     columns=acmap.sr_names
                     )
table.to_csv("table_distances.csv", index=True)

acmap=\
  pr.make_map_from_table(titer_table, dilution_stepsize=1,
                         number_of_optimizations=5000)
  
print(acmap.get_stress(0))