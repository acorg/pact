#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 2026

@author: Sina Tureli
"""

_shown = set()


def warn_once(key, message):
    if key not in _shown:
        _shown.add(key)
        print(message)


def reset_warnings():
    _shown.clear()


def print_coordination_problems(poorly_coordinated, uncoordinated, ndim):
  
  for name in poorly_coordinated["ag"]:
      warn_once(("uncertain_antigen", name),
                f'Warning: antigen {name!r} has exactly {ndim} measured '
                f'titration(s); its position will be uncertain.')
  for name in poorly_coordinated["sr"]:
      warn_once(("uncertain_serum", name),
                f'Warning: serum {name!r} has exactly {ndim} measured '
                f'titration(s); its position will be uncertain.')
  for name in uncoordinated["ag"]:
      warn_once(("uncoordinated_antigen", name),
                f'Warning: antigen {name!r} has fewer than {ndim} measured '
                f'titration(s); its coordinates are set to NaN.')
  for name in uncoordinated["sr"]:
      warn_once(("uncoordinated_serum", name),
                f'Warning: serum {name!r} has fewer than {ndim} measured '
                f'titration(s); its coordinates are set to NaN.')
