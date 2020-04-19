#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:27:44 2020

@author: m4rz910
"""

from model import Fuzzification, InferenceEngine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'

#construct linguistic value databases
#=============================================================================
wms = Fuzzification.wms()
vbbagvg = Fuzzification.vbbagvg()
#=============================================================================

#=============================================================================
plt.figure(); wms.plot(linestyle='--');plt.grid();
plt.figure(); vbbagvg.plot(linestyle='--');plt.grid();
#=============================================================================

domain=np.arange(2,
                 3+0.01,
                 0.01)

x = Fuzzification.trapezoid(z=domain,
                             c_l=domain.min(),tc=domain.min(),Tc=domain.min(),c_u=domain.max()) #lower is better
plt.figure()
plt.plot(domain,x);plt.grid()

# =============================================================================
x = Fuzzification.trapezoid(z=domain,
                             c_l=domain.min(),tc=domain.max(),Tc=domain.max(),c_u=domain.max()) #higher is better
plt.figure()
plt.plot(domain,x);plt.grid()
# =============================================================================



## N

