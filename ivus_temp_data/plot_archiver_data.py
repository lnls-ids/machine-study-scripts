#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:59:43 2024

@author: sergio.lordano
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


fname1 = '20241029_IVU18-1.xlsx'
fname2 = '20241029_IVU18-2.xlsx'



df1 = pd.read_excel(fname1, sheet_name=None, usecols='A:B')
df1.pop('Sheet Info')
keys1 = list(df1.keys())


plt.figure()
for key in keys1:
    plt.plot(df1[key]['x'], df1[key]['y'])
# plt.figure()
# plt.plot(df1[0])










