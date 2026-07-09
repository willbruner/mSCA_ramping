#!/usr/bin/env python
# coding: utf-8

# # freezing 8/24, regularization stuff, theres a somewehat useful script at the bottom

# In[4]:


from src.MPOptoClass import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.helpers.experiment import *
from src.wiener_filter import *

from concurrent.futures import *

import copy
import time
import mat73
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from itertools import permutations, compress, product

script_name = 'regSweep'
reg_terms = np.logspace(0,6,15)
reg_combos = list(product(reg_terms, reg_terms))

session_path = '../../data/co6/co6_10122023'
session_name = session_path.split('/')[-1]
session = MPOptoClass(session_path, post=100)

pre = 100
psth_bounds = copy.deepcopy(session.laser['laser_bounds'])
psth_bounds[:,0] = psth_bounds[:,0] - pre
psth_ctrl_bounds = copy.deepcopy(session.laser['ctrl_bounds'])
psth_ctrl_bounds[:,0] = psth_ctrl_bounds[:,0] - pre


omitLaserBounds = omitBoundInBounds(session.climbing_bounds, psth_bounds)
omitLaserBounds = omitBoundInBounds(omitLaserBounds, psth_ctrl_bounds)
omitLaserBounds = minimumBoundSize(omitLaserBounds)

(session.X, session.Y), pca_objs  = session.format_nlags_PCA(bounds=omitLaserBounds, binsize=10, nlags=10)
session.X_laser, session.Y_laser = session.transformat(pca_objs, bounds=psth_bounds)
session.X_ctrl, session.Y_ctrl = session.transformat(pca_objs, bounds=psth_ctrl_bounds)

def train_and_test_regterms(reg, session):
    C = np.zeros(session.X.shape[1] + 1)
    C[1:(session.num_CFA * 10)+1] = reg[0]
    C[(session.num_CFA * 10)+1:] = reg[1]

    h = train_wiener_filter(session.X, session.Y, C=C)

    predic_laser = test_wiener_filter(session.X_laser, h)
    laser_r2 = weighted_r2(session.Y_laser, predic_laser)
    predic_ctrl = test_wiener_filter(session.X_ctrl, h)
    ctrl_r2 = weighted_r2(session.Y_ctrl, predic_ctrl)

    return (h, laser_r2, ctrl_r2)
    
result = [None] * len(reg_combos)
executor = ProcessPoolExecutor(max_workers=3)    
for idx, reg in enumerate(reg_combos):
    result[idx] = executor.submit(train_and_test_regterms, reg, session)
wait(result, timeout=None, return_when=ALL_COMPLETED)
output=[]
for this_result in result:
    output.append(this_result.result())

pdump(output, '../../picklejar/sweepRegs.pickle')


