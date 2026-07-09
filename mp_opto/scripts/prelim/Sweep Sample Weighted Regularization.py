#!/usr/bin/env python
# coding: utf-8

# In[6]:


from src.MPOptoClass import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.helpers.experiment import *
from src.wiener_filter import *
from src.modeller import *

import copy
import time
import mat73 
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from itertools import permutations, compress, product



# In[7]:


pre=100
post=100
session_path = '../data/vgat2_06062023'
session = MPOptoClass(session_path, post=post)

num_RFA_PCs = 100

(all_nlags_PCA, all_cut_PCA), (CFA_PCObj, RFA_PCObj) = session.format_nlags_PCA(bounds=session.climbing_bounds, binsize=10, nlags=10)


# In[8]:


weights = np.logspace(0, 6, 15)
weight_combos = list(product(weights, weights))


# In[9]:


psth_bounds = session.get_highlaser_bounds()
psth_ctrl_bounds = session.laser['ctrl_bounds']
psth_ctrl_bounds[:,0] = psth_ctrl_bounds[:,0] - pre
psth_bounds[:,0] = psth_bounds[:,0] - pre
psth_newbounds, climbing_duration = reBoundInBounds(session.climbing_bounds, psth_bounds)
psth_logical = bounds2Logical(psth_newbounds, duration=climbing_duration)
psth_logical_trials = unstitchSeams(psth_logical, getSeamsFromBounds(session.climbing_bounds, binsize=1))

sw_list = []
for trial in psth_logical_trials:
    temp = bin_timeseries(trial, binsize=10)
    sw_list.append(format_single_array(temp))
sw = (np.hstack(sw_list))[:-1]

sw = (sw*4)+1


# In[ ]:


h_list = []
for weights in tqdm(weight_combos):
    C = np.zeros(all_nlags_PCA.shape[1] + 1)
    C[1:(session.num_CFA * 10)+1] = weights[0]
    C[(session.num_CFA * 10)+1:] = weights[1]
    print(C.shape)
    print(all_nlags_PCA.shape)
    
    h_list.append(weighted_parameter_fit(all_nlags_PCA, all_cut_PCA, c=C, sw=sw))
    time.sleep(20)
pdump(h_list, '../picklejar/h_sw_sweep')

