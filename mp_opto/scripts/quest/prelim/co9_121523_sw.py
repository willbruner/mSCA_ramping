#!/usr/bin/env python
# coding: utf-8

# # freeizng 12 21 23

# # contains fitting CFA vs RFA PCs for MP datasets/Reaching datsets, and network modelling of opto for new CFA inactivation, dual CFA/RFA recording animals CO7 and CO9
# 

# In[1]:


from src.MPOptoClass import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.helpers.experiment import *
from src.wiener_filter import *
from src.modeller import *
from src.MPRecordClass import *
from src.ReachingClass import *
import scipy
import copy
import time
import mat73 
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import permutations, compress, product



# # 2. looking at two new cfa inactivate animals co7/co9

# In[2]:


session_path = '../../data/co/co9/co9_12152023'
opto_session = MPOptoClass(session_path, laser_channel=3)


# # are stims enough to make the model fit better?

# In[11]:


#num_PCs=20
pre=100
#(all_nlags_PCA, all_cut_PCA), (CFA_PCObj, RFA_PCObj) = session.format_nlags_PCA(bounds=session.climbing_bounds, binsize=10, nlags=10, num_PCs=0)
#all_cut_CFAPCA = all_cut_PCA[:, :session.num_CFA]

laser_bounds = copy.deepcopy(opto_session.laser['laser_bounds'])
laser_bounds[:,0] = laser_bounds[:,0] - pre
train_lasers, test_lasers = train_test_split(laser_bounds, train_size=0.8)

omitTestLaserBounds = omitBoundInBounds(opto_session.climbing_bounds, test_lasers)
omitAllLaserBounds = omitBoundInBounds(omitTestLaserBounds, train_lasers)

(climb_nlags_PCA, climb_cut_PCA), climb_PCObjs = opto_session.format_nlags_PCA(bounds=omitAllLaserBounds, binsize=10, nlags=10)
climb_RFA_PCA = climb_cut_PCA[:, opto_session.num_CFA:]

(laser_nlags_PCA, laser_cut_PCA), laser_PCObjs = opto_session.format_nlags_PCA(bounds=omitTestLaserBounds, binsize=10, nlags=10)
laser_RFA_PCA = laser_cut_PCA[:, opto_session.num_CFA:]

climb_true_y, climb_predic_y, climb_h = wiener_kfolds(climb_nlags_PCA, climb_RFA_PCA, c=(2,5), sweep='log') 
laser_true_y, laser_predic_y, laser_h = wiener_kfolds(laser_nlags_PCA, laser_RFA_PCA, c=(2,5), sweep='log') 

climbtest_nlags_PCA, climbtest_cut_PCA = opto_session.transformat(climb_PCObjs, bounds=test_lasers, binsize=10, nlags=10)
climbtest_RFA_PCA = climbtest_cut_PCA[:, opto_session.num_CFA:]
lasertest_nlags_PCA, lasertest_cut_PCA = opto_session.transformat(laser_PCObjs,
        bounds=test_lasers, binsize=10, nlags=10)
lasertest_RFA_PCA = lasertest_cut_PCA[:, opto_session.num_CFA:]

climbtest_predic_y = test_wiener_filter(climbtest_nlags_PCA, climb_h)
lasertest_predic_y = test_wiener_filter(lasertest_nlags_PCA, laser_h)

climb_xval_score = weighted_r2(climb_true_y, climb_predic_y)
laser_xval_score = weighted_r2(laser_true_y, laser_predic_y)

climb_test_score = weighted_r2(climbtest_RFA_PCA, climbtest_predic_y)
laser_test_score = weighted_r2(lasertest_RFA_PCA, lasertest_predic_y)


# In[9]:


mod_train_lasers_bounds, duration = reBoundInBounds(omitTestLaserBounds, train_lasers)
laser_logical = bounds2Logical(mod_train_lasers_bounds, duration=duration)
laser_logical_trials = unstitchSeams(laser_logical, getSeamsFromBounds(omitTestLaserBounds, binsize=1))
sw_list = []
sw_multiplier=3

for trial in laser_logical_trials:
    temp = bin_timeseries(trial, binsize=10)
    sw_list.append(format_single_array(temp))
    sw = (np.hstack(sw_list))[:-1]
    sw = (sw*sw_multiplier)+1


# In[10]:


nada1, nada2, sw_h = wiener_swkfolds(laser_nlags_PCA, laser_RFA_PCA, sw=sw,
        c=(2,5), n_l2=5, sweep='log')


# In[ ]:


sw_climb_predic_y = test_wiener_filter(laser_nlags_PCA, sw_h)
sw_test_predic_y = test_wiener_filter(lasertest_nlags_PCA, sw_h)
sw_climb_score = weighted_r2(laser_RFA_PCA, sw_climb_predic_y)
sw_test_score = weighted_r2(lasertest_RFA_PCA, sw_test_predic_y)
print(sw_climb_score)
print(sw_test_score)


# In[ ]:


fig22, ax22 = plt.subplots()
ax22.scatter(['only climbing fit'], climb_xval_score, color='tab:blue', label='trained on only climbing')
ax22.scatter(['with stims fit'], laser_xval_score, color='tab:orange', label='trained with stims')
ax22.scatter(['with weighted stims fit'], sw_climb_score, color='tab:green')
ax22.scatter(['only climbing test'], climb_test_score, color='tab:blue')
ax22.scatter(['with stims test'], laser_test_score, color='tab:orange')
ax22.scatter(['with weighted stims test'], sw_test_score, color='tab:green')
ax22.set_ylabel('weighted_r2')
ax22.set_title('predicting held out lasers, w/ laser training vs without')
fig22.tight_layout()
fig22.savefig('with without training.png')


# In[ ]:


print(opto_session.weights_per_region(sw_h))


# In[ ]:


print(opto_session.weights_per_region(h))


# In[ ]:




