#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.MPOptoClass import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.helpers.experiment import *
from src.wiener_filter import *
from src.modeller import *

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



# # lets try on 16 good Lasers, 

# In[2]:


pre=100
post=100
session_path = '../data/vgat2_06062023'
session = MPOptoClass(session_path, post=post)


# In[6]:

sw_multiplier_list = np.linspace(0, 5, 11)

weight_combos = pload('../picklejar/good_reg_terms.pickle')
weight_combos = weight_combos.tolist()
weight = weight_combos[3]

# In[4]:


og_psth_bounds = session.get_highlaser_bounds()
og_psth_ctrl_bounds = session.laser['ctrl_bounds']
og_psth_ctrl_bounds[:,0] = og_psth_ctrl_bounds[:,0] - pre
og_psth_bounds[:,0] = og_psth_bounds[:,0] - pre

omitCtrlBounds = omitBoundInBounds(session.climbing_bounds, og_psth_ctrl_bounds)
psth_newbounds, climbing_duration = reBoundInBounds(omitCtrlBounds, og_psth_bounds)


# In[5]:


def heldoutLaser_sw(session, omitCtrlBounds, psth_newbounds, og_psth_bounds, og_psth_ctrl_bounds, weight, sw_multiplier):
    num_lasers = np.arange(og_psth_bounds.shape[0])
    big_laser_true = []
    big_laser_predic = []

    big_ctrl_true = []
    big_ctrl_predic = []
    print('now running')
    for held_out_num in num_lasers:
        train_psth_bounds = copy.deepcopy(psth_newbounds)
        train_psth_bounds = np.delete(train_psth_bounds, held_out_num, axis=0)
        test_psth_bounds = psth_newbounds[held_out_num, :]
        test_psth_bounds = test_psth_bounds[None, :]


        trainBounds = omitBoundInBounds(omitCtrlBounds, test_psth_bounds)
        train_psth_bounds, duration = reBoundInBounds(trainBounds, train_psth_bounds)
        laser_logical = bounds2Logical(train_psth_bounds, duration=duration)
        laser_logical_trials = unstitchSeams(laser_logical, getSeamsFromBounds(trainBounds, binsize=1))

        sw_list = []
        for trial in laser_logical_trials:
            temp = bin_timeseries(trial, binsize=10)
            sw_list.append(format_single_array(temp))
        sw = (np.hstack(sw_list))[:-1]

        sw = (sw*sw_multiplier)+1
        print('formatting!')
        (all_nlags_PCA, all_cut_PCA), PCA_Objs = session.format_nlags_PCA(bounds=trainBounds, binsize=10, nlags=10)
        
        #C = np.zeros(all_nlags_PCA.shape[1] + 1)
        #C[1:(session.num_CFA * 10)+1] = weights[0]
        #C[(session.num_CFA * 10)+1:] = weights[1]
        
        keeper = np.arange(all_nlags_PCA.shape[0])
        prob = copy.deepcopy(sw)
        prob[prob>1] = prob[prob > 1] * 100
        prob = prob/sum(prob)

        C = np.zeros(all_nlags_PCA.shape[1] + 1)
        C[1:(session.num_CFA * 10)+1] = weight[0]
        C[(session.num_CFA * 10)+1:] = weight[1]
 
        keep = np.random.choice(keeper, int(sw.size//2.5), False, prob)
        print('fitting!')
        
        h = weighted_parameter_fit(all_nlags_PCA[keep, :], all_cut_PCA[keep,
            :], c=C, sw=sw[keep])

        temp_bounds = og_psth_bounds[held_out_num,:]
        temp_bounds = temp_bounds[None,:]

        laser_input, laser_true = session.transformat(PCA_Objs, bounds=temp_bounds)
        ctrl_input, ctrl_true = session.transformat(PCA_Objs, bounds=og_psth_ctrl_bounds)

        big_laser_true.append(laser_true)
        big_laser_predic.append(test_wiener_filter(laser_input, h))

        big_ctrl_true.append(ctrl_true)
        big_ctrl_predic.append(test_wiener_filter(ctrl_input, h))
        print('finished_laser')
    
    

    big_laser_true = np.vstack(big_laser_true)
    big_laser_predic = np.vstack(big_laser_predic)
    big_ctrl_true = np.vstack(big_ctrl_true)
    big_ctrl_predic = np.vstack(big_ctrl_predic)

    big_laser_true = big_laser_true[:, session.num_CFA:]
    big_laser_predic = big_laser_predic[:, session.num_CFA:]
    big_ctrl_true = big_ctrl_true[:, session.num_CFA:]
    big_ctrl_predic = big_ctrl_predic[:, session.num_CFA:]
    
    laser_r2 = weighted_r2(big_laser_true, big_laser_predic)
    ctrl_r2 = weighted_r2(big_ctrl_true, big_ctrl_predic)
    print(f'job complete! laser_r2: {laser_r2}, ctrl_r2: {ctrl_r2}')
    
    return [sw, laser_r2, ctrl_r2, h]
 


# In[6]:


num_lasers = np.arange(og_psth_bounds.shape[0])
result= [None] * len(sw_multiplier_list)
executor = ProcessPoolExecutor(max_workers=3)
i=0
for sw_multiplier in sw_multiplier_list:
    #heldoutLaser(session, omitCtrlBounds, psth_newbounds, og_psth_bounds, og_psth_ctrl_bounds, weight)
    result[i] = executor.submit(heldoutLaser_sw, session, omitCtrlBounds,
            psth_newbounds, og_psth_bounds, og_psth_ctrl_bounds, weight, sw_multiplier)
    i=i+1
    print('job submitted')
wait(result, timeout=None, return_when=ALL_COMPLETED)

output=[]
for this_result in result:
    output.append(this_result.result())
#print('now everything finished!')


pdump(output, '../picklejar/LaserSWSweep/goodreg_output.pickle')

#pdump(laser_r2, '../picklejar/run0/laser_r2.pickle')
#pdump(ctrl_r2, '../picklejar/run0/ctrl_r2.pickle')
#pdump(weight_combos, '../picklejar/run0/weight_combos.pickle')


# In[ ]:




