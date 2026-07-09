# this was ran ~10/2023 in order to sweep through differential regularization
# terms for CFA and RFA linear dynamical models.`
from src.MPOptoClass import *
from src.modeller import *
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

script_name = 'toSweepRegs'
num_workers = 40
reg_terms = np.logspace(1, 9, 12)
reg_combos = list(product(reg_terms, reg_terms))

session_path = '../../data/to/to3/to3_06052024'
session_name = session_path.split('/')[-1]
session = MPOptoClass(session_path)

nlags=20
post=50
binsize=5
num_PCs=20
pre = nlags*binsize

THAL_depth_mask = session.mask_depths(greater=300, less=1000)
new_THAL = []
for idx in np.arange(len(THAL_depth_mask)):
    if THAL_depth_mask[idx] == True:
        new_THAL.append(session.RFA['train'][idx])

session.RFA['train'] = new_THAL

omit_laser_bounds, _ = session.adjustLaserBounds(pre=0, post=200)
laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=pre, post=post)
powers = session.getPowers(laser_bounds, num_powers=5, pre=pre, post=post)
high_lasers = laser_bounds[powers>=4,:]
climb_bounds = omitBoundInBounds(session.climbing_bounds, omit_laser_bounds)
climb_bounds = omitBoundInBounds(climb_bounds, ctrl_bounds)
all_bounds = omitBoundInBounds(session.climbing_bounds, ctrl_bounds)

session.all_X, session.all_Y, all_PCs = session.generate_trainset(all_bounds, binsize, nlags, num_PCs)
print(session.all_X.shape)

session.climb_X, session.climb_Y, climb_PCs = session.generate_trainset(climb_bounds, binsize, nlags, num_PCs)
print(session.climb_X.shape)

session.ctrl_aX, session.ctrl_aY = session.generate_testset(all_PCs, ctrl_bounds, binsize, nlags, num_PCs)

session.ctrl_cX, session.ctrl_cY = session.generate_testset(climb_PCs, ctrl_bounds, binsize, nlags, num_PCs)

session.laser_aX, session.laser_aY = session.generate_testset(all_PCs, laser_bounds, binsize, nlags, num_PCs)

session.laser_cX, session.laser_cY = session.generate_testset(climb_PCs, laser_bounds, binsize, nlags, num_PCs)

inputs = (session.laser_aX, session.ctrl_aX), (session.laser_cX, session.ctrl_cX)

session.sw = session.getSWs(bigBound = all_bounds, smallBound = high_lasers,
        sw_weight=10, binsize=binsize, nlags=nlags)

def analyze(reg, session):

    nlags=20
    post=50
    binsize=5
    num_PCs=20
    pre = nlags*binsize

    C = np.zeros(session.all_X.shape[1] + 1)
    c_cfa_mask, c_thal_mask = session.mask_h_region(C, nlags=nlags, num_CFA=num_PCs)
    C[c_cfa_mask] = reg[0]
    C[(c_thal_mask)] = reg[1]

    h_all = train_wiener_filter(session.all_X, session.all_Y, c=C,
            sw=session.sw)
    h_climb = train_wiener_filter(session.climb_X, session.climb_Y, c=C)

    laser_aYhat = test_wiener_filter(session.laser_aX, h_all)
    ctrl_aYhat = test_wiener_filter(session.ctrl_aX, h_all)

    laser_cYhat = test_wiener_filter(session.laser_cX, h_climb)
    ctrl_cYhat = test_wiener_filter(session.ctrl_cX, h_climb)

    laser_ar2 = weighted_r2(session.laser_aY, laser_aYhat)
    ctrl_ar2 = weighted_r2(session.ctrl_aY, ctrl_aYhat)

    laser_cr2 = weighted_r2(session.laser_cY, laser_cYhat)
    ctrl_cr2 = weighted_r2(session.ctrl_cY, ctrl_cYhat)

    return (h_all, laser_ar2, ctrl_ar2), (h_climb, laser_cr2, ctrl_cr2)

output = multipool(analyze, session, iterable=reg_combos,
        num_workers=num_workers)
       
pdump(output, '../../picklejar/to3_06052024_highlasers_10sw_SweepRegs.pickle')
pdump(inputs, '../../picklejar/to3_06052024_highlasers_10sw_SweepRegs_inputs.pickle')

