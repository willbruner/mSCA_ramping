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
import re

post= 50
num_PCs = 3

def analyze_session(reg, session, num_PCs, post):
    nlag=20
    binsize = 5
    sigma=5

    pre = nlag * binsize
    laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)
    train_lasers, test_lasers = train_test_split(laser_bounds,
            train_size=.8, random_state=1)
    #remove lasers/ctrl from fit
    omitLaserBounds = omitBoundInBounds(session.climbing_bounds,
            test_lasers)
    omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
    omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 

    (X, _), X_PCs = session.format_nlags_PCA(bounds=omitLaserBounds,
            binsize=binsize, nlags=nlag, num_PCs=num_PCs)
    (_, Y), _  = session.format_nlags_PCA_smooth(bounds=omitLaserBounds,
            binsize=binsize, nlags=nlag, smooth_type='future', sigma=sigma,
            num_PCs=num_PCs)

    C = np.zeros(X.shape[1] + 1)
    C[1:(num_PCs * nlag)+1] = reg[0]
    C[(num_PCs * nlag)+1:] = reg[1]

    sw = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
            sw_weight=25, binsize=binsize, nlags=nlag)


    if session.inactivate == 'rfa':
        #predict CFA
        Y = Y[:, :num_PCs]
    else:
        #predict RFA
        Y = Y[:, num_PCs:]

    print(len(sw))
    print(X.shape)

    h = train_wiener_filter(X, Y, c=C)
    h_sw = train_wiener_filter(X, Y, c=C, sw=sw)

    #generate test data (laser and ctrls)

    X_laser, _ = session.transformat(X_PCs, bounds=test_lasers,
            binsize=binsize, nlags=nlag, num_PCs=num_PCs)
    _, Y_laser = session.transformat_smooth(X_PCs, bounds=test_lasers,
            binsize=binsize, nlags=nlag, sigma=sigma, smooth_type='future',
            num_PCs=num_PCs)

    X_ctrl, _ = session.transformat(X_PCs, bounds=ctrl_bounds,
            binsize=binsize, nlags=nlag, num_PCs=num_PCs)
    _, Y_ctrl = session.transformat_smooth(X_PCs, bounds=ctrl_bounds,
            binsize=binsize, nlags=nlag, sigma=sigma, smooth_type='future',
            num_PCs = num_PCs)

    if session.inactivate == 'rfa':
        #predict CFA
        Y_laser = Y_laser[:, :num_PCs]
        Y_ctrl = Y_ctrl[:, :num_PCs]
    else:
        Y_laser = Y_laser[:, num_PCs:]
        Y_ctrl = Y_ctrl[:, num_PCs:]

    # test on data

    Yhat_laser = test_wiener_filter(X_laser, h)
    Yhat_ctrl = test_wiener_filter(X_ctrl, h)

    Yhat_laser_sw = test_wiener_filter(X_laser, h_sw)
    Yhat_ctrl_sw= test_wiener_filter(X_ctrl, h_sw)


    laser_r2 = weighted_r2(Y_laser, Yhat_laser)
    ctrl_r2 = weighted_r2(Y_ctrl,Yhat_ctrl)

    laser_r2_sw = weighted_r2(Y_laser, Yhat_laser_sw)
    ctrl_r2_sw = weighted_r2(Y_ctrl,Yhat_ctrl_sw)
    return (h, laser_r2, ctrl_r2), (h_sw, laser_r2_sw, ctrl_r2_sw), reg, session.inactivate

session_path = '../../data/co/co9/co9_12122023'
session = MPOptoClass(session_path)
reg_terms = np.logspace(1,6,3)
reg_combos = list(product(reg_terms, reg_terms))

for reg in reg_combos:
    analyze_session(reg, session, num_PCs=3, post=50)


#pdump(session_output, f'../../picklejar/test_bulkSweepRegsSW_{post}post.pickle')
    

