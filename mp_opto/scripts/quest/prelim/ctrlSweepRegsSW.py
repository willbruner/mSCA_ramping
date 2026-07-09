from src.MPOptoClass import *
from src.modeller import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.helpers.experiment import *
from src.wiener_filter import *
import traceback

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

script_name = 'ctrlSweepRegsSW'
num_workers = 25
session_path_list = []
session_name_list = []
co_path = '../../data/co'

post= 50
num_PCs = 20
nlag=20
binsize=5
sigma=5

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session_asymps(session, num_PCs, post, nlag, binsize, sigma):
    try:
        pre = nlag * binsize

        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)
        train_lasers, test_lasers = train_test_split(laser_bounds,
                train_size=.8, random_state=1)
        #remove lasers/ctrl from fit
        num_trials = len(train_lasers)
        train_lasers = session.generate_new_ctrls(num_trials, pre=pre, post=post)
        omitLaserBounds = omitBoundInBounds(session.climbing_bounds,
                test_lasers)
        omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
        omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 

        (X, _), X_PCs = session.format_nlags_PCA(bounds=omitLaserBounds,
                binsize=binsize, nlags=nlag, num_PCs=num_PCs)
        (_, Y), _  = session.format_nlags_PCA_smooth(bounds=omitLaserBounds,
                binsize=binsize, nlags=nlag, smooth_type='future', sigma=sigma,
                num_PCs=num_PCs)

        X_cfa_mask, X_rfa_mask = session.mask_input_region(X, nlags=nlag, num_CFA=num_PCs)

        X_cfa = X[:, X_cfa_mask]
        X_rfa = X[:, X_rfa_mask]

        if session.inactivate == 'rfa':
            #predict CFA
            Y = Y[:, :num_PCs]
        else:
            #predict RFA
            Y = Y[:, num_PCs:]


        h_cfa = train_wiener_filter(X_cfa, Y, c=(2,6), sweep='log')
        h_rfa = train_wiener_filter(X_rfa, Y, c=(2,6), sweep='log')

        X_laser, _ = session.transformat(X_PCs, bounds=test_lasers,
                binsize=binsize, nlags=nlag, num_PCs=num_PCs)
        _, Y_laser = session.transformat_smooth(X_PCs, bounds=test_lasers,
                binsize=binsize, nlags=nlag, sigma=sigma, smooth_type='future',
                num_PCs=num_PCs)

        X_laser_cfa = X_laser[:, X_cfa_mask]
        X_laser_rfa = X_laser[:, X_rfa_mask]

        X_ctrl, _ = session.transformat(X_PCs, bounds=ctrl_bounds,
                binsize=binsize, nlags=nlag, num_PCs=num_PCs)
        _, Y_ctrl = session.transformat_smooth(X_PCs, bounds=ctrl_bounds,
                binsize=binsize, nlags=nlag, sigma=sigma, smooth_type='future',
                num_PCs = num_PCs)

        X_ctrl_cfa = X_ctrl[:, X_cfa_mask]
        X_ctrl_rfa = X_ctrl[:, X_rfa_mask]


        if session.inactivate == 'rfa':
            #predict CFA
            Y_laser = Y_laser[:, :num_PCs]
            Y_ctrl = Y_ctrl[:, :num_PCs]
        else:
            Y_laser = Y_laser[:, num_PCs:]
            Y_ctrl = Y_ctrl[:, num_PCs:]

        # test on data

        Yhat_cfa_laser = test_wiener_filter(X_laser_cfa, h_cfa)
        Yhat_rfa_laser = test_wiener_filter(X_laser_rfa, h_rfa)
        Yhat_cfa_ctrl = test_wiener_filter(X_ctrl_cfa, h_cfa)
        Yhat_rfa_ctrl = test_wiener_filter(X_ctrl_rfa, h_rfa)

        laser_cfa_r2 = weighted_r2(Y_laser, Yhat_cfa_laser)
        laser_rfa_r2 = weighted_r2(Y_laser, Yhat_rfa_laser)
 
        ctrl_cfa_r2 = weighted_r2(Y_ctrl, Yhat_cfa_ctrl)
        ctrl_rfa_r2 = weighted_r2(Y_ctrl, Yhat_rfa_ctrl)

        return (laser_cfa_r2, ctrl_cfa_r2, h_cfa), (laser_rfa_r2, ctrl_rfa_r2,
                h_rfa)
        #need to finish

    except Exception as e:
        print(e)
        return 0


def analyze_session(reg, session, num_PCs, post, nlag, binsize, sigma):
    try:
        pre = nlag * binsize
        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)
        train_lasers, test_lasers = train_test_split(laser_bounds,
                train_size=.8, random_state=1)
        #remove lasers/ctrl from fit
        num_trials=len(train_lasers)
        train_lasers = session.generate_new_ctrls(num_trials, pre=pre, post=post)
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
        c_cfa_mask, c_rfa_mask = session.mask_h_region(C, nlags=nlag, num_CFA=num_PCs)
        C[c_cfa_mask] = reg[0]
        C[(c_rfa_mask)] = reg[1]

        sw = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
                sw_weight=25, binsize=binsize, nlags=nlag)


        if session.inactivate == 'rfa':
            #predict CFA
            Y = Y[:, :num_PCs]
        else:
            #predict RFA
            Y = Y[:, num_PCs:]


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

        cfa_weight, rfa_weight = session.weights_per_region(h, nlags=nlag, num_CFA=num_PCs)
        weight_ratio = np.log10(cfa_weight / rfa_weight)

        cfa_weight_sw, rfa_weight_sw = session.weights_per_region(h_sw, nlags=nlag, 
                num_CFA=num_PCs)
        weight_ratio_sw = np.log10(cfa_weight_sw / rfa_weight_sw)

        cfa_l_input, rfa_l_input = session.relative_input(h, X_laser, nlags=nlag, 
                num_CFA=num_PCs)
        input_l_ratio = np.log10(cfa_l_input / rfa_l_input)

        cfa_c_input, rfa_c_input = session.relative_input(h, X_ctrl, nlags=nlag, 
                num_CFA=num_PCs)
        input_c_ratio = np.log10(cfa_c_input / rfa_c_input)

        cfa_l_input_sw, rfa_l_input_sw = session.relative_input(h_sw, X_laser, 
                nlags=nlag, num_CFA=num_PCs)
        input_l_ratio_sw = np.log10(cfa_l_input_sw / rfa_l_input_sw)

        cfa_c_input_sw, rfa_c_input_sw = session.relative_input(h_sw, X_ctrl, 
                nlags=nlag, num_CFA=num_PCs)
        input_c_ratio_sw = np.log10(cfa_c_input_sw / rfa_c_input_sw)

        return (h, weight_ratio, input_l_ratio, input_c_ratio, laser_r2,
        ctrl_r2), (h_sw, weight_ratio_sw, input_l_ratio_sw, input_c_ratio_sw, laser_r2_sw, ctrl_r2_sw), reg, session.inactivate

    except Exception as e:
        print('analyze session unsucessful')
        print(traceback.format_exc())
        return 0

session_output = []
session_asymps = []
for session_path in session_path_list:
    try:
        session = MPOptoClass(session_path)
        reg_terms = np.logspace(1,9,8)
        reg_combos = list(product(reg_terms, reg_terms))
        
        session_asymps.append(analyze_session_asymps(session, num_PCs, post, 
            nlag, binsize, sigma))
        output = multipool(analyze_session, session, num_PCs, post, nlag, 
                binsize, sigma, iterable=reg_combos, 
                num_workers=num_workers)
        

        session_output.append(output)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        session_output.append(0)
        session_asymps.append(0)

pdump(session_output, f'../../picklejar/ctrlSweepRegsSW_{nlag}nlag_{post}post.pickle')
pdump(session_asymps, f'../../picklejar/ctrlSweepRegsSWasymps_{nlag}nlag_{post}post.pickle')
