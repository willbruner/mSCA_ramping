from src.MPOptoClass import *
from src.modeller import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.experiment import *
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

script_name = 'coSweepRegsSW'
num_workers = 30
session_path_list = []
session_name_list = []
co_path = '../../data/co'

post=50
num_PCs = 20
nlag= 20
binsize=5
sigma=5
random_state=2
sw_weight=50

#session_path_list, get all sessions

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
                train_size=.8, random_state=random_state)
        #remove lasers/ctrl from fit
        omitLaserBounds = omitBoundInBounds(session.climbing_bounds,
                test_lasers)
        omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
        omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 

        (X,Y), PCs = session.generate_trainset(omitLaserBounds, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs) 

        sw = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
                sw_weight=sw_weight, binsize=binsize, nlags=nlag)

        X_r1_mask, X_r2_mask = session.mask_input_region(X, nlags=nlag,
                num_region1=num_PCs)

        X_r1 = X[:, X_r1_mask]
        X_r2 = X[:, X_r2_mask]

        h_r1 = train_wiener_filter(X_r1, Y, c=(2,6), sweep='log')
        h_r2 = train_wiener_filter(X_r2, Y, c=(2,6), sweep='log')
        h_sw_r1 = train_wiener_filter(X_r1, Y, c=(2,6), sw=sw, sweep='log')
        h_sw_r2 = train_wiener_filter(X_r2, Y, c=(2,6), sw=sw, sweep='log')



        Xtest, Ytest = session.generate_testset(PCs, ctrl_bounds, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs)

        Xtest_r1_mask, Xtest_r2_mask = session.mask_input_region(Xtest,
                nlags=nlag, num_region1=num_PCs)

        Xtest_r1 = Xtest[:, Xtest_r1_mask]
        Xtest_r2 = Xtest[:, Xtest_r2_mask]

        Yhat_r1 = test_wiener_filter(Xtest_r1, h_r1)
        Yhat_r2 = test_wiener_filter(Xtest_r2, h_r2)
        Yhat_sw_r1 = test_wiener_filter(Xtest_r1, h_sw_r1)
        Yhat_sw_r2 = test_wiener_filter(Xtest_r2, h_sw_r2)

        r1_rsquare = weighted_r2(Ytest, Yhat_r1)
        r2_rsquare = weighted_r2(Ytest, Yhat_r2)
        r1_sw_rsquare = weighted_r2(Ytest, Yhat_sw_r1)
        r2_sw_rsquare = weighted_r2(Ytest, Yhat_sw_r2)

        #generate test data (laser and ctrls)
        X_l, Y_l = session.generate_testset(PCs, test_lasers, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs)

        X_c = Xtest
        Y_c = Ytest

        return (r1_rsquare, r1_sw_rsquare), (r2_rsquare, r2_sw_rsquare), (X_l,
                X_c), (Y_l, Y_c)

    except Exception as e:
        print(f'{session.name} failed asymps, shame them!')
        print(e)
        return 0


def analyze_session(reg, session, num_PCs, post, nlag, binsize, sigma):
    try:
        pre = nlag * binsize
        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)
        train_lasers, test_lasers = train_test_split(laser_bounds,
                train_size=.8, random_state=random_state)
        #remove lasers/ctrl from fit
        omitLaserBounds = omitBoundInBounds(session.climbing_bounds,
                test_lasers)
        omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
        omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 

        (X,Y), PCs = session.generate_trainset(omitLaserBounds, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs) 

        C = np.zeros(X.shape[1] + 1)
        c_r1_mask, c_r2_mask = session.mask_h_region(C, nlags=nlag, num_region1=num_PCs)
        C[c_r1_mask] = reg[0]
        C[(c_r2_mask)] = reg[1]

        sw = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
                sw_weight=sw_weight, binsize=binsize, nlags=nlag)

        h = train_wiener_filter(X, Y, c=C)
        h_sw = train_wiener_filter(X, Y, c=C, sw=sw)

        #generate test data (laser and ctrls)
        X_l, Y_l = session.generate_testset(PCs, test_lasers, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs)

        X_c, Y_c = session.generate_testset(PCs, ctrl_bounds, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs)

        # test on data

        Yhat_l = test_wiener_filter(X_l, h)
        Yhat_c = test_wiener_filter(X_c, h)

        Yhat_lsw = test_wiener_filter(X_l, h_sw)
        Yhat_csw= test_wiener_filter(X_c, h_sw)

        l_rsquare = weighted_r2(Y_l, Yhat_l)
        c_rsquare = weighted_r2(Y_c, Yhat_c)

        l_rsquare_sw = weighted_r2(Y_l, Yhat_lsw)
        c_rsquare_sw = weighted_r2(Y_c,Yhat_csw)

        return (h, l_rsquare, c_rsquare), (h_sw, l_rsquare_sw, 
                c_rsquare_sw), (reg, session.name, session.inactivate)

    except Exception as e:
        print(e)
        print(f'{session.name} failed analyze, asshole')
        return 0

session_output = []
session_asymps = []
for session_path in session_path_list[::2]:
    try:
        session = MPOptoClass(session_path)
        ustream = np.logspace(0,8,12)
        dstream = np.logspace(0,5,12)
        reg_combos = list(product(dstream, ustream))
 
        session_asymps.append(analyze_session_asymps(session, num_PCs, post, 
            nlag, binsize, sigma))
        output = multipool(analyze_session, session, num_PCs, post, nlag, 
                binsize, sigma, iterable=reg_combos, 
                num_workers=num_workers)
        
        session_output.append(output)
       
    except Exception as e:
        print('failed instantiatiion?')
        print(e)
        session_output.append(0)
        session_asymps.append(0)

pdump(session_output,
        f'../../picklejar/coSweepRegsSW_{nlag}nlag_{post}post_{sw_weight}sw.pickle')
pdump(session_asymps,
        f'../../picklejar/coSweepRegsSWasymps_{nlag}nlag_{post}post_{sw_weight}sw.pickle')
