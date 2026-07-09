from src.MPOptoClass import *
from src.modeller import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.experiment import *
from src.wiener_filter import *
from src.analysis import *

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

script_name = 'coSubSweep'
num_workers = 20
session_path_list = []
session_name_list = []
co_path = '../../data/co'

post=50
num_PCs = 20
nlag= 5
binsize=5
sigma=5
random_state=0
sw_weight=50
subsamp_percent = .5

#session_path_list, get all sessions

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session_asymps(session):
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

        #generate laser training data
        X_laser, Y_laser = session.generate_testset(PCs, train_lasers,
                binsize=binsize, nlags=nlag, num_PCs=num_PCs)

        sw = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
                sw_weight=sw_weight, binsize=binsize, nlags=nlag)

        new_ctrls = session.generate_new_ctrls(len(train_lasers), pre=pre,
                post=post)

        cw = session.getSWs(bigBound=omitLaserBounds, smallBound=new_ctrls,
                sw_weight=50, binsize=binsize, nlags=nlag)

        X_r1_mask, X_r2_mask = mask_input_region(X, nlags=nlag,
                num_region1=num_PCs)

        X_r1 = X[:, X_r1_mask]
        X_r2 = X[:, X_r2_mask]

        X_laser_r1 = X_laser[:, X_r1_mask]
        X_laser_r2 = X_laser[:, X_r2_mask]

        h_r1 = train_wiener_filter(X_r1, Y, c=(2,6), sweep='log')
        h_r2 = train_wiener_filter(X_r2, Y, c=(2,6), sweep='log')
        h_sw_r1 = train_wiener_filter(X_r1, Y, c=(2,6), sw=sw, sweep='log')
        h_sw_r2 = train_wiener_filter(X_r2, Y, c=(2,6), sw=sw, sweep='log')
        h_cw_r1 = train_wiener_filter(X_r1, Y, c=(2,6), sw=cw, sweep='log')
        h_cw_r2 = train_wiener_filter(X_r2, Y, c=(2,6), sw=cw, sweep='log')
        h_laser_r1 = train_wiener_filter(X_laser_r1, Y_laser, c=(2,6),
                sweep='log')
        h_laser_r2 = train_wiener_filter(X_laser_r2, Y_laser, c=(2,6),
                sweep='log')


        #generate test data (laser and ctrls)
        X_l, Y_l = session.generate_testset(PCs, test_lasers, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs)

        X_c, Y_c = session.generate_testset(PCs, ctrl_bounds, binsize=binsize,
                nlags=nlag, num_PCs=num_PCs)

        X_c_r1_mask, X_c_r2_mask = mask_input_region(X_c,
                nlags=nlag, num_region1=num_PCs)

        X_c_r1 = X_c[:, X_c_r1_mask]
        X_c_r2 = X_c[:, X_c_r2_mask]

        X_l_r1 = X_l[:, X_c_r1_mask]
        X_l_r2 = X_l[:, X_c_r2_mask]

        Yhat_c_r1 = test_wiener_filter(X_c_r1, h_r1)
        Yhat_c_r2 = test_wiener_filter(X_c_r2, h_r2)
        Yhat_c_sw_r1 = test_wiener_filter(X_c_r1, h_sw_r1)
        Yhat_c_sw_r2 = test_wiener_filter(X_c_r2, h_sw_r2)
        Yhat_c_cw_r1 = test_wiener_filter(X_c_r1, h_cw_r1)
        Yhat_c_cw_r2 = test_wiener_filter(X_c_r2, h_cw_r2)
        Yhat_c_laser_r1 = test_wiener_filter(X_c_r1, h_laser_r1)
        Yhat_c_laser_r2 = test_wiener_filter(X_c_r2, h_laser_r2)

        r1_c_rsquare = weighted_r2(Y_c, Yhat_c_r1)
        r2_c_rsquare = weighted_r2(Y_c, Yhat_c_r2)
        r1_c_sw_rsquare = weighted_r2(Y_c, Yhat_c_sw_r1)
        r2_c_sw_rsquare = weighted_r2(Y_c, Yhat_c_sw_r2)
        r1_c_cw_rsquare = weighted_r2(Y_c, Yhat_c_cw_r1)
        r2_c_cw_rsquare = weighted_r2(Y_c, Yhat_c_cw_r2)
        r1_c_laser_rsquare = weighted_r2(Y_c, Yhat_c_laser_r1)
        r2_c_laser_rsquare = weighted_r2(Y_c, Yhat_c_laser_r2)

        Yhat_l_r1 = test_wiener_filter(X_l_r1, h_r1)
        Yhat_l_r2 = test_wiener_filter(X_l_r2, h_r2)
        Yhat_l_sw_r1 = test_wiener_filter(X_l_r1, h_sw_r1)
        Yhat_l_sw_r2 = test_wiener_filter(X_l_r2, h_sw_r2)
        Yhat_l_cw_r1 = test_wiener_filter(X_l_r1, h_cw_r1)
        Yhat_l_cw_r2 = test_wiener_filter(X_l_r2, h_cw_r2)
        Yhat_l_laser_r1 = test_wiener_filter(X_l_r1, h_laser_r1)
        Yhat_l_laser_r2 = test_wiener_filter(X_l_r2, h_laser_r2)

        r1_l_rsquare = weighted_r2(Y_l, Yhat_l_r1)
        r2_l_rsquare = weighted_r2(Y_l, Yhat_l_r2)
        r1_l_sw_rsquare = weighted_r2(Y_l, Yhat_l_sw_r1)
        r2_l_sw_rsquare = weighted_r2(Y_l, Yhat_l_sw_r2)
        r1_l_cw_rsquare = weighted_r2(Y_l, Yhat_l_cw_r1)
        r2_l_cw_rsquare = weighted_r2(Y_l, Yhat_l_cw_r2)
        r1_l_laser_rsquare = weighted_r2(Y_l, Yhat_l_laser_r1)
        r2_l_laser_rsquare = weighted_r2(Y_l, Yhat_l_laser_r2)

        #i literally cant figure out how to not fuck up autoindent

        return ((r1_l_rsquare, r1_c_rsquare), (r2_l_rsquare, r2_c_rsquare)), ((r1_l_sw_rsquare, r1_c_sw_rsquare), (r2_l_sw_rsquare, r2_c_sw_rsquare)), ((r1_l_cw_rsquare, r1_c_cw_rsquare), (r2_l_cw_rsquare, r2_c_cw_rsquare)), ((r1_l_laser_rsquare, r1_c_laser_rsquare), (r2_l_laser_rsquare, r2_c_laser_rsquare)), (X_l, X_c), (Y_l, Y_c)
    except Exception as e:
        print(f'{session.name} failed asymps, shame them!')
        print(e)
        return 0


def analyze_session(reg, session):
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

        #generate laser training data
        X_laser, Y_laser = session.generate_testset(PCs, train_lasers,
                binsize=binsize, nlags=nlag, num_PCs=num_PCs)

        C = np.zeros(X.shape[1] + 1)
        c_r1_mask, c_r2_mask = mask_h_region(C, nlags=nlag, num_region1=num_PCs)
        C[c_r1_mask] = reg[0]
        C[(c_r2_mask)] = reg[1]

        sw = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
                sw_weight=sw_weight, binsize=binsize, nlags=nlag)

        new_ctrls = session.generate_new_ctrls(len(train_lasers), pre=pre,
                post=post)

        cw = session.getSWs(bigBound=omitLaserBounds, smallBound=new_ctrls,
                sw_weight=50, binsize=binsize, nlags=nlag)

        h = train_wiener_filter(X, Y, c=C)
        h_sw = train_wiener_filter(X, Y, c=C, sw=sw)
        h_cw = train_wiener_filter(X, Y, c=C, sw=cw)
        h_laser = train_wiener_filter(X_laser, Y_laser, c=C)

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

        Yhat_lcw = test_wiener_filter(X_l, h_cw)
        Yhat_ccw= test_wiener_filter(X_c, h_cw)

        Yhat_llaser = test_wiener_filter(X_l, h_laser)
        Yhat_claser = test_wiener_filter(X_c, h_laser)

        l_rsquare = weighted_r2(Y_l, Yhat_l)
        c_rsquare = weighted_r2(Y_c, Yhat_c)

        l_rsquare_sw = weighted_r2(Y_l, Yhat_lsw)
        c_rsquare_sw = weighted_r2(Y_c,Yhat_csw)

        l_rsquare_cw = weighted_r2(Y_l, Yhat_lcw)
        c_rsquare_cw = weighted_r2(Y_c,Yhat_ccw)

        l_rsquare_laser = weighted_r2(Y_l, Yhat_llaser)
        c_rsquare_laser = weighted_r2(Y_c, Yhat_claser)

        return (h, l_rsquare, c_rsquare), (h_sw, l_rsquare_sw, 
                c_rsquare_sw), (h_cw, l_rsquare_cw, c_rsquare_cw), (h_laser, l_rsquare_laser, c_rsquare_laser), (reg, session.name, session.inactivate)

    except Exception as e:
        print(e)
        print(f'{session.name} failed analyze, asshole')
        return 0

session_output = []
session_asymps = []
for session_path in session_path_list:
    try:
        session = MPOptoClass(session_path)
        session.subsampleNeurons(percent_region1=.5, random_state=random_state)
        ustream = np.logspace(0,8,12)
        dstream = np.logspace(0,5,12)
        reg_combos = list(product(dstream, ustream))
 
        session_asymps.append(analyze_session_asymps(session))
        output = multipool(analyze_session, session, iterable=reg_combos, 
                num_workers=num_workers)
        
        session_output.append(output)
       
    except Exception as e:
        print('failed instantiatiion?')
        print(e)
        session_output.append(0)
        session_asymps.append(0)

pdump(session_output,
        f'../../picklejar/coSubSweep_{nlag}nlag_{post}post_{sw_weight}sw.pickle')
pdump(session_asymps,
        f'../../picklejar/coSubSweep_{nlag}nlag_{post}post_{sw_weight}sw.pickle')
