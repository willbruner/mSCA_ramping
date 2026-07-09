# script ran on 2/5/24ish, using mp_opto master
# this was to run on all CO mice to quantify basic properties of laser effects

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
import re

script_name = 'bulkModel'
num_workers = 5
post=50

session_path_list = []
session_name_list = []
co_path = '../../data/co'
for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(session_path, post):
    try:
        nlag = 20
        binsize=5
        sigma=5
        session = MPOptoClass(session_path)
        pre = nlag * binsize 

        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)

        omitLaserBounds = omitBoundInBounds(session.climbing_bounds,
                laser_bounds)
        omitLaserBounds = omitBoundInBounds(omitLaserBounds,
                ctrl_bounds)
        omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size=pre+post)

        (X, _), X_PCs = session.format_nlags_PCA(bounds=omitLaserBounds,
                binsize=binsize, nlags=nlag)
        (_, Y), _  = session.format_nlags_PCA_smooth(bounds=omitLaserBounds,
                binsize=binsize, nlags=nlag, smooth_type='future', sigma=sigma)

        if session.inactivate == 'rfa':
            #predict CFA
            Y = Y[:, :session.num_CFA]
        else:
            #predict RFA
            Y = Y[:, session.num_CFA:]

        X_rfa = X[:, session.num_CFA*nlag:]
        X_cfa = X[:, :session.num_CFA*nlag]

        h = train_wiener_filter(X, Y, c=(2,5), sweep='log')
        h_rfa = train_wiener_filter(X_rfa, Y, c=(2,5), sweep='log')
        h_cfa = train_wiener_filter(X_cfa, Y, c=(2,5), sweep='log')

        X_laser, _ = session.transformat(X_PCs, bounds=laser_bounds,
                binsize=binsize, nlags=nlag)
        _, Y_laser = session.transformat_smooth(X_PCs, bounds=laser_bounds,
                binsize=binsize, nlags=nlag, sigma=sigma, smooth_type='future')

        X_ctrl, _ = session.transformat(X_PCs, bounds=ctrl_bounds,
                binsize=binsize, nlags=nlag)
        _, Y_ctrl = session.transformat_smooth(X_PCs, bounds=ctrl_bounds,
                binsize=binsize, nlags=nlag, sigma=sigma, smooth_type='future')

        if session.inactivate == 'rfa':
            #predict CFA
            Y_laser = Y_laser[:, :session.num_CFA]
            Y_ctrl = Y_ctrl[:, :session.num_CFA]
        else:
            Y_laser = Y_laser[:, session.num_CFA:]
            Y_ctrl = Y_ctrl[:, session.num_CFA:]

        X_laser_rfa = X_laser[:, session.num_CFA*nlag:]
        X_laser_cfa = X_laser[:, :session.num_CFA*nlag]

        X_ctrl_rfa = X_ctrl[:, session.num_CFA*nlag:]
        X_ctrl_cfa = X_ctrl[:, :session.num_CFA*nlag]

        Yhat_laser = test_wiener_filter(X_laser, h)
        Yhat_laser_rfa = test_wiener_filter(X_laser_rfa, h_rfa)
        Yhat_laser_cfa = test_wiener_filter(X_laser_cfa, h_cfa)

        Yhat_ctrl = test_wiener_filter(X_ctrl, h)
        Yhat_ctrl_rfa = test_wiener_filter(X_ctrl_rfa, h_rfa)
        Yhat_ctrl_cfa = test_wiener_filter(X_ctrl_cfa, h_cfa)


        laser_r2 = weighted_r2(Y_laser, Yhat_laser)
        laser_rfa_r2 = weighted_r2(Y_laser, Yhat_laser_rfa)
        laser_cfa_r2 = weighted_r2(Y_laser, Yhat_laser_cfa)

        ctrl_r2 = weighted_r2(Y_ctrl,Yhat_ctrl)
        ctrl_rfa_r2 = weighted_r2(Y_ctrl, Yhat_ctrl_rfa)
        ctrl_cfa_r2 = weighted_r2(Y_ctrl, Yhat_ctrl_cfa)
        
        return (laser_r2, laser_cfa_r2, laser_rfa_r2), (ctrl_r2, ctrl_rfa_r2,
                ctrl_cfa_r2)

    except Exception as e:
        print(e)
        return 0

output = multipool(analyze_session, post, iterable=session_path_list,
        num_workers=num_workers)

print('finished!')
pdump(output, f'../../picklejar/bulkQuantify_{post}post.pickle')


