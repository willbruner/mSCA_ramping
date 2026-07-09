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

script_name = 'bulkSweepRegs'
num_workers = 20
session_path_list = []
session_name_list = []
co_path = '../../data/co'

post= 50
num_PCs = 20

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(reg, session, num_PCs, post):
    try:
        nlag=20
        binsize = 5
        sigma=5

        pre = nlag * binsize
        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)
        #remove lasers/ctrl from fit
        omitLaserBounds = omitBoundInBounds(session.climbing_bounds, laser_bounds)
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


        if session.inactivate == 'rfa':
            #predict CFA
            Y = Y[:, :num_PCs]
        else:
            #predict RFA
            Y = Y[:, num_PCs:]

        h = train_wiener_filter(X, Y, c=C)

        #generate test data (laser and ctrls)

        X_laser, _ = session.transformat(X_PCs, bounds=laser_bounds,
                binsize=binsize, nlags=nlag, num_PCs=num_PCs)
        _, Y_laser = session.transformat_smooth(X_PCs, bounds=laser_bounds,
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

        laser_r2 = weighted_r2(Y_laser, Yhat_laser)
        ctrl_r2 = weighted_r2(Y_ctrl,Yhat_ctrl)

        return (h, laser_r2, ctrl_r2, reg, session.inactivate)

    except Exception as e:
        print(e)
        return 0

session_output = []
for session_path in session_path_list:
    try:
        session = MPOptoClass(session_path)
        reg_terms = np.logspace(1,6,12)
        reg_combos = list(product(reg_terms, reg_terms))

        output = multipool(analyze_session, session, num_PCs, post, iterable=reg_combos,
                num_workers=num_workers)
        

        session_output.append(output)
    except Exception as e:
        print(e)
        session_output.append(0)

pdump(session_output, f'../../picklejar/bulkSweepRegs_{post}post.pickle')
    

