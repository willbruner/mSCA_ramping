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

script_name = 'bulkSweepLags'
num_workers = 2
session_path_list = []
session_name_list = []
co_path = '../../data/co'

post= 50

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(nlag, session, post):
    try:
        binsize = 5
        sigma=5

        pre = nlag * binsize
        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post)
        #remove lasers/ctrl from fit
        omitLaserBounds = omitBoundInBounds(session.climbing_bounds, laser_bounds)
        omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
        omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 

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

        h = train_wiener_filter(X, Y, c=(2,5), sweep='log')

        #generate test data (laser and ctrls)

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

        # test on data

        Yhat_laser = test_wiener_filter(X_laser, h)
        Yhat_ctrl = test_wiener_filter(X_ctrl, h)

        laser_r2 = weighted_r2(Y_laser, Yhat_laser)
        ctrl_r2 = weighted_r2(Y_ctrl,Yhat_ctrl)

        return h, laser_r2, ctrl_r2, nlag, session.inactivate

    except Exception as e:
        print(e)
        return 0

session_output = []
for session_path in session_path_list:
    try:
        session = MPOptoClass(session_path)
        nlags = np.arange(2, 22, 2) 
        output = multipool(analyze_session, session, post, iterable=nlags ,
                num_workers=num_workers)

        session_output.append(output)
    except Exception as e:
        print(e)
        session_output.append(0)

pdump(session_output, f'../../picklejar/bulkSweepLags_post{post}.pickle')
    

