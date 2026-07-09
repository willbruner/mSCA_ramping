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

script_name = 'sweepRegsSWCo9'
num_workers = 20

#reg_terms = np.power(np.linspace(-20,40,15), 3)
#reg_terms = reg_terms + np.abs((reg_terms[0]))
#reg_combos = list(product(reg_terms, reg_terms))

reg_terms = np.logspace(0,9, 20)
reg_combos = list(product(reg_terms, reg_terms))

binsize=10
nlags=10

session_path = '../../data/co/co9/co9_12122023'
session_name = session_path.split('/')[-1]
session = MPOptoClass(session_path, laser_channel=3, post=100)

pre = 100
laser_bounds = copy.deepcopy(session.laser['laser_bounds'])
laser_bounds[:,0] = laser_bounds[:,0] - pre
ctrl_bounds = copy.deepcopy(session.laser['ctrl_bounds'])
ctrl_bounds[:,0] = ctrl_bounds[:,0] - pre
train_lasers, test_lasers = train_test_split(laser_bounds, train_size=0.8, 
    random_state=1)

omitTestLaserBounds = omitBoundInBounds(session.climbing_bounds,
        ctrl_bounds)
omitTestLaserBounds = omitBoundInBounds(omitTestLaserBounds, test_lasers)
omitTestLaserBounds = minimumBoundSize(omitTestLaserBounds)

omitLaserBounds = omitBoundInBounds(session.climbing_bounds, ctrl_bounds)
omitLaserBounds = omitBoundInBounds(omitLaserBounds, laser_bounds)
omitLaserBounds = minimumBoundSize(omitLaserBounds)

(session.X, session.Y), pca_objs  = session.format_nlags_PCA(bounds=omitLaserBounds, 
    binsize=binsize, nlags=nlags)
session.Y = session.Y[:, session.num_CFA:]

(session.Xstim, session.Ystim), pca_objs_stim  = \
        session.format_nlags_PCA(bounds=omitTestLaserBounds, binsize=binsize,
        nlags=nlags)
session.Ystim = session.Ystim[:, session.num_CFA:]

session.X_laser, session.Y_laser = session.transformat(pca_objs,
        bounds=test_lasers)
session.Y_laser = session.Y_laser[:, session.num_CFA:]
session.X_ctrl, session.Y_ctrl = session.transformat(pca_objs,
        bounds=ctrl_bounds)
session.Y_ctrl = session.Y_ctrl[:, session.num_CFA:]

session.X_stimlaser, session.Y_stimlaser = session.transformat(pca_objs_stim,
        bounds=test_lasers)
session.Y_stimlaser = session.Y_stimlaser[:, session.num_CFA:]
session.X_stimctrl, session.Y_stimctrl = session.transformat(pca_objs_stim,
        bounds=ctrl_bounds)
session.Y_stimctrl = session.Y_stimctrl[:, session.num_CFA:]

session.sw = session.getSWs(bigBound = omitTestLaserBounds, smallBound = train_lasers,
        sw_weight=25, binsize=binsize, nlags=nlags) 

print('starting workers')






def train_and_test_regterms(reg, session):
 
    C = np.zeros(session.X.shape[1] + 1)
    C[1:(session.num_CFA * 10)+1] = reg[0]
    C[(session.num_CFA * 10)+1:] = reg[1]

    h = train_wiener_filter(session.X, session.Y, C=C)

    predic_laser = test_wiener_filter(session.X_laser, h)
    laser_r2 = weighted_r2(session.Y_laser, predic_laser)
    predic_ctrl = test_wiener_filter(session.X_ctrl, h)
    ctrl_r2 = weighted_r2(session.Y_ctrl, predic_ctrl)

    stim_h = train_wiener_filter(session.Xstim, session.Ystim,
            C=C, sw=session.sw)

    predic_stimlaser = test_wiener_filter(session.X_stimlaser, stim_h)
    stimlaser_r2 = weighted_r2(session.Y_stimlaser, predic_stimlaser)
    predic_stimctrl = test_wiener_filter(session.X_stimctrl, stim_h)
    stimctrl_r2 = weighted_r2(session.Y_stimctrl, predic_stimctrl)

    print('success')

    return (h, laser_r2, ctrl_r2, stim_h, stimlaser_r2, stimctrl_r2)
    
result = [None] * len(reg_combos)
executor = ProcessPoolExecutor(max_workers=num_workers)    
for idx, reg in enumerate(reg_combos):
    result[idx] = executor.submit(train_and_test_regterms, reg, session)
wait(result, timeout=None, return_when=ALL_COMPLETED)
output=[]
for this_result in result:
    output.append(this_result.result())

pdump(output, '../../picklejar/sweepRegsSW_9log_co9.pickle')


