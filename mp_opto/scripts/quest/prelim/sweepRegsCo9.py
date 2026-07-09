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

script_name = 'regSweepCo9'
num_workers = 15
reg_terms = np.logspace(0,9,20)
reg_combos = list(product(reg_terms, reg_terms))

session_path = '../../data/co/co9/co9_12122023'
session_name = session_path.split('/')[-1]
session = MPOptoClass(session_path, laser_channel=3, post=100)

pre = 100
binsize=5
nlags = 4

psth_bounds = copy.deepcopy(session.laser['laser_bounds'])
psth_bounds[:,0] = psth_bounds[:,0] - pre
psth_ctrl_bounds = copy.deepcopy(session.laser['ctrl_bounds'])
psth_ctrl_bounds[:,0] = psth_ctrl_bounds[:,0] - pre


omitLaserBounds = omitBoundInBounds(session.climbing_bounds, psth_bounds)
omitLaserBounds = omitBoundInBounds(omitLaserBounds, psth_ctrl_bounds)
omitLaserBounds = minimumBoundSize(omitLaserBounds)

(session.X, session.Y), pca_objs  = session.format_nlags_PCA(bounds=omitLaserBounds, 
    binsize=binsize, nlags=nlags)
session.Y = session.Y[:, session.num_CFA:]
session.X_laser, session.Y_laser = session.transformat(pca_objs, bounds=psth_bounds)
session.Y_laser = session.Y_laser[:, session.num_CFA:]
session.X_ctrl, session.Y_ctrl = session.transformat(pca_objs, bounds=psth_ctrl_bounds)
session.Y_ctrl = session.Y_ctrl[:, session.num_CFA:]

def train_and_test_regterms(reg, session):
    C = np.zeros(session.X.shape[1] + 1)
    C[1:(session.num_CFA * nlags)+1] = reg[0]
    C[(session.num_CFA * nlags)+1:] = reg[1]

    h = train_wiener_filter(session.X, session.Y, C=C)

    predic_train = test_wiener_filter(session.X, h)
    #predic_train = predic_train[:, :session.num_CFA]
    train_r2 = weighted_r2(session.Y, predic_train)
    predic_laser = test_wiener_filter(session.X_laser, h)
    #predic_laser = predic_laser[:, :session.num_CFA]
    laser_r2 = weighted_r2(session.Y_laser, predic_laser)
    predic_ctrl = test_wiener_filter(session.X_ctrl, h)
    #predic_ctrl = predic_ctrl[:, :session.num_CFA]
    ctrl_r2 = weighted_r2(session.Y_ctrl, predic_ctrl)

    return (h, laser_r2, ctrl_r2, train_r2)
    
result = [None] * len(reg_combos)
executor = ProcessPoolExecutor(max_workers=num_workers)    
for idx, reg in enumerate(reg_combos):
    result[idx] = executor.submit(train_and_test_regterms, reg, session)
wait(result, timeout=None, return_when=ALL_COMPLETED)
output=[]
for this_result in result:
    output.append(this_result.result())

pdump(output, '../../picklejar/sweepRegsco9bin5.pickle')


