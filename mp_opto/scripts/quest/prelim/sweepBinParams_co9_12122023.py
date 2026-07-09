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

script_name = 'sweepBinParams_co9_12122023'
session_path = '../../data/co/co9/co9_12122023'
num_workers=20

session_name = session_path.split('/')[-1]
session = MPOptoClass(session_path, laser_channel=3, post=60)

nlags = np.arange(2, 22, 2)

laser_bounds = copy.deepcopy(session.laser['laser_bounds'])
ctrl_bounds = copy.deepcopy(session.laser['ctrl_bounds'])
train_lasers, test_lasers = train_test_split(laser_bounds, train_size=0.8,
    random_state=1)

session.train_lasers = train_lasers
session.test_lasers = test_lasers
session.ctrl_bounds = ctrl_bounds

omitLaserBounds = omitBoundInBounds(session.climbing_bounds, laser_bounds)
omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
session.omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size=160)

omitTestLaserBounds = omitBoundInBounds(session.climbing_bounds, test_lasers)
omitTestLaserBounds = omitBoundInBounds(omitTestLaserBounds, ctrl_bounds)
session.omitTestLaserBounds = minimumBoundSize(omitTestLaserBounds,
    min_size=160)



def sweepNlags(nlag, session):
    try:

        binsize=5
        sigma=5

        pre=nlag * binsize

        train_lasers = copy.deepcopy(session.train_lasers)
        train_lasers[:,0] = train_lasers[:,0] - pre

        test_lasers = copy.deepcopy(session.test_lasers)
        test_lasers[:,0] = test_lasers[:,0] - pre

        ctrl_bounds = copy.deepcopy(session.ctrl_bounds)
        ctrl_bounds[:,0] = ctrl_bounds[:,0] - pre



        tbound = copy.deepcopy(session.omitTestLaserBounds)
        lbound = copy.deepcopy(session.omitLaserBounds)
        tbound[:,0]= tbound[:,0] - pre
        lbound[:,0] = lbound[:,0] - pre
        
        (climb_nlags_PCA, nada), climb_PCObjs = session.format_nlags_PCA(bounds=lbound, binsize=binsize, 
                nlags=nlag)

        (nada, climb_cut_PCA), nada = session.format_nlags_PCA_smooth(bounds=lbound, binsize=binsize, nlags=nlag,
                smooth_type='future', sigma=sigma)	
        climb_RFA_PCA = climb_cut_PCA[:, session.num_CFA:]

        climb_h = train_wiener_filter(climb_nlags_PCA, climb_RFA_PCA, c=(2,5), sweep='log')

        climbtest_nlags_PCA, nada = session.transformat(climb_PCObjs,bounds=test_lasers, 
                binsize=binsize, nlags=nlag)
        nada, climbtest_cut_PCA = session.transformat_smooth(climb_PCObjs, 
                bounds=test_lasers, binsize=binsize, nlags=nlag,
                sigma=sigma)	

        climbctrl_nlags_PCA, nada = session.transformat(climb_PCObjs,bounds=ctrl_bounds, 
                binsize=binsize, nlags=nlag)
        nada, climbctrl_cut_PCA = session.transformat_smooth(climb_PCObjs, 
                bounds=ctrl_bounds, binsize=binsize, nlags=nlag,
                sigma=sigma)	
 
        
        climbtest_RFA_PCA = climbtest_cut_PCA[:, session.num_CFA:]
        climbctrl_RFA_PCA = climbctrl_cut_PCA[:, session.num_CFA:]

        climbctrl_predic_y = test_wiener_filter(climbctrl_nlags_PCA, climb_h)
        climbtest_predic_y = test_wiener_filter(climbtest_nlags_PCA, climb_h)
        climbctrl_r2 = weighted_r2(climbctrl_RFA_PCA, climbctrl_predic_y)
        climbtest_r2 = weighted_r2(climbtest_RFA_PCA, climbtest_predic_y)

        #now retrain with lasers

        (laser_nlags_PCA, nada), laser_PCObjs = session.format_nlags_PCA(bounds=tbound, 
                binsize=binsize, nlags=nlag)

        (nada, laser_cut_PCA), nada = session.format_nlags_PCA_smooth(bounds=tbound, 
                binsize=binsize, nlags=nlag, smooth_type='future',
                sigma=sigma)	
        laser_RFA_PCA = laser_cut_PCA[:, session.num_CFA:]

        sw = session.getSWs(bigBound=tbound, smallBound=train_lasers,
                sw_weight=10, binsize=binsize, nlags=nlag)

        laser_h = train_wiener_filter(laser_nlags_PCA, laser_RFA_PCA,
                c=(2,5), sw=sw, sweep='log')

        lasertest_nlags_PCA, nada = session.transformat(laser_PCObjs,bounds=test_lasers, 
                binsize=binsize, nlags=nlag)
        nadaa, lasertest_cut_PCA = session.transformat_smooth(laser_PCObjs,
                bounds=test_lasers, binsize=binsize, nlags=nlag,
                sigma=sigma)	
        lasertest_RFA_PCA = lasertest_cut_PCA[:, session.num_CFA:]


        laserctrl_nlags_PCA, nada = session.transformat(laser_PCObjs,
                bounds=ctrl_bounds, binsize=binsize, nlags=nlag)
        nadaa, laserctrl_cut_PCA = session.transformat_smooth(laser_PCObjs,
                bounds=ctrl_bounds, binsize=binsize, nlags=nlag,
                sigma=sigma)	
        laserctrl_RFA_PCA = laserctrl_cut_PCA[:, session.num_CFA:]

        laserctrl_predic_y = test_wiener_filter(laserctrl_nlags_PCA, laser_h)
        lasertest_predic_y = test_wiener_filter(lasertest_nlags_PCA, laser_h)
        laserctrl_r2 = weighted_r2(laserctrl_RFA_PCA, laserctrl_predic_y)
        lasertest_r2 = weighted_r2(lasertest_RFA_PCA, lasertest_predic_y)



        print('success')

        return climb_h, laser_h, (climbctrl_r2, climbtest_r2), (laserctrl_r2,
                lasertest_r2), nlag
    except Exception as e:
        print(e)
        return 0

result = [None] * len(nlags)
executor = ProcessPoolExecutor(max_workers=num_workers)    
for idx, nlag in enumerate(nlags):
    result[idx] = executor.submit(sweepNlags, nlag, session)
wait(result, timeout=None, return_when=ALL_COMPLETED)
output=[]
for this_result in result:
    output.append(this_result.result())

pdump(output, '../../picklejar/sweepBinParamsco9_10sw_post60.pickle')


