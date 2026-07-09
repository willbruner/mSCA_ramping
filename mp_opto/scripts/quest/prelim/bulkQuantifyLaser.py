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

script_name = 'bulkQuantifyLaser'
num_workers = 20
session_path_list = []
session_name_list = []
co_path = '../../data/co'
for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(session_path):
    try:
        session = MPOptoClass(session_path)
        #laser_bounds = session.laser['laser_bounds']
        #ctrl_bounds = session.laser['ctrl_bounds']

        laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=-50, post=100)

        omitLaserBounds = omitBoundInBounds(session.climbing_bounds,
                laser_bounds)

        omitLaserBounds = omitBoundInBounds(omitLaserBounds,
                ctrl_bounds)

        CFA_laser, RFA_laser = session.binner(bounds=laser_bounds, binsize=5,
                concat=False)
        CFA_ctrl, RFA_ctrl = session.binner(bounds=ctrl_bounds, binsize=5,
                concat=False)

        avg_cfa = np.average(CFA_ctrl) - np.average(CFA_laser)
        avg_rfa = np.average(RFA_ctrl) - np.average(RFA_laser)
        
        avg_tuple = (avg_cfa, avg_rfa) 

        fractional_cfa = np.average(CFA_ctrl, axis=(0,1)) - \
        np.average(CFA_ctrl, axis=(0,1))

        print(fractional_cfa.shape)

        fractional_rfa = np.average(RFA_ctrl, axis=(0,1)) - \
        np.average(RFA_ctrl, axis=(0,1))

        fractional = (fractional_cfa, fractional_rfa)

        (CFA_PCs, RFA_PCs), (CFA_PCObj, RFA_PCObj)  =\
        session.apply_PCA(bounds = omitLaserBounds, binsize=5,
                smooth_type='two sided', sigma=5, concat=True)

        (_,_), (CFA_l_PCObj, RFA_l_PCObj)  =\
        session.apply_PCA(bounds = laser_bounds, 
                binsize=5, smooth_type='two sided', sigma=5, concat=True)

        (_,_), (CFA_c_PCObj, RFA_c_PCObj)  =\
        session.apply_PCA(bounds = ctrl_bounds, 
                binsize=5, smooth_type='two sided', sigma=5, concat=True)

        CFA_climb, RFA_climb = session.smoother(bounds = omitLaserBounds,
                binsize=5, smooth_type='two sided', sigma=5, concat=True)

        CFA_l_PCs = CFA_l_PCObj.transform(CFA_climb)
        RFA_l_PCs = RFA_l_PCObj.transform(RFA_climb)

        CFA_c_PCs = CFA_c_PCObj.transform(CFA_climb)
        RFA_c_PCs = RFA_c_PCObj.transform(RFA_climb)

        l_cfa_divergence = session.divergence(CFA_PCs, CFA_l_PCs)
        l_rfa_divergence = session.divergence(RFA_PCs, RFA_l_PCs)
        
        c_cfa_divergence = session.divergence(CFA_PCs, CFA_c_PCs)
        c_rfa_divergence = session.divergence(RFA_PCs, RFA_c_PCs)

        l_divergence = (l_cfa_divergence, l_rfa_divergence)
        c_divergence = (c_cfa_divergence, c_rfa_divergence)
        print('success')



        
        
        return avg_tuple, fractional, l_divergence, c_divergence, session.inactivate

    except Exception as e:
        print(e)
        return 0

output = multipool(analyze_session, session_path_list, num_workers)

print('finished!')
pdump(output, '../../picklejar/bulkQuantify50to100post.pickle')


