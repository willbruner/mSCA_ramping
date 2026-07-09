# script ran on 12/18/23 ish, using mp_opto master
# this was to run a big sweep on mp_record data to get cfa vs rfa vs both

from src.MPOptoClass import *
from src.MPRecordClass import *
from src.ReachingClass import *
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

script_name = 'mprCortFit'
num_workers = 20
session_path_list = []
session_name_list = []
mp_path = '../../data/mp_record'
for root, dirs, files in os.walk(mp_path, topdown=True):
    for name in dirs:
        if re.search('mp.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(session_path):
    try:
        nlag = 10
        binsize = 10
        num_PCs=20
 
        session = MPRecordClass(session_path)

        (X, Y), pca_objs  = session.format_nlags_PCA(bounds=session.climbing_bounds, 
        binsize=binsize, nlags=nlag, num_PCs=num_PCs)

        CFA_mask, RFA_mask = session.mask_input_region(X, nlags=nlag,
                num_CFA=num_PCs)

        X_CFA = X[:, CFA_mask]
        X_RFA = X[:, RFA_mask]


        session_output = {}

        true_y, predic_y, h = wiener_kfolds(X, Y, c=(2,5), k=4, n_l2=10, 
            sweep='log')
        session_output['all'] = weighted_r2(true_y, predic_y)
        session_output['allcfa'] = weighted_r2(true_y[:, :num_PCs], 
            predic_y[:, :num_PCs])
        session_output['allrfa'] = weighted_r2(true_y[:, num_PCs:], 
            predic_y[:, num_PCs:])
        session_output['h'] = h


        true_y, predic_y_CFA, nada = wiener_kfolds(X_CFA, Y, c=(2,5), k=4, n_l2=10, 
            sweep='log')
        session_output['cfacfa'] = weighted_r2(true_y[:, :num_PCs],
            predic_y_CFA[:, :num_PCs])
        session_output['cfarfa'] = weighted_r2(true_y[:, num_PCs:], 
            predic_y_CFA[:, num_PCs:])

        true_y, predic_y_RFA, nada = wiener_kfolds(X_RFA, Y, c=(2,5), k=4, n_l2=10, 
            sweep='log')
        session_output['rfacfa'] = weighted_r2(true_y[:, :num_PCs], 
            predic_y_RFA[:, :num_PCs])
        session_output['rfarfa'] = weighted_r2(true_y[:, num_PCs:], 
            predic_y_RFA[:, num_PCs:])


        session_output['num_CFA'] = num_PCs
        session_output['num_RFA'] = num_PCs

        #maybe a lot of steps to get mean FR, but lets do it
        CFA, RFA = session.binner(bounds = session.climbing_bounds, binsize=10, concat=True)

        session_output['mean_FRs'] = (np.average(CFA), np.average(RFA))
        session_output['name'] = session_path.split('/')[-1]

        return session_output
    except Exception as e:
        print(e)
        return 0 
result = [None] * len(session_path_list)
executor = ProcessPoolExecutor(max_workers=num_workers)    
for idx, session_path in enumerate(session_path_list):
    result[idx] = executor.submit(analyze_session, session_path)
wait(result, timeout=None, return_when=ALL_COMPLETED)
output = []
for this_result in result:
    
    output.append(this_result.result())

pdump(output, '../../picklejar/mprCortFit_k4.pickle')


