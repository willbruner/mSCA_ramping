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

script_name = 'rrCortFit'
num_workers = 20
session_path_list = []
session_name_list = []
reaching_path = '../../data/reaching'
for root, dirs, files in os.walk(reaching_path, topdown=True):
    for name in dirs:
        if re.search('2020', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(session_path):
    try:
        nlag = 10
        binsize = 10
        num_PCs=20
        session = ReachingClass(session_path)

        (X, Y), pca_objs  = session.format_nlags_PCA(bounds=session.reach_bounds, 
        binsize=binsize, nlags=nlag, num_PCs = num_PCs)

        CFA_mask, RFA_mask = session.mask_input_region(X, nlags=nlag,
                num_CFA=num_PCs)

        X_CFA = X[:, CFA_mask]
        X_RFA = X[:, RFA_mask]

        session_output = {}

        true_y, predic_y,_ = wiener_kfolds(X, Y, c=(2,5), k=4, n_l2=10, sweep='log')
        session_output['all'] = (true_y, predic_y)

        true_y, predic_y_CFA,_ = wiener_kfolds(X_CFA, Y, c=(2,5), k=4, n_l2=10,
            sweep='log')
        session_output['cfa_only'] = (true_y, predic_y_CFA)

        true_y, predic_y_RFA,_ = wiener_kfolds(X_RFA, Y, c=(2,5), k=4, n_l2=10,
            sweep='log')
        session_output['rfa_only'] = (true_y, predic_y_RFA)

        session_output['num_CFA'] = session.num_CFA
        session_output['num_RFA'] = session.num_RFA

        #maybe a lot of steps to get mean FR, but lets do it
        CFA, RFA = session.binner(bounds = session.reach_bounds, binsize=10, concat=True)

        session_output['mean_FRs'] = (np.average(CFA), np.average(RFA))
        name = session_path.split('/')[-1]
        session_output['name'] = name
        print(f'analyzed {name}')

        return session_output
    except Exception as e:
        print(e)
        return 0
result = [None] * len(session_path_list)
executor = ProcessPoolExecutor(max_workers=num_workers)    
for idx, session_path in enumerate(session_path_list):
    result[idx] = executor.submit(analyze_session, session_path)
wait(result, timeout=None, return_when=ALL_COMPLETED)
print('finished!')
output = []
for this_result in result:
    
    output.append(this_result.result())

pdump(output, '../../picklejar/rrCortFit_k4.pickle')


