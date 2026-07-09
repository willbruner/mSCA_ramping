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

session_path_list = session_path_list[:10]
session_name_list = session_name_list[:10]

def analyze_session(session_path):
    try:
        session = MPRecordClass(session_path)
        (X, Y), pca_objs  = session.format_nlags_PCA(bounds=session.climbing_bounds, 
        binsize=10, nlags=10)
        X_CFA = X[:, :session.num_CFA*10]
        X_RFA = X[:, session.num_CFA*10:]

        session_output={}

        true_y, predic_y = wiener_kfolds(X, Y, c=(2,5), k=2, n_l2=2, 
            sweep='log')
        session_output['all'] = weighted_r2(true_y, predic_y)

        true_y, predic_y_CFA = wiener_kfolds(X_CFA, Y, c=(2,5), k=2, n_l2=2, 
            sweep='log')
        session_output['cfa_only'] = weighted_r2(true_y, predic_y_CFA)

        true_y, predic_y_RFA = wiener_kfolds(X_RFA, Y, c=(2,5), k=2, n_l2=2, 
            sweep='log')
        session_output['rfa_only'] = weighted_r2(true_y, predic_y_RFA)

        session_output['num_CFA'] = session.num_CFA
        session_output['num_RFA'] = session.num_RFA

        #maybe a lot of steps to get mean FR, but lets do it
        CFA, RFA = session.binner(bounds = session.climbing_bounds, fs=10, concat=True)

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
print('made it to end?')
output = []
for this_result in result:
    
    output.append(this_result.result())

pdump(output, '../../picklejar/test_mprCortFit_k4.pickle')


