#!/usr/bin/env python
# coding: utf-8

# # freezing 8/24, regularization stuff, theres a somewehat useful script at the bottom

# In[4]:


from src.MPOptoClass import *
from src.MPRecordClass import *
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

script_name = 'mp_corts'
num_workers = 10
session_path_list = []
session_name_list = []
mp_path = '../../data/mp_record'
for root, dirs, files in os.walk(mp_path, topdown=True):
    for name in dirs:
        if re.search('mp.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

def analyze_session(session_path):
    session = MPRecordClass(session_path)
    if (session.CFA == 0 or session.RFA == 0):
        return 0

    print('continuing!')
    (X, Y), pca_objs  = session.format_nlags_PCA(bounds=session.climbing_bounds, 
    binsize=10, nlags=10)
    X_CFA = X[:, :session.num_CFA*10]
    X_RFA = X[:, session.num_CFA*10:]

    true_y, predic_y = wiener_kfolds(X, Y, c=(3,6), k=10)
    session_output['all'] = (true_y, predic_y)

    true_y, predic_y_CFA = wiener_kfolds(X_CFA, Y, c=(3,6), k=10)
    session_output['cfa_only'] = (true_y, predic_y_CFA)

    true_y, predic_y_RFA = wiener_kfolds(X_RFA, Y, c=(3,6), k=10)
    session_output['rfa_only'] = (true_y, predic_y_RFA)

    session_output['num_CFA'] = session.num_CFA
    session_output['num_RFA'] = session.num_RFA

    #maybe a lot of steps to get mean FR, but lets do it
    CFA, RFA = session.binner(bounds = session.climbing_bounds, fs=10, concat=True)

    session_output['mean_FRs'] = (np.average(CFA), np.average(RFA))

    return session_output
result = [None] * len(session_path_list)
executor = ProcessPoolExecutor(max_workers=num_workers)    
for idx, session_path in enumerate(session_path_list):
    result[idx] = executor.submit(analyze_session, session_path)
wait(result, timeout=None, return_when=ALL_COMPLETED)
output = {}
for i in range(len(result)):
    
    output[f'{session_name_list[i]}'] = result[i].result()

pdump(output, '../../picklejar/mp_corts.pickle')


