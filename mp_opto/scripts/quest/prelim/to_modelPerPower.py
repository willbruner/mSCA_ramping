from src.MPOptoClass import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.helpers.experiment import *
from src.wiener_filter import *
from src.modeller import *
from src.MPRecordClass import *
from src.ReachingClass import *
from src.analysis import *
import scipy
import copy
import time
import mat73 
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import permutations, compress, product
from tqdm import tqdm

session_path = '../../data/to/to3/to3_06042024'
session = MPOptoClass(session_path)

nlags = 20
binsize=5
pre = nlags * binsize
post=50
num_PCs=20

laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre, post, only_climb=True)
omit_lasers, omit_ctrls = session.adjustLaserBounds(pre, 200, only_climb=True)
climb_bounds = omitBoundInBounds(session.climbing_bounds, omit_lasers)
climb_bounds = omitBoundInBounds(climb_bounds, omit_ctrls)

X_climb, Y_climb, PCs_climb = session.generate_trainset(climb_bounds, binsize, nlags, num_PCs)
h_climb = train_wiener_filter(X_climb, Y_climb)

powers = session.getPowers(laser_bounds, num_powers=5, pre=pre, post=post)

output = {}
for i in np.arange(1,6):
    output[i] = []

for i in np.arange(1,6):
    this_laser_bounds = laser_bounds[powers==i,:]
    rest_laser_bounds = laser_bounds[powers!=i,:]
    
    Y_test_list = []
    Y_test_hat_list = []

    Y_atest_list = []
    Y_atest_hat_list = []

    Y_ctest_list = []
    Y_ctest_hat_list = []

    for train_index, test_index in leaveBoundOut(this_laser_bounds):
        train_bounds = this_laser_bounds[train_index]
        all_bounds = np.vstack((rest_laser_bounds, train_bounds))
        test_bounds = this_laser_bounds[test_index]

        X, Y, PCs = session.generate_trainset(train_bounds, binsize, nlags, num_PCs)
        X_test, Y_test = session.generate_testset(PCs, test_bounds, binsize, nlags, num_PCs)

        X_all, Y_all, PCs_all = session.generate_trainset(all_bounds, binsize, nlags, num_PCs)
        X_test_all, Y_test_all = session.generate_testset(PCs_all, test_bounds, binsize, nlags, num_PCs)

        X_test_climb, Y_test_climb = session.generate_testset(PCs_climb, test_bounds, binsize, nlags, num_PCs)

        h = train_wiener_filter(X, Y, c=(1,5))
        h_all = train_wiener_filter(X_all, Y_all, c=(1,5))

        Y_test_hat = test_wiener_filter(X_test, h)
        Y_test_list.append(Y_test)
        Y_test_hat_list.append(Y_test_hat)
        
        Y_atest_hat = test_wiener_filter(X_test_all, h_all)
        Y_atest_list.append(Y_test_all)
        Y_atest_hat_list.append(Y_atest_hat)
        
        Y_ctest_hat = test_wiener_filter(X_test_climb, h_climb)
        Y_ctest_list.append(Y_test_climb)
        Y_ctest_hat_list.append(Y_ctest_hat) 

    temp1 = np.array(Y_test_list)
    temp2 = np.array(Y_test_hat_list)

    temp3 = np.array(Y_atest_list)
    temp4 = np.array(Y_atest_hat_list)

    temp5 = np.array(Y_ctest_list)
    temp6 = np.array(Y_ctest_hat_list)

    p = np.reshape(temp1, (temp1.shape[0]*temp1.shape[1], temp1.shape[2]))
    phat = np.reshape(temp2, (temp2.shape[0]*temp2.shape[1], temp2.shape[2]))

    p_all = np.reshape(temp3, (temp3.shape[0]*temp3.shape[1], temp3.shape[2]))
    phat_all = np.reshape(temp4, (temp4.shape[0]*temp4.shape[1], temp4.shape[2]))

    p_climb = np.reshape(temp5, (temp5.shape[0]*temp5.shape[1], temp5.shape[2]))
    phat_climb = np.reshape(temp4, (temp6.shape[0]*temp6.shape[1], temp6.shape[2]))

    output[i].append(weighted_r2(p, phat))
    output[i].append(weighted_r2(p_all, phat_all))
    output[i].append(weighted_r2(p_climb, phat_climb))

output_path = '/home/diya/Documents/mp_opto/picklejar/toPowerSweep.pickle'
pdump(output, output_path) 
