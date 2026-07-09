import numpy as np
from src.wiener_filter import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import copy
from tqdm import tqdm

def leaveBoundOut(bounds):
    k=len(bounds)
    kf = KFold(n_splits=k)

    return kf.split(bounds)


def LR_kfolds(X, Y, k=10):
    kf = KFold(n_splits=k)

    output = {}
    
    output['test_sets'] = []
    output['test_scores'] = []
    output['fit_scores'] =[]
    output['test_variance'] = []
    output['fit_variance'] = []
    output['models'] = [] 
    num_features = Y.shape[1]

    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index, :], Y[test_index, :]
        output['models'].append(LinearRegression())
        output['models'][-1].fit(train_x, train_y)
        
        num_samples = test_x.shape[0]
        predic_train_y = output['models'][-1].predict(train_x)
        predic_test_y = output['models'][-1].predict(test_x)
        
        temp = r2_score(test_y, predic_test_y, multioutput='raw_values')
        temp[temp < -1] = -1
        output['test_scores'].append(temp)
        output['test_variance'].append(np.var(predic_test_y, axis=0))
        
        temp = r2_score(train_y, predic_train_y, multioutput='raw_values')
        temp[temp < -1] = -1
        output['fit_scores'].append(temp)        
        output['fit_variance'].append(np.var(predic_train_y, axis=0))
        
        output['test_sets'].append((test_x, test_y))
    return output


def wiener_kfolds(X, Y, c, sw=None, k=4, n_l2=10, sweep='log'):
    kf = KFold(n_splits=k)

    all_test_y =[]
    all_predic = []
    temp = []

    for train_index, test_index in tqdm(kf.split(X)):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index, :], Y[test_index, :]
        if sw is not None:
            train_sw = sw[train_index]
        else:
            train_sw = None
        h = train_wiener_filter(train_x, train_y, c=c, sw=train_sw, n_l2=n_l2, sweep=sweep)
        predic = test_wiener_filter(test_x, h)

        all_test_y.append(test_y)
        all_predic.append(predic)
        temp.append(weighted_r2(test_y, predic))

    return np.vstack(all_test_y), np.vstack(all_predic), h

def deprec_wiener_swkfolds(X, Y, c, sw, k=4, n_l2=10, sweep='log'):
    kf = KFold(n_splits=k)

    all_test_y =[]
    all_predic = []

    for train_index, test_index in tqdm(kf.split(X)):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index, :], Y[test_index, :]
        train_sw = sw[train_index]

        h = train_weighted_wiener_filter(train_x, train_y, sw=train_sw, c=c, n_l2=n_l2, sweep=sweep)
        predic = test_wiener_filter(test_x, h)

        all_test_y.append(test_y)
        all_predic.append(predic)

    return np.vstack(all_test_y), np.vstack(all_predic), h


def Ridge_kfolds_redo(X, Y, alpha=0.1, k=10):
    kf = KFold(n_splits=k)

    output = {}
    
    models=[]
    best_score = -10000
    best_model_idx = 0

    all_train = []
    all_test = []

    all_train_predic = []
    all_test_predic = []

    for train_index, test_index in tqdm(kf.split(X)):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index, :], Y[test_index, :]

        all_train.append(train_y)
        all_test.append(test_y)

        models.append(Ridge(alpha=alpha))
        models[-1].fit(train_x, train_y)
        
        predic_train_y = models[-1].predict(train_x)
        predic_test_y = models[-1].predict(test_x)

        all_train_predic.append(predic_train_y)
        all_test_predic.append(predic_test_y)
        
        temp = r2_score(test_y, predic_test_y, multioutput='raw_values')
        temp[temp < -1] = -1
        test_var = np.var(predic_test_y, axis=0)
        test_score = np.average(temp, weights = test_var)
        if test_score > best_score:
            best_score = test_score
            best_model_idx = len(models)-1

    all_train = np.vstack(all_train)
    all_test = np.vstack(all_test)
    all_train_predic = np.vstack(all_train_predic)
    all_test_predic = np.vstack(all_test_predic)

    output['train_sets'] = (all_train, all_train_predic)
    output['test_sets'] = (all_test, all_test_predic)
    output['fit_score'] = weighted_r2(all_train, all_train_predic)
    output['test_score'] = weighted_r2(all_test, all_test_predic)
    output['best_model'] = models[best_model_idx]


    return output


def Ridge_kfolds(X, Y, alpha=0.1, k=10):
    kf = KFold(n_splits=k)

    output = {}
    
    output['test_sets'] = []
    output['test_scores'] = []
    output['fit_scores'] =[]
    output['test_variance'] = []
    output['fit_variance'] = []
    output['models'] = [] 
    num_features = Y.shape[1]

    for train_index, test_index in tqdm(kf.split(X)):
        train_x, test_x = X[train_index, :], X[test_index,:]
        train_y, test_y = Y[train_index, :], Y[test_index, :]
        output['models'].append(Ridge(alpha=alpha))
        output['models'][-1].fit(train_x, train_y)
        
        num_samples = test_x.shape[0]
        predic_train_y = output['models'][-1].predict(train_x)
        predic_test_y = output['models'][-1].predict(test_x)
        
        temp = r2_score(test_y, predic_test_y, multioutput='raw_values')
        temp[temp < -1] = -1
        output['test_scores'].append(temp)
        output['test_variance'].append(np.var(predic_test_y, axis=0))
        
        temp = r2_score(train_y, predic_train_y, multioutput='raw_values')
        temp[temp < -1] = -1
        output['fit_scores'].append(temp)        
        output['fit_variance'].append(np.var(predic_train_y, axis=0))
        
        output['test_sets'].append((test_x, test_y))
    return output

def smooth_other_features(binned_lags, smoothed_lags, binned_true, nlags, c):
    #assert(binned_lags.shape == smoothed_lags.shape), 'something weird with\
    #lags'

    features = np.arange(0, binned_true.shape[1])
    h_list = []
    r2_list = []
    var_list = []
    predic_list=[]
    for feature in tqdm(features):
        X = copy.deepcopy(smoothed_lags)
        X[:, feature*nlags:(feature*nlags)+nlags] = binned_lags[:,\
                feature*nlags:(feature*nlags)+nlags]
        Y = binned_true[:, feature]

        h_list.append(train_wiener_filter(X, Y, c=c))
        predic = test_wiener_filter(X, h_list[-1])
        predic_list.append(predic)
        r2_list.append(r2_score(Y, predic))
        var_list.append(np.var(Y))
    
    r2_array = np.array(r2_list)
    r2_array[r2_array<-1]=-1
    return h_list, np.average(r2_array, weights=var_list), np.vstack(predic_list)

#def test_smooth_other_features(h_list, binned_lags, smoothed_lags,
#        binned_true):


 
