import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import least_squares
from tqdm import tqdm

def flatten_list(X):
    """
    Converting list containing multiple ndarrays into a large ndarray
    X: a list
    return: a numpy ndarray
    """
    n_col = np.size(X[0],1)
    Y = np.empty((0, n_col))
    for each in X:
        Y = np.vstack((Y, each))
    return Y

def vaf(x,xhat, round_values=True):
    """
    Calculating vaf value
    x: actual values, a numpy array
    xhat: predicted values, a numpy array
    """
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    if round_values is True:
        return np.round((1-(np.sum(np.square(x -
            xhat))/np.sum(np.square(x)))),2)
    else:
        return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))
def weighted_r2(x, xhat, remove_lows=True):
    score = r2_score(x, xhat, multioutput='raw_values')
    if remove_lows:
        score[score<-1]=-1
    variance = np.var(xhat, axis=0)
    return np.average(score, weights=variance)

def format_data(x, y, N=10):
    spike_N_lag = []
    emg_N_lag = []
    for i in range(np.size(x, 0) - N):
        temp = x[i:i+N, :]
        temp = temp.reshape((np.size(temp)))
        spike_N_lag.append(temp)
        if y.ndim == 2:
            emg_N_lag.append(y[i+N-1, :])
        elif y.ndim == 1:
            emg_N_lag.append(y[i+N-1])
    return np.asarray(spike_N_lag), np.asarray(emg_N_lag)

def format_data_ar(x, nlags=10):
    x_nlags = []
    for i in range(np.size(x, 0) - nlags):
        temp = x[i:i+nlags, :]
        temp = temp.reshape((np.size(temp)))
        x_nlags.append(temp)
    x_nlags = np.array(x_nlags)
    x_cut = x[nlags:, :]

    return x_nlags, x_cut

def format_and_stitch(x, y, N=10):
    #format_data, working only on trials
    #if array, data must be trial_length x neurons x trials
    #if list, must be a list of trial_length x neurons np arrays
    assert type(x) is list, 'use format_data if just time series'
    num_trials = len(x)
    
    x_nlags = []
    y_cut = []
    for i in range(num_trials):
        temp_x = x[i]
        temp_y = y[i]

        temp1, temp2 = format_data(temp_x, temp_y, N=N)
        if temp1.shape[0] > 0:
            x_nlags.append(temp1)
            y_cut.append(temp2)
        else:
            print('bounds too small for history')
    return flatten_list(x_nlags), flatten_list(y_cut)

    
def format_and_stitch_ar(x, nlags=10):
    #format_data, working only on trials
    #if array, data must be trial_length x neurons x trials
    #if list, must be a list of trial_length x neurons np arrays
    assert type(x) is list, 'use format_data if just time series'
    num_trials = len(x)
    
    x_nlags = []
    x_cut = []
    for i in range(num_trials):
        temp_x = x[i]

        temp1, temp2 = format_data_ar(temp_x, nlags=nlags)
        if temp1.shape[0] > 0:
            x_nlags.append(temp1)
            x_cut.append(temp2)
        else:
            print('bounds too small for history')
    return flatten_list(x_nlags), flatten_list(x_cut)



def format_single_array(x, N=10):
    data_N_lag = []
    for i in range(np.size(x)-N):
        data_N_lag.append(x[i+N-1])

    return np.asarray(data_N_lag)

def parameter_fit(x, y, c, sw=None, zscore=False):
    """
    c : L2 regularization coefficient
    I : Identity Matrix
    Linear Least Squares (code defaults to this if c is not passed)
    H = ( X^T * X )^-1 * X^T * Y
    Ridge Regression
    R = c * I
    ridge regression doesn't penalize x
    R[0,0] = 0
    H = ( (X^T * SW*X) + R )^-1 * X^T * SW*Y
    sw = saple weights
    """
    temp = np.c_[np.ones((np.size(x, 0), 1)), x]
    x_t = temp.T
    if sw is not None:
        assert sw.size == x.shape[0], f'something weird with sample\
        weight, sw size={sw.size}, x shape = {x.shape[0]}'
        
        x = sw[:, None] * x
        y = sw[:, None] * y
    if zscore:
        x = StandardScaler().fit_transform(x)
    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    R = c * np.eye( x_plus_bias.shape[1] )
    R[0,0] = 0;
    temp = np.linalg.inv(np.dot(x_t, x_plus_bias) + R)
    temp2 = np.dot(temp,x_t)
    H = np.dot(temp2,y)
    return H #code is a little awkward, i tacked sw on top of stuff. 

def deprecated_weighted_parameter_fit(x, y, c, sw):
    """
    c : L2 regularization coefficient
    I : Identity Matrix
    Linear Least Squares (code defaults to this if c is not passed)
    H = ( X^T * X )^-1 * X^T * Y
    Ridge Regression
    R = c * I
    ridge regression doesn't penalize x
    R[0,0] = 0
    H = ( (X^T * X) + R )^-1 * X^T * Y
    """
    assert sw.size == x.shape[0], 'something weird with sample weight'
    x_weighted = np.dot(diag, x)
    y_weighted = np.dot(diag, y)
    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    x_weighted_plus_bias = np.c_[np.ones((np.size(x_weighted, 0), 1)),
            x_weighted]
    R = c * np.eye( x_weighted_plus_bias.shape[1] )
    R[0,0] = 0;
    temp = np.linalg.inv(np.dot(x_plus_bias.T, x_weighted_plus_bias) + R)
    temp2 = np.dot(temp,x_plus_bias.T)
    H = np.dot(temp2,y_weighted)
    return H

def deprecated_weighted_parameter_fit_with_sweep( x, y, C, sw):
    #now weighted by variance, might not be correct for all thigns
    reg_r2 = []
    train_x, test_x, train_y, test_y, train_sw, nada = train_test_split(x, y,
            sw, test_size=.2)
    for c in C:
        #print( 'Testing c= ' + str(c) )
        cv_r2 = []
        # fit decoder
        H = weighted_parameter_fit(train_x, train_y, c, train_sw)
        #print( H.shape )
        # predict
        test_y_pred = test_wiener_filter(test_x, H)
        # evaluate performance
        reg_r2.append(weighted_r2(test_y, test_y_pred))
        print(reg_r2[-1])
        # append mean of CV decoding for output

    reg_r2 = np.asarray(reg_r2)        
    best_c = C[ np.argmax( reg_r2 ) ] 
    return best_c


def parameter_fit_with_sweep( x, y, c, sw=None, zscore=False):
    #now weighted by variance, might not be correct for all thigns
    reg_r2 = []
    if sw is not None:
        train_x, test_x, train_y, test_y, sw, nada = train_test_split(x, y, sw, test_size=.2)
    else:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)
    for c_ in c:
        #print( 'Testing c= ' + str(c) )
        cv_r2 = []
        # fit decoder
        H = parameter_fit(train_x, train_y, c_, sw, zscore)
        #print( H.shape )
        # predict
        test_y_pred = test_wiener_filter(test_x, H)
        # evaluate performance
        reg_r2.append(weighted_r2(test_y, test_y_pred))
        # append mean of CV decoding for output

    reg_r2 = np.asarray(reg_r2)        
    best_c = c[ np.argmax( reg_r2 ) ] 
    return best_c

def train_wiener_filter(x, y, c = 0, n_l2=10, sw = None, sweep='log',
        zscore=False):
    """
    To train a linear decoder
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    C: if tuple, then sweep betweenlower log bound, upper low boudn
    else, if an int, then use as reg term
    """
    #print(sw)
    if type(c) is tuple:
        #n_l2 = 10 #edited
        if sweep == 'log':
            c = np.logspace(c[0], c[1], n_l2 )#edited to be larger numbers
        else:
            c = np.linspace(c[0], c[1], n_l2)
        best_c = parameter_fit_with_sweep( x, y, c, sw, zscore)
        print(f'best_c: {best_c}')
    else:
        best_c = c
    H_reg = parameter_fit( x, y, best_c, sw, zscore)
    #print(vaf(y, temp))
    return H_reg

def deprecated_train_weighted_wiener_filter(x, y, sw, C = 0, n_l2=10, sweep='log'):
    """
    To train a linear decoder
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    C: if tuple, then sweep betweenlower log bound, upper low boudn
    else, if an int, then use as reg term
    """
    if type(C) is tuple:
        #n_l2 = 10 #edited
        if sweep == 'log':
            C = np.logspace(C[0], C[1], n_l2 )#edited to be larger numbers
        else:
            C = np.linspace(C[0], C[1], n_l2)
        best_c = weighted_parameter_fit_with_sweep( x, y, C, sw)
    else:
        best_c = C
    print(f'best_c: {best_c}')
    H_reg = weighted_parameter_fit( x, y, best_c, sw)
    temp = test_wiener_filter(x, H_reg)
    #print(vaf(y, temp))
    return H_reg


def deprecated_train_wiener_filter_fine(x, y, C = 0):
    """
    To train a linear decoder
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    C: if tuple, then sweep betweenlower log bound, upper low boudn
    else, if an int, then use as reg term
    """
    if type(C) is tuple:
        n_l2 = 10 #edited
        C = np.linspace(C[0], C[1], n_l2 )#edited to be larger numbers
        kfolds = 4
        kf = KFold( n_splits = kfolds )
        best_c = parameter_fit_with_sweep( x, y, C, kf )
    else:
        best_c = C
    print(f'best_c: {best_c}')
    H_reg = parameter_fit( x, y, best_c )
    return H_reg 
def test_wiener_filter(x, H, zscore_scaler=None):
    """
    To get predictions from input data x with linear decoder
    x: input data
    H: parameter vector obtained by training
    """

    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    y_pred = np.dot(x_plus_bias, H)
    return y_pred    

def nonlinearity(p, y):
    return p[0]+p[1]*y+p[2]*y*y
    
def nonlinearity_residue(p, y, z):
    return (nonlinearity(p, y) - z).reshape((-1,))

def train_nonlinear_wiener_filter(x, y, l2 = 0):
    """
    To train a nonlinear decoder
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    l2: 0 or 1, switch for turning L2 regularization on or off
    """
    if l2 == 1:
        n_l2 = 20
        C = np.logspace( 1, 5, n_l2 )
        kfolds = 4
        kf = KFold( n_splits = kfolds )
        best_c = parameter_fit_with_sweep( x, y, C, kf )
    else:
        best_c = 0
    H_reg = parameter_fit( x, y, best_c )
    y_pred = test_wiener_filter(x, H_reg)
    res_lsq = least_squares(nonlinearity_residue, [0.1,0.1,0.1], args = (y_pred, y))
    return H_reg, res_lsq

def test_nonlinear_wiener_filter(x, H, res_lsq):  
    """
    To get predictions from input data x with nonlinear decoder
    x: input data
    H: parameter vector obtained by training
    res_lsq: nonlinear components obtained by training
    """
    y1 = test_wiener_filter(x, H)
    y2 = nonlinearity(res_lsq.x, y1)
    return y2    
    
def unformat_h(h):
    h_flip = h.T
    h_flip = h_flip[:,1:]
    output=[]

    for rates in h_flip:
        output.append(np.array.split(rates, 10))

    return output

