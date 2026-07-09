#genutils

import pickle
import yaml
import numpy as np
from concurrent.futures import *

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalizeDataToRange(data, newRange):
    assert newRange != (0,1), 'just use normalizeData instead i think'
    norm = normalizeData(data)
    return norm * (newRange[1] - newRange[0]) + newRange[0]

def varexplained(data):
    #samples x features

    return np.var(data, axis=0) / sum(np.var(data, axis=0))
   
def pload(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
    

def pdump(object, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(object, f)

def load_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
                print(exc)
    return data

def nanlog(x):
    return np.log(x, out=np.zeros_like(x), where=(x!=0))

def nanaverage(A, axis, weights=1):
    return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

def multipool(function, *args, iterable, num_workers):
    result = [None] * len(iterable)
    executor = ProcessPoolExecutor(max_workers = num_workers)

    for idx, i in enumerate(iterable):
        result[idx] = executor.submit(function, i, *args)

    wait(result, timeout=None, return_when=ALL_COMPLETED)

    output = []
    for this_result in result:
        output.append(this_result.result())

    return output

def squeezer(*args):
    output = []
    for arg in args:
        output.append(np.squeeze(arg))
    return tuple(output)

def avgstd(data, axis=None):
    return np.average(data, axis=axis), np.std(axis=axis)
