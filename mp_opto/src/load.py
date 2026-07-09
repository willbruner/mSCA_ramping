import mat73
import scipy.io as sio
import numpy as np
import os

def load_to_neural(mouse_session_path, region_name):
    """G
    matlab raw files -> python, no semicolons, 1-based indexing allowed boys

    args:
        mouse_session_path: string path to folder that contains 'preprocess'
        directory

    returns:
        cfa: cfa dict
        rfa: rfa dict
        analogin: analog numpy array
    
    """
    preprocess_path = f"{mouse_session_path}/preprocess"
    region_path = f'{preprocess_path}/neurons_{region_name}.mat'

    region_temp = mat73.loadmat(region_path)
    region_temp = region_temp[f'neurons_{region_name}']
    region = region_temp[0]
    region['train'] = region['train']
    region['depth'] = np.array(region['depth'])
    region['name'] = region_name

    if region_name == 'THAL':
        greater = 300
        less = 1000
        depth_mask = np.logical_and(region['depth'] > greater, region['depth']  < less)

        print(f'masking THAL, {greater}<x<{less}')

        new_train = []

        for idx in np.arange(len(depth_mask)):
            if depth_mask[idx] == True:
                new_train.append(region['train'][idx])
        
        region['train'] = new_train



    region['num_neurons'] = len(region['train'])

    return region

def load_co_neural(mouse_session_path, region_name):
    """
    matlab raw files -> python, no semicolons, 1-based indexing allowed boys

    args:
        mouse_session_path: string path to folder that contains 'preprocess'
        directory

    returns:
        cfa: cfa dict
        rfa: rfa dict
        analogin: analog numpy array
    
    """
    preprocess_path = f"{mouse_session_path}/preprocess"
    region_path = f'{preprocess_path}/neurons_{region_name}.mat'
    region_temp = sio.loadmat(region_path)
    region = {}
    temp = np.squeeze(region_temp[f'neurons_{region_name}']['train']).tolist()
    region['train'] = temp
    temp = np.squeeze(region_temp[f'neurons_{region_name}']['width'])
    region['name'] = region_name
    width_list = []
    for t in temp:
        width_list.append(t[0][0])
    region['width'] = width_list
    region['num_neurons'] = len(region['train'])

    return region
 
   

def load_analogin(mouse_session_path):
    preprocess_path = f"{mouse_session_path}/preprocess"
    analogin_path = f"{preprocess_path}/analog.mat"
    analogin = sio.loadmat(analogin_path)
    analogin = analogin['analog']

    return analogin



def load_climb(mouse_session_path):
    preprocess_path = f"{mouse_session_path}/preprocess"
    isclimbing_path = f'{preprocess_path}/isclimbing.mat'
    isclimbing = sio.loadmat(isclimbing_path)
    isclimbing = isclimbing['isclimbing']

    return isclimbing


def load_vgat(mouse_session_path):
    """
    matlab raw files -> python, no semicolons, 1-based indexing allowed boys

    args:
        mouse_session_path: string path to folder that contains 'preprocess'
        directory

    returns:
        cfa: cfa dict
        rfa: rfa dict
        analogin: analog numpy array
    
    """
    preprocess_path = f"{mouse_session_path}/preprocess"

    cfa_path = f'{preprocess_path}/neurons_probe1_curated.mat' #for vgat_
    rfa_path = f'{preprocess_path}/neurons_probe2_curated.mat' #for vgat_

    cfa_temp = mat73.loadmat(cfa_path)
    CFA = cfa_temp['neurons_probe1']
    #CFA['train'] = np.array((CFA['train']))
    
    rfa_temp = mat73.loadmat(rfa_path)
    RFA = rfa_temp['neurons_probe2']   
    #RFA['train'] = np.array((RFA['train']))

    #lets also load smoothed FRs for ease
    #cfa_fr_path= f"{preprocess_path}/CFA_FR.mat"
    #rfa_fr_path = f"{preprocess_path}/RFA_FR.mat"
    #cfa_temp = mat73.loadmat(cfa_fr_path)
    #rfa_temp = mat73.loadmat(rfa_fr_path)
    #CFA['firingrate'] = cfa_temp['CFA_FR']
    #RFA['firingrate'] = rfa_temp['RFA_FR']

    analogin_path = f"{preprocess_path}/analogin.mat"
    analogin = mat73.loadmat(analogin_path)
    analogin = analogin['analogin']

    isclimbing_path = f'{preprocess_path}/isclimbing.mat'
    isclimbing = mat73.loadmat(isclimbing_path)
    isclimbing = isclimbing['isclimbing']

    return CFA, RFA, analogin, isclimbing

def load_co(mouse_session_path):
    """
    matlab raw files -> python, no semicolons, 1-based indexing allowed boys

    args:
        mouse_session_path: string path to folder that contains 'preprocess'
        directory

    returns:
        cfa: cfa dict
        rfa: rfa dict
        analogin: analog numpy array
    
    """
    preprocess_path = f"{mouse_session_path}/preprocess"

    cfa_path = f'{preprocess_path}/neurons_CFA.mat'
    rfa_path = f'{preprocess_path}/neurons_RFA.mat'

    cfa_temp = sio.loadmat(cfa_path)
    CFA = {}
    #temp = cfa_temp['neurons_CFA']
    temp = np.squeeze(cfa_temp['neurons_CFA']['train']).tolist()
    CFA['train'] = temp
    temp = np.squeeze(cfa_temp['neurons_CFA']['width'])
    width_list = []
    for t in temp:
        width_list.append(t[0][0])
    CFA['width'] = width_list
 
    
    rfa_temp = sio.loadmat(rfa_path)
    RFA = {}
    temp = np.squeeze(rfa_temp['neurons_RFA']['train']).tolist()
    RFA['train'] = temp
    temp = np.squeeze(rfa_temp['neurons_RFA']['width'])
    width_list = []
    for t in temp:
        width_list.append(t[0][0])
    RFA['width'] = width_list

    analogin_path = f"{preprocess_path}/analog.mat"
    analogin = sio.loadmat(analogin_path)
    analogin = analogin['analog']

    isclimbing_path = f'{preprocess_path}/isclimbing.mat'
    isclimbing = sio.loadmat(isclimbing_path)
    isclimbing = isclimbing['isclimbing']

    return CFA, RFA, analogin, isclimbing

def load_mprecord(mouse_session_path):
    """
    matlab raw files -> python, no semicolons, 1-based indexing allowed boys

    args:
        mouse_session_path: string path to folder that contains 'preprocess'
        directory

    returns:
        cfa: cfa dict
        rfa: rfa dict
        analogin: analog numpy array
    
    """
    preprocess_path = f"{mouse_session_path}" #/preprocess"

    cfa_path = f'{preprocess_path}/neurons_CFA.mat'
    rfa_path = f'{preprocess_path}/neurons_RFA.mat'
    dls_path = f'{preprocess_path}/neurons_DLS.mat'
    ms_path = f'{preprocess_path}/neurons_MS.mat'
    s1_path = f'{preprocess_path}/neurons_S1.mat'
    mpfc_path = f'{preprocess_path}/neurons_mPFC.mat'

    cfa_temp = sio.loadmat(cfa_path)
    CFA = {}
    temp = np.squeeze(cfa_temp['regiondata']['train']).tolist()
    CFA['train'] = temp
 
    rfa_temp = sio.loadmat(rfa_path)
    RFA = {}
    temp = np.squeeze(rfa_temp['regiondata']['train']).tolist()
    RFA['train'] = temp

    dls_temp = sio.loadmat(dls_path)
    DLS = {}
    temp = np.squeeze(dls_temp['regiondata']['train']).tolist()
    DLS['train'] = temp

    ms_temp = sio.loadmat(ms_path)
    MS = {}
    temp = np.squeeze(ms_temp['regiondata']['train']).tolist()
    MS['train'] = temp

    s1_temp = sio.loadmat(s1_path)
    S1 = {}
    temp = np.squeeze(s1_temp['regiondata']['train']).tolist()
    S1['train'] = temp

    mpfc_temp = sio.loadmat(mpfc_path)
    MPFC = {}
    temp = np.squeeze(mpfc_temp['regiondata']['train']).tolist()
    MPFC['train'] = temp

    analogin_path = f"{preprocess_path}/analogin.mat"
    analogin = mat73.loadmat(analogin_path)
    analogin = analogin['analogin']
    
    try: 
        isclimbing_path = f'{preprocess_path}/isclimbing.mat'
        isclimbing = mat73.loadmat(isclimbing_path)
        isclimbing = isclimbing['isclimbing']

    except:

        isclimbing_path = f'{preprocess_path}/isclimbing.mat'
        isclimbing = sio.loadmat(isclimbing_path)
        isclimbing = isclimbing['isclimbing']

    return CFA, RFA, DLS, MS, S1, MPFC, analogin, isclimbing

def load_reaching(mouse_session_path):
    session_name = mouse_session_path.split('/')[-1] 
    neurons_path =\
    f'{mouse_session_path}/{session_name}_pyr_CFA_RFA_trains_sort.mat'
    neurons = sio.loadmat(neurons_path)
    neurons = neurons['pyr_CFA_RFA_trains_sort']

    CFA_temp = neurons[0,0][:,0]
    RFA_temp = neurons[0,1][:,0]

    CFA={}
    RFA={}

    CFA['train'] = CFA_temp.tolist()
    RFA['train'] = RFA_temp.tolist()

    reach_bounds_path =\
    f'{mouse_session_path}/{session_name}_reach_bounds_edit.mat'
    reach_bounds = sio.loadmat(reach_bounds_path)
    reach_bounds = reach_bounds['reach_bounds_edit']

    return CFA, RFA, reach_bounds
