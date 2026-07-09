import numpy as np
from src.load import *
from src.experiment import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.wiener_filter import *
from src.neural import *
#from Neural_Decoding.preprocessing_funcs import bin_spikes
import copy
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import pynapple as nap
import itertools

class MPOptoClass:
    ''' MPOptoClass designed to handle data with simultaneous reg1/reg2 
    recordings and stim reg2

    session_path = MOUSE_SESSIONDATE, like co6_10122023

    updated 07/24, 
    '''
    def __init__(self, session_path):
        #second region is stim region
        yaml_data = load_yaml(f'{session_path}/session_info.yaml')
        self.name = yaml_data['session_name']

        regions = yaml_data['record']
        stim_regions = yaml_data['stim']

        assert len(regions)==2, 'this class for two region only'
        assert len(stim_regions)==1, 'this class for one stim region only'
        assert stim_regions[0] == regions[1], 'second region should be stim region'

        laser_channel = yaml_data['analogin']['stim_pulse']
        ctrl_channel = yaml_data['analogin']['ctrl_pulse']
        loader = yaml_data['loader']

        if loader == 'TO':
            load_neural = load_to_neural
            laser_thres=.3
            self.num_powers = yaml_data['num_powers']
        elif loader == 'CO':
            load_neural = load_co_neural
            laser_thres=.1

        analogin = load_analogin(session_path)
        isclimbing = load_climb(session_path)

        self.region1 = load_neural(session_path, regions[0])
        self.region2 = load_neural(session_path, regions[1])

        self.num_region1 = self.region1['num_neurons']
        self.num_region2 = self.region2['num_neurons']

        self.inactivate = yaml_data['record'][-1]
        self.duration = analogin.shape[1] 
         
        self.climbing_bounds = climbing_bounds_from_logical(isclimbing)
        self.climbing_logical = np.squeeze(isclimbing.astype(int))

        self.laser = {}
        self.laser['laser_ts'] = analogin[laser_channel,:]
        self.laser['ctrl_ts'] = analogin[ctrl_channel,:]
        self.laser['laser_bounds'], self.laser['laser_powers'] =\
        getLaserBounds(self.laser['laser_ts'], laser_thres)

        self.laser['ctrl_bounds'], nada =\
        getLaserBounds(self.laser['ctrl_ts'], laser_thres)

        self.num_trials = self.laser['laser_bounds'].shape[0]
        self.num_ctrls = self.laser['ctrl_bounds'].shape[0]


        print('love')

    def subsampleNeurons(self, percent_region1, percent_region2=None,
            random_state=None):
        rng = np.random.default_rng(seed=random_state)
        if percent_region2 is None:
            percent_region2 = percent_region1
        subsamp_region1 = rng.choice(self.num_region1, 
                size=int(percent_region1*self.num_region1))
        subsamp_region2 = rng.choice(self.num_region2, 
                int(percent_region2*self.num_region2))
        
        self.region1['orig_train'] = self.region1['train']
        self.region2['orig_train'] = self.region2['train']



        region1_newtrain = []

        region1_newtrain = [self.region1['train'][idx] for idx in
                subsamp_region1]

        region2_newtrain = [self.region2['train'][idx] for idx in
                subsamp_region2]

        self.region1['train'] = region1_newtrain
        self.region2['train'] = region2_newtrain
        self.num_region1 = len(region1_newtrain)
        self.num_region2 = len(region2_newtrain)

        print(f'new num region1: {self.num_region1}')
        print(f'new num region2: {self.num_region2}')

        return

    def adjustLaserBounds(self, pre, post, only_climb=False):
        #need to fix good lasers at some point
        laser_bounds = adjustBounds(self.laser['laser_bounds'], pre, post)
        ctrl_bounds = adjustBounds(self.laser['ctrl_bounds'], pre, post)

        if only_climb:
            laser_mask = goodLasers(laser_bounds, self.climbing_logical)
            ctrl_mask = goodLasers(ctrl_bounds, self.climbing_logical)
            laser_bounds = laser_bounds[laser_mask, :]
            ctrl_bounds = ctrl_bounds[ctrl_mask, :]

        return laser_bounds, ctrl_bounds

    def getPowers(self, bounds, num_powers=None, pre=0, post=0):
        #awkward, but most of the times laser bounds include every stuff
        #pre post adjusts for this

        if num_powers is None:
            num_powers = self.num_powers
        ts_list = []
        temp = self.laser['laser_ts']
        for bound in bounds:
            start = bound[0]+pre
            end = bound[1]-post
            ts_list.append(temp[start:end])
        sums = np.sum(ts_list, axis=1)
        hist, bin_edges = np.histogram(sums, bins=num_powers)

        return np.digitize(sums, bin_edges)
            

    def binner(self, bounds=None, binsize=1, concat=False):

        if bounds is None:
            bounds = np.array([[0, self.duration]])
            concat=True
        else:
            assert bounds.shape[1] == 2, 'bounds looks weird'

        region1_binned = bin_spiketimes_bounds(self.region1['train'], binsize, bounds)
        region2_binned = bin_spiketimes_bounds(self.region2['train'], binsize, bounds)
        if concat:
            region1_binned = np.vstack(region1_binned)
            region2_binned = np.vstack(region2_binned)
        return region1_binned, region2_binned

        #probably inscrutable....


    def threshold_FRs(self, threshold=.5, bounds=None):
        #default threshold is .5 hz
        if bounds is None:
            bounds = self.climbing_bounds
        

        region1, region2 = self.binner(binsize=1, bounds=bounds, concat=True)

        region1_hz = (np.sum(region1, axis=0) / region1.shape[0]) * 1000
        region2_hz = (np.sum(region2, axis=0) / region2.shape[0]) * 1000

        return region1_hz > threshold, region2_hz > threshold

    def smoother(self, bounds=None, sigma=10, binsize=1, concat=False,
            smooth_type='causal, future, or anything for twosided'):
        # some assumptions, but lets bin/smooth all data, and then truncate
        #concat only works if bounds are even, will fix later
        #edgesmooth = 100//binsize #we smooth a little past bounds, and then remove
        
        if bounds is None:
            print('smoothing entire thing')
            region1_binned, region2_binned = self.binner(binsize=binsize, concat=True)

            if smooth_type == 'causal':
                gausmooth = gaussian_filter1d_oneside
            elif smooth_type == 'future':
                gausmooth = gaussian_filter1d_future
            else:
                gausmooth = gaussian_filter1d

            region1_smooth = gausmooth(region1_binned, sigma=sigma//binsize, axis=0,
                    mode='constant')

            region2_smooth = gausmooth(region2_binned, sigma=sigma//binsize, axis=0, 
                    mode='constant')

            return region1_smooth, region2_smooth

        mod_bounds, edgesmooth = self._mod_bounds(bounds, sigma)
        region1_binned, region2_binned = self.binner(bounds=mod_bounds, 
                binsize=binsize, concat=False)

        region1_smooth = self._smooth_in_modbounds(region1_binned, sigma, binsize,
                edgesmooth, smooth_type)
        region2_smooth = self._smooth_in_modbounds(region2_binned, sigma, binsize,
                edgesmooth, smooth_type)

        if concat:
            region1_smooth = np.vstack(region1_smooth)
            region2_smooth = np.vstack(region2_smooth)

        return region1_smooth, region2_smooth

    def _mod_bounds(self, bounds, sigma):
        edgesmooth = sigma*2

        mod_bounds = copy.deepcopy(bounds)
        mod_bounds[:,0]  = mod_bounds[:,0] - edgesmooth
        mod_bounds[:,1] = mod_bounds[:,1] + edgesmooth

        return mod_bounds, edgesmooth

 

    def _smooth_in_modbounds(self, neural_data, sigma, binsize,edgesmooth, smooth_type):

        if smooth_type == 'causal':
            gausmooth = gaussian_filter1d_oneside
        elif smooth_type == 'future':
            gausmooth = gaussian_filter1d_future
        else:
            gausmooth = gaussian_filter1d

        edgesmooth_bin = edgesmooth//binsize
        output = []

        for trial_num in range(len(neural_data)):
            trial = neural_data[trial_num]
            temp = gausmooth(trial, sigma=sigma//binsize,
                    axis=0, mode='constant')
       
            temp = temp[edgesmooth_bin:-edgesmooth_bin]
            output.append(temp)

        return output
   
    def apply_PCA(self, bounds=None, binsize=1, smooth_type = None, sigma = 10,
            num_PCs = 0, concat=False):

        if bounds is None:
            bounds = self.climbing_bounds

        if smooth_type is None:
            region1, region2 = self.binner(bounds=bounds, binsize=binsize, concat=True)
        else:
            region1, region2 = self.smoother(bounds=bounds, binsize=binsize, 
                    smooth_type=smooth_type, sigma=sigma, concat=True)
        
        region1_PCAobj = PCA(n_components = self.num_region1)
        region2_PCAobj = PCA(n_components = self.num_region2)

        region1_PCs = region1_PCAobj.fit_transform(region1)
        region2_PCs = region2_PCAobj.fit_transform(region2)

        if num_PCs > 0:
            region1_PCs=region1_PCs[:,:num_PCs]
            region2_PCs=region2_PCs[:,:num_PCs]


        if concat is False:
           region1_PCs = unstitchSeams(region1_PCs, getSeamsFromBounds(bounds,
               binsize=binsize)) 
           region2_PCs = unstitchSeams(region2_PCs, getSeamsFromBounds(bounds,
               binsize=binsize)) 

        return (region1_PCs, region2_PCs), (region1_PCAobj, region2_PCAobj)

    def format_nlags(self, bounds=None, binsize=10, nlags=10):
        if bounds is None:
            bounds = self.climbing_bounds

        region1_binned, region2_binned = self.binner(bounds=bounds, binsize=binsize, concat=False)
        
        all_binned=[]
        for i in range(len(region1_binned)):
            all_binned.append(np.hstack((region1_binned[i], region2_binned[i])))

        all_nlags, all_cut  = format_and_stitch_ar(all_binned, nlags=nlags)

        return all_nlags, all_cut

    def transformat(self, pca_objs, bounds=None, binsize=10,  nlags=10,
            num_PCs=0):
        #little helper if we already have PCA objs, to transform data 
        #and then format it.

        if bounds is None:
            bounds=self.climbing_bounds

        assert type(pca_objs) is tuple, 'something weird with pca objs'

        region1_PCObj = pca_objs[0]
        region2_PCObj = pca_objs[1]

        region1, region2 = self.binner(bounds=bounds, binsize=binsize, concat=True)

        region1_PCs = region1_PCObj.transform(region1)
        region2_PCs = region2_PCObj.transform(region2)

        if num_PCs > 0:
            region1_PCs=region1_PCs[:,:num_PCs]
            region2_PCs=region2_PCs[:,:num_PCs]


        all_PCs = np.hstack((region1_PCs, region2_PCs))
        all_PCs = unstitchSeams(all_PCs, getSeamsFromBounds(bounds=bounds, binsize=binsize))

        X,Y = format_and_stitch_ar(all_PCs, nlags=nlags)

        return X, Y

    def transformat_smooth(self, pca_objs, bounds=None, binsize=10,  nlags=10,
            sigma=10, num_PCs=0, smooth_type='causal'):
        #little helper if we already have PCA objs, to transform data 
        #and then format it.

        if bounds is None:
            bounds=self.climbing_bounds

        assert type(pca_objs) is tuple, 'something weird with pca objs'

        mod_bounds, edgesmooth = self._mod_bounds(bounds, sigma)

        region1_PCObj = pca_objs[0]
        region2_PCObj = pca_objs[1]

        region1, region2 = self.binner(bounds=mod_bounds, binsize=binsize, concat=True)

        region1_PCs = region1_PCObj.transform(region1)
        region2_PCs = region2_PCObj.transform(region2)

        if num_PCs > 0:
            region1_PCs=region1_PCs[:,:num_PCs]
            region2_PCs=region2_PCs[:,:num_PCs]


        all_PCs = np.hstack((region1_PCs, region2_PCs))
        all_PCs = unstitchSeams(all_PCs, 
                getSeamsFromBounds(bounds=mod_bounds, binsize=binsize))
        
        smooth_output = self._smooth_in_modbounds(all_PCs, sigma, binsize,
                edgesmooth, smooth_type)


        X,Y = format_and_stitch_ar(smooth_output, nlags=nlags)


        return X, Y


 
    def format_nlags_PCA(self, bounds=None, binsize=10, nlags=10, num_PCs=0):
        if bounds is None:
            bounds = self.climbing_bounds

        region1_binned, region2_binned = self.binner(bounds=bounds, binsize=binsize,
                concat=True)

        region1_PCAObject = PCA(n_components = region1_binned.shape[1])
        region1_PCs = region1_PCAObject.fit_transform(region1_binned)

        region2_PCAObject = PCA(n_components = region2_binned.shape[1])
        region2_PCs = region2_PCAObject.fit_transform(region2_binned)

        if num_PCs > 0:
            region1_PCs=region1_PCs[:,:num_PCs]
            region2_PCs=region2_PCs[:,:num_PCs]

        all_PCs = np.hstack((region1_PCs, region2_PCs))
        
        seams = getSeamsFromBounds(bounds, binsize=binsize)
        all_PCs = unstitchSeams(all_PCs, seams)

        all_nlags, all_cut  = format_and_stitch_ar(all_PCs, nlags=nlags)
        
        return (all_nlags, all_cut), (region1_PCAObject, region2_PCAObject)

    def format_nlags_PCA_smooth(self, bounds=None, binsize=10, sigma=100,
            nlags=10, smooth_type='causal', num_PCs=0):
        if bounds is None:
            bounds = self.climbing_bounds

        mod_bounds, edgesmooth = self._mod_bounds(bounds, sigma)

        region1_binned, region2_binned = self.binner(bounds=mod_bounds, binsize=binsize,
                concat=True)

        region1_PCAObject = PCA(n_components = region1_binned.shape[1])
        region1_PCs = region1_PCAObject.fit_transform(region1_binned)

        region2_PCAObject = PCA(n_components = region2_binned.shape[1])
        region2_PCs = region2_PCAObject.fit_transform(region2_binned)
 
        if num_PCs > 0:
            region1_PCs=region1_PCs[:,:num_PCs]
            region2_PCs=region2_PCs[:,:num_PCs]

        all_PCs = np.hstack((region1_PCs, region2_PCs))       
        
        seams = getSeamsFromBounds(mod_bounds, binsize=binsize)
        all_PCs = unstitchSeams(all_PCs, seams)

        smooth_output = self._smooth_in_modbounds(all_PCs, sigma, binsize,
                edgesmooth, smooth_type)
        all_nlags, all_cut  = format_and_stitch_ar(smooth_output, nlags=nlags)

        return (all_nlags, all_cut), (region1_PCAObject, region2_PCAObject)

    def weights_per_region(self, h, nlags=10, num_region1=None):

        #move to analysis.md, soon to be deprecated
        if num_region1 is None:
            num_region1 = self.num_region1
        region1_mask, region2_mask = self.mask_h_region(h, nlags, num_region1,
                has_bias=True)
        region1 = np.sum(np.abs(h[region1_mask,:]), axis=0)
        region2 = np.sum(np.abs(h[region2_mask,:]), axis=0)

        return np.average(region1), np.average(region2)

    def weights_per_region_ridgeobject(self, coef_, nlags=10, num_region1=None):
        #deprecated
        if num_region1 is None:
            num_region1 = self.num_region1
        region1 = np.sum(np.abs(coef_[0:num_region1*nlags, :]), axis=0)
        region2 = np.sum(np.abs(coef_[num_region1*nlags:, :]), axis=0)

        return np.average(region1), np.average(region2)

    def relative_input(self, h, X_format, nlags, num_region1=0):
        #probably should be redone, and moved to analysis.md
        if num_region1 == 0:
            num_region1 = self.num_region1	

        region1_mask, region2_mask = self.mask_input_region(X_format, nlags, num_region1)

        region1 = X_format[:, region1_mask]
        region2 = X_format[:, region2_mask]

        region1_h_mask, region2_h_mask = self.mask_h_region(h, nlags,
                num_region1, has_bias=True)
        
        region1_h = h[region1_h_mask, :]
        #region1_h = region1_h[1:,:]
        region2_h = h[region2_h_mask, :]
        #region2_h = region2_h[1:,:]

        region1_drive = np.sum(np.abs(region1@region1_h),axis=0)
        region2_drive = np.sum(np.abs(region2@region2_h),axis=0)

        return np.average(region1_drive), np.average(region2_drive)


    def getSWs(self, bigBound, smallBound, sw_weight=50, binsize=10, nlags=10):
        # confusing, but if we're grabbing data in bigBound
        # and we want to weight extra stuff in smallBound
        # which i guess is a subset of all in bigBOund
        # then run this, it'll give a logical that should be
        # the same duration as whatever operation in BigBOund

        sw_list = []

        mod_smallBound, duration  = reBoundInBounds(bigBound, smallBound)
        mod_smallBound_logical = bounds2Logical(mod_smallBound, duration=duration)
        mod_smallBound_trials = unstitchSeams(mod_smallBound_logical,
                getSeamsFromBounds(bigBound, binsize=1))

        for trial in mod_smallBound_trials:
            temp = bin_timeseries(trial, binsize=binsize)
            sw_list.append(format_single_array(temp, N=nlags))
        sw = np.hstack(sw_list)
        sw[sw >= 1] = sw_weight - 1
        sw = sw+1

        return sw

    

    def weights_per_lag(self, h, nlags=10):
        #deprecated i think
        new_h = copy.deepcopy(h)
        new_h = new_h[1:,:]
        assert new_h.shape[0] % nlags == 0, 'something wrnog'

        iterator = np.arange(new_h.shape[0]) % nlags
        lag_weights = np.arange(nlags)
        output = []
        for lag in lag_weights:
            this_mask = np.argwhere(iterator==lag)
            output.append(np.sum(np.abs(new_h[this_mask,:])))
        
        output = normalizeData(np.array(output))
        return output

    def deprec_divergence(self, PCs1, PCs2):

        #need to move to analysis.py

        difference = np.abs(np.var(PCs2, axis=0) - np.var(PCs1, axis=0))
        summy = np.var(PCs2, axis=0) + np.var(PCs1, axis=0)
        weights = np.var(PCs1, axis=0)

        return np.average(difference / summy, weights=weights)

    def mask_input_region(self, X_format, nlags, num_region1=0):

        #need to move to analysis.py
        if num_region1 == 0:
            num_region1 = self.num_region1

        features = X_format.shape[1]
        repeats = features/nlags
        assert features % repeats == 0, 'something wrong'

        temp =  np.arange(features) % repeats
        region1_mask = temp < num_region1
        region2_mask = temp >= num_region1 

        return region1_mask, region2_mask

    def mask_h_region(self, h, nlags, num_region1=None, has_bias=True):

        #need to move to analysis.py
        if num_region1 is None:
            num_region1 = self.num_region1
        if has_bias:
            features = h.shape[0] - 1 
        else:
            features = h.shape[0]
        repeats = features / nlags
        assert features % repeats ==0, 'something wrong'

        temp = np.arange(features) % repeats
        region1_mask = temp < num_region1
        region2_mask = temp >= num_region1

        if has_bias:

            region1_mask = np.hstack([False, temp < num_region1])
            region2_mask = np.hstack([False, temp >= num_region1])

        return region1_mask, region2_mask

    def generate_new_ctrls(self, num_trials, pre, post=50, omit_post=250):
        _, ctrl_bounds = self.adjustLaserBounds(pre, post)
        laser_bounds, _ = self.adjustLaserBounds(pre, omit_post)
        bounds = omitBoundInBounds(self.climbing_bounds, laser_bounds)
        bounds = omitBoundInBounds(bounds, ctrl_bounds)

        trial_length = ctrl_bounds[0][1] - ctrl_bounds[0][0]

        bounds = minimumBoundSize(bounds, min_size=trial_length)


        num_bouts = len(bounds)
        rng = np.random.default_rng(1)
        temp =rng.choice(num_bouts, size=num_trials, replace=True)
        random_bouts_idxs, counts = np.unique(temp, return_counts=True)
        random_bouts = bounds[random_bouts_idxs, :]

        start_idx_list = []

        for idx, bout in enumerate(random_bouts):
            start = bout[0]
            end = bout[1]-trial_length

            for c in np.arange(counts[idx]):
                start_idx = rng.choice(np.arange(start, end))
                start_idx_list.append(start_idx)

        output = np.zeros((num_trials, 2))
        output[:,0] = start_idx_list
        output[:,1] = [x + trial_length for x in start_idx_list] 

        return np.sort(output, axis=0).astype(int)

    def generate_trainset(self, bounds, binsize, nlags, num_PCs):

        (X, _), PCs = self.format_nlags_PCA(bounds=bounds, 
                binsize=binsize, nlags=nlags, num_PCs=num_PCs)
        (_, Y) = self.transformat_smooth(PCs, 
                bounds=bounds, binsize=binsize, nlags=nlags, 
                num_PCs=num_PCs, sigma=binsize, smooth_type='future')
        if num_PCs==0:
            Y=Y[:, :self.num_region1]
        else:
            Y=Y[:,:num_PCs]

        return (X, Y), PCs

    def generate_testset(self, PCs, bounds, binsize, nlags, num_PCs):

        (X, _)= self.transformat(PCs, bounds=bounds, 
                binsize=binsize, nlags=nlags, num_PCs=num_PCs)
        (_, Y) = self.transformat_smooth(PCs, 
                bounds=bounds, binsize=binsize, nlags=nlags, 
                num_PCs=num_PCs, sigma=binsize, smooth_type='future')
        if num_PCs==0:
            Y=Y[:, :self.num_region1]
        else:
            Y=Y[:,:num_PCs]

        return X, Y
    
            
                

