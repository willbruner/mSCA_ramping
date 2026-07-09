import torch
import numpy as np
import scipy as sp
from random import shuffle
from collections import namedtuple
from sklearn.model_selection import KFold


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Performs Gaussian smoothing of time-series.

    Arguments
    ----------
    x : np.ndarray
        Time x N_i array of neural activity for region i
    s : float
        Standard deviation to use in Gaussian.

    Returns
    -------
    x_smoothed : np.ndarray
        Neurons x Time array of smoothed neural
        activity.
    """

    return sp.ndimage.gaussian_filter1d(x, sigma=sigma, axis=0, mode="constant")


def presmooth(X: dict, sigma: float = 5.0) -> dict:
    """
    Function that smooths neural activity with respect
    to time, using a Gaussian filter.

    Arguments
    ----------
    X : dict
        Format described in quickstart.ipynb
    sigma : float
        Standard deviation to use in Gaussian. Units
        are in bins.

    Returns
    -------
    X_smoothed : dict
        Same format as X, but now smoothed.
    """
    # Iterate through regions / trials and smooth
    X_smooth = {}
    for r_name, x_r in X.items():
        X_smooth[r_name] = [
            gaussian_smooth(x_r_i.astype("float32"), sigma) for x_r_i in x_r
        ]
    return X_smooth


def region_sizes(X_dict: dict, cumulative: bool = True) -> list:
    """
    Returns an list of the number of neurons in each region

    Parameters
    ----------
    X_dict : dict
        Format described in quickstart.ipynb
    cumulative : bool
        Whether or not to compute the cumulative number of neurons
    """
    region_sizes = np.cumsum([0] + [v.shape[1] for _, v in X_dict.items()])
    if cumulative:
        return region_sizes.tolist()
    else:
        return np.diff(region_sizes).tolist()


def split_into_regions(X_concat: np.ndarray, X_dict: dict) -> dict:
    """
    Splits an array of size (T x (N_1 + N_2 + ... + N_i)) into region dictionary
    described in quickstart.ipynb

    Parameters
    ----------
    X_concat : np.ndarray
        Concatenated neural data across all regions (time x total_neurons).
    X_dict : dict
        Dictionary mapping region names to original neural data arrays (used for region sizes).

    Returns
    -------
    X_out : dict
        Dictionary mapping region names to their corresponding slices of X_dict.
    """
    X_out = {}
    sizes = region_sizes(X_dict)
    for region, start, end in zip(X_dict.keys(), sizes[:-1], sizes[1:]):
        X_out[region] = X_concat[:, start:end]
    return X_out


def dict_of_lists_torchify(X_dict):
    """
    Arguments
    ----------
    X_dict : dict
        Described in ./msca/models.py:fit()

    Returns
    -------
    X_dict : dict
        Same as inputs, but now the trials
        are torch tensors instead of numpy
        arrays.
    """
    return {
        k: [torch.tensor(x, dtype=torch.float32) for x in v] for k, v in X_dict.items()
    }


def trim(X, filter_len, num_convs=1):
    """
    Arguments
    ----------
    X: dict
        Keys are region names and values are torch tensors
        of shape [batch x time x neurons]
    filt_len : int
        The length of the filter used in the convolutions
        in mSCA.

    Returns
    -------
    X_t: dict
        Same as inputs, but now truncated on either side
        to account for convolving with no padding
    """
    trunc = int(np.floor(filter_len / 2)) * num_convs
    return {k: v[:, trunc:-trunc] for k, v in X.items()}


def mask(Z, X_inp, X_tgt, M_c, M_f, filter_len):
    """
    This function handles the masking of the latents,
    reconstructions, and targets.

    Arguments
    ----------
    Z : torch.tensor [batch x time x latents]
        We don't want to apply the sparsity penalty
        to portions of the trial that we padded with
        zeros.
    X_inp : dict
        This contains the predicted reconstructions. The
        keys are region names and values are torch.tensors().
        We don't want to reconstruct the portions of the
        input that have been padded with zeros.
    X_tgt : dict
        This contains the targets. The keys are region names
        and values are torch.tensors(). We have to trim this
        to account for not using padding in the convolutions.
    M : dict
        Contains tensors for appropriate masking.

    Returns
    -------
    Z : torch.tensor [batch x time x latents]
        Original Z, but with latents zeroed out
        in the padded portions.
    X_inp : dict
        Origional X_inp, but with reconstructions zeroed out
        at the padded portions.
    X_tgt : dict
        Original X_tgt, but truncated to match the result
        post-conv x2

    """
    # Trim the targets according to the convolution length
    X_tgt_t = trim(X_tgt, filter_len, num_convs=2)

    # Trim the masks to account for the conv length for recon
    M_t_i = trim(M_c, filter_len, num_convs=2)

    # Trim the masks to account for the conv length for latent
    M_t_z = trim(M_f, filter_len)

    # Multiply the mask with the reconstruction
    X_inp_t = {k: v * M_t_i[k] for k, v in X_inp.items()}

    # Multiply the mask with the latent

    # THIS IS WRONG - NEED TO FILL IN M_t_z
    k = list(M_t_z.keys())[0]
    Z_t = M_t_z[k][:, :, : Z.shape[-1]] * Z

    return Z_t, X_inp_t, X_tgt_t


def detect_deflect(Z: dict, msca: object, trial_num=0) -> np.ndarray:
    """
    Sorts signals by peak deflection time

    Parameters
    ----------
    Z : dict[str, list[np.ndarray]]
        Keys are region names, values are lists of trials where each trial
        is a np.ndarray of shape truncated(t_j) x n_components
    msca : object
        Trained mSCA object
    trial_num : int
        The trial number to order in terms of latent deflection time

    Returns
    -------
    idxs : np.ndarray
        Indices ordering the latents by deflection time.
    """
    # Grab the latent for the current trial
    Z = {k: v[trial_num] for k, v in Z.items()}

    # Amount of squared activity each dimension explains in SCA
    sq_activity = {}
    for k, v in Z.items():
        sq_activity[k] = [
            np.sum(
                (
                    v[:, i : i + 1]
                    @ np.array(msca.model.decoder.model.weight.data)[:, i : i + 1].T  # type: ignore
                )
                ** 2,
                axis=1,
            )
            for i in range(msca.n_components)  # type: ignore
        ]

    # Sum squared activity across regions
    summed_sq_activity = sum([np.array(v) for v in sq_activity.values()])

    # Get the deflection times
    deflection_times = np.argmax(
        np.abs(summed_sq_activity - summed_sq_activity.mean(axis=1)[:, None])  # type: ignore
        > summed_sq_activity.std(axis=1)[:, None],  # type: ignore
        axis=1,
    )

    return np.argsort(deflection_times)


def to_list_of_dicts(X):
    """
    Arguments
    ----------
    X : dict
        Dictionary where keys are region names and values are tensors where
        the first dimension is the number of trials in the dataset

    Returns
    -------
    X : list of dicts
        Same as inputs, but now the trials are lists of dicts
    """
    n_trials = len(next(iter(X.values())))
    return [{region: X[region][i] for region in X} for i in range(n_trials)]


def truncate(X, trunc):
    """
    Arguments
    ----------
    X : dict
        Described in ./msca/models.py:fit()
    trunc : slice
        The slice to truncate the data to

    Returns
    -------
    X : dict
        Same as inputs, but now truncated
    """
    if isinstance(X, dict):
        return {k: X[k][:, trunc] for k in X.keys()}
    else:
        return X[:, trunc]


def torchify(X):
    """
    Parameters
    ----------
    X : dict
        Format described in quickstart.ipynb

    Returns
    -------
    output : dict
        The same format as X, but now each np.ndarray is a torch.tensor
    """
    if isinstance(X, dict):
        return {
            k: [torch.tensor(x, dtype=torch.float32) for x in v] for k, v in X.items()
        }
    elif isinstance(X, list):
        return [torch.tensor(x, dtype=torch.float32) for x in X]
    else:
        raise NotImplementedError


def pad_trials(X_tensor):
    """
    Utility function for padding all trials to be the same length as the
    longest trial -- needed for batching during training

    Parameters
    ----------
    X_tensor : dict
        Dictionary with format described in quickstary.ipynb - values are lists
        of torch.tensors instead of np.ndarrays

    Returns
    -------
    X_padded : dict
        Values are now torch.tensors where the first dimension is the total number
        of trials in the dataset
    lengths : list
        A list of all the pre-padding trial lengths in the dataset
    """

    # Retrieve the lengths of each trial
    k0 = list(X_tensor.keys())[0]
    lengths = [x.shape[0] for x in X_tensor[k0]]

    # Pad the trials to be the same length
    X_padded = {
        k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        for k, v in X_tensor.items()
    }

    return X_padded, lengths


def to_named_tuples(X_list_of_dicts, trial_lengths):
    """
    Converts list of dictionaries to list of named tuples.
    This is used for batching during training

    Parameters
    ----------
    X_list_of_dicts: list
        Each entry is a dictionary where the keys are the region names and the values
        are the neural activity for each region on that trial

    Returns
    -------
    output : list
        A list of named tuples including the data and the trial length for each
        corresponding trial. This is used for masking during training.
    """

    Trial = namedtuple("Trial", ["X", "trial_length"])
    return [
        Trial(X, trial_length)
        for X, trial_length in zip(X_list_of_dicts, trial_lengths)
    ]


def inv_softplus(x: np.ndarray, beta=5.0) -> np.ndarray:
    """
    Inverse softplus function used to initialize the decoder biases.
    """
    return (1 / beta) * np.log(np.exp(beta * x) - 1)


def bi_cross_validation_neuron_indices(X, n_splits=5):
    """
    This will create bi-cross-validation indices (across neurons)

    X : dict
        Dictionary of neural data described in quickstart.ipynb
    n_splits : int
        The number of splits across neurons to do
    """

    # Get the number of neurons for each region
    k0 = list(X.keys())[0]
    if isinstance(X[k0], list):
        n_neurons = {k: v[0].shape[1] for k, v in X.items()}
    else:
        n_neurons = {k: v.shape[1] for k, v in X.items()}

    # Create random splits for each region
    idxs = {k: np.arange(v) for k, v in n_neurons.items()}
    bcv_train_idxs = {k: [] for k in X.keys()}
    bcv_test_idxs = {k: [] for k in X.keys()}
    for k in X.keys():
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(idxs[k]):
            bcv_train_idxs[k].append(train_index)
            bcv_test_idxs[k].append(test_index)

    return bcv_train_idxs, bcv_test_idxs


def select_lam_level(lams, mean_scores, sds, c=3.0):
    """Model selection for the lam_region CD-score curve: the LAST lam still on
    the performance plateau -- the most-regularized model before the score
    departs the plateau.

    The plateau level and noise are estimated robustly (median + MAD) over the
    grid levels left of the knee; the selected level is the LARGEST non-zero lam
    whose mean score is still within `c` robust-SDs of the plateau median. This
    is more stable than "one grid step below the knee": the knee (max-curvature
    corner) can sit a variable distance into the shoulder, so knee-1 lands on the
    plateau for some datasets but one step into the decline for others. Keying on
    "where the score leaves the plateau" selects the same conceptual point
    regardless of exactly where the corner is detected.

    The knee is still computed and returned (it delimits the plateau region and
    is drawn on the plots). A one-SD(lam=0) floor guards against selecting an
    already-degraded point, and the choice is clamped to the first non-zero lam.

    Pure function of the aggregate curve, so it can re-select a stored run
    (ramp_scores.pt's selection_full) without retraining.

    Parameters
    ----------
    lams : array-like         lam_region grid (may include 0.0 as the anchor).
    mean_scores : array-like  per-lam mean CD score.
    sds : array-like          per-lam SD across bootstraps.
    c : float                 plateau-band half-width in robust SDs (default 3).

    Returns
    -------
    dict with keys: selected_level, selected_lam, knee_level, knee_lam.
    """
    lams = np.asarray(lams, dtype=float)
    mean = np.asarray(mean_scores, dtype=float)
    sds = np.asarray(sds, dtype=float)
    n = len(lams)

    nz = lams > 0
    first_nz = int(np.argmax(nz)) if nz.any() else 0

    # ---- knee on log10(lam), lam=0 excluded (delimits the plateau region) ----
    knee_level = n - 1
    knee_lam = float(lams[knee_level])
    if nz.sum() >= 2:
        try:
            from kneed import KneeLocator

            kl = KneeLocator(
                np.log10(lams[nz]),
                mean[nz],
                curve="concave",
                direction="decreasing",
                interp_method="polynomial",
                S=1.0,
                online=True,
            )
        except Exception:
            kl = None
        if kl is not None and kl.knee is not None:
            knee_lam = 10 ** float(kl.knee)
            knee_level = int(np.argmin(np.abs(lams - knee_lam)))

    # ---- plateau band: robust median/MAD over non-zero levels left of knee ---
    region = [i for i in range(n) if nz[i] and i < knee_level]
    if len(region) >= 3:
        # get the cd-score performances for all values of lam_region
        # between 0 and the knee-level
        vals = mean[region]

        # find the median
        med = float(np.median(vals))

        # get the median deviation of each performance level from
        # the median performance
        std = float(np.std(vals - med))

        # find the point at which performance falls below c stddevs of
        thresh = med - c * std

        # largest non-zero lam whose score is still within the plateau band
        within = [i for i in range(n) if nz[i] and mean[i] >= thresh]
        selected_level = max(within) if within else max(knee_level - 1, first_nz)
    else:
        # too few plateau points to estimate -- fall back to the level below knee
        selected_level = max(knee_level - 1, first_nz)

    # ---- one-SD(lam=0) degradation floor ------------------------------------
    if n > 0:
        floor = mean[0] - sds[0]
        while selected_level > first_nz and mean[selected_level] < floor:
            selected_level -= 1
    selected_level = max(int(selected_level), first_nz)

    return {
        "selected_level": int(selected_level),
        "selected_lam": float(lams[selected_level]),
        "knee_level": int(knee_level),
        "knee_lam": float(knee_lam),
    }
