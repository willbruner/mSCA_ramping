from typing import Callable
from sklearn.decomposition import PCA
from scipy.stats import entropy

from .utils import *
from .loss_funcs import *

### TESTING MORE SOPHISTICATED INITIALIZATION
from mvlearn.embed import MCCA
from sklearn.decomposition import PCA
from scipy.linalg import null_space


def _init_shared(X: dict, n_components: int, loss_func: str):
    # Concatenate X across regions
    X_r_concat = np.concatenate(list(X.values()), axis=1)

    # Fit PCA
    pca = PCA(n_components=n_components).fit(X_r_concat)

    # Compute latents
    Z = pca.transform(X_r_concat)

    # Compute reconstructions
    X_reconstruction = pca.inverse_transform(Z)

    # Use ReLU to constrain reconstructions >= 0
    if loss_func == "Poisson":
        X_reconstruction = np.maximum(X_reconstruction, 0)

    # Partitions back into regions
    X_reconstruction = split_into_regions(X_reconstruction, X)

    # Split U and get V
    U = split_into_regions(pca.components_, X)
    V = {k: u.T for k, u in U.items()}

    return (
        Z,
        U,
        V,
        X_reconstruction,
    )  # , b_dec_init


def _init_unique(X: dict, n_components: int, loss_func: str):
    # Grab the total number of neurons
    N = sum([v.shape[1] for v in X.values()])

    #### TESTING UNIQUE INITIALZIATION
    U_cat, X_reconstruction, Z = [], [], []
    col_counter_beg = 0
    for i, (k, v) in enumerate(X.items()):
        # Fit PCA to region data
        pca_k = PCA(n_components=n_components // len(X)).fit(v)

        # Make block sparse - evenly splitting number of latents across regions
        row_counter_beg = i * n_components // len(X)
        row_counter_end = (i + 1) * n_components // len(X)

        # For iterating over neurons
        col_counter_end = col_counter_beg + v.shape[1]

        # For block-sparse matrix
        container = np.zeros((n_components, N))
        container[row_counter_beg:row_counter_end, col_counter_beg:col_counter_end] = (
            pca_k.components_
        )
        col_counter_beg = col_counter_end

        # Get the latent for this region
        Z_k = pca_k.transform(v)
        Z.append(Z_k)

        # Get the reconstruction
        X_reconstruction_k = pca_k.inverse_transform(Z_k)
        X_reconstruction.append(X_reconstruction_k)

        # Save the components for this region
        U_cat.append(container)

    # Concatenate across regions
    Z = np.concatenate(Z, axis=1)
    X_reconstruction = np.concatenate(X_reconstruction, axis=1)
    U = sum(U_cat)

    # Use ReLU to constrain reconstructions >= 0
    if loss_func == "Poisson":
        X_reconstruction = np.maximum(X_reconstruction, 0)

    # Partitions back into regions
    X_reconstruction = split_into_regions(X_reconstruction, X)

    # # Split U and V into region dicts
    U = split_into_regions(U, X)
    V = {k: u.T for k, u in U.items()}

    return (
        Z,
        U,
        V,
        X_reconstruction,
    )  # , b_dec_init


def _init_unique_v2(X: dict, n_components: Union[int, list], loss_func: str):

    if isinstance(n_components, list):
        n_shared, n_unique = n_components
    else:
        n_shared = n_components // 2
        n_unique = (n_components // 2) // len(X)  # this is n_unique PER region

        # Add one more to shared
        if (n_shared + (n_unique * len(X))) < n_components:
            n_shared += 1

    # Split X into a list
    X_list = [v for v in X.values()]

    # Fit mcca and find latents from projections into shared space
    mcca = MCCA(n_components=n_shared)
    mcca = mcca.fit(X_list)

    # Get the loading matrices for each
    V_shared = mcca.loadings_

    # Get unique components for each region
    V_unq = [null_space(V.T) for V in V_shared]

    # Project each region into private subspace
    X_priv = [(x @ v @ v.T) for x, v in zip(X_list, V_unq)]

    # Fit PCA to private projections
    PCA_unq = [PCA(n_components=n_unique).fit(x).components_ for x in X_priv]

    # Container for final V matrix
    V_final = {k: np.ones((x.shape[1], n_components)) * np.inf for k, x in X.items()}

    # Set the shared loadings in V_final
    for i, k in enumerate(V_final.keys()):
        V_final[k][:, :n_shared] = V_shared[i]

    # Set the unique loadings
    for i, k1 in enumerate(X.keys()):
        for j, k2 in enumerate(X.keys()):
            if i == j:
                # set unique loadings to correct block
                V_final[k2][
                    :, (n_shared + (n_unique * i)) : (n_shared + (n_unique * (i + 1)))
                ] = PCA_unq[i].T
            else:
                V_final[k2][
                    :, (n_shared + (n_unique * i)) : (n_shared + (n_unique * (i + 1)))
                ] = (np.random.normal(size=PCA_unq[j].T.shape) / 100)

    # Normalize
    norm_factor = np.linalg.norm(
        np.concatenate([v for v in V_final.values()], axis=0), axis=0
    )
    V = {k: v / norm_factor[None, :] for k, v in V_final.items()}

    # Set encoder values
    U = {k: v.T for k, v in V.items()}

    # Compute initial latents
    Z_r = {k: X[k] @ U[k].T for k, _ in U.items()}
    Z = sum(Z_r.values())

    # Compute the initial reconstructions
    X_reconstruction = {k: Z_r[k] @ V[k].T for k, _ in U.items()}

    # Use ReLU to constrain reconstructions >= 0
    if loss_func == "Poisson":
        X_reconstruction = np.maximum(X_reconstruction, 0)

    # Partitions back into regions
    # X_reconstruction = split_into_regions(X_reconstruction, X)
    # X_reconstruction = X

    # # Split U and V into region dicts
    # U = split_into_regions(U, X)
    # U = {k: v.T for k, v in V_final.items()}

    return (
        Z,
        U,
        V,
        X_reconstruction,
    )  # , b_dec_init


def _pca_reconstruction(
    X: dict, n_components: int, loss_func: str, init="unique"
) -> tuple[np.ndarray, dict, dict, dict, dict]:
    """
    Computes the reconstruction of the neural activity using principal component analysis

    Parameters
    ----------
    X : dict
        Format described in quickstart.ipynb, but now all trials are concatenated into a np.ndarray
        for each region
    n_components : int
        The number of latent factors used in PCA
    loss_func : str
        The loss function - need to make reconstructions >= 0 for spiking data
    init : str
        Whether to initialize the model using PCA shared across all regions or use a block sparse
        initialization where each block is PCA applied to each region separately.

    Returns
    -------
    Z : np.ndarray
        latents computed using PCA
    U : dict
        PCA loadings to be used as mSCA's initial encoder
    V : dict
        PCA loadings to be used as mSCA's initial decoder
    X_reconstruction : dict
        Reconstruction from using PCA
    b_enc_init : np.ndarray
        Initial bias term to use in PCA's encoder
    b_dec_init : dict
        Initial bias term to use in PCA's decoder
    """
    # Concatenate X across regions
    X_r_concat = np.concatenate(list(X.values()), axis=1)

    if init == "shared":
        Z, U, V, X_reconstruction = _init_shared(X, n_components, loss_func)
    elif init == "unique":
        Z, U, V, X_reconstruction = _init_unique_v2(X, n_components, loss_func)
    else:
        raise NotImplementedError

    # TODO: does it make more sense to have this before the encoding?
    if loss_func == "Poisson":
        b_dec_init = inv_softplus(X_r_concat.mean(axis=0), beta=5.0)
    else:
        b_dec_init = X_r_concat.mean(axis=0)

    # Split for multiple brain regions
    b_dec_init = split_into_regions(b_dec_init.reshape(1, -1), X)

    # Return results
    return Z, U, V, X_reconstruction, b_dec_init


def _compute_relative_reconstruction_loss(
    X_concat: dict, X_reconstruction: dict, eval_func: Callable
) -> dict:
    """
    Parameters
    ----------
    X_concat : dict
        Format described in quickstart.ipynb, but now all trials are concatenated into a np.ndarray
        for each region
    X_reconstruction : dict
        Same as X_concat, except for reconstructions, not the original neural activity

    Returns
    -------
    l_rel : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted off.
    """
    # Concatenate ground-truth data across regions
    X_r_concat = np.concatenate(list(X_concat.values()), axis=1)

    # Concatenate PCA + presmoothing reconstruction across regions
    X_r_reconstruction = np.concatenate(list(X_reconstruction.values()), axis=1)

    # Compute the reconstruction loss using "perfect" reconstructions
    l_perf = eval_func(X_r_concat, X_r_concat, mode="evaluate")
    l_perf = {k: v.sum() for k, v in split_into_regions(l_perf, X_concat).items()}

    # Compute the reconstruction loss using the real reconstructions
    l_real = eval_func(X_r_reconstruction, X_r_concat, mode="evaluate")
    l_real = {k: v.sum() for k, v in split_into_regions(l_real, X_concat).items()}

    # Compute the relative reconstruction loss
    l_rel = {k: (l_real[k] - l_perf[k]) for k in l_real.keys()}

    return l_rel

def _compute_lam_supervised(old_lam_sup, X, X_recon, z, Y):
    for region in X:
        X[region] = X[region] = np.hstack(X[region])
        X_recon[region] = np.hstack(X_recon[region])
    l_rel = _compute_relative_reconstruction_loss(X, X_recon, poisson_f)
    supervised_loss = 0
    for region in z:
        for i in range(len(z[region])):
            supervised_loss += (torch.tensor(z[region][i]) - Y).pow(2).sum()
    return (old_lam_sup * (sum(l_rel.values()))) / supervised_loss


def _compute_region_weights(relative_reconstruction_loss: dict) -> dict:
    """
    Computes weights for balancing the reconstruction loss across brain regions. More specifically,
    it weights each regions' reconstruction loss such that it's equal to the mean reconstruction
    loss across regions.

    Parameters
    ----------
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted off.

    Returns
    -------
    region_weights : dict
        A dictionary where the weights are to be applied to the reconstruction loss for each region
        so that mSCA doesn't preferentially find latents more representative of one region over
        another.
    """
    # Compute egion weights to balance loss across regions
    mean_rel_error = sum(relative_reconstruction_loss.values()) / len(
        relative_reconstruction_loss.values()
    )
    return {
        k: (mean_rel_error / v).item() for k, v in relative_reconstruction_loss.items()
    }


def _compute_lam_sparse(
    Z: np.ndarray,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the sparsity penalty (lam_sparse) as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        Latent representations computed using PCA (samples x n_components).
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default sparsity level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Sparsity penalty (lam_sparse) computed relative to the reconstruction loss.
    """

    # Compute the L1 norm of the latents
    L_sparse = np.abs(Z).sum()

    # If the user has not manually passed a sparsity value to use
    if pct is None:
        pct = 1.0 if loss_func == "Gaussian" else 0.025

    # Make lambda sparse such that L1 is pct% of reconstruction
    return sum(relative_reconstruction_loss.values()) * pct / L_sparse


def _compute_lam_orthog(
    n_components: int,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial orthogonality loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Orthogonality penalty (lam_orthog) computed relative to the reconstruction loss.
    """
    # Compute "initial" orthogonality loss estimate
    L_orth = np.sum(n_components * (n_components - 1) * 0.01)

    # If the user has not manually passed a orthogonality value to use defaults
    if pct is None:
        pct = 0.1 if loss_func == "Gaussian" else 0.01

    # Make lambda sparse such that L1 is pct% of reconstruction
    return sum(relative_reconstruction_loss.values()) * pct / L_orth


def _make_C_matrix(n_components, n_regions, init):
    init_scaling = np.ones((n_components, n_regions))
    if init == "unique":
        # Set the number of shared and unique dimensions
        n_shared = n_components // 2
        n_unique = (n_components // 2) // n_regions  # this is n_unique PER region

        # Add one more to shared
        if (n_shared + (n_unique * n_regions)) < n_components:
            n_shared += 1

        # Make unique dims smaller in dims initialized as region-specific
        for i in range(n_regions):
            row = slice(n_shared + (i * n_unique), n_shared + ((i + 1) * n_unique))
            if i == 0:
                init_scaling[row, (i + 1) :] = 0.5
            elif (i > 0) and (i < n_regions - 1):
                init_scaling[row, :i] = 0.5
                init_scaling[row, (i + 1) :] = 0.5
            else:
                init_scaling[row, :i] = 0.5

    # Normalize
    init_scaling /= np.linalg.norm(init_scaling, axis=1)[:, None]
    return init_scaling


def _compute_lam_region_v1(
    Z: np.ndarray,
    n_components: int,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
    init: str = "shared",
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        The latents inferred using PCA
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial region sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Region sparsity penalty (lam_region) computed relative to the reconstruction loss.
    """
    # Set the magnitude function - using lambda function in case we change it
    # mag_f = lambda x: np.std(x, axis=0)
    # mag_f = lambda x: np.std(x, axis=0)

    # Set the function for computing the magnitude of the latents
    mag_f = lambda x: np.abs(x).sum(axis=0)

    # Compute the magnitude across all latents
    mag_z = mag_f(Z)

    # Penalty is applied to tensor of (n_components x n_regions)
    n_regions = len(relative_reconstruction_loss.keys())

    # Set the C params according to the initialization method
    C = _make_C_matrix(Z.shape[1], len(relative_reconstruction_loss), init)

    # Find total contribution - want sparsity across regions (each row)
    L_region = (np.abs(mag_z[:, None]).sum(axis=1)[:, None] * C).sum()

    # If the user has not manually passed a region-sparsity value use defaults
    if pct is None:
        # TODO: determine if we want this to be defaulted or nah
        # pct = 0.1 if loss_func == "Gaussian" else 0.01
        pct = 0.0

    # Compute the region-sparsity penalty
    return sum(relative_reconstruction_loss.values()) * pct / L_region


def _compute_lam_region_v2(
    Z: np.ndarray,
    n_components: int,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
    init: str = "shared",
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        The latents inferred using PCA
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial region sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Region sparsity penalty (lam_region) computed relative to the reconstruction loss.
    """
    # Set the magnitude function - using lambda function in case we change it
    # mag_f = lambda x: np.std(x, axis=0)
    # mag_f = lambda x: np.std(x, axis=0)

    # Set the function for computing the magnitude of the latents
    mag_f = lambda x: np.abs(x).sum(axis=0)

    # Compute the magnitude across all latents
    mag_z = mag_f(Z)

    # Penalty is applied to tensor of (n_components x n_regions)
    n_regions = len(relative_reconstruction_loss.keys())

    # Set the C params according to the initialization method
    C = _make_C_matrix(Z.shape[1], len(relative_reconstruction_loss), init)

    # Find total contribution - want sparsity across regions (each row)
    L_region = (np.abs(mag_z[:, None]).sum(axis=1)[:, None] * C).sum()

    # Multiply by two for encoder scaling as well
    L_region *= 2

    # If the user has not manually passed a region-sparsity value use defaults
    if pct is None:
        # TODO: determine if we want this to be defaulted or nah
        # pct = 0.1 if loss_func == "Gaussian" else 0.01
        pct = 0.0

    # Compute the region-sparsity penalty
    return sum(relative_reconstruction_loss.values()) * pct / L_region


def _compute_lam_region_group_sparse(
    relative_reconstruction_loss: dict,
    V: Union[None, dict] = None,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        The latents inferred using PCA
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial region sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Region sparsity penalty (lam_region) computed relative to the reconstruction loss.
    """
    # Set the magnitude function - using lambda function in case we change it

    # Concatenate the entries of the decoder + gather region-sizes
    V_cat = np.concatenate(list(V.values()), axis=0)
    rs = [v.shape[0] for v in V.values()]

    # Compute the group-sparsity loss
    L_region = group_sparsity(torch.tensor(V_cat), torch.tensor(rs)).numpy()

    # Compute the region-sparsity penalty
    return sum(relative_reconstruction_loss.values()) * pct / L_region


def _compute_lam_region_group_scaled(
    Z: np.ndarray,
    relative_reconstruction_loss: dict,
    V: Union[None, dict] = None,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        The latents inferred using PCA
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial region sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Region sparsity penalty (lam_region) computed relative to the reconstruction loss.
    """
    # Set the magnitude function - using lambda function in case we change it
    mag_f = lambda x: np.abs(x).sum(axis=0)

    # Compute the magnitude across all latents
    mag_z = mag_f(Z)

    # Compute the group-sparsity loss weighted by the size of each latent
    L_region = (
        np.abs(np.stack([np.linalg.norm(v, axis=0) for k, v in V.items()])).sum(axis=0)
        * mag_z
    ).sum()

    # Compute the region-sparsity penalty
    return sum(relative_reconstruction_loss.values()) * pct / L_region


def _compute_lam_region_entropy(
    Z: np.ndarray,
    relative_reconstruction_loss: dict,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        The latents inferred using PCA
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial region sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Region sparsity penalty (lam_region) computed relative to the reconstruction loss.
    """
    # Set the magnitude function - using lambda function in case we change it
    mag_f = lambda x: np.abs(x).sum(axis=0)

    # Compute the magnitude across all latents
    mag_z = mag_f(Z)

    # Make an initially shared distribution of region-scalars
    C = np.ones((Z.shape[1], len(relative_reconstruction_loss))) * 0.5

    # Compute the group-sparsity loss weighted by the size of each latent
    L_region = (entropy(C, axis=1) * mag_z).sum()

    # Compute the region-sparsity penalty
    return sum(relative_reconstruction_loss.values()) * pct / L_region


def _initialize(
    X: dict,
    n_components: int,
    loss_func: str,
    lam_sparse: Union[None, float],
    lam_orthog: Union[None, float],
    lam_region: Union[None, float],
    init: str,
    region_sparsity_method: str,
) -> tuple[
    dict,  # encoder
    dict,  # decoder
    dict,  # decoder bias
    float,  # lam_sparse
    float,  # lam_orthog
    float,  # lam_region
    dict,  # region weights
    dict,  # region sizes
]:

    ### HACK: change supervised-poisson to just poisson
    ###       will want to change this later if we want
    ###       to make ramping-project-specific initialization
    if loss_func == "Supervised_Poisson" or loss_func == "Supervised_Regression_Poisson":
        loss_func = "Poisson"

    # Check for valid loss function
    if loss_func not in ["Gaussian", "Poisson"]:
        raise NotImplementedError

    # Pre-smooth for PCA if using spiking data
    X_smoothed = presmooth(X) if "Poisson" in loss_func else X  # type: ignore

    # Concatenate non-smoothed data to use as target in reconstruction loss
    X_concat = {region: np.concatenate(trials) for region, trials in X.items()}

    # Concatenate smoothed data for computing PCA reconstruction
    X_smoothed_concat = {
        region: np.concatenate(trials) for region, trials in X_smoothed.items()
    }

    # Compute the PCA reconstruction
    Z, U, V, X_reconstruction, b_dec_init = _pca_reconstruction(
        X_smoothed_concat, n_components=n_components, loss_func=loss_func, init=init
    )

    # Compute the relative reconstruction loss
    eval_func_name = f"{loss_func.lower()}_f"

    # RAMPING: truncate to use poisson_f not supervised-poisson_f
    eval_func_name = eval_func_name.split("-")[-1]

    eval_func = eval(eval_func_name)
    relative_reconstruction_loss = _compute_relative_reconstruction_loss(
        X_concat, X_reconstruction, eval_func
    )

    # Balance the reconstruction loss across regions
    rws = _compute_region_weights(relative_reconstruction_loss)

    # Compute the latent sparsity loss as a function of initial reconstruction loss
    if lam_sparse != "adaptive":
        lam_sparse = _compute_lam_sparse(
            Z, relative_reconstruction_loss, loss_func, pct=lam_sparse
        )
    else:
        lam_sparse = None

    # Compute the orthogonality loss as a function of the initial reconstruction loss
    lam_orthog = _compute_lam_orthog(
        n_components, relative_reconstruction_loss, loss_func, pct=lam_orthog
    )

    # Compute the region sparsity loss as a function of initial reconstruction loss
    if "decoder_scaling" in region_sparsity_method:
        if "entropy" in region_sparsity_method:
            lam_region = _compute_lam_region_entropy(
                Z,
                relative_reconstruction_loss,
                pct=lam_region,
            )
        else:
            lam_region = _compute_lam_region_v1(
                Z,
                n_components,
                relative_reconstruction_loss,
                loss_func,
                pct=lam_region,
                init=init,
            )

    elif "group_sparse" in region_sparsity_method:
        # TODO: double-check this
        lam_region = _compute_lam_region_group_sparse(
            relative_reconstruction_loss,
            V,
            pct=lam_region,
        )
    elif "group_scaled" in region_sparsity_method:
        # TODO: double-check this
        lam_region = _compute_lam_region_group_scaled(
            Z,
            relative_reconstruction_loss,
            V,
            pct=lam_region,
        )

    # Compute the region sizes and convert to dictionary
    rs = region_sizes(X_concat, cumulative=False)
    rs = {k: rs[i] for i, k in enumerate(rws.keys())}

    return U, V, b_dec_init, lam_sparse, lam_orthog, lam_region, rws, rs
