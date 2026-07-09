import numpy as np
from typing import Union, Callable

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.functional import poisson_nll_loss

### TESTING
from torch.distributions import Categorical


def gaussian_f(
    X_reconstruction: Union[torch.Tensor, np.ndarray],
    X: Union[torch.Tensor, np.ndarray],
    mode: str = "train",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Wrapper function that evaluates reconstructions against X using the Gaussian
    SSE loss

    Arguments
    ----------
    X_reconstruction : torch.tensor or np.array
        Reconstructed neural activity
    X : torch.tensor or np.array
        Grount-truth neural activity

    Returns
    -------
    loss : torch.float32
        Gaussian loss
    """
    # Check if input is a numpy array
    is_numpy = isinstance(X_reconstruction, np.ndarray)

    # Convert to torch tensors if not already
    if is_numpy:
        X = torch.tensor(X)
        X_reconstruction = torch.tensor(X_reconstruction)

    # Sum the loss over neurons and time-points when training / initializing
    if mode == "train":
        loss = (((X_reconstruction - X)) ** 2).sum()

    # Compute the elementwise loss for bootstrapping if evaluating
    elif mode == "evaluate":
        loss = ((X_reconstruction - X)) ** 2

    # Convert back to numpy array if needed
    return loss.numpy() if is_numpy else loss


def poisson_f(
    X_reconstruction: Union[torch.Tensor, np.ndarray],
    X: Union[torch.Tensor, np.ndarray],
    mode: str = "train",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Wrapper function that evaluates reconstructions against X using the Poisson
    negative log-likelihood

    Arguments
    ----------
    X_reconstruction : torch.tensor or np.array
        Reconstructed neural activity
    X : torch.tensor or np.array
        Grount-truth neural activity

    Returns
    -------
    loss : torch.float32
        Poisson loss
    """
    # Check if input is a numpy array
    is_numpy = isinstance(X_reconstruction, np.ndarray)

    # Convert to torch tensors if not already
    if is_numpy:
        X = torch.tensor(X)
        X_reconstruction = torch.tensor(X_reconstruction)

    # Sum the loss over neurons and time-points when training / initializing
    if mode == "train":
        loss = poisson_nll_loss(
            X_reconstruction, X, log_input=False, reduction="sum"  # type: ignore
        )
    # Compute the elementwise loss for bootstrapping if evaluating
    elif mode == "evaluate":
        loss = poisson_nll_loss(
            X_reconstruction, X, log_input=False, reduction="none"  # type: ignore
        )

    # Convert back to numpy array if needed
    return loss.numpy() if is_numpy else loss


def reconstruction_loss(
    X_inp: dict[str, torch.Tensor],
    X_tgt: dict[str, torch.Tensor],
    eval_func: Callable,
    mode: Union[str, None] = None,
) -> list[torch.Tensor]:
    """
    Computes the reconstruction loss of the PCA
    reconstructions using the specified evaluation fn.

    Parameters
    ----------
    X_inp : dict
        Dict of reconstructed neural activity, concatenated
        across trials - keys are region names
    X_tgt : dict
        Dict of ground-truth neural activity, concatenated
        across trials - keys are region names
    eval_func : function
        Function used to compute reconstruction performance.
        Defined in utils.

    Returns
    -------
    recon_loss : list
        Each entry is the reconstruction loss for the
        corresponding region.

    """
    return [
        eval_func(x_i, x_h, mode=mode)
        for x_i, x_h in zip(X_inp.values(), X_tgt.values())
    ]


def region_sparsity_loss(z: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes the region-sparsity loss. Weights the penalty on the region-scaling
    parameter by the magnitude of the latent to focus gradients on signals
    that are meaningful for reconstruction.

    Parameters
    ----------
    z : torch.Tensor
        The latent after combining across regions (see mSCA_architecture.forward())
    scaling : torch.Tensor
        A region-specific scalar parameter used to encourage region-specific solutions
        in the latents signals.
    """
    # Set the magnitude function - using lambda in case we decide to change
    mag_f = lambda x: x.abs().sum(dim=(0, 1))

    # Detaching magnitude calculation so this loss function doesn't affect latent shape
    z_mag = mag_f(z).detach()

    # Computing region-sparsity loss weighting scalars by latent magnitude
    L = (scaling * z_mag[:, None]).abs().sum(axis=1).sum()
    return L


def entropy_region_sparsity_loss(
    z: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    # Clamp scaling to prevent infinite compensation for pre-filter sparsity
    scaling = torch.clamp(scaling, -1, 1)

    # Predicting logits
    scaling = F.softplus(scaling, beta=3.0)

    # Create distribution across regions and compute entropy
    gs = torch.stack(
        [Categorical(logits=scaling[i]).entropy() for i in range(scaling.shape[0])]
    )

    # Scale the region-sparsity term by the magnitude of the latents
    mag_f = lambda x: x.abs().sum(dim=(0, 1))

    # Detaching magnitude calculation so this loss function doesn't affect latent shape
    z_mag = mag_f(z).detach()

    return (z_mag * gs).sum()


def region_sparsity_loss_handler(z, decoder_scaling, region_sparsity_method, V, rs):
    if "decoder_scaling" in region_sparsity_method:
        if "entropy" in region_sparsity_method:
            L = entropy_region_sparsity_loss(z, decoder_scaling)
        else:
            L = region_sparsity_loss(z, decoder_scaling)
    elif "group_sparse" in region_sparsity_method:
        L = group_sparsity(V, rs)
    elif "group_scaled" in region_sparsity_method:
        L = group_sparsity_scaled(V, rs, z)

    return L


def gaussian_loss(
    input: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rws: dict[str, float],
    z: torch.Tensor,
    lam_sparse: float,
    lam_region: float,
    lam_orthog: float,
    decoder_scaling: torch.Tensor,
    V: torch.Tensor,
    encoder: object,
    ## EXPERIMENTAL
    region_sparsity_method=None,
    ## END EXPERIMENTAL
    mode: Union[str, None] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Arguments
    ----------
    input : dict
        Keys are region names, values are torch tensors
        containing reconstructed neural activity.
    target : dict
        Keys are region names, values are torch tensors
        containing the neural activity we're trying to
        reconstruct.
    rws : dict
        Keys are region names, values are the weights
        to apply to each region's reconstruction loss.
    z : torch.tensor
        Latent prior to smoothing / time-shifting which
        is where we will apply the L1 sparsity penalty.
    lam_sparse: float
        Weight to apply to sparsity penalty - the
        higher this is, the sparser the latents will be.
    V : torch.tensor
        Decoder matrix used to reconstruct the neural
        activity. We wil apply group-sparsity to this.

    Returns
    ----------
    loss : torch.tensor
        Returns the weighted sum of the Gaussian
        reconstruction loss, L1 sparsity loss, and the
        region-sparsity loss.
    """

    # Compute Gaussian SSE loss
    rc = reconstruction_loss(input, target, gaussian_f, mode)

    # Weight each region separately
    rc = sum([x * rw for x, rw in zip(rc, rws.values())])

    # Compute sparsity loss
    l1 = torch.sum(torch.abs(z))

    # l1 = torch.sum(torch.log(1 + torch.abs(z) / 1e-4))

    # Compute the group-sparsity loss
    gs = region_sparsity_loss_handler(
        z,
        decoder_scaling,
        region_sparsity_method,
        V,
        [v.shape[-1] for v in input.values()],
    )

    # Compute the orthogonality loss
    orth = torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device)) ** 2

    return (
        rc + l1 * lam_sparse + gs * lam_region + orth * lam_orthog,
        {
            "reconstruction": rc.item(),  # type: ignore
            "latent_sparsity": l1.item() * lam_sparse,
            "region_sparsity": gs.item() * lam_region,
            "orthogonality": orth.item() * lam_orthog,
        },
    )


def poisson_loss(
    input: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rws: dict[str, float],
    z: torch.Tensor,
    lam_sparse: float,
    lam_region: float,
    lam_orthog: float,
    decoder_scaling: torch.Tensor,
    V: torch.Tensor,
    encoder: object,
    ## EXPERIMENTAL
    region_sparsity_method=None,
    ## END EXPERIMENTAL
    mode: Union[str, None] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Arguments
    ----------
    input : dict
        Keys are region names, values are torch tensors
        containing reconstructed neural activity.
    target : dict
        Keys are region names, values are torch tensors
        containing the neural activity we're trying to
        reconstruct.
    rws : dict
        Keys are region names, values are the weights
        to apply to each region's reconstruction loss.
    z : torch.tensor
        Latent prior to smoothing / time-shifting which
        is where we will apply the L1 sparsity penalty.
    lam_sparse: float
        Weight to apply to sparsity penalty - the
        higher this is, the sparser the latents will be.
    V : torch.tensor
        Decoder matrix used to reconstruct the neural
        activity. We wil apply group-sparsity to this.

    Returns
    ----------
    loss : torch.tensor
        Returns the weighted sum of the Poisson
        reconstruction loss, L1 sparsity loss, and the
        region-sparrsity loss.
    """

    # Compute Poisson reconstruction loss
    rc = reconstruction_loss(input, target, poisson_f, mode)

    # Weight each region separately
    rc = sum([x * rw for x, rw in zip(rc, rws.values())])

    # Compute sparsity loss
    l1 = torch.sum(torch.abs(z))

    # Compute the group-sparsity loss
    gs = region_sparsity_loss_handler(
        z,
        decoder_scaling,
        region_sparsity_method,
        V,
        [v.shape[-1] for v in input.values()],
    )

    # Compute the orthogonality loss
    orth = torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device)) ** 2

    return (
        rc + l1 * lam_sparse + gs * lam_region + orth * lam_orthog,
        {
            "reconstruction": rc.item(),  # type: ignore
            "latent_sparsity": l1.item() * lam_sparse,
            "region_sparsity": gs.item() * lam_region,
            "orthogonality": orth.item() * lam_orthog,
        },
    )


def supervised_poisson_loss(
    input: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rws: dict[str, float],
    z: torch.Tensor,
    lam_sparse: float,
    lam_region: float,
    lam_orthog: float,
    decoder_scaling: torch.Tensor,
    V: torch.Tensor,
    encoder: object,
    ## EXPERIMENTAL
    Y: torch.Tensor,
    lam_supervised: float,
    region_sparsity_method=None,
    ## END EXPERIMENTAL
    mode: Union[str, None] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Arguments
    ----------
    input : dict
        Keys are region names, values are torch tensors
        containing reconstructed neural activity.
    target : dict
        Keys are region names, values are torch tensors
        containing the neural activity we're trying to
        reconstruct.
    rws : dict
        Keys are region names, values are the weights
        to apply to each region's reconstruction loss.
    z : torch.tensor
        Latent prior to smoothing / time-shifting which
        is where we will apply the L1 sparsity penalty.
    lam_sparse: float
        Weight to apply to sparsity penalty - the
        higher this is, the sparser the latents will be.
    V : torch.tensor
        Decoder matrix used to reconstruct the neural
        activity. We wil apply group-sparsity to this.

    Returns
    ----------
    loss : torch.tensor
        Returns the weighted sum of the Poisson
        reconstruction loss, L1 sparsity loss, and the
        region-sparsity loss.
    """

    # Compute Poisson reconstruction loss
    rc = reconstruction_loss(input, target, poisson_f, mode)

    # Weight each region separately
    rc = sum([x * rw for x, rw in zip(rc, rws.values())])

    # Compute the group-sparsity loss
    supervised_loss = (z - Y).pow(2).sum()

    # Compute the orthogonality loss
    orth = torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device)) ** 2
    return (
        rc + orth * lam_orthog + supervised_loss * lam_supervised,
        {
            "reconstruction": rc.item(),  # type: ignore
            "orthogonality": orth.item() * lam_orthog,
            "supervised": supervised_loss.item() * lam_supervised,
        },
    )


#### BEGIN TESTING CODE ####
def group_sparsity(V, rs):
    """
    Computes the group-sparsity on the loadings of the
    decoder.

    Arguments
    ----------
    V : torch.tensor
        This is the linear decoder read-out matrix

    Returns
    ----------
    loss : torch.tensor
        Group-sparsity: L1 of L2 norms applied to
        region-specific portions of the decoder loadings

    """
    loss, d0, d1 = [], 0, 0
    for r in rs:  # V.rs:
        d1 = d0 + r
        # loss.append((V.model.weight[d0:d1]**2).sum(axis=0).sqrt())
        loss.append((V[d0:d1] ** 2 + 1e-6).sum(axis=0).sqrt())
        d0 = d1
    return torch.stack(loss, axis=0).abs().sum()


def group_sparsity_scaled(V, rs, z):
    """
    Computes the group-sparsity on the loadings of the
    decoder.

    Arguments
    ----------
    V : torch.tensor
        This is the linear decoder read-out matrix

    Returns
    ----------
    loss : torch.tensor
        Group-sparsity: L1 of L2 norms applied to
        region-specific portions of the decoder loadings

    """
    # Compute the magnitude of the latents
    mag_f = lambda x: x.abs().sum(dim=(0, 1))

    # Detaching magnitude calculation so this loss function doesn't affect latent shape
    z_mag = mag_f(z).detach()

    loss, d0, d1 = [], 0, 0
    for r in rs:  # V.rs:
        d1 = d0 + r
        loss.append(torch.linalg.norm(V[d0:d1], axis=0))
        d0 = d1
    loss = torch.stack(loss) * z_mag[None, :]
    return loss.abs().sum()

def supervised_regression_poisson_loss(
    input: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    regression_predictions: dict[str, list],
    rws: dict[str, float],
    regression_dict: dict[str, list],
    lam_supervised: float,
    mode: Union[str, None] = None,
) -> tuple[float, dict[str, float]]:

    # Compute Poisson reconstruction loss
    rc = reconstruction_loss(input, target, poisson_f, mode)

    # Weight each region separately
    rc = sum([x * rw for x, rw in zip(rc, rws.values())])

    supervised_loss = 0
    for region in regression_predictions:
        goal = torch.tensor([i % 2 for i in range(len(regression_predictions[region]))], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss()
        regs = torch.stack(regression_predictions[region])
        supervised_loss += criterion(regs.flatten(), goal)

    return (
        rc + supervised_loss * lam_supervised,
        {
            "reconstruction": float(rc),
            "supervised": float(supervised_loss * lam_supervised),
        },
    )