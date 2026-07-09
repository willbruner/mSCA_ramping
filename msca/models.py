from typing import Union
import numpy as np
from tqdm import tqdm
from kneed import KneeLocator

from .coordinated_dropout import *
from .initializations import _initialize, _compute_lam_supervised
from .architectures import *
from .loss_funcs import *
from .utils import *


def convert_to_dataloader(X, Y, batch_size=64, shuffle=True):
    """
    Utility function for converting to PyTorch dataloader

    Parameters
    ----------
    X : dict
        Formatting described in quickstart.ipynb

    Y : np.ndarray
        Numpy array of shape (T x k) - for matching latents

    Returns
    -------
    output : PyTorch dataloader object
    """
    # Convert Y into same shape as X
    k0 = list(X.keys())[0]
    Y = [Y] * len(X[k0])

    # Convert to torch tensors
    X_tensor = torchify(X)
    Y_tensor = torchify(Y)

    ### BOOKMARK

    # Pad trials to be the same length
    X_padded, trial_lengths = pad_trials(X_tensor)

    ### RAMPING: add Y as a key in
    X_padded.update({"Y": Y_tensor})

    # Convert X_padded to list of dicts
    X_list_of_dicts = to_list_of_dicts(X_padded)

    # Store trial lengths + tensors
    X_named_tuples = to_named_tuples(X_list_of_dicts, trial_lengths)

    # Convert to data loader
    train_loader = torch.utils.data.DataLoader(
        X_named_tuples, batch_size=batch_size, shuffle=shuffle  # type: ignore
    )
    return train_loader, trial_lengths


class mSCA:
    """
    The interface for mSCA: training, latent inference, and prediction

    n_components : int
        Number of latent dimensions.
    n_epochs : int, optional
        Number of training epochs (default: 8000).
    loss_func : str
        Loss function to use: 'Gaussian' (for pre-smoothed data) or 'Poisson' (for binned data).
    lr : float, optional
        Learning rate for training (default: 1e-3).
    filter_len : int, optional
        Length of smoothing filter (default: 21).
    linear : bool, optional
        If True, use a linear encoder; if False, use a nonlinear encoder (default: False).
    region_weights : dict, optional
        Per-region weights for reconstruction loss. Keys are region names.
    lam_sparse : float, optional
        Weight for L1 sparsity loss. If None, defaults to 10% of reconstruction loss for Gaussian model;
        4% for the Poisson model
    lam_orthog : float, optional
        Weight for orthogonality loss. If None, defaults to 10% of reconstruction loss for Gaussian model;
        1% for the Poisson model
    lam_region : float, optional
        Weight for region-sparsity loss. If None, defaults to 0%.
    batch_size : int, optional
        Batch size for training (default: 64).
    cd_rate : float, optional
        Coordinated dropout rate (default: 0.0).

    TODO:
        - Confirm defaults for hyperparameters (lam_x) work properly
        - Finalize defaults for lam_sparse (Gaussian and Poisson)
        - Finalize default for lam_region (Gaussian and Poisson)
        - Add GPU support
    """

    def __init__(
        self,
        n_components,
        n_epochs: int = 8000,
        loss_func: str = "Gaussian",
        lr: float = 1e-3,
        filter_len: int = 31,
        linear: bool = False,
        region_weights: Union[None, list[np.ndarray]] = None,
        lam_sparse: float = 0.0,
        lam_orthog: Union[None, float] = None,
        lam_region: float = 0.0,
        batch_size: int = 64,
        cd_rate: float = 0.5,
        device: str = "cpu",
        cd_mode: str = "both",
        post_hoc_epoch: int = -1,
        balance_interval: int = 100,
        init: str = "shared",
        region_sparsity_method: Union[None, str] = None,
        ## RAMPING PROJECT SPECIFIC
        lam_supervised: float = 0.0,
    ):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.loss_func = loss_func
        self.lr = lr
        self.filter_len = filter_len
        self.linear = linear
        self.region_weights = region_weights
        self.lam_sparse = lam_sparse
        self.lam_orthog = lam_orthog
        self.lam_region = lam_region
        self.batch_size = batch_size
        self.cd_rate = cd_rate
        self.device = device
        self.cd_mode = cd_mode
        self.post_hoc_epoch = post_hoc_epoch
        self.balance_interval = balance_interval
        self.init = init

        ## RAMPING PROEJCT SPECIFIC
        self.lam_supervised = lam_supervised
        self.old_lam_sup = 0

        ## EXPERIMENTAL
        self.region_sparsity_method = "decoder_scaling_unnormalized"

    def fit(
        self,
        X: dict[str, list[np.ndarray]],
        Y: list[np.ndarray],
        load: bool = False,
        lam_region: Union[None, float] = None,
    ) -> tuple[object, dict[str, np.ndarray]]:
        # Store the names of the regions
        self.region_names = list(X.keys())

        # Compute initial model parameters
        (
            init_encoder,
            init_decoder,
            init_decoder_bias,
            auto_lam_sparse,
            auto_lam_orthog,
            auto_lam_region,
            auto_region_weights,
            region_sizes,
        ) = _initialize(
            X,
            self.n_components,
            self.loss_func,
            self.lam_sparse,
            self.lam_orthog,
            self.lam_region,
            self.init,
            ### EXPERIMENTAL
            self.region_sparsity_method,
        )

        # Set per-region weights on reconstruction loss
        self.region_weights = (
            auto_region_weights if self.region_weights is None else self.region_weights
        )
        #print(f"Using region-weights = {auto_region_weights}")

        # Se lam
        if self.lam_sparse == "adaptive":
            self.lam_sparse_mode = "adaptive"
        else:
            self.lam_sparse = (
                auto_lam_sparse
                if self.cd_rate == 0.0
                else auto_lam_sparse * self.cd_rate
            )
            self.lam_sparse_mode = "fixed"
        #print(f"Using lam_sparse = {self.lam_sparse}")

        # Automatically set the orthogonality penalty
        self.lam_orthog = (
            auto_lam_orthog if self.cd_rate == 0.0 else auto_lam_orthog * self.cd_rate
        )
        #print(f"Using lam_orthog = {self.lam_orthog}")

        # ---- lam_region: FIXED, ABSOLUTE penalty ---------------------------
        # NOTE: This block previously re-interpreted self.lam_region as a
        if lam_region is not None:
            self.lam_region = lam_region
        self.lam_region = 0.0 if self.lam_region is None else float(self.lam_region)
        #print(f"Using fixed (absolute) lam_region = {self.lam_region}")


        # Instantiate model architecture and set initial params
        self.model = mSCA_architecture(
            init_encoder,
            init_decoder,
            init_decoder_bias,
            region_sizes,
            self.n_components,
            linear=self.linear,
            loss_func=self.loss_func,
            filter_length=self.filter_len,
            init_mode=self.init,
            # EXPERIMENTAL
            region_sparsity_method=self.region_sparsity_method,

        )

        # Convert input data to dataloader
        data_loader, _ = convert_to_dataloader(
            X, Y, batch_size=self.batch_size, shuffle=True
        )

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Make coordinated dropout object
        self.cd = CoordinatedDropout(
            self.n_components,
            self.cd_rate,
            self.filter_len,
            mode=self.cd_mode,
        )

        # Used for truncating trials - post x1 convolutions
        self.half_trunc = slice(self.filter_len // 2, -(self.filter_len // 2))

        # Used for truncating trials - post x2 convolutions
        self.trunc = slice(self.filter_len - 1, -self.filter_len + 1)

        # Set up loss function
        train_criterion = eval(f"{self.loss_func}_loss".lower())

        # Initialize loss tracking
        train_loss_dicts = {
            "reconstruction": [],
            "supervised": [],
            "region_sparsity": [],
            "orthogonality": [],
            "total_loss": [],
        }

        if not load:
            # Iterate through training loop
            for epoch in tqdm(range(self.n_epochs)):

                if epoch < (self.n_epochs - self.post_hoc_epoch):
                    # Every balance_interval epochs, update lam_sparse
                    if (epoch % self.balance_interval == 0) and (
                        self.lam_sparse_mode == "adaptive"
                    ):
                        new_lam_sparse = self.loop(data_loader, mode="balance-sparsity")
                        self.lam_sparse = new_lam_sparse

                    # Training step
                    self.criterion = train_criterion
                    _, _, loss_dict = self.loop(data_loader, mode="train")

                # Allows for non-unit norm decoder scaling with encoder fixed
                elif epoch == self.n_epochs - self.post_hoc_epoch:
                    self._init_post_hoc()
                else:
                    _, _, loss_dict = self.loop(data_loader, mode="post-hoc-scaling")

                # Store training loss
                train_loss_dicts["reconstruction"].append(loss_dict["reconstruction"])
                train_loss_dicts["supervised"].append(loss_dict["supervised"])
                # train_loss_dicts["region_sparsity"].append(loss_dict["region_sparsity"])
                train_loss_dicts["orthogonality"].append(loss_dict["orthogonality"])
                train_loss_dicts["total_loss"].append(loss_dict["total_loss"])

                if epoch == 0:
                    self.old_lam_sup = self.lam_supervised
                    sup_loss = loss_dict["supervised"]
                    recon_loss = loss_dict["reconstruction"]
                    self.lam_supervised = (self.lam_supervised * recon_loss) / sup_loss


            # Concatenate training losses over all epochs
            train_loss_dicts = {k: np.array(v) for k, v in train_loss_dicts.items()}


            return self, train_loss_dicts

        else:
            # This is used to return the initialized model (containers for model exist)
            return self  # type: ignore

    def loop(
        self, data_loader: torch.utils.data.DataLoader, mode: str = "train"
    ) -> tuple[dict[str, list], dict[str, list], dict[str, float]]:

        # Initialize containers for latents
        latents = {k: [] for k in self.region_names}
        reconstructions = {k: [] for k in self.region_names}
        masked_outputs = {k: [] for k in self.region_names}
        masked_masks = {
            k: [] for k in self.region_names
        }  # CD held-out masks (bootstrap)

        # Object we will use to store training loss for this epoch
        epoch_loss_dict = {
            "reconstruction": 0.0,
            # "region_sparsity": 0.0,
            "orthogonality": 0.0,
            "supervised": 0.0,
            "total_loss": 0.0,
        }

        # Set the reconstruction loss function if post-hoc training
        if mode in ["post-hoc-scaling", "balance-sparsity"]:
            r_grads, l1_grads, orth_grads, r_orth_grads = [], [], [], []
            r_loss_f = eval(self.loss_func.lower() + "_f")

        # Iterate over trials in the data_loader
        for _, (X_target, trial_length) in enumerate(data_loader):
            # Apply the mask to the inputs and outputs
            X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                self.cd.forward(
                    X_target,
                    trial_length,
                )
            )

            # Forward pass
            Z, Z_r, X_reconstruction = self.model(X_input_masked)

            # Apply the output mask to the reconstructions
            X_reconstruction_masked = self.cd.mask(
                X_reconstruction, truncate(output_mask, self.trunc)
            )

            # Mask the latents to account for trial length
            Z_masked = self.cd.mask(Z, truncate(Z_mask, self.half_trunc))



            if mode == "train":
                # Compute the loss

                #Logisitic Classifier
                if self.criterion == supervised_regression_poisson_loss:
                    logits = self.model.logreg(Z_r)
                    loss, loss_dict = self.criterion(
                        X_reconstruction_masked,
                        truncate(X_output_masked, self.trunc),
                        logits,
                        self.region_weights,
                        self.model.logreg.weights,
                        self.lam_supervised,
                        mode=mode
                    )

                else:
                    loss, loss_dict = self.criterion(
                        X_reconstruction_masked,
                        truncate(X_output_masked, self.trunc),
                        self.region_weights,
                        Z_masked,
                        self.lam_sparse,
                        self.lam_region,
                        self.lam_orthog,
                        self.model.decoder_scaling,
                        self.model.decoder.model.weight,
                        self.model.encoder.model,
                        ### EXPERIMENTAL
                        X_target["Y"][:, self.half_trunc],
                        self.lam_supervised,
                        self.region_sparsity_method,
                        ### END EXPERIMENTAL
                        mode=mode,
                    )

                # Backpropagation and optimizer step (if training)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Accumulate loss
            if mode == "train":
                epoch_loss_dict["reconstruction"] += loss_dict["reconstruction"]
                epoch_loss_dict["supervised"] += loss_dict["supervised"]
                # epoch_loss_dict["region_sparsity"] += loss_dict["region_sparsity"]
                #epoch_loss_dict["orthogonality"] += loss_dict["orthogonality"]
                epoch_loss_dict["total_loss"] += loss.item()

            # Store latents
            if mode in ["evaluate", "bootstrap"]:
                # Mask the post-convolution latents for visualization
                Z_r_masked = self.cd.mask(Z_r, truncate(Z_r_mask, self.trunc))

                # Append to latents
                [latents[k].append(Z_r_masked[k].squeeze()) for k in self.region_names]  # type: ignore

                # Append reconstructions
                [
                    reconstructions[k].append(X_reconstruction_masked[k].squeeze())  # type: ignore
                    for k in self.region_names
                ]

                # Append masked outputs (ground-truth)
                [
                    masked_outputs[k].append(truncate(X_output_masked[k], self.trunc).squeeze())  # type: ignore
                    for k in self.region_names
                ]

                # Append the (truncated) coordinated-dropout mask. evaluate()
                # uses this to select the held-out entries by mask, instead of
                # by value != 0 -- the latter silently drops held-out bins whose
                # true value is 0 (common for Poisson spike counts) and breaks
                # the recon/output length match.
                output_mask_trunc = truncate(output_mask, self.trunc)
                [
                    masked_masks[k].append(output_mask_trunc[k].squeeze())  # type: ignore
                    for k in self.region_names
                ]

        if mode in ["train", "evaluate", "post-hoc-scaling"]:
            return latents, reconstructions, epoch_loss_dict

        elif mode == "balance-sparsity":
            # Stack gradients for reconstruction and sparsity across trials
            r_grads = torch.stack(r_grads)
            l1_grads = torch.stack(l1_grads)

            # Adjust lam_sparse to match reconstruction learning rate
            lam_sparse = r_grads.mean() / l1_grads.mean()

            return lam_sparse

        elif mode == "bootstrap":
            return reconstructions, masked_outputs, masked_masks

    @torch.no_grad()
    def transform(
        self,
        X: dict[str, list[np.ndarray]],
        Y: list[np.ndarray],
        mode: str = "evaluate",
    ) -> dict[str, list[np.ndarray]]:
        # Convert inputs to data loader maintaining trial ordering
        data_loader, trial_lengths = convert_to_dataloader(
            X, Y, batch_size=1, shuffle=False
        )

        # IMPORTANT: Disable coordinated dropout for finding latents
        self.cd.cd_rate = 0.0

        # Run model to get latents
        latents, _, _ = self.loop(data_loader, mode=mode)

        # Convert latents to numpy array
        latents = {k: [x.numpy() for x in v] for k, v in latents.items()}

        # Compute the magnitude of the latents within each region
        loadings = self.model.decoder.r_loadings()
        latents = {k: [z_i * loadings[k] for z_i in v] for k, v in latents.items()}

        # Remove the padding from the latents
        trial_lengths = np.array(trial_lengths) - (self.filter_len - 1) * 2
        latents = {
            k: [x[:t] for x, t in zip(v, trial_lengths)] for k, v in latents.items()
        }

        # Return latents
        return latents

    @torch.no_grad()
    def predict(
        self, X: dict[str, list[np.ndarray]], Y, mode: str = "evaluate"
    ) -> dict[str, list[np.ndarray]]:
        # Convert inputs to data loader
        data_loader, trial_lengths = convert_to_dataloader(
            X, Y, batch_size=1, shuffle=False
        )

        # IMPORTANT: Disable coordinated dropout for finding latents
        self.cd.cd_rate = 0.0

        # Run model to get latents
        _, reconstructions, _ = self.loop(data_loader, mode=mode)

        # Convert latents to numpy array
        reconstructions = {
            k: [x.numpy() for x in v] for k, v in reconstructions.items()
        }

        # Remove the padding from the latents
        trial_lengths = np.array(trial_lengths) - (self.filter_len - 1) * 2
        reconstructions = {
            k: [x[:t] for x, t in zip(v, trial_lengths)]
            for k, v in reconstructions.items()
        }

        # Return latents
        return reconstructions

    @torch.no_grad()
    def evaluate(
        self,
        X: dict[str, list[np.ndarray]],
        num_bootstraps: int = 100,
        cd_rate: float = 0.5,
    ):
        # Convert input data to dataloader
        data_loader, _ = convert_to_dataloader(
            X, batch_size=self.batch_size, shuffle=True
        )

        # Make coordinated dropout object
        self.cd = CoordinatedDropout(
            self.n_components,
            cd_rate,
            self.filter_len,
            mode="neurons",  # TESTING: BOTH
        )

        # Adjust the scaling factor
        self.cd.scaling_factor = 1.0

        # Scorer depends on the noise model: standard R2 for Gaussian, Poisson
        # deviance pseudo-R2 for spike counts (sklearn r2_score is invalid for
        # counts -- it assumes Gaussian errors).
        is_poisson = self.loss_func == "Poisson"
        if is_poisson:
            from .evaluations import pseudo_r2

            # Null model = each neuron's mean firing rate over all time + trials.
            mean_fr = {
                k: np.concatenate([np.asarray(x) for x in X[k]], axis=0).mean(axis=0)
                for k in self.region_names
            }

        from sklearn.metrics import r2_score

        r2_scores = {k: [] for k in X.keys()}
        for _ in tqdm(range(num_bootstraps), desc="bootstrapping", leave=False):
            # Return reconstructions
            masked_reconstructions, masked_outputs, masked_masks = self.loop(
                data_loader, mode="bootstrap"
            )

            for k in self.region_names:
                # Flatten every trial's masked values into one array per region.
                recon = (
                    torch.cat(
                        [x.detach().reshape(-1) for x in masked_reconstructions[k]]
                    )
                    .cpu()
                    .numpy()
                )
                output = (
                    torch.cat([x.detach().reshape(-1) for x in masked_outputs[k]])
                    .cpu()
                    .numpy()
                )
                mask = (
                    torch.cat([m.detach().reshape(-1) for m in masked_masks[k]])
                    .cpu()
                    .numpy()
                )

                # Select the held-out entries by the coordinated-dropout MASK
                # (mask != 0), NOT by value != 0. The value-based test drops
                # held-out bins whose true value is 0 -- fine for mean-centered
                # Gaussian data, but wrong for Poisson counts (many true zeros),
                # where it left recon/output with mismatched lengths.
                held_out = mask != 0

                if is_poisson:
                    # Per-element neuron-mean null, aligned with the flattened
                    # (..., N) ordering (neurons are the last axis of each trial).
                    null = np.concatenate(
                        [
                            np.broadcast_to(mean_fr[k], tuple(x.shape)).reshape(-1)
                            for x in masked_outputs[k]
                        ]
                    )
                    r2_scores[k].append(
                        float(
                            pseudo_r2(output[held_out], recon[held_out], null[held_out])
                        )
                    )
                else:
                    r2_scores[k].append(r2_score(output[held_out], recon[held_out]))

        return {k: np.array(v) for k, v in r2_scores.items()}

    def save(self, f: str):
        """
        Method for saving trained model weights in mSCA

        Parameters
        ----------
        f : str
            Path to save weights to
        """
        torch.save(self.model.state_dict(), f)

    def load(self, f: str, X: dict, Y):
        """
        Method for loading trained model weights in mSCA

        Parameters
        ----------
        f : str
            Path to load weights from
        X : dict
            Described above
        """
        self.n_epochs = 1
        self.fit(X, Y, load=True)

        # Switch mSCA's decoder
        pre_decoder_w = self.model.decoder.model.weight.data
        pre_decoder_b = self.model.decoder.model.bias

        # Set new decoder
        new_decoder = nn.Linear(*reversed(pre_decoder_w.shape))
        new_decoder.weight.data = pre_decoder_w
        new_decoder.bias.data = pre_decoder_b
        self.model.decoder.model = new_decoder

        self.model.load_state_dict(torch.load(f))

    def _init_post_hoc(self):
        """
        This function intiailizes a separate optimizer to be used for
        training an un-constrained (not unit-norm) decoder during the
        last post-hoc-epoch epochs of training.
        """
        # Freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Switch mSCA's decoder
        pre_decoder_w = self.model.decoder.model.weight.data
        pre_decoder_b = self.model.decoder.model.bias

        # Set new decoder
        new_decoder = nn.Linear(*reversed(pre_decoder_w.shape))
        new_decoder.weight.data = pre_decoder_w
        new_decoder.bias.data = pre_decoder_b
        self.model.decoder.model = new_decoder

        # Add new optimizer
        self.optimizer_post_hoc = torch.optim.Adam(
            [{"params": self.model.decoder.model.parameters(), "lr": 1e-3}]
        )

    #### TESTING FOR REGION-SPECIFICITY ####
    def fit_ramp_lam_region(
        self,
        X,
        max_warmup_epochs=3000,  # Phase 1: settle at base_lam_region
        growth=0.5,
        k=1000,
        max_levels=20,
        n_boot=500,
        eval_cd_rate=0.5,
    ):
        # ---- identical init block to fit(), lines 128-231 -------------------
        self.region_names = list(X.keys())
        (
            init_encoder,
            init_decoder,
            init_decoder_bias,
            auto_lam_sparse,
            auto_lam_orthog,
            auto_lam_region,
            auto_region_weights,
            region_sizes,
        ) = _initialize(
            X,
            self.n_components,
            self.loss_func,
            self.lam_sparse,
            self.lam_orthog,
            self.lam_region,
            self.init,
            self.region_sparsity_method,
        )

        # Auto region-weights
        self.region_weights = (
            auto_region_weights if self.region_weights is None else self.region_weights
        )

        # Auto sparsity
        if self.lam_sparse == "adaptive":
            self.lam_sparse_mode = "adaptive"
        else:
            self.lam_sparse = (
                auto_lam_sparse
                if self.cd_rate == 0.0
                else auto_lam_sparse * self.cd_rate
            )
            self.lam_sparse_mode = "fixed"

        # Auto orthogonality
        self.lam_orthog = (
            auto_lam_orthog if self.cd_rate == 0.0 else auto_lam_orthog * self.cd_rate
        )
        self.lam_region = (
            auto_lam_region if self.cd_rate == 0.0 else auto_lam_region * self.cd_rate
        )
        base_lam_region = self.lam_region  # effective starting scale to ramp from

        # Instantiate architecture
        self.model = mSCA_architecture(
            init_encoder,
            init_decoder,
            init_decoder_bias,
            region_sizes,
            self.n_components,
            linear=self.linear,
            loss_func=self.loss_func,
            filter_length=self.filter_len,
            region_sparsity_method=self.region_sparsity_method,
            init_mode=self.init,
        )

        # Create the data loader
        data_loader, _ = convert_to_dataloader(
            X, batch_size=self.batch_size, shuffle=True
        )

        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Create the coordinated dropout object
        self.cd = CoordinatedDropout(
            self.n_components, self.cd_rate, self.filter_len, mode=self.cd_mode
        )
        self.half_trunc = slice(self.filter_len // 2, -(self.filter_len // 2))
        self.trunc = slice(self.filter_len - 1, -self.filter_len + 1)
        train_criterion = eval(f"{self.loss_func}_loss".lower())
        # ---------------------------------------------------------------------

        # =====================================================================
        # Phase 1 — warmup / settle at base_lam_region (no ramp, no eval gating)
        # =====================================================================
        self.lam_region = base_lam_region
        print("Warmup training ...")
        reconstruction_warmup = []

        # Training for full warmup period
        for epoch in tqdm(range(max_warmup_epochs)):
            if (epoch % self.balance_interval == 0) and (
                self.lam_sparse_mode == "adaptive"
            ):
                self.lam_sparse = self.loop(data_loader, mode="balance-sparsity")
            self.criterion = train_criterion
            _, _, loss_dict = self.loop(data_loader, mode="train")

            reconstruction_warmup.append(loss_dict["reconstruction"])

        # =====================================================================
        # Phase 2 — gated geometric ramp (your existing logic)
        # =====================================================================
        history = []

        ### Adding checkpointing - this could potentially lead to RAM issue
        ### NOTE: checkpoints is a list kept position-for-position aligned with
        ### `lams`/`scores`, so the knee look-up below can index it directly.
        checkpoints = []
        lams, scores = [], []
        sds = []  # CLAUDE: per-lam error bar (SD across bootstraps) for selection/plot

        # Get an initial score using lam_region = 0.
        per_region = self.evaluate(X, num_bootstraps=n_boot, cd_rate=eval_cd_rate)
        self.cd = CoordinatedDropout(
            self.n_components, self.cd_rate, self.filter_len, mode=self.cd_mode
        )
        all_boot = np.stack([per_region[k_] for k_ in self.region_names])  # (R, n_boot)
        init_score = all_boot.mean()
        init_sd = all_boot.mean(axis=0).std()  # SD across bootstraps (n_boot-invariant)

        history.append((-1, 0.0, init_score, init_sd))
        lams.append(0.0)
        scores.append(init_score)
        sds.append(init_sd)  # CLAUDE: SD at lam_region = 0 (used by the rule)
        checkpoints.append(
            {k_: v.detach().cpu().clone() for k_, v in self.model.state_dict().items()}
        )

        print("Iterating over lam_region levels ...")
        for level in tqdm(range(max_levels)):
            self.lam_region = base_lam_region * (1 + growth) ** level

            # recalibrate sparsity ONCE for this lam_region, then hold it fixed
            # TODO: possibly change this to adapt within one value of lam_region a
            #       well
            if self.lam_sparse_mode == "adaptive":
                self.lam_sparse = self.loop(data_loader, mode="balance-sparsity")

            print(f"Training for {k} epochs with lam_region = {self.lam_region}")
            current_reconstruction_losses = []
            for epoch in tqdm(range(k)):
                # Only updating lam_sparse during the warmup period
                self.criterion = train_criterion
                _, _, loss_dict = self.loop(data_loader, mode="train")

                # Save the reconstruction losses from within the inner loop
                current_reconstruction_losses.append(loss_dict["reconstruction"])

            # evaluate on bootstrapped held-out masks; evaluate() clobbers self.cd
            per_region = self.evaluate(X, num_bootstraps=n_boot, cd_rate=eval_cd_rate)
            self.cd = CoordinatedDropout(
                self.n_components, self.cd_rate, self.filter_len, mode=self.cd_mode
            )

            # stack the bootstrapped r2s across lam_regions
            all_boot = np.stack(
                [per_region[k_] for k_ in self.region_names]
            )  # (R, n_boot)
            score = all_boot.mean()
            sd = all_boot.mean(axis=0).std()  # SD across bootstraps (n_boot-invariant)
            history.append((level, self.lam_region, score, sd))

            # Save the current value of lam_region
            lams.append(self.lam_region)
            scores.append(score)
            sds.append(sd)  # CLAUDE: per-lam error bar for selection/plot

            # Checkpoint the model (appended in curve order; aligns with `lams`)
            checkpoints.append(
                {
                    k_: v.detach().cpu().clone()
                    for k_, v in self.model.state_dict().items()
                }
            )

        # CLAUDE: ===== model selection (Option A: last point before the knee) ==
        # CLAUDE: select the grid level just BELOW the knee of the score-vs-
        # CLAUDE: log10(lam) curve -- the most-regularized model that has not yet
        # CLAUDE: started to lose performance. Shared rule: utils.select_lam_level.
        lams_arr = np.array(lams, dtype=float)
        scores_arr = np.array(scores, dtype=float)
        sds_arr = np.array(sds, dtype=float)
        nz = lams_arr > 0  # CLAUDE: used by the log-x plot below

        _sel = select_lam_level(lams_arr, scores_arr, sds_arr)
        knee_level = _sel["knee_level"]
        knee_lam = _sel["knee_lam"]
        selected_level = _sel["selected_level"]
        selected_lam = _sel["selected_lam"]
        print(
            f"[select] knee_lam={knee_lam:g} (level {knee_level}); "
            f"selected lam_region={selected_lam:g} (level {selected_level})"
        )

        # CLAUDE: load the SELECTED model (was: the knee checkpoint).
        self.model.load_state_dict(checkpoints[selected_level])

        # CLAUDE: ---- Plot 1: lam_region vs score (+/- SD), selection marked ----
        import matplotlib.pyplot as plt  # CLAUDE: lazy import for inline plotting

        eps = lams_arr[nz].min() / 3.0  # CLAUDE: position for lam=0 on a log x-axis
        x_plot = np.where(lams_arr > 0, lams_arr, eps)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.errorbar(x_plot, scores_arr, yerr=sds_arr, fmt="o", capsize=3)
        ax1.set_xscale("log")
        ax1.axvline(
            x_plot[selected_level],
            ls=":",
            color="k",
            label=f"selected lam_region={selected_lam:g}",
        )
        if np.isfinite(knee_lam):
            ax1.axvline(knee_lam, ls="--", color="0.6", label=f"knee={knee_lam:g}")
        ax1.set_xlabel("lam_region (lam=0 shown at far left)")
        ax1.set_ylabel("held-out R2 (mean over regions)")
        ax1.set_title("lam_region sweep: score +/- SD")
        ax1.legend()
        fig1.tight_layout()
        plt.show()

        # Perform post-hoc scaling for convergence interpretation +
        if self.post_hoc_epoch > 0:
            self._init_post_hoc()
            for _ in range(self.post_hoc_epoch):
                self.loop(data_loader, mode="post-hoc-scaling")

        # Compute latents for each region (on the SELECTED model).
        Z = self.transform(X)

        # Rank the latents in each region by variance (ascending).
        var_idxs = {k: np.concatenate(v).var(axis=0).argsort() for k, v in Z.items()}

        # CLAUDE: seed `boot` with the un-ablated bootstrap R2 ARRAYS (previously
        # CLAUDE: the scalar mean) so every entry is an array(n_boot) for plotting.
        boot = {r: [v] for r, v in self.evaluate(X, cd_rate=eval_cd_rate).items()}

        # CLAUDE: clone (not reference) the decoder scaling so it can be restored
        # CLAUDE: after the ablation sweep zeroes the columns.
        orig_decoder_scaling = self.model.decoder_scaling.data.clone()

        ### TODO: Iteratively prune dimensions based on their variance
        with torch.no_grad():
            for i in tqdm(range(self.n_components)):
                # iterating over regions
                for region_col, region_name in enumerate(Z.keys()):
                    self.model.decoder_scaling[
                        int(var_idxs[region_name][i]), region_col
                    ] = 0

                # get the change in performance on pseudo-held-out data
                r2s = self.evaluate(X, cd_rate=0.5)

                for r in Z.keys():
                    boot[r].append(r2s[r])

        # CLAUDE: restore the decoder scaling that the ablation sweep zeroed, so
        # CLAUDE: the returned model is the selected (un-ablated) one.
        with torch.no_grad():
            self.model.decoder_scaling.data = orig_decoder_scaling

        # CLAUDE: ---- Plot 2: ablation curves, one subplot per region ----------
        # CLAUDE: x = number of dimensions ablated (0..n_components); y = held-out
        # CLAUDE: R2 mean +/- SD across bootstraps, from `boot`.
        region_names = list(Z.keys())
        x_ab = np.arange(self.n_components + 1)
        fig2, axes = plt.subplots(
            1, len(region_names), figsize=(6 * len(region_names), 4), squeeze=False
        )
        for ax2, r in zip(axes[0], region_names):
            arr = np.stack(
                [np.asarray(b).reshape(-1) for b in boot[r]]
            )  # (n_ab, n_boot)
            m, s = arr.mean(axis=1), arr.std(axis=1)
            ax2.errorbar(x_ab, m, yerr=s, fmt="-o", capsize=3)
            ax2.set_title(f"region: {r}")
            ax2.set_xlabel("# dimensions ablated")
            ax2.set_ylabel("Coordinated dropout R2")
        fig2.suptitle(f"ablation @ selected lam_region={selected_lam:g}")
        fig2.tight_layout()
        plt.show()

        # CLAUDE: richer return -- model + selection diagnostics + ablation boot.
        results = {
            "lams": lams_arr,
            "scores": scores_arr,
            "sds": sds_arr,
            "knee_lam": float(knee_lam),
            "knee_level": int(knee_level),
            "selected_lam": float(selected_lam),
            "selected_level": int(selected_level),
            "boot": boot,
            "kl": kl,
        }
        return self, results

    # CLAUDE: New method added for the region_sparsity_v3 fit-vs-ramp comparison.
    # CLAUDE: Explicit-grid ramp variant of fit_ramp_lam_region. Differences:
    # CLAUDE:   * ramps through an EXPLICIT list of lam_region values so it lines
    # CLAUDE:     up one-to-one with the fixed-lam sweep (config.LAM_REGIONS);
    # CLAUDE:   * warm-starts each level from the previous one (continuation),
    # CLAUDE:     after a single warmup at lam_region = 0;
    # CLAUDE:   * does NOT do knee-selection / post-hoc scaling / pruning -- it
    # CLAUDE:     only trains, evaluates, and checkpoints at every lam so the
    # CLAUDE:     downstream bi-CV + thresholding can use each checkpoint.
    def fit_ramp_lam_region_beta(
        self,
        X,
        lam_regions,  # CLAUDE: explicit lam_region grid (e.g. config.LAM_REGIONS)
        max_warmup_epochs=3000,  # CLAUDE: Phase 1 warmup at lam_region = 0
        k=500,  # CLAUDE: warm-started epochs per lam_region level
        n_boot=500,  # CLAUDE: bootstraps for the internal evaluate() curve
        eval_cd_rate=0.5,
    ):
        """
        CLAUDE: Ramp lam_region through an explicit grid, checkpointing each level.

        Returns
        -------
        self
        checkpoints : list[dict]
            CPU state_dicts, one per entry of `lam_regions`, position-aligned
            with `lams` / `scores`.
        lams : list[float]
            The lam_region grid actually used (echoes `lam_regions`).
        scores : list[dict]
            Per-lam {region: np.ndarray(n_boot)} held-out reconstruction R2 from
            evaluate() -- the curve used for visualizing ramp performance.
        losses : list[list[float]]
            Per-lam reconstruction-loss trace (length k; empty for a skipped
            lam == 0 level that reuses the warmup model).
        selection : dict
            Model-selection result (same criteria as fit_ramp_lam_region):
            {selected_level, selected_lam, knee_level, knee_lam, mean_scores,
            sds}. self.model is left at the selected checkpoint. Dimension
            thresholding is NOT applied here (done downstream).
        """
        # CLAUDE: ---- init block, mirrors fit() / fit_ramp_lam_region ----------
        self.region_names = list(X.keys())
        (
            init_encoder,
            init_decoder,
            init_decoder_bias,
            auto_lam_sparse,
            auto_lam_orthog,
            auto_lam_region,
            auto_region_weights,
            region_sizes,
        ) = _initialize(
            X,
            self.n_components,
            self.loss_func,
            self.lam_sparse,
            self.lam_orthog,
            self.lam_region,
            self.init,
            self.region_sparsity_method,
        )

        self.region_weights = (
            auto_region_weights if self.region_weights is None else self.region_weights
        )

        if self.lam_sparse == "adaptive":
            self.lam_sparse_mode = "adaptive"
        else:
            self.lam_sparse = (
                auto_lam_sparse
                if self.cd_rate == 0.0
                else auto_lam_sparse * self.cd_rate
            )
            self.lam_sparse_mode = "fixed"

        self.lam_orthog = (
            auto_lam_orthog if self.cd_rate == 0.0 else auto_lam_orthog * self.cd_rate
        )

        self.model = mSCA_architecture(
            init_encoder,
            init_decoder,
            init_decoder_bias,
            region_sizes,
            self.n_components,
            linear=self.linear,
            loss_func=self.loss_func,
            filter_length=self.filter_len,
            region_sparsity_method=self.region_sparsity_method,
            init_mode=self.init,
        )

        data_loader, _ = convert_to_dataloader(
            X, batch_size=self.batch_size, shuffle=True
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.cd = CoordinatedDropout(
            self.n_components, self.cd_rate, self.filter_len, mode=self.cd_mode
        )
        self.half_trunc = slice(self.filter_len // 2, -(self.filter_len // 2))
        self.trunc = slice(self.filter_len - 1, -self.filter_len + 1)
        train_criterion = eval(f"{self.loss_func}_loss".lower())

        # CLAUDE: ---- Phase 1: warmup at lam_region = 0 ----------------------
        self.lam_region = 0.0
        print("Warmup training (lam_region = 0) ...")
        for epoch in tqdm(range(max_warmup_epochs), desc="warmup"):
            if (epoch % self.balance_interval == 0) and (
                self.lam_sparse_mode == "adaptive"
            ):
                self.lam_sparse = self.loop(data_loader, mode="balance-sparsity")
            self.criterion = train_criterion
            self.loop(data_loader, mode="train")

        # CLAUDE: ---- Phase 2: ramp through the explicit lam_region grid -----
        checkpoints: list[dict] = []
        lams: list[float] = []
        scores: list[dict] = []
        losses: list[list] = []
        mean_scores: list[float] = []  # CLAUDE: per-lam mean score (for selection)
        sds: list[float] = []  # CLAUDE: per-lam SD across bootstraps (for selection)

        print("Ramping over explicit lam_region grid ...")
        for lam in tqdm(lam_regions, desc="lam_region grid"):
            self.lam_region = float(lam)

            # CLAUDE: recalibrate lam_sparse ONCE per lam, then hold it fixed.
            if self.lam_sparse_mode == "adaptive":
                self.lam_sparse = self.loop(data_loader, mode="balance-sparsity")

            # CLAUDE: warm-start k epochs at this lam. If the first grid value is
            # CLAUDE: 0.0 the warmup already trained it, so skip (anchor = warmup).
            n_train = 0 if (self.lam_region == 0.0 and len(lams) == 0) else k
            lam_losses = []
            for _ in range(n_train):
                self.criterion = train_criterion
                _, _, loss_dict = self.loop(data_loader, mode="train")
                lam_losses.append(loss_dict["reconstruction"])

            # CLAUDE: internal pseudo-held-out curve. evaluate() rebuilds self.cd,
            # CLAUDE: so restore the training dropout afterwards.
            per_region = self.evaluate(X, num_bootstraps=n_boot, cd_rate=eval_cd_rate)
            self.cd = CoordinatedDropout(
                self.n_components, self.cd_rate, self.filter_len, mode=self.cd_mode
            )

            # CLAUDE: aggregate score (mean over regions per bootstrap) -> mean + SD.
            all_boot = np.stack(
                [per_region[r] for r in self.region_names]
            )  # (R, n_boot)
            mean_scores.append(float(all_boot.mean()))
            sds.append(float(all_boot.mean(axis=0).std()))

            lams.append(self.lam_region)
            scores.append({r: np.asarray(v) for r, v in per_region.items()})
            losses.append(lam_losses)
            checkpoints.append(
                {
                    kk: vv.detach().cpu().clone()
                    for kk, vv in self.model.state_dict().items()
                }
            )

        # CLAUDE: ---- model selection (Option A: last point before the knee) ---
        # CLAUDE: select the grid level just BELOW the knee of the score-vs-
        # CLAUDE: log10(lam) curve -- the most-regularized model that has not yet
        # CLAUDE: started to lose performance. See utils.select_lam_level (shared
        # CLAUDE: with fit_ramp_lam_region and the plotting code so stored runs can
        # CLAUDE: be re-selected without retraining). NOTE: dimension-deletion /
        # CLAUDE: thresholding is intentionally NOT done here.
        lams_arr = np.array(lams, dtype=float)
        mean_arr = np.array(mean_scores, dtype=float)
        sds_arr = np.array(sds, dtype=float)

        _sel = select_lam_level(lams_arr, mean_arr, sds_arr)
        knee_level = _sel["knee_level"]
        knee_lam = _sel["knee_lam"]
        selected_level = _sel["selected_level"]
        selected_lam = _sel["selected_lam"]

        # CLAUDE: leave self.model at the selected checkpoint.
        self.model.load_state_dict(checkpoints[selected_level])

        selection = {
            "selected_level": int(selected_level),
            "selected_lam": selected_lam,
            "knee_level": int(knee_level),
            "knee_lam": float(knee_lam),
            "mean_scores": mean_arr,
            "sds": sds_arr,
        }

        return self, checkpoints, lams, scores, losses, selection
