from mp_opto.src.load import load_mprecord
from mp_opto.src.neural import bin_spiketimes_bounds
from mp_opto.src.experiment import climbing_bounds_from_logical

import os, sys

sys.path.append(os.getcwd())
from msca import mSCA

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import os
from msca.loss_funcs import reconstruction_loss, poisson_f
from scipy.stats import pearsonr

def plot_trial_all_neurons(
    X,
    X_hat,
    region,
    trial_idx,
    neuron_list,
    save_path="plot",
    start=0,
    stop=50
):
    """
    Plot all neurons in a given trial in a roughly square grid.

    Parameters
    ----------
    X : dict
        Original data.
    X_hat : dict
        Reconstructed data.
    region : str
        Region to plot.
    trial_idx : int
        Trial to plot.
    neuron_list : list of int
        Neuron indices to plot.
    save_path : str
        Output image path.
    """

    def to_array(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
        return np.asarray(item)

    n_neurons = len(neuron_list)
    n_cols = math.ceil(np.sqrt(n_neurons))
    n_rows = math.ceil(n_neurons / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 2.5 * n_rows)
    )

    axes = np.array(axes).reshape(-1)

    x_trial = to_array(X[region][trial_idx])
    x_hat_trial = to_array(X_hat[region][trial_idx])
    _stop = min(stop, x_trial.shape[0], x_hat_trial.shape[0])
    t_axis = np.arange(start, _stop)

    for i, neuron_idx in enumerate(neuron_list):
        ax = axes[i]

        ax.plot(t_axis,
                x_trial[start:_stop, neuron_idx],
                label="Actual",
                color="black",
                linewidth=1)

        ax.plot(t_axis,
                x_hat_trial[start:_stop, neuron_idx],
                label="Reconstructed",
                color="tab:red",
                linestyle="--",
                linewidth=1)

        ax.set_title(f"Neuron {neuron_idx}", fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

    for ax in axes[n_neurons:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{region} — Trial {trial_idx}", fontsize=14)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
def main():
    experiment_path = "./mp37_05172023"

    # SETTTINGS
    filter_len = 11

    # load_mprecord is a function from Diya's mp_opto library. If you want to modify which
    # regions it returns recordings from, I suggest modifying mp_opto.src.load.mp_record
    cfa, rfa, dls, ms, s1, mpfc, analogin, isclimbing = load_mprecord(
        f"{experiment_path}"
    )

    # This function will retrieve the bounds for climbing bouts defined by the onset/offset
    # of the movement of the wheel (I believe).
    bounds = climbing_bounds_from_logical(isclimbing)


    # Make bounds precede movement by 1 second
    pre_bounds = np.zeros_like(bounds)

    # Set the end of the trial to be the onset of movement
    pre_bounds[:, 1] = bounds[:, 0]

    # Set the beginning of the trial to 1s preceding movement onset
    pre_bounds[:, 0] = pre_bounds[:, 1] - 1000

    # Adjust end of trial bound to slightly precede movement onset
    pre_bounds[:, 1] -= 200

    # Adjust the bounds to account for the filter length
    pre_bounds[:, 0] -= (11) * 10
    pre_bounds[:, 1] += (11) * 10

    # Bin the spikes using 10 ms bins + our "trial" bounds
    cfa_binned = bin_spiketimes_bounds(cfa["train"], 10, pre_bounds)
    rfa_binned = bin_spiketimes_bounds(rfa["train"], 10, pre_bounds)
    dls_binned = bin_spiketimes_bounds(dls["train"], 10, pre_bounds)
    ms_binned = bin_spiketimes_bounds(ms["train"], 10, pre_bounds)
    s1_binned = bin_spiketimes_bounds(s1["train"], 10, pre_bounds)
    mpfc_binned = bin_spiketimes_bounds(mpfc["train"], 10, pre_bounds)

    # Put all the data into a dictionary for mSCA
    X = {
        "CFA": cfa_binned,
        "RFA": rfa_binned,
        "DLS": dls_binned,
        "MS": ms_binned,
        "S1": s1_binned,
        "MPFC": mpfc_binned,
    }

    # Instantiate the mSCA object - including some good defaults
    k = 5  # Number of latent dimensions
    msca = mSCA(
        n_components=k,  # This determines the dimensionality of the latent space
        filter_len=11,  # This determines the max time-delay the model can find
        lam_orthog=0.0,  # Don't worry about this
        n_epochs=1,  # This is the number of epochs used to train the model - I recommend setting this higher, check losses for convergence
        loss_func="Supervised_Poisson",
        device="cpu",  # Can set this to cuda:0 if working on a GPU, but kind-of inconsequential
        lam_supervised=0.1,  # This will control how much you weight the supervised loss (it's set quite high right now)
    )

    #os.makedirs(f"lam_sup={msca.lam_supervised}_recons", exist_ok=True)


    # Making the supervision target (the ramp)
    T = X["CFA"][0].shape[0]
    Y = np.arange(T)[::-1] * -1
    Y = np.stack([Y] * k).T.astype("float64")  # Set the
    Y -= Y.mean(axis=0)

    rng = np.random.default_rng(seed=0)
    train_idxs = np.sort(rng.choice(137, size=110, replace=False))

    all_idxs = set(range(137))
    test_idxs = sorted(all_idxs - set(train_idxs))

    X_train = X.copy()
    X_test = X.copy()


    for region in X.keys():
        X_train[region] = [X_train[region][i] for i in train_idxs]
        X_test[region] = [X_test[region][i] for i in test_idxs]


    # This line will actually train the model
    #msca.load(f"lam_sup={msca.lam_supervised}/model.pt", X_test, Y)
    #_, losses = msca.fit(X_train, Y)
    #reconstructions = msca.predict(X_train, Y)
    '''
    msca.save(f'lam_sup={msca.lam_supervised}/model.pt')

    torch.save(losses, f'lam_sup={msca.lam_supervised}/loss.pt')

    supervised_loss = losses["supervised"][100:]
    recon_loss = losses["reconstruction"][100:]

    fig, ax1 = plt.subplots()
    ax1.plot(supervised_loss, color="tab:blue", label="Supervised loss")
    ax1.set_ylabel("Supervised loss", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(recon_loss, color="tab:red", label="Reconstruction loss")
    ax2.set_ylabel("Reconstruction loss", color="tab:red")

    ax1.set_xlabel("Epoch")
    ax1.set_xlim(left=100)
    fig.legend(loc="upper right")
    plt.savefig(f"lam_sup={msca.lam_supervised}/loss_curves.png", dpi=150)
    plt.close()

    for i in range(10):
        plot_X_vs_Xhat_trials(
            X_test, reconstructions,
            region="CFA",
            n_trials=20,
            neuron_idx=i,
            save_path=f"lam_sup={msca.lam_supervised}/CFA_neuron_{i}.png",
            start=msca.filter_len,
            stop=T - msca.filter_len
        )


    # Let's plot the latents for slightly outside the bounds we applied the ramping
    pre_bounds[:, 0] -= (10) * 10
    pre_bounds[:, 1] += (10) * 10

    # We want to embed slightly outside the bounds we applied the penalty
    cfa = bin_spiketimes_bounds(cfa["train"], 10, pre_bounds)
    rfa = bin_spiketimes_bounds(rfa["train"], 10, pre_bounds)
    dls = bin_spiketimes_bounds(dls["train"], 10, pre_bounds)
    ms = bin_spiketimes_bounds(ms["train"], 10, pre_bounds)
    s1 = bin_spiketimes_bounds(s1["train"], 10, pre_bounds)
    mpfc = bin_spiketimes_bounds(mpfc["train"], 10, pre_bounds)
    X_embed = {"CFA": cfa, "RFA": rfa, "DLS": dls, "MS": ms, "S1": s1, "MPFC": mpfc}

    # Compute the low-D representations with the new bounds
    for region in X.keys():
        X_embed[region] = [X_embed[region][i] for i in test_idxs]
'''
    #z = msca.transform(X_embed, Y)
    lam_supervised_list = [0, 0.001, 0.01, 0.1, 1.0]
    lst_of_correlations_train = []
    lst_of_correlations_test = []
    for i in lam_supervised_list:
        msca.lam_supervised = i
        msca.load(f"lam_sup={msca.lam_supervised}/model.pt", X_train, Y)
        z_test = msca.transform(X_test, Y)
        z_train = msca.transform(X_train, Y)
        for j in range(len(z_train["CFA"])):
            all_data_trial = np.concatenate([z_train[key][j] for key in z_train], axis=1)
            Y_tiled_trial = np.tile(Y[msca.trunc], (1, len(z_train)))
            trial_corr, _ = pearsonr(all_data_trial.flatten(), Y_tiled_trial.flatten())
            lst_of_correlations_train.append(trial_corr)

        for j in range(len(z_test["CFA"])):
            all_data_trial = np.concatenate([z_test[key][j] for key in z_test], axis=1)
            Y_tiled_trial = np.tile(Y[msca.trunc], (1, len(z_test)))
            trial_corr, _ = pearsonr(all_data_trial.flatten(), Y_tiled_trial.flatten())
            lst_of_correlations_test.append(trial_corr)

        plt.figure(figsize=(8, 5))
        plt.hist(lst_of_correlations_train, bins=20, color="tab:blue", edgecolor="black", alpha=0.6, label="Train")
        plt.hist(lst_of_correlations_test, bins=20, color="tab:orange", edgecolor="black", alpha=0.6, label="Test")
        plt.axvline(np.mean(lst_of_correlations_train), color="tab:blue", linestyle="--",
                    label=f"Train mean = {np.mean(lst_of_correlations_train):.3f}")
        plt.axvline(np.mean(lst_of_correlations_test), color="tab:orange", linestyle="--",
                    label=f"Test mean = {np.mean(lst_of_correlations_test):.3f}")
        plt.xlabel("Correlation with ramp")
        plt.ylabel("Number of trials")
        plt.title(f"Distribution of per-trial correlations with lam_sup={msca.lam_supervised}")
        plt.legend()
        plt.savefig(f"correlation_histogram_lam_sup={msca.lam_supervised}.png", dpi=150, bbox_inches="tight")
        plt.close()

    '''
        n_trials = len(z["CFA"])

        all_data = np.concatenate([
            np.concatenate([z[key][i] for i in range(n_trials)], axis=0)
            for key in z
        ], axis=1)

        Y_tiled = np.tile(Y[msca.trunc], (n_trials, len(z)))
        corr, _ = pearsonr(all_data.flatten(), Y_tiled.flatten())
        lst.append((msca.lam_supervised, corr))

        msca.load(f"lam_sup={msca.lam_supervised}/model.pt", X_train, Y)
        z_2 = msca.transform(X_train, Y)
        n_trials = len(z_2["CFA"])

        all_data = np.concatenate([
            np.concatenate([z_2[key][i] for i in range(n_trials)], axis=0)
            for key in z_2
        ], axis=1)  # (n_trials*102, k*n_regions)

        Y_tiled = np.tile(Y[msca.trunc], (n_trials, len(z_2)))  # (n_trials*102, k*n_regions)
        corr, _ = pearsonr(all_data.flatten(), Y_tiled.flatten())
        lst_2.append((msca.lam_supervised, corr))

    x1, y1 = zip(*lst)
    x2, y2 = zip(*lst_2)

    plt.scatter(x1, y1, color="tab:blue", label="Testing Data")
    plt.scatter(x2, y2, color="tab:red", label="Training Data")
    plt.legend()
    plt.xscale("log")
    plt.savefig(f"correlations.png")
    plt.close()


    lst_of_correlations = []

    for i in range(len(z["CFA"])):
        corr = 0
        for key in z:
            for j in range(k):
                region_corr, _ = pearsonr(z[key][i][:, j], Y[:, j])
                corr += region_corr
        lst_of_correlations.append(corr)
    sorted_corrs = sorted(enumerate(lst_of_correlations), key=lambda x: x[1])
    sorted_corrs.reverse()

    sorted_corrs = sorted_corrs[:10]
    for j in sorted_corrs:
        trial_num = j[0]
        figure(figsize=(4, 12), dpi=80)
        for i in range(k):
            plt.subplot(k, 1, i + 1)
            plt.plot(z['CFA'][trial_num][:, i], label="CFA")
            plt.plot(z['RFA'][trial_num][:, i], label="RFA")
            plt.plot(z['DLS'][trial_num][:, i], label="DLS")
            plt.plot(z['MS'][trial_num][:, i], label="MS")
            plt.plot(z['S1'][trial_num][:, i], label="S1")
            plt.plot(z['MPFC'][trial_num][:, i], label="MPFC")
            plt.plot(Y[0:102, i], label="Ramp")

            if i == (k - 1):
                plt.legend()
        rank = sorted_corrs.index(j) + 1
        plt.tight_layout()
        plt.savefig(f"lam_sup={msca.lam_supervised}/latents_trial_{trial_num}_rank_{rank}.png")
        plt.close()
        
   
        num = len(reconstructions["CFA"])

        lst_of_losses = []

        X_data = X_train.copy()
        recons = reconstructions.copy()

        for i in range(num):
            for region in X_train:
                X_data[region] = torch.tensor(X_train[region][i][msca.trunc])
                recons[region] = torch.tensor(reconstructions[region][i])
            loss = (reconstruction_loss(recons, X_data, poisson_f, "train"))
            lst_of_losses.append(loss)
        sorted_losses = sorted(enumerate(lst_of_losses), key=lambda x: x[1])
        sorted_losses = sorted_losses[1:11]

        for j in sorted_losses:
            trial = j[0]
            rank = sorted_losses.index(j) + 1
            plot_trial_all_neurons(X_train,
                                   reconstructions,
                                   region="MPFC",
                                   trial_idx=trial,
                                   neuron_list=list(range(30)),
                                   save_path=f"lam_sup={msca.lam_supervised}_recons/MPFC_trial_{trial}_rank_{rank}.png",
                                    start= msca.filter_len,
                                    stop=T - msca.filter_len)


    '''

    # Example cd eval call - use this to evaluate the model for different
    # settings of lam_sparse/lam_supervised/dimensionsality
    # cd_eval_loss = msca.cd_eval(X, criterion="Poisson", num_runs=10)


if __name__ == "__main__":
    main()
