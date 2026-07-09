import pickle
import os, sys
from msca import mSCA
from msca import *
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from msca.loss_funcs import reconstruction_loss, poisson_f
from scipy.stats import pearsonr
import matplotlib.patches as mpatches

graph_losses = False
graph_latents = False
graph_reconstructions = False
graph_correlations = False
bootstrapping = False
delays = False
decode = False
spike_count = False
trial_lengths = True

with open("spike_dict.pkl", "rb") as f:
    X = pickle.load(f)

list_of_neuron_firings = [0] * X["simple lobule"][0].shape[1]
for trial in X["simple lobule"]:
    temp = np.vstack((trial, list_of_neuron_firings))
    list_of_neuron_firings = np.sum(temp, axis=0)

list_of_neuron_firings.flatten()
sorted_neurons = sorted(enumerate(list_of_neuron_firings), key=lambda x: x[1])
sorted_neurons = sorted_neurons[-12:]





k = 5  # Number of latent dimensions
msca = mSCA(
    n_components=k,  # This determines the dimensionality of the latent space
    filter_len=11,  # This determines the max time-delay the model can find
    lam_orthog=0.0,  # Don't worry about this
    n_epochs=1000,  # This is the number of epochs used to train the model - I recommend setting this higher, check losses for convergence
    loss_func="Supervised_Poisson",
    device="cpu",  # Can set this to cuda:0 if working on a GPU, but kind-of inconsequential
    lam_supervised=0.000562 ,  # This will control how much you weight the supervised loss (it's set quite high right now)
    )

model_path = f"new_data_models/lambda={msca.lam_supervised}.pt"

# Making the supervision target (the ramp)
T = X["mPFC"][0].shape[0]
Y = np.arange(T)[::-1] * -1
Y = np.stack([Y] * k).T.astype("float64")  # Set the
Y -= Y.mean(axis=0)


all_regions = ['mPFC', 'secondary motor', 'hippocampus', 'mediodorsal',
                   'lateral habenula', 'globus pallidus', 'primary motor',
                   'DCN', 'retrosplenial', 'simple lobule', 'visual']

rng = np.random.default_rng(seed=0)
train_idxs = np.sort(rng.choice(208, size=165, replace=False))

all_idxs = set(range(208))
test_idxs = sorted(all_idxs - set(train_idxs))

X_train = X.copy()
X_train = X_train
X_test = X.copy()

for region in X.keys():
    X_train[region] = [X_train[region][i] for i in train_idxs]
    X_test[region] = [X_test[region][i] for i in test_idxs]

msca.load(model_path, X_train, Y)
losses = torch.load(f"new_data_models/lam_sup={msca.lam_supervised}_loss.pt", weights_only=False)
supervised_loss = losses["supervised"]
recon_loss = losses["reconstruction"]
reconstructions = msca.predict(X_test, Y)
z = msca.transform(X_test, Y)

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

#os.makedirs(f"lam_sup={msca.lam_supervised}", exist_ok=True)

if graph_losses:
    fig, ax1 = plt.subplots()
    ax1.plot(supervised_loss, color="tab:blue", label="Supervised loss")
    ax1.set_ylabel("Supervised loss", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(recon_loss, color="tab:red", label="Reconstruction loss")
    ax2.set_ylabel("Reconstruction loss", color="tab:red")

    ax1.set_xlabel("Epoch")
    ax1.set_xlim(left=500)
    fig.legend(loc="upper right")
    plt.savefig(f"lam_sup={msca.lam_supervised}/loss_curves.png", dpi=150)
    plt.close()

if graph_reconstructions:
    num = len(reconstructions["mPFC"])

    lst_of_losses = []

    X_data = X_test.copy()
    recons = reconstructions.copy()

    for i in range(num):
        for region in X_train:
            X_data[region] = torch.tensor(X_test[region][i][msca.trunc])
            recons[region] = torch.tensor(reconstructions[region][i])

        loss = (reconstruction_loss(recons, X_data, poisson_f, "train"))
        lst_of_losses.append(loss)
    sorted_losses = sorted(enumerate(lst_of_losses), key=lambda x: x[1])
    sorted_losses = sorted_losses[:10]

    for j in range(10):
        #trial = j[0]
        #rank = sorted_losses.index(j) + 1
        plot_trial_all_neurons(X_train,
                               reconstructions,
                               region="simple lobule",
                               trial_idx=j,
                               neuron_list=[x[0] for x in sorted_neurons],
                               save_path=f"lam_sup={msca.lam_supervised}/trial_{j}.png",
                               start=msca.filter_len,
                               stop=T - msca.filter_len)

if graph_latents:
    lst_of_correlations = []

    for i in range(len(z["mPFC"])):
        corr = 0
        for key in z:
            for j in range(k):
                region_corr, _ = pearsonr(z[key][i][:, j], Y[:, j][msca.trunc])
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
            for region in all_regions:
                plt.plot(z[region][trial_num][:, i], label=region)
            plt.plot(Y[:, i], label="Ramp")

            if i == (k - 1):
                plt.legend()
        rank = sorted_corrs.index(j) + 1
        plt.tight_layout()
        plt.savefig(f"lam_sup={msca.lam_supervised}/latents_trial_{trial_num}_rank_{rank}.png")
        plt.close()

if graph_correlations:
    lam_supervised_list = [msca.lam_supervised]
    lst_of_correlations_train = []
    lst_of_correlations_test = []
    for i in lam_supervised_list:
        msca.lam_supervised = i
        msca.load(f"new_data_models/lambda={msca.lam_supervised}.pt", X_train, Y)
        z_test = msca.transform(X_test, Y)
        z_train = msca.transform(X_train, Y)
        for j in range(len(z_train["mPFC"])):
            all_data_trial = np.concatenate([z_train[key][j] for key in z_train], axis=1)
            Y_tiled_trial = np.tile(Y[msca.trunc], (1, len(z_train)))
            trial_corr, _ = pearsonr(all_data_trial.flatten(), Y_tiled_trial.flatten())
            lst_of_correlations_train.append(trial_corr)

        for j in range(len(z_test["mPFC"])):
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
        plt.savefig(f"lam_sup={msca.lam_supervised}/correlation_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()

if bootstrapping:
    delay_effects_train = bootstrap_delays_decoder(msca, X_train, Y)
    delay_effects_test = bootstrap_delays_decoder(msca, X_test, Y)

    threshold = 0
    offset = 0.2  # horizontal shift so violins don't overlap

    dims_train = list(delay_effects_train.keys())
    dims_test = list(delay_effects_test.keys())

    fig, ax = plt.subplots()

    vp_train = ax.violinplot(
        [v for v in delay_effects_train.values()],
        positions=[k - offset for k in dims_train],
        widths=0.3,
    )
    vp_test = ax.violinplot(
        [v for v in delay_effects_test.values()],
        positions=[k + offset for k in dims_test],
        widths=0.3,
    )

    # Color the two groups differently
    for body in vp_train['bodies']:
        body.set_facecolor('tab:blue')
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    for body in vp_test['bodies']:
        body.set_facecolor('tab:orange')
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    # violinplot doesn't give a labeled legend by default, so build one manually
    train_patch = mpatches.Patch(color='tab:blue', label='X_train')
    test_patch = mpatches.Patch(color='tab:orange', label='X_test')
    ax.legend(handles=[train_patch, test_patch])

    ax.grid(ls=":")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("delay effect score")
    ax.axhline(y=threshold, ls=":", c="k")
    ax.set_title("Delay effect distributions")

    plt.savefig(f"validating_delays_lam_sup={msca.lam_supervised}.png")
    plt.close()

    # Iterate through dimensions and prune out those delays that don't fit our criterion
    #for i in range(msca.n_components):
        #mean, lower = mean_confidence_interval(delay_effects[i])
        #if lower <= threshold:
            #msca.model.filters.mus.data[i] = 0

if delays:
    delays = msca.model.filters.mus

    n_rows, n_cols = delays.shape  # 5, 11
    # x positions for each group (column)
    row_names = ['1', '2', '3', '4', '5']
    colors = plt.cm.tab10(np.linspace(0, 1, n_rows))
    group_positions = np.arange(n_cols)
    bar_width = 0.15  # width of each individual bar within a group

    fig, ax = plt.subplots(figsize=(12, 5))

    for row_idx in range(n_rows):
        # offset each of the 5 bars within a group so they sit side by side
        offset = (row_idx - (n_rows - 1) / 2) * bar_width
        ax.bar(group_positions + offset, delays[row_idx, :].detach().numpy(), width=bar_width,
               label=row_names[row_idx], color=colors[row_idx])

    ax.set_xticks(group_positions)
    ax.set_xticklabels(all_regions, rotation=45, ha='right')
    ax.set_xlabel('Region')
    ax.set_ylabel('Delay')
    ax.legend(title='Latent Number')
    ax.set_title('Grouped delays by region')
    plt.tight_layout()
    plt.savefig("Delays.png")
    plt.close()

if decode:
    region_sizes = {region: [] for region in all_regions}
    for region in all_regions:
        region_sizes[region] = X[region][0].shape[1]
    region_names = list(region_sizes.keys())
    sizes = list(region_sizes.values())
    import numpy as np
    import matplotlib.pyplot as plt

    W = msca.model.decoder.model.weight.data.numpy()
    boundaries = np.concatenate([[0], np.cumsum(sizes)])

    n_regions = len(region_names)
    n_latents = W.shape[1]

    region_avg = np.zeros((n_regions, n_latents))

    start = 0
    for i, size in enumerate(sizes):
        end = start + size
        region_avg[i, :] = W[start:end, :].mean(axis=0)  # average across neurons in this region
        start = end

    vmax = 1

    fig, ax = plt.subplots(figsize=(10, 6))

    # x edges: one per latent boundary (latents themselves stay uniform width)
    x_edges = np.arange(n_latents + 1)
    # y edges: cumulative neuron counts -- this is what gives each region proportional height
    y_edges = boundaries

    im = ax.pcolormesh(x_edges, y_edges, region_avg, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Avg weight value')

    # label each region at its midpoint (in neuron-count space)
    midpoints = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(sizes))]
    ax.set_yticks(midpoints)
    ax.set_yticklabels(region_names, fontsize=11)

    # flip y-axis so first region is at top, matching matshow convention
    ax.invert_yaxis()

    ax.set_xticks(np.arange(0, n_latents, max(1, n_latents // 15)))
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.set_xlabel('Latent dimension')
    ax.set_title('Average decoder weight per latent, by region', pad=20)

    for b in boundaries[1:-1]:
        ax.axhline(y=b, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig("decoder_avg.png")
    plt.close()


    # midpoints for labeling each region block
    midpoints = []
    start = 0
    for size in sizes:
        midpoints.append(start + size / 2)
        start += size

    # --- plot ---
    vmax = 1
    fig, ax = plt.subplots(figsize=(10, 12))  # taller figure gives labels more room
    im = ax.matshow(W, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Weight value', fraction=0.046, pad=0.04, extend='both')

    # region demarcation lines (horizontal, since neurons are on rows)
    for b in boundaries[:-1]:
        ax.axhline(y=b - 0.5, color='black', linestyle='--', linewidth=1.2)

    # only label region midpoints, not every neuron -- this fixes the smushing
    ax.set_yticks(midpoints)
    ax.set_yticklabels(region_names, fontsize=11)

    # x-axis: latent indices, spaced out reasonably instead of every single one
    n_latents = W.shape[1]
    ax.set_xticks(np.arange(0, n_latents, max(1, n_latents // 15)))  # ~15 ticks max
    ax.xaxis.set_ticks_position('bottom')  # move ticks below plot instead of matshow default (top)
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    ax.set_xlabel('Latent dimension')
    ax.set_title('Decoder weights by neuron/region', pad=20)

    plt.tight_layout()
    plt.savefig("decoder.png")
    plt.close()

if spike_count:
    os.makedirs("firing_histograms", exist_ok=True)
    for region in all_regions:
        total_neurons = X[region][0].shape[1]
        firings_per_trial = []
        for trial in X[region]:
            firings_per_trial.append(np.sum(trial))
        mean_val = np.mean(firings_per_trial)

        fig, ax = plt.subplots()
        plt.hist(firings_per_trial, bins=10)
        plt.xlabel('Number of Firings')
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
        plt.ylabel('Frequency')
        plt.title(f'Neuron Firings By Trial in the {region}')

        avg_per_neuron = round(mean_val / total_neurons, 2)
        info_text = (f"Average firings per trial: {round(mean_val, 2)}\n"
                     f"Number of neurons: {total_neurons}\n"
                     f"Average firings per trial per neuron: {avg_per_neuron}")
        ax.text(0.95, 0.95, info_text,
                transform=ax.transAxes,  # coordinates are in axes-fraction, not data units
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.savefig(f"firing_histograms/{region}.png")
        plt.close()

if trial_lengths:
    with h5py.File('climbing_data.mat', 'r') as f:
        bin_size = float(f['bin_size'][()].flatten()[0])

        #get indices of which neurons are in which region
        ui_refs = f['unit_indices'][()].flatten()
        unit_indices = {}
        time_vectors = []
        for i, ref in enumerate(ui_refs):
            unit_indices[all_regions[i]] = f[ref][()].flatten().astype(bool)

        sequences = f['selfpaced_sequences']

        for seq_idx in range(sequences.shape[1]):
            time_vec = f[sequences[1, seq_idx]][()].flatten()  # row 1 = time vector
            time_vectors.append(time_vec)
    lengths = []
    for trial in time_vectors:
        lengths.append(trial[-1] - trial[0])
    mean_val = np.mean(lengths)
    plt.hist(lengths, bins=10)
    plt.xlabel('Length (s)')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    plt.ylabel('Frequency')
    plt.title(f'Trial Lengths')
    plt.savefig("trial_lengths.png")
    plt.close()