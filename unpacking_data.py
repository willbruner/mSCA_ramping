import h5py
from msca import mSCA
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle



def main():
    """
    filter_len = 11
    new_bin_size = 0.01
    all_regions = ['mPFC', 'secondary motor', 'hippocampus', 'mediodorsal',
                   'lateral habenula', 'globus pallidus', 'primary motor',
                   'DCN', 'retrosplenial', 'simple lobule', 'visual']

    spike_data = {region: [] for region in all_regions}

    with h5py.File('climbing_data.mat', 'r') as f:
        bin_size = float(f['bin_size'][()].flatten()[0])
        old_bins_per_new_bin = int(new_bin_size / bin_size)

        #get indices of which neurons are in which region
        ui_refs = f['unit_indices'][()].flatten()
        unit_indices = {}
        for i, ref in enumerate(ui_refs):
            unit_indices[all_regions[i]] = f[ref][()].flatten().astype(bool)

        sequences = f['selfpaced_sequences']

        for seq_idx in range(sequences.shape[1]):
            spike_matrix = f[sequences[0, seq_idx]][()].T  # row 0 = spike data
            time_vec = f[sequences[1, seq_idx]][()].flatten()  # row 1 = time vector
            if time_vec[0] < -1.11:
                #adjust bounds to account for filter length
                pre_movement_mask = ((time_vec >= -1 - filter_len * new_bin_size) &
                                     (time_vec <= -0.2 + filter_len * new_bin_size))
                #put into new bin sizes
                for region in all_regions:
                    region_spikes = spike_matrix[unit_indices[region], :].T
                    region_spikes_bounded = region_spikes[pre_movement_mask]
                    new_bin_number = int(region_spikes_bounded.shape[0] // old_bins_per_new_bin)
                    region_spikes_binned = np.zeros((new_bin_number, region_spikes.shape[1]))
                    for i in range(new_bin_number):
                        region_spikes_binned[i] = np.sum(
                            region_spikes_bounded[i * old_bins_per_new_bin: (i + 1) * old_bins_per_new_bin], axis=0)
                    spike_data[region].append(region_spikes_binned)
    X = spike_data

    with open("spike_dict.pkl", "wb") as f:
        pickle.dump(spike_data, f)

    """

    with open("spike_dict.pkl", "rb") as f:
        X = pickle.load(f)

    k = 5  # Number of latent dimensions
    msca = mSCA(
        n_components=k,  # This determines the dimensionality of the latent space
        filter_len=11,  # This determines the max time-delay the model can find
        lam_orthog=0.0,  # Don't worry about this
        n_epochs=100,
        # This is the number of epochs used to train the model - I recommend setting this higher, check losses for convergence
        loss_func="Supervised_Poisson",
        device="cpu",  # Can set this to cuda:0 if working on a GPU, but kind-of inconsequential
        lam_supervised=0.5,  # This will control how much you weight the supervised loss (it's set quite high right now)
    )

    # os.makedirs(f"lam_sup={msca.lam_supervised}_recons", exist_ok=True)

    # Making the supervision target (the ramp)
    T = X["mPFC"][0].shape[0]
    Y = np.arange(T)[::-1] * -1
    Y = np.stack([Y] * k).T.astype("float64")  # Set the
    Y -= Y.mean(axis=0)

    rng = np.random.default_rng(seed=0)
    train_idxs = np.sort(rng.choice(208, size=165, replace=False))

    all_idxs = set(range(208))
    test_idxs = sorted(all_idxs - set(train_idxs))

    X_train = X.copy()
    X_test = X.copy()

    for region in X.keys():
        X_train[region] = [X_train[region][i] for i in train_idxs]
        X_test[region] = [X_test[region][i] for i in test_idxs]

    _, losses = msca.fit(X_train, Y)
    supervised_loss = losses["supervised"]
    recon_loss = losses["reconstruction"]

    msca.save(f"new_data_models/lambda={msca.old_lam_sup}.pt")
    torch.save(losses, f'new_data_models/lam_sup={msca.old_lam_sup}_loss.pt')

    fig, ax1 = plt.subplots()
    ax1.plot(supervised_loss, color="tab:blue", label="Supervised loss")
    ax1.set_ylabel("Supervised loss", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(recon_loss, color="tab:red", label="Reconstruction loss")
    ax2.set_ylabel("Reconstruction loss", color="tab:red")

    ax1.set_xlabel("Epoch")
    ax1.set_xlim(left=0)
    fig.legend(loc="upper right")
    plt.savefig(f"test.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()






