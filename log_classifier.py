import numpy as np
import torch
import torch.nn as nn
from msca import mSCA
import pickle
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_bins=82, num_latents=5, learning_rate=0.1, epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.u = nn.Parameter(torch.randn(num_bins) * 0.01)
        self.v = nn.Parameter(torch.randn(num_latents) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        logits = torch.einsum(X, self.u, self.v) + self.b
        return logits

k = 5  # Number of latent dimensions
msca = mSCA(
    n_components=k,  # This determines the dimensionality of the latent space
    filter_len=11,  # This determines the max time-delay the model can find
    lam_orthog=0.0,  # Don't worry about this
    n_epochs=1000,  # This is the number of epochs used to train the model - I recommend setting this higher, check losses for convergence
    loss_func="Supervised_Poisson",
    device="cpu",  # Can set this to cuda:0 if working on a GPU, but kind-of inconsequential
    lam_supervised=0.00001,  # This will control how much you weight the supervised loss (it's set quite high right now)
    )

model_path = f"new_data_models/lambda={msca.lam_supervised}.pt"

with open("spike_dict.pkl", "rb") as f:
    X = pickle.load(f)

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
latents = msca.transform(X_test, Y)

print(len(latents["mPFC"]))
print(latents["mPFC"][0].shape)



model = LogisticRegression()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), model.learning_rate)

for epoch in range(model.epochs):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
