import matplotlib.pyplot as plt
import numpy as np

def plot_avgstd(x, y, axis=None, ax=None):
    if ax==None:
        fig, ax = plt.subplots()
    avg = np.average(y, axis) 
    std = np.std(y, axis)
    ax.errorbar(x, avg, yerr=std, fmt="o")
    return ax
