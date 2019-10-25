#!/usr/bin/env python

"""
    plot_hub.py: the plot tool

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plt_fidelity_vs_iter(fidelities,losses,config,indx=0):

    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel('Epoch')
    axs1.set_ylabel('Fidelity between real and fake states')
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Wasserstein Loss')
    plt.tight_layout()
    plt.savefig('{}/{}qubit_{}_{}.png'.format(config.figure_path,config.system_size, config.label, indx))