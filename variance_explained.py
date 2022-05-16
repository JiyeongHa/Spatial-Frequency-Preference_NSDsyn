import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
import sfp_nsd_utils as utils
import matplotlib.pyplot as plt
import seaborn as sns

def load_R2(sn, R2_path='func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/R2_nsdsynthetic.nii.gz'):
    """load a variance explained file (nii.gz 3D) for a subject"""

    subj = utils.sub_number_to_string(sn)
    R2_dir=f'/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/{subj}/'
    R2_path = os.path.join(R2_dir, R2_path)
    R2_file = nib.load(f'{R2_path}').get_fdata()
    return R2_file

def load_R2_all_subj(sn_list, R2_path='func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/R2_nsdsynthetic.nii.gz'):
    all_subj_R2 = {}
    for sn in sn_list:
        all_subj_R2[utils.sub_number_to_string(sn)] = load_R2(sn, R2_path=R2_path)
    return all_subj_R2

def R2_histogram(sn_list, all_subj_R2, n_col=2, n_row=4, xlimit=100, n_bins=200,
                 save_fig=True, save_file_name='R2_hist.png',
                 save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/'):

    subj_list = [utils.sub_number_to_string(i) for i in sn_list]
    kwargs = dict(alpha=0.5, bins=n_bins, density=True, stacked=True)
    color_list = sns.color_palette("hls", len(sn_list))
    fig, axes = plt.subplots(n_col, n_row, figsize=(12,6), sharex=True, sharey=True)
    max_list = []
    if xlimit == 'final_max':
        for xSN in subj_list:
            x = all_subj_R2[xSN]
            max_list.append(x[~np.isnan(x)].max())
            xlimit = max(max_list)

    for i, (ax, xSN) in enumerate(zip(axes.flatten(), subj_list)):
        x = all_subj_R2[xSN]
        ax.hist(x[~np.isnan(x)], **kwargs, label=xSN, color=color_list[i])
        ax.set_xlim([0, xlimit])
        #ax.set_xlabel('Variance Explained (%)', fontsize=20)
        #ax.set_ylabel('Density', fontsize=20)
        ax.legend(fontsize=15)

    fig = axes[0, 0].figure
    plt.suptitle('Probability Histogram of Variance Explained', y=1.05, size=16)
    fig.text(0.5, 0.04, "Variance Explained (%)", ha="center", va="center", fontsize=20)
    fig.text(0.05, 0.5, "Density", ha="center", va="center", rotation=90, fontsize=20)
    #plt.tight_layout()
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + 'density_histogram' + '_vs_' + 'R2')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{save_file_name}')
        plt.savefig(save_path)
    plt.show()
    return
