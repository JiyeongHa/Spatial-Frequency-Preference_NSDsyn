import sys
#sys.path.append('../../')
import os
import seaborn as sns
import sfp_nsd_utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import warnings
import argparse
import itertools
import re
import functools
from scipy import stats
from torch.utils import data as torchdata
from hessian import hessian
import binning_eccen as binning


def break_down_phase():
    dv_to_group = ['subj', 'freq_lvl', 'names_idx', 'voxel', 'hemi']
    df = df.groupby(dv_to_group).mean().reset_index()

    return df

class Forward():
    """ Define parameters used in forward model"""
    def __init__(self, params, params_idx, subj_df):
        self.params_df = params.iloc[params_idx]
        self.sigma = self.params_df['sigma']
        self.amp = self.params_df['amp']
        self.intercept = self.params_df['intercept']
        self.p_1 = self.params_df['p_1']
        self.p_2 = self.params_df['p_2']
        self.p_3 = self.params_df['p_3']
        self.p_4 = self.params_df['p_4']
        self.A_1 = self.params_df['A_1']
        self.A_2 = self.params_df['A_2']
        self.A_3 = self.params_df['A_3']
        self.A_4 = self.params_df['A_4']
        self.subj_df = subj_df.copy()
        self.theta_l = self.subj_df['local_ori']
        self.theta_v = self.subj_df['angle']
        self.r_v = self.subj_df['eccentricity'] # voxel eccentricity (in degrees)
        self.w_l = self.subj_df['local_sf'] # in cycles per degree


    def get_Av(self):
        """ Calculate A_v (formula no. 7 in Broderick et al. (2022)) """

        Av = 1 + self.A_1*np.cos(2*self.theta_l) + \
             self.A_2*np.cos(4*self.theta_l) + \
             self.A_3*np.cos(2*(self.theta_l - self.theta_v)) + \
             self.A_4*np.cos(4*(self.theta_l-self.theta_v))
        return Av

    def get_Pv(self):
        """ Calculate p_v (formula no. 6 in Broderick et al. (2022)) """
        ecc_dependency = self.amp*self.r_v + self.intercept
        Pv = ecc_dependency*(1 + self.A_1*np.cos(2*self.theta_l) +
                             self.A_2*np.cos(4*self.theta_l) +
                             self.A_3*np.cos(2*(self.theta_l - self.theta_v)) +
                             self.A_4*np.cos(4*(self.theta_l-self.theta_v)))
        return Pv

    def two_dim_prediction(self):
        """ Return predicted BOLD response in eccentricity (formula no. 5 in Broderick et al. (2022)) """
        Av = self.get_Av()
        Pv = self.get_Pv()
        return Av*np.exp(-(np.log2(self.w_l)+np.log2(Pv))**2/(2*self.sigma**2))

def normalize(df, to_norm, group_by=["voxel"]):
    """calculate L2 norm for each voxel """

    if all(df.groupby(group_by).size() == 28) is False:
        raise Exception('There are more than 28 conditions for one voxel!\n')
    l2_norm = df.groupby(group_by)[to_norm].apply(lambda x: x / np.linalg.norm(x))

    return l2_norm


def beta_comp(sn, df, to_subplot="vroinames", to_label="eccrois",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred',
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
              n_row=4, legend_out=True,
              save_fig=False, save_dir='/Users/auna/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred.png'):
    subj = utils.sub_number_to_string(sn)
    cur_df = df.query('subj == @subj')
    col_order = utils.sort_a_df_column(cur_df[to_subplot])
    grid = sns.FacetGrid(cur_df,
                         col=to_subplot,
                         col_order=col_order,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         col_wrap=n_row,
                         legend_out=legend_out,
                         sharex=True, sharey=True)
    g = grid.map(sns.scatterplot, dp_to_x_axis, dp_to_y_axis)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1), labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    grid.fig.suptitle(f'{subj}', fontsize=18, fontweight="bold")
    grid.tight_layout()
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{sn}_{save_file_name}')
        plt.savefig(save_path)
    plt.show()
