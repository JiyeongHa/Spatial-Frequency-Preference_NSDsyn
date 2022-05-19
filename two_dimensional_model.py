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


def break_down_phase(df):
    dv_to_group = ['subj', 'freq_lvl', 'names_idx', 'voxel', 'hemi']
    df = df.groupby(dv_to_group).mean().reset_index()

    return df

class Forward():
    """ Define parameters used in forward model"""
    def __init__(self, params, params_idx, subj_df):
        self.params_df = params.iloc[params_idx]
        self.sigma = self.params_df['sigma']
        self.amp = self.params_df['slope']
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
              n_row=4, legend_out=True, alpha=0.5, set_max=True,
              save_fig=False, save_dir='/Users/auna/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred.png'):
    subj = utils.sub_number_to_string(sn)
    cur_df = df.query('subj == @subj')
    col_order = utils.sort_a_df_column(cur_df[to_subplot])
    grid = sns.FacetGrid(cur_df,
                         col=to_subplot, col_order=col_order,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         col_wrap=n_row,
                         legend_out=legend_out,
                         sharex=True, sharey=True)
    if set_max:
        max_point = cur_df[[dp_to_x_axis, dp_to_y_axis]].max().max()
        grid.set(xlim=(0, max_point+0.025), ylim=(0, max_point+0.025))

    g = grid.map(sns.scatterplot, dp_to_x_axis, dp_to_y_axis, alpha=alpha)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1),
                    labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
        #ax.set_xlim(ax.get_ylim())
        #ax.set_xticks(ax.get_yticks())
        ax.set_aspect('equal', adjustable='box')
    grid.fig.subplots_adjust(top=0.8, right=0.9)  # adjust the Figure in rp
    grid.fig.suptitle(f'{subj}', fontsize=18, fontweight="bold")
    #grid.tight_layout()
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{sn}_{save_file_name}')
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return grid

def beta_2Dhist(sn, df, to_subplot="vroinames", to_label="eccrois",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred',
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
              n_row=4, legend_out=True, alpha=0.5, bins=30, set_max=False,
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
    if set_max:
        max_point = cur_df[[dp_to_x_axis, dp_to_y_axis]].max().max()
        grid.set(xlim=(0, max_point), ylim=(0, max_point))
    g = grid.map(sns.histplot, dp_to_x_axis, dp_to_y_axis, bins=bins, alpha=alpha)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1), labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
        ax.set_xlim(ax.get_ylim())
        ax.set_aspect('equal')
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

def beta_1Dhist(sn, df, to_subplot="vroinames",
              x_axis_label='Beta', y_axis_label="Probability",
              legend_title="Beta type", labels=['measured betas', 'model prediction'],
              n_row=4, legend_out=True, alpha=0.5, bins=30,
              save_fig=False, save_dir='/Users/auna/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred.png'):
    subj = utils.sub_number_to_string(sn)
    cur_df = df.query('subj == @subj')
    melt_df = pd.melt(cur_df, id_vars=['subj', 'voxel', 'vroinames'], value_vars=['norm_betas', 'norm_pred'],
                      var_name='beta_type', value_name='beta_value')
    col_order = utils.sort_a_df_column(cur_df[to_subplot])
    grid = sns.FacetGrid(melt_df,
                         col=to_subplot,
                         col_order=col_order,
                         hue="beta_type",
                         palette=sns.color_palette("husl"),
                         col_wrap=n_row,
                         legend_out=legend_out)
    g = grid.map(sns.histplot, "beta_value", stat="probability")
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

# numpy to torch function
def _cast_as_tensor(x):
    """ Change numpy vector to torch vector. The input x should be either a column of dataframe,
     a list, or numpy vector.You can also pass a torch vector but it will print out warnings."""
    if type(x) == pd.Series:
        x = x.values
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32)

def _cast_as_param(x, requires_grad=True):
    """ Change input x to """
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)

def _cast_args_as_tensors(args, on_cuda=False):

    return_args = []
    for v in args:
        if not torch.is_tensor(v):
            v = _cast_as_tensor(v)
        if on_cuda:
            v = v.cuda()
        return_args.append(v)
    return return_args

# SF model class
class SpatialFrequencyModel(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.sigma = _cast_as_param(np.random.random(1))
        self.slope = _cast_as_param(np.random.random(1))
        self.intercept = _cast_as_param(np.random.random(1))
        self.p_1 = _cast_as_param(np.random.random(1))
        self.p_2 = _cast_as_param(np.random.random(1))
        self.p_3 = _cast_as_param(np.random.random(1))
        self.p_4 = _cast_as_param(np.random.random(1))
        self.A_1 = _cast_as_param(np.random.random(1))
        self.A_2 = _cast_as_param(np.random.random(1))
        self.A_3 = 0
        self.A_4 = 0


    def get_Av(self, theta_l, theta_v):
        """ Calculate A_v (formula no. 7 in Broderick et al. (2022)) """
        theta_l = _cast_as_tensor(theta_l)
        theta_v = _cast_as_tensor(theta_v)
        Av = 1 + self.A_1 * torch.cos(2 * theta_l) + \
             self.A_2 * torch.cos(4 * theta_l) + \
             self.A_3 * torch.cos(2 * (theta_l - theta_v)) + \
             self.A_4 * torch.cos(4 * (theta_l - theta_v))
        return Av

    def get_Pv(self, theta_l, theta_v, r_v):
        """ Calculate p_v (formula no. 6 in Broderick et al. (2022)) """
        theta_l = _cast_as_tensor(theta_l)
        theta_v = _cast_as_tensor(theta_v)
        r_v = _cast_as_tensor(r_v)
        ecc_dependency = self.slope * r_v + self.intercept
        Pv = ecc_dependency * (1 + self.A_1 * torch.cos(2 * theta_l) +
                               self.A_2 * torch.cos(4 * theta_l) +
                               self.A_3 * torch.cos(2 * (theta_l - theta_v)) +
                               self.A_4 * torch.cos(4 * (theta_l - theta_v)))
        return Pv

    def forward(self, theta_l, theta_v, r_v, w_l):
        """ Return predicted BOLD response in eccentricity (formula no. 5 in Broderick et al. (2022)) """
        w_l = _cast_as_tensor(w_l)

        Av = self.get_Av(theta_l, theta_v)
        Pv = self.get_Pv(theta_l, theta_v, r_v)
        return Av * torch.exp(-(torch.log2(w_l) + torch.log2(Pv)) ** 2 / (2 * self.sigma ** 2))


    def objective_func(self, x, y):
        mse_loss = torch.nn.functional.mse_loss(x, torch.from_numpy(y).to(torch.float32))

        #Lv = (1/self.sigma**2)
        return mse_loss

def count_nan_in_torch_vector(x):
    return torch.nonzero(torch.isnan(torch.log2(x).view(-1)))