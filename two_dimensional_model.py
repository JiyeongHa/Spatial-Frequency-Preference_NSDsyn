import sys
# sys.path.append('../../')
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
from timeit import default_timer as timer


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
        self.r_v = self.subj_df['eccentricity']  # voxel eccentricity (in degrees)
        self.w_l = self.subj_df['local_sf']  # in cycles per degree

    def get_Av(self, full_ver):
        """ Calculate A_v (formula no. 7 in Broderick et al. (2022)) """
        if full_ver is True:
            Av = 1 + self.A_1 * np.cos(2 * self.theta_l) + \
                 self.A_2 * np.cos(4 * self.theta_l) + \
                 self.A_3 * np.cos(2 * (self.theta_l - self.theta_v)) + \
                 self.A_4 * np.cos(4 * (self.theta_l - self.theta_v))
        elif full_ver is False:
            Av = 1
        return Av

    def get_Pv(self, full_ver):
        """ Calculate p_v (formula no. 6 in Broderick et al. (2022)) """
        ecc_dependency = self.amp * self.r_v + self.intercept
        if full_ver is True:
            Pv = ecc_dependency * (1 + self.A_1 * np.cos(2 * self.theta_l) +
                                   self.A_2 * np.cos(4 * self.theta_l) +
                                   self.A_3 * np.cos(2 * (self.theta_l - self.theta_v)) +
                                   self.A_4 * np.cos(4 * (self.theta_l - self.theta_v)))
        elif full_ver is False:
            Pv = ecc_dependency
        return Pv

    def two_dim_prediction(self, full_ver=True):
        """ Return predicted BOLD response in eccentricity (formula no. 5 in Broderick et al. (2022)) """
        Av = self.get_Av(full_ver=full_ver)
        Pv = self.get_Pv(full_ver=full_ver)
        return Av * np.exp(-(np.log2(self.w_l) + np.log2(Pv)) ** 2 / (2 * self.sigma ** 2))




def normalize(voxel_info, to_norm, group_by=["voxel"], for_two_dim_model=True):
    """calculate L2 norm for each voxel and normalized using the L2 norm"""

    if type(voxel_info) == pd.DataFrame:
        if for_two_dim_model is True:
            if all(voxel_info.groupby(group_by).size() == 28) is False:
                raise Exception('There are more than 28 conditions for one voxel!\n')
        normed = voxel_info.groupby(group_by)[to_norm].apply(lambda x: x / np.linalg.norm(x))

    elif type(voxel_info) == torch.Tensor:
        normed = torch.empty(to_norm.shape, dtype=torch.float64)
        for idx in voxel_info.unique():
            voxel_idx = voxel_info == idx
            normed[voxel_idx] = to_norm[voxel_idx] / torch.linalg.norm(to_norm[voxel_idx])
    return normed


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
        grid.set(xlim=(0, max_point + 0.025), ylim=(0, max_point + 0.025))

    g = grid.map(sns.scatterplot, dp_to_x_axis, dp_to_y_axis, alpha=alpha)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1),
                    labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
        # ax.set_xlim(ax.get_ylim())
        # ax.set_xticks(ax.get_yticks())
        ax.set_aspect('equal', adjustable='box')
    grid.fig.subplots_adjust(top=0.8, right=0.9)  # adjust the Figure in rp
    grid.fig.suptitle(f'{subj}', fontsize=18, fontweight="bold")
    # grid.tight_layout()
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


def count_nan_in_torch_vector(x):
    return torch.nonzero(torch.isnan(torch.log2(x).view(-1)))


class SpatialFrequencyDataset:
    def __init__(self, df, beta_col='avg_betas'):
        self.tmp = df[['voxel', 'local_ori', 'angle', 'eccentricity', 'local_sf', beta_col, 'sigma_v']]
        self.my_tensor = torch.tensor(self.tmp.to_numpy())
        self.voxel_info = self.my_tensor[:, 0]
        self.target = self.my_tensor[:, 5]
        self.sigma_v = self.my_tensor[:, 6]

    def df_to_numpy(self):
        return self.tmp.to_numpy()

    def df_to_tensor(self):
        tmp = self.df_to_numpy()
        self.my_tensor = torch.tensor(tmp)
        return self.my_tensor


class SpatialFrequencyModel(torch.nn.Module):
    def __init__(self, subj_tensor, full_ver):
        """ The input subj_df should be across-phase averaged prior to this class."""
        super().__init__()  # Allows us to avoid using the base class name explicitly
        self.sigma = _cast_as_param(np.random.random(1))
        self.slope = _cast_as_param(np.random.random(1))
        self.intercept = _cast_as_param(np.random.random(1))
        self.full_ver = full_ver
        if full_ver is True:
            self.p_1 = _cast_as_param(np.random.random(1))
            self.p_2 = _cast_as_param(np.random.random(1))
            self.p_3 = _cast_as_param(np.random.random(1))
            self.p_4 = _cast_as_param(np.random.random(1))
            self.A_1 = _cast_as_param(np.random.random(1))
            self.A_2 = _cast_as_param(np.random.random(1))
            self.A_3 = 0
            self.A_4 = 0
        self.subj_tensor = subj_tensor
        self.voxel = self.subj_tensor[:, 0]
        self.theta_l = self.subj_tensor[:, 1]
        self.theta_v = self.subj_tensor[:, 2]
        self.r_v = self.subj_tensor[:, 3]
        self.w_l = self.subj_tensor[:, 4]
        self.target = self.subj_tensor[:, 5]
        self.sigma_v = self.subj_tensor[:,6]
        # self.theta_l = _cast_as_tensor(self.subj_df.iloc[0:]['local_ori'])
        # self.theta_v = _cast_as_tensor(self.subj_df.iloc[0:]['angle'])
        # self.r_v = _cast_as_tensor(self.subj_df.iloc[0:]['eccentricity'])  # voxel eccentricity (in degrees)
        # self.w_l = _cast_as_tensor(self.subj_df.iloc[0:]['local_sf'])  # in cycles per degree

    def get_Av(self):
        """ Calculate A_v (formula no. 7 in Broderick et al. (2022)) """
        # theta_l = _cast_as_tensor(theta_l)
        # theta_v = _cast_as_tensor(theta_v)
        if self.full_ver is True:
            Av = 1 + self.A_1 * torch.cos(2 * self.theta_l) + \
             self.A_2 * torch.cos(4 * self.theta_l) + \
             self.A_3 * torch.cos(2 * (self.theta_l - self.theta_v)) + \
             self.A_4 * torch.cos(4 * (self.theta_l - self.theta_v))
        elif self.full_ver is False:
            Av = 1
        return Av

    def get_Pv(self):
        """ Calculate p_v (formula no. 6 in Broderick et al. (2022)) """
        # theta_l = _cast_as_tensor(theta_l)
        # theta_v = _cast_as_tensor(theta_v)
        # r_v = _cast_as_tensor(r_v)
        ecc_dependency = self.slope * self.r_v + self.intercept
        if self.full_ver is True:
            Pv = ecc_dependency * (1 + self.A_1 * torch.cos(2 * self.theta_l) +
                                   self.A_2 * torch.cos(4 * self.theta_l) +
                                   self.A_3 * torch.cos(2 * (self.theta_l - self.theta_v)) +
                                   self.A_4 * torch.cos(4 * (self.theta_l - self.theta_v)))
        elif self.full_ver is False:
            Pv = ecc_dependency
        return Pv

    def forward(self):
        """ In the forward function we accept a Variable of input data and we must
        return a Variable of output data. Return predicted BOLD response
        in eccentricity (formula no. 5 in Broderick et al. (2022)) """
        # w_l = _cast_as_tensor(w_l)

        Av = self.get_Av()
        Pv = self.get_Pv()
        pred = Av * torch.exp(-(torch.log2(self.w_l) + torch.log2(Pv)) ** 2 / (2 * self.sigma ** 2))
        return pred

    def get_pred_into_tensor_df(self, pred):
        update_tensor = torch.cat((self.subj_tensor, pred.reshape((-1, 1))), 1)
        return update_tensor


def loss_fn(voxel_info, sigma_v_info, prediction, target):
    """"""
    norm_pred = normalize(voxel_info=voxel_info, to_norm=prediction)
    norm_measured = normalize(voxel_info=voxel_info, to_norm=target)
    voxel_list = voxel_info.unique()
    loss_all_voxels = torch.empty(voxel_list.shape, dtype=torch.float64)
    for i, idx in zip(range(voxel_list.shape[0]), voxel_list):
        voxel_idx = voxel_info == idx
        n = sum(voxel_idx)
        sigma_v = sigma_v_info[voxel_idx]
        loss_v = (1/n) * torch.dot(sigma_v, ((norm_pred[voxel_idx] - norm_measured[voxel_idx]) ** 2))
        loss_all_voxels[i] = torch.mean(loss_v)
    return loss_all_voxels

def fit_model(model, dataset, learning_rate=1e-4, max_epoch=1000, loss_all_voxels=True, anomaly_detection=True):
    """Fit the model. This function will allow you to run a for loop for N times set as max_epoch,
    and return the output of the training; loss history, model history."""
    torch.autograd.set_detect_anomaly(anomaly_detection)
    # [sigma, slope, intercept, p_1, p_2, p_3, p_4, A_1, A_2]
    my_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(my_parameters, lr=learning_rate)
    losses_history = []
    loss_history = []
    model_history = []
    start = timer()

    for t in range(max_epoch):

        pred = model.forward()  # predictions should be put in here
        losses = loss_fn(dataset.voxel_info, dataset.sigma_v, prediction=pred, target=dataset.target)  # loss should be returned here
        loss = torch.mean(losses)
        if loss_all_voxels is True:
            losses_history.append(losses.detach().numpy())
        model_values = [p.detach().numpy().item() for p in model.parameters() if p.requires_grad]  # output needs to be put in there
        loss_history.append(loss.item())
        model_history.append(model_values)  # more than one item here
        if (t + 1) % 10 == 1:
            print(f'**epoch no.{t} loss: {np.round(loss.item(), 3)}')

        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients
        model.eval()
    end = timer()
    elapsed_time = end - start
    print(f'**epoch no.{max_epoch}: Finished! final model params...\n{model_values}')
    print(f'Elapsed time: {np.round(end - start, 2)} sec')

    return loss_history, model_history, elapsed_time, losses_history


def melt_history_df(history_df):
    return pd.concat(history_df).reset_index().rename(columns={'level_0': 'subj', 'level_1': 'epoch'})


def plot_loss_history(loss_history_df, to_x_axis="epoch", to_y_axis="loss", n_rows=4,
                      x_axis_label="Epoch", y_axis_label="Loss", to_label=None, to_subplot=None,
                      legend_title=None, labels=None, title="Loss change over time (N = 9)",
                      save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                      save_file_name='.png', ci=68, n_boot=100):
    grid = sns.FacetGrid(loss_history_df,
                         col=to_subplot,
                         hue=to_label,
                         col_wrap=n_rows,
                         palette=sns.color_palette("rocket"),
                         legend_out=True,
                         sharex=True, sharey=True)
    g = grid.map(sns.lineplot, to_x_axis, to_y_axis, linewidth=2, ci=ci, n_boot=n_boot)
    grid.fig.set_figwidth(10)
    grid.fig.set_figheight(6)
    grid.set_axis_labels(x_axis_label, y_axis_label, fontsize=18)
    grid.add_legend(bbox_to_anchor=(1, 0.75))
    #grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1), labels=labels, fontsize=18)
    grid.fig.suptitle(f'{title}', fontsize=20, fontweight="bold")
    grid.fig.subplots_adjust(top=0.85, right=0.85)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    plt.semilogy()
    #plt.yscale('log')
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{save_file_name}')
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_parameters(model_history_df, to_x_axis="param", to_y_axis="value",
                    to_label="study_type", legend_title="Study", hue_order=None,
                    x_axis_label="Parameter", y_axis_label="Parameter Value",
                    title="Final parameter values (N = 9)",
                    save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                    save_file_name='.png', rotate_ticks=True):
    sns.set(font_scale=1.3)
    grid = sns.FacetGrid(model_history_df,
                         palette=sns.color_palette("rocket", n_colors=model_history_df[to_label].nunique()),
                         hue=to_label,
                         hue_order=hue_order,
                         legend_out=True,
                         sharex=True, sharey=True)
    grid.map(sns.lineplot,
             to_x_axis, to_y_axis, lw=30, markersize=8, alpha=0.8,
             marker='o', linestyle='', err_style='bars', ci=68)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 0.92), fontsize=15)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    if rotate_ticks:
        plt.xticks(rotation=45)
    grid.fig.subplots_adjust(top=0.9, right=0.75)  # adjust the Figure in rp
    grid.fig.suptitle(f'{title}', fontweight="bold")
    #grid.tight_layout()
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{save_file_name}')
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    pass
