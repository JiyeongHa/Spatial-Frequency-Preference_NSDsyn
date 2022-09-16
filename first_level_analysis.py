import sys
sys.path.append('../../')
import os
import numpy as np
import itertools
import pandas as pd
import torch
import sfp_nsd_utils as utils
from timeit import default_timer as timer
import two_dimensional_model as model
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe

def torch_log_norm_pdf(x, slope, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = slope * torch.exp(-(torch.log2(x) - torch.log2(mode)) ** 2 / (2 * sigma ** 2))

    return pdf


def np_log_norm_pdf(x, amp, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = amp * np.exp(-(np.log2(x) - np.log2(mode)) ** 2 / (2 * sigma ** 2))
    return pdf


def _track_loss_df(values, create_loss_df=True, loss_df=None,
                   keys=["subj", "vroinames", "eccrois", "alpha", "epoch", "start_loss", "final_loss"]):
    if create_loss_df:
        loss_df = utils.create_empty_df(col_list=keys)
    else:
        if loss_df is None:
            raise Exception("Dataframe for saving loss log is not defined!")

    loss_df = loss_df.append(dict(zip(keys, values)), ignore_index=True)
    return loss_df


def _set_initial_params(init_list):
    if init_list == "random":
        init_list = np.random.random(3)

    slope = torch.tensor([init_list[0]], dtype=torch.float32, requires_grad=True)
    mode = torch.tensor([init_list[1]], dtype=torch.float32, requires_grad=True)
    sigma = torch.tensor([init_list[2]], dtype=torch.float32, requires_grad=True)
    return slope, mode, sigma


def _df_column_to_torch(df, column):
    torch_column_val = torch.from_numpy(df[column].values)
    return torch_column_val


def fit_1D_model(df, sn, stim_class, varea, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):
    subj = utils.sub_number_to_string(sn)
    eroi_list = utils.sort_a_df_column(df[ecc_bins])

    # Initialize output df
    loss_history_df = {}
    model_history_df = {}

    # start fitting process
    cur_df = df.query('(subj == @subj) & (names == @stim_class) & (vroinames == @varea)')
    print(f'##{subj}-{stim_class}-{varea} start!##')
    start = timer()
    for cur_ecc in eroi_list:
        tmp_df = cur_df[cur_df[ecc_bins] == cur_ecc]
        x = _df_column_to_torch(tmp_df, "local_sf")
        y = _df_column_to_torch(tmp_df, "betas")
        # set initial parameters
        slope, mode, sigma = _set_initial_params(initial_val)
        # select an optimizer
        optimizer = torch.optim.Adam([slope, mode, sigma], lr=lr)
        # set a loss function - mean squared error
        criterion = torch.nn.MSELoss()

        loss_history = []
        model_history = []
        for t in range(epoch):
            optimizer.zero_grad()
            loss = criterion(torch_log_norm_pdf(x, slope, mode, sigma), y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            model_history.append([slope.item(), mode.item(), sigma.item()])
            if (t + 1) % n_print == 0:
                print(f'Loss at epoch {str(t + 1)}: {str(round(loss.item(), 3))}')

        print(f'###Eccentricity bin {str(eroi_list.index(cur_ecc) + 1)} out of {len(eroi_list)} finished!###')
        print(
            f'Final parameters: slope {slope.item()}, mode {mode.item()}, sigma {sigma.item()}\n')
        model_history_df[cur_ecc] = pd.DataFrame(model_history, columns=['slope', 'mode', 'sigma'])
        loss_history_df[cur_ecc] = pd.DataFrame(loss_history, columns=['loss'])
    end = timer()
    elapsed_time = end - start
    print(f'###{subj}-{stim_class}-{varea} finished!###')
    print(f'Elapsed time: {np.round(end - start, 2)} sec')
    model_history_df = pd.concat(model_history_df).reset_index().rename(
        columns={'level_0': "ecc_bin", 'level_1': "epoch"})
    loss_history_df = pd.concat(loss_history_df).reset_index().rename(
        columns={'level_0': "ecc_bin", 'level_1': "epoch"})
    loss_history_df['lr'] = lr
    return model_history_df, loss_history_df, elapsed_time


def run_1D_model(df, sn_list, stim_class_list, varea_list, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):
    model_history_df = {}
    loss_history_df = {}
    for sn in sn_list:

        subj = utils.sub_number_to_string(sn)
        m_m_df = {}
        l_l_df = {}
        for stim_class in stim_class_list:
            m_df = {}
            l_df = {}
            for varea in varea_list:
                m_df[varea], l_df[varea], e_time = fit_1D_model(df, sn=sn, stim_class=stim_class,
                                                                varea=varea, ecc_bins=ecc_bins, n_print=n_print,
                                                                initial_val=initial_val, epoch=epoch, lr=lr)

            m_m_df[stim_class] = pd.concat(m_df)
            l_l_df[stim_class] = pd.concat(l_df)
        model_history_df[subj] = pd.concat(m_m_df)
        loss_history_df[subj] = pd.concat(l_l_df)
    col_replaced = {'level_0': 'subj', 'level_1': 'names', 'level_2': 'vroinames'}
    model_history_df = pd.concat(model_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)
    loss_history_df = pd.concat(loss_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)

    return model_history_df, loss_history_df

def sim_fit_1D_model(cur_df, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):

    eroi_list = utils.sort_a_df_column(cur_df[ecc_bins])

    # Initialize output df
    loss_history_df = {}
    model_history_df = {}

    # start fitting process
    start = timer()
    for cur_ecc in eroi_list:
        tmp_df = cur_df[cur_df[ecc_bins] == cur_ecc]
        x = _df_column_to_torch(tmp_df, "local_sf")
        y = _df_column_to_torch(tmp_df, "betas")
        # set initial parameters
        amp, mode, sigma = _set_initial_params(initial_val)
        # select an optimizer
        optimizer = torch.optim.Adam([amp, mode, sigma], lr=lr)
        # set a loss function - mean squared error
        criterion = torch.nn.MSELoss()

        loss_history = []
        model_history = []
        for t in range(epoch):
            optimizer.zero_grad()
            loss = criterion(torch_log_norm_pdf(x, amp, mode, sigma), y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            model_history.append([amp.item(), mode.item(), sigma.item()])
            if (t + 1) % n_print == 0:
                print(f'Loss at epoch {str(t + 1)}: {str(round(loss.item(), 3))}')

        print(f'###Eccentricity bin {str(eroi_list.index(cur_ecc) + 1)} out of {len(eroi_list)} finished!###')
        print(
            f'Final parameters: slope {amp.item()}, mode {mode.item()}, sigma {sigma.item()}\n')
        model_history_df[cur_ecc] = pd.DataFrame(model_history, columns=['amp', 'mode', 'sigma'])
        loss_history_df[cur_ecc] = pd.DataFrame(loss_history, columns=['loss'])
    end = timer()
    elapsed_time = end - start
    print(f'Elapsed time: {np.round(end - start, 2)} sec')
    model_history_df = pd.concat(model_history_df).reset_index().rename(
        columns={'level_0': "bins", 'level_1': "epoch"})
    loss_history_df = pd.concat(loss_history_df).reset_index().rename(
        columns={'level_0': "bins", 'level_1': "epoch"})
    loss_history_df['lr'] = lr
    return model_history_df, loss_history_df, elapsed_time

def bin_ecc(df, bin_list, to_bin='eccentricity', bin_labels=None):
    if bin_labels is None:
        bin_labels = [f'{str(a)}-{str(b)}deg' for a, b in zip(bin_list[:-1], bin_list[1:])]
    ecc_bin = pd.cut(df[to_bin], bins=bin_list, include_lowest=True, labels=bin_labels)
    return ecc_bin

def get_bin_labels(e1, e2, enum):

    if 'log' in enum:
        enum_only = enum[3:]
        bin_list = np.round(np.logspace(np.log2(float(e1)), np.log2(float(e2)), num=int(enum_only)+1, base=2), 2)
    else:
        bin_list = np.round(np.linspace(float(e1), float(e2), int(enum)+1), 2)
    bin_labels = [f'{str(a)}-{str(b)} deg' for a, b in zip(bin_list[:-1], bin_list[1:])]
    return bin_list, bin_labels

def summary_stat_for_ecc_bin(df, to_bin=["betas", "local_sf"], central_tendency="mode"):

    group = ['subj', 'ecc_bin', 'freq_lvl', 'names', 'vroinames']
    if central_tendency == "mode":
        c_df = df.groupby(group)[to_bin].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    else:
        c_df = df.groupby(group)[to_bin].agg(central_tendency).reset_index()
    # this should be fixed for cases where there are more than two central tendencies.
    return c_df


class LogGaussianTuningDataset:
    """Tranform dataframes to pivot style. x axis represents ecc_bin, y axis is freq_lvl."""
    def __init__(self, df):
        self.target = torch.tensor(df.pivot('ecc_bin', 'freq_lvl', 'betas').to_numpy())
        self.sf = torch.tensor(df.pivot('ecc_bin', 'freq_lvl', 'local_sf').to_numpy())


# numpy to torch function
def _cast_as_tensor(x):
    """ Change numpy vector to torch vector. The input x should be either a column of dataframe,
     a list, or numpy vector.You can also pass a torch vector but it will print out warnings."""
    if type(x) == pd.Series:
        x = x.values
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32)


def _cast_as_param(x, requires_grad=True):
    """ Change input x to parameter"""
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)

class LogGaussianTuningModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = _cast_as_param(np.random.random(1))
        self.mode = _cast_as_param(np.random.random(1)+0.5)
        self.sigma = _cast_as_param(np.random.random(1)+0.5)


    def forward(self, x):
        """the pdf of the log normal distribution, with a scale factor
        """
        # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
        # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
        # the preferred period, the inverse of the mode / the peak spatial frequency.
        pdf = self.slope * torch.exp(-(torch.log2(x) - torch.log2(self.mode)) ** 2 / (2 * self.sigma ** 2))
        return pdf


def fit_tuning_curves(my_model, my_dataset, learning_rate=1e-4, max_epoch=5000, print_every=100,
                      anomaly_detection=True, amsgrad=False, eps=1e-8):
    """Fit log normal Gaussian tuning curves.
    This function will allow you to run a for loop for N times set as max_epoch,
    and return the output of the training; loss history, model history."""
    torch.autograd.set_detect_anomaly(anomaly_detection)
    my_parameters = [p for p in my_model.parameters() if p.requires_grad]
    params_col = [name for name, param in my_model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(my_parameters, lr=learning_rate, amsgrad=amsgrad, eps=eps)
    loss_fn = torch.nn.MSELoss()
    loss_history = []
    model_history = []
    start = timer()
    param_values = [p.detach().numpy().item() for p in my_model.parameters() if p.requires_grad]
    print(f'**epoch no.{max_epoch}: Finished! final model params...\n {dict(zip(params_col, param_values))}\n')
    for t in range(max_epoch):
        optimizer.zero_grad()  # clear previous gradients
        pred = my_model.forward(x=my_dataset.sf)
        loss = loss_fn(pred, my_dataset.target)
        param_values = [p.detach().numpy().item() for p in my_model.parameters() if p.requires_grad]
        loss_history.append(loss.item())
        model_history.append(param_values)  # more than one item here
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients

        if (t + 1) % print_every == 0 or t == 0:
            content = f'**epoch no.{t} loss: {np.round(loss.item(), 3)} \n'
            print(content)

    elapsed_time = timer() - start
    print(f'**epoch no.{max_epoch}: Finished! final model params...\n {dict(zip(params_col, param_values))}\n')
    print(f'Elapsed time: {np.round(elapsed_time, 2)} sec \n')
    loss_history = pd.DataFrame(loss_history, columns=['loss']).reset_index().rename(columns={'index': 'epoch'})
    model_history = pd.DataFrame(model_history, columns=params_col).reset_index().rename(columns={'index': 'epoch'})
    return loss_history, model_history


def fit_tuning_curves_for_each_bin(bin_labels, df, learning_rate=1e-4, max_epoch=5000, print_every=100,
                                   anomaly_detection=True, amsgrad=False, eps=1e-8):
    loss_history = {}
    model_history = {}
    for bin in bin_labels:
        c_df = df.query('ecc_bin == @bin')
        my_dataset = LogGaussianTuningDataset(c_df)
        my_model = LogGaussianTuningModel()
        loss_history[bin], model_history[bin] = fit_tuning_curves(my_model, my_dataset, learning_rate, max_epoch, print_every,
                                                                  anomaly_detection, amsgrad, eps)
    loss_history = pd.concat(loss_history).reset_index().drop(columns='level_1').rename(columns={'level_0': 'ecc_bin'})
    model_history = pd.concat(model_history).reset_index().drop(columns='level_1').rename(columns={'level_0': 'ecc_bin'})
    return loss_history, model_history


def load_history_df_1D(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, df_dir):
    subj = utils.sub_number_to_string(sn, dataset=dset)
    f_name = f'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr_rate}_eph-{max_epoch}_{roi}_vs-pRFcenter_e{e1}-{e2}_nbin-{nbin}.h5'
    return pd.read_hdf(os.path.join(df_dir, f_name))

def load_history_1D_all_subj(sn_list, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, df_dir):
    df = pd.DataFrame({})
    for sn in sn_list:
        tmp = load_history_df_1D(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, df_dir)
        df = pd.concat((df, tmp), axis=0)
    return df

def load_binned_df_1D(sn, dset, stat, roi, e1, e2, nbin, df_dir):
    subj = utils.sub_number_to_string(sn, dataset=dset)
    f_name = f'binned_e{e1}-{e2}_nbin-{nbin}_{subj}_stim_voxel_info_df_vs-pRFcenter_{roi}_{stat}.csv'
    df = pd.read_csv(os.path.join(df_dir, f_name))
    if 'subj' not in df.columns.tolist():
        df['subj'] = subj
    return df

def load_binned_df_1D_all_subj(sn_list, dset, stat, roi, e1, e2, nbin, df_dir):
    df = pd.DataFrame({})
    for sn in sn_list:
        tmp = load_binned_df_1D(sn, dset, stat, roi, e1, e2, nbin, df_dir)
        df = pd.concat((df, tmp), axis=0)
    return df

def _merge_fitting_output_df_to_subj_df(model_history, binned_df, merge_on=["subj","vroinames", "ecc_bin", 'names']):
    max_epoch = model_history.epoch.max()
    model_history = model_history.query('epoch == @max_epoch')
    merged_df = binned_df.merge(model_history, on=merge_on)
    return merged_df

def load_and_merge_1D_df(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, input_dir, output_dir):
    model_history = load_history_df_1D(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, output_dir)
    binned_df = load_binned_df_1D(sn, dset, stat, roi, e1, e2, nbin, input_dir)
    return _merge_fitting_output_df_to_subj_df(model_history, binned_df, merge_on=['ecc_bin','names'])

def load_and_merge_1D_df_all_subj(sn_list, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, input_dir, output_dir):
    model_history = load_history_1D_all_subj(sn_list, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, output_dir)
    binned_df = load_binned_df_1D_all_subj(sn_list, dset, stat, roi, e1, e2, nbin, input_dir)
    return _merge_fitting_output_df_to_subj_df(model_history, binned_df, merge_on=['subj', 'vroinames', 'ecc_bin','names'])



def plot_datapoints(df, col='names', hue='ecc_bin', lgd_title='Eccentricity', height=5, subplot_right=0.9, sup_title=None,
                save_fig=False, save_path='/Volumes/server/Project/sfp_nsd/derivatives/figures/1D_results.png'):
    col_order = utils.sort_a_df_column(df[col])
    sns.set_context("notebook", font_scale=1.5)
    grid = sns.FacetGrid(df,
                         col='names',
                         hue=hue,
                         hue_order=df[hue].unique(),
                         palette=sns.color_palette("rocket", df[hue].nunique()),
                         col_wrap=4,
                         height=height,
                         legend_out=True,
                         sharex=True, sharey=True)
    g = grid.map(sns.scatterplot, 'local_sf', 'betas', edgecolor="gray")
    grid.set_axis_labels('Spatial Frequency', 'Beta')
    grid.fig.legend(title=lgd_title, labels=df[hue].unique())
    if sup_title is not None:
        grid.fig.suptitle(sup_title)
        grid.fig.subplots_adjust(right=subplot_right, top=0.86)
    else:
        grid.fig.subplots_adjust(right=subplot_right)
    plt.xscale('log')
    utils.save_fig(save_fig, save_path)
    return grid

def _get_x_and_y_prediction(min, max, fnl_param_df):
    x = np.linspace(min, max, 30)
    y = [np_log_norm_pdf(k, fnl_param_df['slope'].item(), fnl_param_df['mode'].item(), fnl_param_df['sigma'].item()) for k in x]
    return x, y

def plot_curves(df, fnl_param_df, col='names', save_fig=False, save_path='/Volumes/server/Project/sfp_nsd/derivatives/figures/figure.png'):
    subplot_list = df[col].unique()
    fig, axes = plt.subplots(1, len(subplot_list), figsize=(22, 5.5), dpi=300, sharex=True, sharey=True)
    ecc_list = df['ecc_bin'].unique()
    colors = mpl.cm.magma(np.linspace(0, 1, len(ecc_list)))

    for g in range(len(subplot_list)):
        for ecc in range(len(ecc_list)):
            tmp = df.query('names == @subplot_list[@g] & ecc_bin == @ecc_list[@ecc]')
            x = tmp['local_sf']
            y = tmp['betas']
            axes[g].scatter(x, y, s=28, color=colors[ecc,:], alpha=0.9, label=ecc_list[ecc], edgecolors='gray')
            tmp_history = fnl_param_df.query('names == @subplot_list[@g] & ecc_bin == @ecc_list[@ecc]')
            pred_x, pred_y = _get_x_and_y_prediction(x.min(), x.max(), tmp_history)
            axes[g].plot(pred_x, pred_y, color=colors[ecc,:], linewidth=3, path_effects=[pe.Stroke(linewidth=4, foreground='gray'), pe.Normal()])
            axes[g].set_title(subplot_list[g], fontsize=20)
            plt.xscale('log')
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
    axes[len(subplot_list)-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    model.control_fontsize(16, 25, 25)
    fig.supxlabel('Spatial Frequency', fontsize=25)
    fig.supylabel('Beta', fontsize=25)
    plt.tight_layout(w_pad=2)
    fig.subplots_adjust(left=.06, bottom=0.2)
    utils.save_fig(save_fig, save_path)


def plot_param_history(df,
                       to_label=None, label_order=None,
                       lgd_title=None, height=5,
                       save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/.png',
                       ci=68, n_boot=100, log_y=True):
    sns.set_context("notebook", font_scale=1.5)
    to_x = "epoch"
    to_y = "value"
    x_label = "Epoch"
    y_label = "Parameter value"
    n_labels = df[to_label].nunique()
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    pal = sns.color_palette("rocket", n_labels)
    grid = sns.FacetGrid(df,
                         hue=to_label,
                         hue_order=label_order,
                         col="names",
                         row='params',
                         height=height,
                         palette=pal,
                         legend_out=False,
                         sharex=True, sharey=False)
    g = grid.map(sns.lineplot, to_x, to_y, linewidth=2, ci=ci, n_boot=n_boot)
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_fig, save_path)
