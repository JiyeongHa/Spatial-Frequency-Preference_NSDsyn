import sys
sys.path.append('../../')
import os
import itertools
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import re
import itertools
import pandas as pd
import seaborn as sns
import matplotlib as mpl
#mpl.use('module://backend_interagg')
from matplotlib import pyplot as plt
import torch
import binning_eccen
import sfp_nsd_utils as utils

def log_norm_pdf(x, a, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = a * torch.exp(-(torch.log2(x)-torch.log2(mode))**2/(2*sigma**2))

    return pdf
def np_log_norm_pdf(x, a, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = a * np.exp(-(np.log2(x)-np.log2(mode))**2/(2*sigma**2))
    return pdf
def _track_loss_df(values,
                   create_loss_df=True, loss_df=None,
                   keys=["subj", "vroinames", "eccrois", "alpha", "n_epoch", "start_loss", "final_loss"]):
    if create_loss_df:
        loss_df = utils.create_empty_df(col_list=keys)
    else:
        if loss_df is None:
            raise Exception("Dataframe for saving loss log is not defined!")

    loss_df = loss_df.append(dict(zip(keys, values)), ignore_index=True)
    return loss_df

def pytorch_1D_model_fitting(input_df, subj_list=None,
                             vroi_list=None, initial_val=[1, 1, 1], epoch=5000, alpha=0.025,
                             save_output_df=False,
                             output_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                             output_df_name='1D_model_results.csv',
                             save_loss_df=False,
                             loss_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis/loss_track',
                             loss_df_name='1D_model_loss.csv'):

    if subj_list is not None:
        subj_list = [str(i).zfill(2) for i in subj_list]
        subj_list = [f"subj{s}" for s in subj_list]
    elif subj_list is None:
        subj_list = input_df.subj.unique()
    if vroi_list is None:
        vroi_list = utils.sort_a_df_column(input_df['vroinames'])

    # only leave subjs and vrois specified in the df
    input_df = input_df[input_df.vroinames.isin(vroi_list) & input_df.subj.isin(subj_list)]
    eccrois_list = utils.sort_a_df_column(input_df['eccrois'])

    # Create output df
    output_df = pd.DataFrame({})
    loss_df = utils.create_empty_df(["subj", "vroinames", "eccrois", "alpha", "n_epoch", "start_loss", "final_loss"])
    for sn in subj_list:
        output_single_df = pd.DataFrame({})
        for cur_roi, cur_ecc in itertools.product(vroi_list, eccrois_list):
            tmp_single_df = input_df.query('(subj == @sn) & (vroinames == @cur_roi) & (eccrois == @cur_ecc)')
            x = torch.from_numpy(tmp_single_df.local_sf.values) #.to(torch.float32)
            y = torch.from_numpy(tmp_single_df.avg_betas.values) #.to(torch.float32)
            #### Torch part ###
            # set initial parameters
            amp = torch.tensor([initial_val[0]], dtype=torch.float32, requires_grad=True)
            mode = torch.tensor([initial_val[1]], dtype=torch.float32, requires_grad=True)
            sigma = torch.tensor([initial_val[2]], dtype=torch.float32, requires_grad=True)
            # select an optimizer
            optimizer = torch.optim.Adam([amp, mode, sigma], lr=alpha)
            # set a loss function - mean squared error
            criterion = torch.nn.MSELoss()
            losses = torch.empty(epoch)
            for epoch_count in range(epoch):
                optimizer.zero_grad()
                loss = criterion(log_norm_pdf(x, amp, mode, sigma), y)
                loss.backward()
                optimizer.step()
                losses[epoch_count] = loss.item()
                if epoch_count % 500 == 0 or epoch_count == (epoch-1):
                    print('epoch ' + str(epoch_count) + ' loss ' + str(round(loss.item(), 3)))
                if epoch_count == epoch-1:
                    print('*** ' + sn + ' ' + cur_roi + ' ecc ' + str(cur_ecc) + ' finished ***')
                    print(f'amplitude {round(amp.item(),2)}, mode {round(mode.item(),2)}, sigma {round(sigma.item())}\n')
            values = [sn, cur_roi, cur_ecc, alpha, epoch, losses[0].detach().numpy(), losses[-1].detach().numpy()]
            loss_df = _track_loss_df(values=values, create_loss_df=False, loss_df=loss_df)
            tmp_output_df = pd.DataFrame({'subj': sn,
                             'vroinames': cur_roi,
                             'eccrois': cur_ecc,
                             'amp': amp.detach().numpy(),
                             'mode': mode.detach().numpy(),
                             'sigma': sigma.detach().numpy()})
            output_single_df = pd.concat([output_single_df, tmp_output_df], ignore_index=True, axis=0)
            output_df = pd.concat([output_df, tmp_output_df], ignore_index=True, axis=0)
        if save_output_df:
            if not os.path.exists(output_df_dir):
                os.makedirs(output_df_dir)
            output_path = os.path.join(output_df_dir, f'{sn}_lr-{alpha}_ep-{epoch}_{output_df_name}')
            output_single_df.to_csv(output_path, index=False)
        if save_loss_df:
            if not os.path.exists(loss_df_dir):
                os.makedirs(loss_df_dir)
            loss_output_path = os.path.join(loss_df_dir, f'{sn}_lr-{alpha}_ep-{epoch}_{loss_df_name}')
            loss_df.to_csv(loss_output_path, index=False)
    return output_df, loss_df

def _merge_fitting_output_df_to_subj_df(fitting_df, subj_df, merge_on=["subj","vroinames", "eccrois"]):
    merged_df = subj_df.merge(fitting_df, on=merge_on)
    return merged_df
def __get_y_pdf(row):
    y_pdf = np_log_norm_pdf(row['local_sf'], row['amp'], row['mode'], row['sigma'])
    return y_pdf


def _merge_pdf_values(fitting_df, subj_df=None, merge_on_cols=["subj", "vroinames", "eccrois"], merge_output_df=True):
    if merge_output_df:
        merge_df = _merge_fitting_output_df_to_subj_df(fitting_df, subj_df, merge_on=merge_on_cols)
    else:
        merge_df = fitting_df
    merge_df['y_lg_pdf'] = merge_df.apply(__get_y_pdf, axis=1)
    return merge_df


def beta_vs_sf_scatterplot(subj, merged_df, to_subplot="vroinames", n_sp_low=2,
                           legend_out=True, to_label="eccrois",
                           dp_to_x_axis='local_sf', dp_to_y_axis='avg_betas', plot_pdf=True,
                           ln_y_axis="y_lg_pdf", x_axis_label="Spatial Frequency", y_axis_label="Beta",
                           legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                           save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                           save_file_name='.png'):
    sn = utils.sub_number_to_string(subj)

    cur_df = merged_df.query('subj == @sn')
    col_order = utils.sort_a_df_column(cur_df[to_subplot])
    grid = sns.FacetGrid(cur_df,
                         col=to_subplot,
                         col_order=col_order,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         col_wrap=n_sp_low,
                         legend_out=legend_out,
                         xlim=[10 ** -1, 10 ** 2],
                         sharex=True, sharey=True)
    g = grid.map(sns.scatterplot, dp_to_x_axis, dp_to_y_axis)
    if plot_pdf:
        grid.map(sns.lineplot, dp_to_x_axis, ln_y_axis, linewidth=2)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1), labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    plt.xscale('log')
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    grid.fig.suptitle(f'{sn}', fontsize=18, fontweight="bold")
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

    return grid


def plot_beta_all_subj(subj_to_run, merged_df, to_subplot="vroinames", n_sp_low=2, legend_out=True, to_label="eccrois",
                       dp_to_x_axis='local_sf', dp_to_y_axis='avg_betas', plot_pdf=True,
                       ln_y_axis="y_lg_pdf", x_axis_label="Spatial Frequency", y_axis_label="Beta",
                       legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                       save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                       save_file_name='.png'):
    for sn in subj_to_run:
        grid = beta_vs_sf_scatterplot(subj=sn, merged_df=merged_df, to_subplot=to_subplot, n_sp_low=n_sp_low,
                                      legend_out=legend_out, to_label=to_label, dp_to_x_axis=dp_to_x_axis,
                                      dp_to_y_axis=dp_to_y_axis, plot_pdf=plot_pdf, ln_y_axis=ln_y_axis,
                                      x_axis_label=x_axis_label, y_axis_label=y_axis_label, legend_title=legend_title,
                                      labels=labels, save_fig=save_fig, save_dir=save_dir,
                                      save_file_name=save_file_name)
    return grid


def merge_and_plot(subj_to_run, fitting_df, subj_df, merge_on_cols=["subj", "vroinames", "eccrois"],
                   to_subplot="vroinames", n_sp_low=2, legend_out=True, to_label="eccrois",
                   dp_to_x_axis="local_sf", dp_to_y_axis='avg_betas', plot_pdf=True,
                   ln_y_axis="y_lg_pdf", x_axis_label="Spatial Frequency", y_axis_label="Beta",
                   legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                   save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                   save_file_name='.png'):
    merged_df = _merge_pdf_values(fitting_df=fitting_df, subj_df=subj_df, merge_on_cols=merge_on_cols,
                                  merge_output_df=True)
    grid = plot_beta_all_subj(subj_to_run=subj_to_run, merged_df=merged_df, to_subplot=to_subplot, n_sp_low=n_sp_low,
                              legend_out=legend_out, to_label=to_label, dp_to_x_axis=dp_to_x_axis,
                              dp_to_y_axis=dp_to_y_axis, plot_pdf=plot_pdf,
                              ln_y_axis=ln_y_axis, x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                              legend_title=legend_title, labels=labels,
                              save_fig=save_fig, save_dir=save_dir,
                              save_file_name=save_file_name)

    return merged_df, grid
