import sys
sys.path.append('../../')
import os
import seaborn as sns
from matplotlib import pyplot as plt
import sfp_nsd_utils as utils
import pandas as pd
import numpy as np
from first_level_analysis import np_log_norm_pdf

def _merge_fitting_output_df_to_subj_df(model_df, subj_df, merge_on=["subj","vroinames", "eccrois"]):
    merged_df = subj_df.merge(model_df, on=merge_on)
    return merged_df
def _get_y_pdf(row):
    y_pdf = np_log_norm_pdf(row['local_sf'], row['amp'], row['mode'], row['sigma'])
    return y_pdf


def merge_pdf_values(model_df, subj_df=None, merge_on_cols=["subj", "vroinames", "eccrois"], merge_output_df=True):
    if merge_output_df:
        merge_df = _merge_fitting_output_df_to_subj_df(model_df, subj_df, merge_on=merge_on_cols)
    else:
        merge_df = model_df
    merge_df['y_lg_pdf'] = merge_df.apply(_get_y_pdf, axis=1)
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
                         hue_order=labels,
                         palette=sns.color_palette("rocket"),
                         col_wrap=n_sp_low,
                         legend_out=legend_out,
                         xlim=[10 ** -1, 10 ** 2],
                         sharex=True, sharey=True)
    g = grid.map(sns.scatterplot, dp_to_x_axis, dp_to_y_axis)
    if plot_pdf:
        grid.map(sns.lineplot, dp_to_x_axis, ln_y_axis, linewidth=2)
    grid.set_axis_labels(x_axis_label, y_axis_label)
    grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 0.9), labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    plt.xscale('log')
    grid.fig.subplots_adjust(top=0.8, right=0.82)  # adjust the Figure in rp
    grid.fig.suptitle(f'{sn}', fontsize=18, fontweight="bold")
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


def plot_parameter_mean(output_df, subj_to_run=None, to_subplot="vroinames", n_sp_low=4,
                           legend_out=True, to_label="eccrois",
                           dp_to_x_axis='params', dp_to_y_axis='value',
                           x_axis_label="Parameters", y_axis_label="Value",
                           legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                           save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                           save_file_name='.png'):
    #new_output_df = output_df.query('eccrois == @cur_ecc')
    new_output_df = pd.melt(output_df, id_vars=['subj', 'vroinames', 'eccrois'], value_vars=['slope', 'mode', 'sigma'], ignore_index=True).rename(columns={'variable': 'params'})
    col_order = utils.sort_a_df_column(new_output_df[to_subplot])
    grid = sns.FacetGrid(new_output_df,
                         col=to_subplot, row='eccrois',
                         col_order=col_order,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         legend_out=legend_out,
                         sharex=True, sharey=True)
    grid.map(sns.lineplot, dp_to_x_axis, dp_to_y_axis, ci=68, marker='o', linestyle='', err_style='bars', hue_order=utils.sort_a_df_column(output_df[to_label]))
    grid.set_axis_labels(x_axis_label, y_axis_label)
    lgd = grid.fig.legend(title=legend_title, bbox_to_anchor=(1.05, 0.8), loc="upper left", labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #for subplot_title, ax in grid.axes_dict.items():
    #    ax.set_title(f"{subplot_title.title()}")
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    n_subj = len(output_df['subj'].unique())
    grid.fig.suptitle(f'N = {n_subj}', fontsize=18, fontweight="bold")
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, 'ecc_seperately_all_subj')
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()

    return grid


def plot_preferred_period(df, labels, to_subplot="vroinames", n_rows=4, to_label="names",
                          dp_to_x_axis='eccentricity', dp_to_y_axis='mode',
                          x_axis_label="Eccentricity", y_axis_label="Preferred period",
                          legend_title="Stimulus class", title=None, legend=True,
                          save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                          save_file_name='.png', ci=68, estimator=None):

    df['preferred_period'] = 1 / df[dp_to_y_axis]
    col_order = utils.sort_a_df_column(df[to_subplot])

    grid = sns.FacetGrid(df,
                         col=to_subplot,
                         col_order=col_order,
                         hue=to_label,
                         col_wrap=n_rows,
                         hue_order=labels,
                         palette=sns.color_palette("Set1"),
                         legend_out=True,
                         sharex=True, sharey=False)
    grid.map(sns.lineplot, dp_to_x_axis, 'preferred_period', estimator=estimator, ci=ci, marker='o',linestyle='', err_style='bars')
    grid.set(xticks=[0, 1, 2, 3, 4])
    grid.set_axis_labels(x_axis_label, y_axis_label)
    if legend == True:
        grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 0.9), labels=labels, fontsize=15)
    # Put the legend out of the figure
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    grid.fig.subplots_adjust(top=0.8, right=0.82)  # adjust the Figure in rp
    grid.fig.suptitle(f'{title}', fontsize=18, fontweight="bold")
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{save_file_name}')
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
