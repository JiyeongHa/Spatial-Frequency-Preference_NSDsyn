import sys
import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
import pandas as pd
from sfp_nsdsyn.one_dimensional_model import np_log_norm_pdf
from sfp_nsdsyn.one_dimensional_model import _get_x_and_y_prediction
import matplotlib as mpl
from sfp_nsdsyn.visualization import plot_2D_model_results as vis2D


def _get_y_pdf(row):
    y_pdf = np_log_norm_pdf(row['local_sf'], row['slope'], row['mode'], row['sigma'])
    return y_pdf


def merge_pdf_values(subj_df, model_df, on=["sub", "vroinames", "ecc_bin"]):
    merge_df = subj_df.merge(model_df, on=on)
    merge_df['pdf'] = merge_df.apply(_get_y_pdf, axis=1)
    return merge_df


def beta_vs_sf_scatterplot(df, pdf=None, hue="ecc_bin", hue_order=None, lgd_title='Eccentricity',
                           col='names', suptitle=None, height=5,
                           save_path=None, **kwargs):
    col_order = utils.sort_a_df_column(df[col])
    grid = sns.FacetGrid(df,
                         col=col,
                         col_order=col_order,
                         hue=hue,
                         height=height,
                         hue_order=hue_order,
                         sharex=True, sharey=True, **kwargs)
    g = grid.map(sns.scatterplot, 'local_sf', 'betas')
    if pdf is not None:
        grid.map(sns.lineplot, 'local_sf', pdf)
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    grid.fig.suptitle(suptitle, fontweight="bold")
    grid.set(xscale='log')
    utils.save_fig(save_path)

    return grid
def _get_middle_ecc(row):
    label = row['ecc_bin']
    e1 = float(label[0:3])
    e2 = float(label[4:6])
    return np.round((e1+e2)/2, 2)

def plot_curves(df, fnl_param_df, title, save_path=None):
    subplot_list = df['names'].unique()
    fig, axes = plt.subplots(1, len(subplot_list), figsize=(22, 8), dpi=400, sharex=True, sharey=True)
    ecc_list = df['ecc_bin'].unique()
    colors = mpl.cm.magma(np.linspace(0, 1, len(ecc_list)))

    for g in range(len(subplot_list)):
        for ecc in range(len(ecc_list)):
            tmp = df[df.names == subplot_list[g]]
            tmp = tmp[tmp.ecc_bin == ecc_list[ecc]]
            x = tmp['local_sf']
            y = tmp['betas']
            tmp_history = fnl_param_df[fnl_param_df.names == subplot_list[g]]
            tmp_history = tmp_history[tmp_history.ecc_bin == ecc_list[ecc]]
            pred_x, pred_y = _get_x_and_y_prediction(x.min(), x.max(), tmp_history)
            axes[g].plot(pred_x, pred_y, color=colors[ecc,:], linewidth=3, path_effects=[pe.Stroke(linewidth=4, foreground='gray'), pe.Normal()])
            axes[g].scatter(x, y, s=160, color=colors[ecc,:], alpha=0.9, label=ecc_list[ecc], edgecolors='gray')
            axes[g].set_title(subplot_list[g], fontsize=20)
            vis2D.control_fontsize(25, 30, 40)
            plt.xscale('log')
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        axes[g].tick_params(axis='both', labelsize=22)
    axes[len(subplot_list)-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.supxlabel('Spatial Frequency', fontsize=25)
    fig.supylabel('Beta', fontsize=25)
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(w_pad=2)
    fig.subplots_adjust(left=.08, bottom=0.13)
    utils.save_fig(save_path)

def preferred_period(df, hue="names", hue_order=None, lgd_title='Stimulus Class',
                           col=None, col_wrap=None, suptitle=None, height=5,
                           save_path=None):
    sns.set_context("notebook", font_scale=1.5)
    new_df = df.copy()
    new_df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df['pp'] = 1/new_df['mode']
    grid = sns.FacetGrid(new_df,
                         col=col,
                         col_wrap=col_wrap,
                         hue=hue,
                         height=height,
                         hue_order=hue_order,
                         sharex=True, sharey=True)
    g = grid.map(sns.lineplot, 'ecc', 'pp', marker='o', ci=68, err_style='bars')
    if lgd_title is not None:
        g.add_legend(title=lgd_title)
    grid.set_axis_labels('Eccentricity', 'Preferred period')
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    grid.fig.suptitle(suptitle, fontweight="bold")
    utils.save_fig(save_path)
    return grid

def plot_beta_all_subj(subj_to_run, merged_df, to_subplot="vroinames", n_sp_low=2, legend_out=True, to_label="eccrois",
                       dp_to_x_axis='local_sf', dp_to_y_axis='avg_betas', plot_pdf=True,
                       ln_y_axis="y_lg_pdf", x_axis_label="Spatial Frequency", y_axis_label="Beta",
                       legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                       save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                       save_file_name='.png'):
    for sn in subj_to_run:
        grid = beta_vs_sf_scatterplot()
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


def plot_dots(df, y, col, hue, lgd_title, height=5, save_path=None):
    grid = sns.FacetGrid(df,
                         col=col,
                         col_order=df[col].unique(),
                         hue=hue,
                         hue_order=df[hue].unique(),
                         height=5,
                         palette=sns.color_palette("rocket", df[hue].nunique()),
                         sharex=True, sharey=True)
    g = grid.map(sns.lineplot, 'local_sf', y, marker='o', err_style='bars', linestyle='')
    g = grid.map(sns.lineplot, 'local_sf', 'normed_pred', marker='', err_style='bars', linestyle='dotted')
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    grid.fig.legend(title=lgd_title, labels=df[hue].unique())
    plt.xscale('log')


