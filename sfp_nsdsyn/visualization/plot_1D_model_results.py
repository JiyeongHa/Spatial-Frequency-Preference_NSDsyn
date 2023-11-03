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
import matplotlib.patheffects as pe
from sfp_nsdsyn.visualization import plot_2D_model_results as vis2D


def _get_y_pdf(row):
    y_pdf = np_log_norm_pdf(row['local_sf'], row['slope'], row['mode'], row['sigma'])
    return y_pdf



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


def merge_pdf_values(bin_df, model_df, on=["sub", "vroinames", "ecc_bin"]):
    merge_df = bin_df.merge(model_df, on=on)
    merge_df['pdf'] = merge_df.apply(_get_y_pdf, axis=1)
    return merge_df



def plot_curves_sns(df, x, y, hue, hue_order=None,
                    height=5, col=None, col_wrap=None, to_logscale=True, **kwargs):
    sns.set_context("notebook", font_scale=2.5)
    grid = sns.FacetGrid(df,
                         col=col,
                         col_wrap=col_wrap,
                         height=height,
                         hue=hue,
                         hue_order=hue_order,
                         aspect=1,
                         palette=sns.color_palette("tab10"),
                         sharex=True, sharey=False, **kwargs)
    g = grid.map(sns.lineplot, x, y, linestyle='-', marker='o',
                 estimator=np.mean, ci='sd')
    if to_logscale:
        grid.set(xscale='log')


def plot_sf_curves(df, params_df, x, y, col, hue, height=13, lgd_title=None, save_path=None):
    rc = {'axes.labelpad': 20,
          'axes.linewidth': 3,
          'axes.titlepad': 40,
          'axes.titleweight': "bold",
          'xtick.major.pad': 10,
          'ytick.major.pad': 10,
          'xtick.major.width': 3,
          'xtick.minor.width': 3,
          'ytick.major.width': 3,
          'xtick.major.size': 10,
          'xtick.minor.size': 6,
          'ytick.major.size': 10,
          'grid.linewidth': 3,
          'font.family': 'Helvetica',
          'lines.linewidth': 2}
    large_fontsize = 50
    utils.set_fontsize(30, 35, large_fontsize)
    utils.set_rcParams(rc)
    subplot_list = df[col].unique()
    hue_list = np.flip(df[hue].unique())
    fig, axes = plt.subplots(1, len(subplot_list),
                             figsize=(height*1.9, height),
                             sharex=True, sharey=False)

    colors = utils.get_continuous_colors(len(hue_list)+1, '#3f0377')
    colors = colors[1:]
    for g in range(len(subplot_list)):
        subplot_tmp = df[df[col] == subplot_list[g]]
        for c in range(len(hue_list)):
            tmp = subplot_tmp[subplot_tmp[hue] == hue_list[c]]
            xx = tmp[x]
            yy = tmp[y]
            tmp_history = params_df[params_df[col] == subplot_list[g]]
            tmp_history = tmp_history[tmp_history[hue] == hue_list[c]]
            pred_x, pred_y = _get_x_and_y_prediction(xx.min(), xx.max(), tmp_history)
            axes[g].set_title(subplot_list[g])
            axes[g].plot(pred_x, pred_y,
                         color=colors[c],
                         linewidth=5,
                         path_effects=[pe.Stroke(linewidth=5.6, foreground='black'),
                                       pe.Normal()],
                         zorder=0)
            axes[g].scatter(xx, yy,
                            s=200,
                            color=colors[c],
                            alpha=0.95,
                            label=hue_list[c],
                            edgecolors='black',
                            zorder=10)
            plt.xscale('log')
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        if len(axes[g].get_yticks()) > 4:
            axes[g].set_yticks(axes[g].get_yticks()[::2])
        axes[g].tick_params(axis='both')
    axes[len(subplot_list)-1].legend(title=lgd_title, loc='center left', bbox_to_anchor=(1, 0.7), frameon=False)
    fig.supxlabel('Spatial Frequency', fontsize=large_fontsize)
    fig.supylabel('Beta', fontsize=large_fontsize)
    fig.subplots_adjust(wspace=0.4, left=.11, bottom=0.14)
    utils.save_fig(save_path)

def _get_middle_ecc(row):

    label = row['ecc_bin']
    bin_e1 = float(label.split('-')[0])
    bin_e2 = float(label.split('-')[1][:-4])
    return np.round((bin_e1+bin_e2)/2, 2)

def _add_jitter(df, to_jitter, subset, jitter_scale=0.01):
    rand_vals = (0.5-np.random.random(df[subset].nunique()))*jitter_scale
    jitters = dict(zip(df[subset].unique(), rand_vals))
    new_col = df.apply(lambda row: row[to_jitter] + jitters[row[subset]], axis=1)
    return new_col

def plot_preferred_period(df, precision_col=None,
                          hue="names", hue_order=None, lgd_title='Stimulus Class',
                          col=None, col_wrap=None, suptitle=None, height=5,
                          save_path=None):
    rc = {'axes.labelpad': 25}
    sns.set_context("notebook", font_scale=2, rc=rc)
    new_df = df.copy()
    new_df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', hue, jitter_scale=0.08)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', 'ecc_bin', jitter_scale=0.05)
    new_df['pp'] = 1 / new_df['mode']
    if precision_col is not None:
        new_df['value_and_weight'] = [v + w * 1j for v, w in zip(new_df['pp'], new_df[precision_col])]
    grid = sns.FacetGrid(new_df,
                         col=col,
                         col_wrap=col_wrap,
                         height=height,
                         aspect=1.2,
                         palette=sns.color_palette("tab10"),
                         sharex=True, sharey=True)
    g = grid.map(sns.lineplot, 'ecc', 'value_and_weight', hue, hue_order=hue_order, marker='o',
                 lw=4, markersize=20, estimator=utils.weighted_mean, ci=68,
                 err_style='bars', err_kws={'elinewidth': 4})
    grid.set(xticks=[0,1,2,3,4], yticks=[0, 0.5, 1])
    if lgd_title is not None:
        g.add_legend(title=lgd_title, bbox_to_anchor=(1.02, 0.7))
    grid.set_axis_labels('Eccentricity', 'Preferred period')
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    grid.fig.suptitle(suptitle, fontweight="bold")
    # set transparency
    plt.setp(grid.ax.collections, alpha=.85)  # for the markers
    #plt.setp(grid.ax.lines, alpha=.8)  # for the lines
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


def plot_dots(df, y, col, hue, lgd_title, height=5, save_path=None):
    grid = sns.FacetGrid(df,
                         col=col,
                         col_order=df[col].unique(),
                         hue=hue,
                         hue_order=df[hue].unique(),
                         height=height,
                         palette=sns.color_palette("rocket", df[hue].nunique()),
                         sharex=True, sharey=True)
    g = grid.map(sns.lineplot, 'local_sf', y, marker='o', err_style='bars', linestyle='')
    g = grid.map(sns.lineplot, 'local_sf', 'normed_pred', marker='', err_style='bars', linestyle='dotted')
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    grid.fig.legend(title=lgd_title, labels=df[hue].unique())
    plt.xscale('log')


