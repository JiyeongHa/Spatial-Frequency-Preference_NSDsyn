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

mpl.rcParams.update(mpl.rcParamsDefault)
rc = {'text.color': 'black',
      'axes.labelcolor': 'black',
      'xtick.color': 'black',
      'ytick.color': 'black',
      'axes.edgecolor': 'black',
      'font.family': 'Helvetica',
      'axes.linewidth': 1,
      'axes.labelpad': 6,
      'xtick.major.pad': 10,
      'xtick.major.width': 1,
      'ytick.major.width': 1,
      'lines.linewidth': 1,
      'font.size': 12,
      'axes.titlesize': 12,
      'axes.labelsize': 12,
      'xtick.labelsize': 12,
      'ytick.labelsize': 12,
      'legend.title_fontsize': 11,
      'legend.fontsize': 11,
      'figure.titlesize': 15,
      'figure.dpi': 72 * 3,
      'savefig.dpi': 72 * 4
      }
mpl.rcParams.update(rc)

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

def plot_sf_curves(df, params_df, x, y, hue, col, baseline=None,
                   hue_order=None, col_order=None, suptitle=None,
                   lgd_title=None, save_path=None, palette=None):
    rc.update({'axes.linewidth': 1.2,
          'xtick.major.width':1.2,
          'ytick.major.width':1.2,
          'xtick.minor.width':1,
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'xtick.minor.size': 3.5,
          'axes.labelpad': 8,
          'axes.titlepad': 15,
          'axes.titleweight': 'bold',
          'font.family': 'Helvetica',
          'axes.edgecolor': 'black',
          'figure.dpi': 72*2,
          'savefig.dpi': 72*4})
    utils.set_rcParams(rc)
    utils.set_fontsize(11, 11, 15)
    sns.set_theme("notebook", style='ticks', rc=rc)

    if hue_order is None:
        hue_order = df[hue].unique()
    if col_order is None:
        col_order = df[col].unique()
    fig, axes = plt.subplots(1, len(col_order),
                             figsize=(7, 7/1.9),
                             sharex=True, sharey=False)
    if palette is None:
        colors = utils.get_continuous_colors(len(hue_order)+1, '#3f0377')
        colors = colors[1:][::-1]
    else:
        colors = palette
    for i, g in enumerate(range(len(col_order))):
        subplot_tmp = df[df[col] == col_order[g]]
        cur_color = colors[i]
        for c, ls, fc, in zip(range(len(hue_order)), ['--','-'], ['w', cur_color]):
            tmp = subplot_tmp[subplot_tmp[hue] == hue_order[c]]
            xx = tmp[x]
            yy = tmp[y]
            tmp_history = params_df[params_df[col] == col_order[g]]
            tmp_history = tmp_history[tmp_history[hue] == hue_order[c]]
            pred_x, pred_y = _get_x_and_y_prediction(xx.min(), xx.max(),
                                                     tmp_history['slope'].item(),
                                                     tmp_history['mode'].item(),
                                                     tmp_history['sigma'].item(), n_points=1000)
            #np.min(pred_y)

            axes[g].set_title(col_order[g], fontsize=15)
            axes[g].plot(pred_x, pred_y,
                         color=cur_color,
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0)
            axes[g].scatter(xx, yy,
                            s=34,
                            facecolor=fc, #colors[c],
                            alpha=0.95,
                            label=hue_order[c],
                            edgecolor=cur_color, linewidth=1.5,
                            zorder=10)

        if baseline is not None:
            baseline_example_df = baseline[baseline[col] == col_order[g]]
            yy = np.mean(baseline_example_df[y])
            axes[g].axhline([yy], color='grey', linestyle='--', linewidth=1, zorder=20)
        plt.xscale('log')
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        if len(axes[g].get_yticks()) > 4:
            axes[g].set_yticks(axes[g].get_yticks()[::2])
        axes[g].tick_params(axis='both')
    axes[len(col_order)-1].legend(title=lgd_title, loc='center left', bbox_to_anchor=(0.9, 0.9), frameon=False, fontsize=13)
    leg = axes[len(col_order)-1].get_legend()
    leg.legendHandles[0].set_edgecolor('black')
    leg.legendHandles[1].set_color('black')
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")
    fig.supxlabel('Local spatial frequency (cpd)')
    fig.supylabel('Response\n(% BOLD signal change)', ha='center')
    fig.subplots_adjust(wspace=0.5, left=0.1, bottom=0.17)


    utils.save_fig(save_path)
    return fig, axes


def plot_sf_curves_with_broderick(nsd_subj, nsd_subj_df, nsd_tuning_df, nsd_bins_to_plot,
                                  broderick_subj, broderick_subj_df, broderick_tuning_df, broderick_bins_to_plot,
                                  pal, markersize=20,
                                  width=7, height=2.5, save_path=None):

    rc.update({'xtick.major.pad': 3,
               'xtick.labelsize': 9,
               'axes.titlepad': 10,
               'legend.title_fontsize': 10,
               'legend.fontsize': 10,
               })
    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 4, figsize=(width, height),
                             sharex=True, sharey=True)
    for cur_bin, ls, fc in zip(broderick_bins_to_plot, ['--', '-'], ['w', 'gray']):
        tmp_subj_df = broderick_subj_df.query('sub == @broderick_subj & ecc_bin == @cur_bin & vroinames == "V1"')
        tmp_tuning_df = broderick_tuning_df.query('sub == @broderick_subj & ecc_bin == @cur_bin & vroinames == "V1"')
        tmp_subj_df['betas'] = tmp_subj_df['betas'] / tmp_subj_df['betas'].max()
        pred_x, pred_y = _get_x_and_y_prediction(tmp_subj_df['local_sf'].min() * 0.8,
                                                 tmp_subj_df['local_sf'].max() * 1.4,
                                                 tmp_tuning_df['slope'].item(),
                                                 tmp_tuning_df['mode'].item(),
                                                 tmp_tuning_df['sigma'].item())
        pred_y = pred_y / np.max(pred_y)
        axes[0].plot(pred_x, pred_y,
                     color='gray',
                     linestyle=ls,
                     linewidth=2,
                     path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                   pe.Normal()],
                     zorder=0, clip_on=False)
        axes[0].scatter(tmp_subj_df['local_sf'], tmp_subj_df['betas'],
                        s=markersize,
                        facecolor=fc,  # colors[c],

                        alpha=0.95,
                        label=cur_bin,
                        edgecolor='gray', linewidth=1.3,
                        zorder=10, clip_on=False)
        axes[0].set_title('Broderick et al.\n(2022)')

    for i, nsd_roi in enumerate(['V1', 'V2', 'V3']):
        for cur_bin, ls, fc in zip(nsd_bins_to_plot, ['--', '-'], ['w', pal[i]]):
            min_val = nsd_subj_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].min()
            max_val = nsd_subj_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].max()
            tmp_subj_df = nsd_subj_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == @nsd_roi')
            tmp_tuning_df = nsd_tuning_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == @nsd_roi')
            tmp_subj_df['betas'] = tmp_subj_df['betas'] / tmp_subj_df['betas'].max()
            pred_x, pred_y = _get_x_and_y_prediction(min_val * 0.7,
                                                     max_val * 1.2,
                                                     tmp_tuning_df['slope'].item(),
                                                     tmp_tuning_df['mode'].item(),
                                                     tmp_tuning_df['sigma'].item())
            pred_y = pred_y / np.max(pred_y)
            axes[i+1].plot(pred_x, pred_y,
                         color=pal[i],
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0, clip_on=False)
            axes[i+1].scatter(tmp_subj_df['local_sf'], tmp_subj_df['betas'],
                            s=markersize,
                            facecolor=fc,
                            alpha=0.95,
                            label=cur_bin,
                            edgecolor=pal[i], linewidth=1.3,
                            zorder=10, clip_on=False)
            axes[i+1].set_title(f'NSD {nsd_roi}')

    for g in range(len(axes)):
        axes[g].set_xscale('log')
        axes[g].set(ylim=[0, 1.05], yticks=[0, 0.5, 1])
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        axes[g].tick_params(axis='both')
        axes[g].legend(title=None, loc=(-0.1, 0.00), frameon=False,handletextpad=0.08)


    #axes[-1].legend(title='Eccentricity band', bbox_to_anchor=(1, 0.85), frameon=False)
    #leg = axes[-1].get_legend()
    #leg.legendHandles[0].set_edgecolor('black')
    #leg.legendHandles[1].set_color('black')

    fig.supxlabel('Local spatial frequency (cpd)')
    fig.supylabel('BOLD response\n(Normalized amplitude)', ha='center')
    fig.subplots_adjust(wspace=0.3, left=0.1, bottom=0.2)
    utils.save_fig(save_path)
    return fig, axes

def plot_sf_curves_only_for_V1(df, params_df, x, y, hue,
                               hue_order=None, col_title=None, suptitle=None,
                               lgd_title=None, save_path=None, palette=None):
    rc.update({'axes.linewidth': 1.2,
          'xtick.major.width':1.2,
          'ytick.major.width':1.2,
          'xtick.minor.width':1,
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'xtick.minor.size': 3.5,
          'axes.labelpad': 8,
          'axes.titlepad': 15,
          'axes.titleweight': 'bold',
          'font.family': 'Helvetica',
          'axes.edgecolor': 'black',
          'figure.dpi': 72*2,
          'savefig.dpi': 72*4})
    utils.set_rcParams(rc)
    utils.set_fontsize(11, 11, 15)
    sns.set_theme("notebook", style='ticks', rc=rc)

    if hue_order is None:
        hue_order = df[hue].unique()
    fig, ax = plt.subplots(1, 1,
                             figsize=(2.5, 7/1.9),
                             sharex=True, sharey=False)
    if palette is None:
        colors = utils.get_continuous_colors(len(hue_order)+1, '#3f0377')
        colors = colors[1:][::-1]
    else:
        cur_color = palette
        subplot_tmp = df
        for c, ls, fc, in zip(range(len(hue_order)), ['--','-'], ['w', cur_color]):
            tmp = subplot_tmp[subplot_tmp[hue] == hue_order[c]]
            xx = tmp[x]
            yy = tmp[y]
            tmp_history = params_df
            tmp_history = tmp_history[tmp_history[hue] == hue_order[c]]
            pred_x, pred_y = _get_x_and_y_prediction(xx.min(), xx.max(),
                                                     tmp_history['slope'].item(),
                                                     tmp_history['mode'].item(),
                                                     tmp_history['sigma'].item(), n_points=1000)
            ax.plot(pred_x, pred_y,
                         color=cur_color,
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0)
            ax.scatter(xx, yy,
                            s=34,
                            facecolor=fc, #colors[c],
                            alpha=0.95,
                            label=hue_order[c],
                            edgecolor=cur_color, linewidth=1.5,
                            zorder=10)
        plt.xscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(ax.get_yticks()) > 4:
            ax.set_yticks(ax.get_yticks()[::2])
        ax.tick_params(axis='both')
    ax.set_title(col_title, fontsize=15)
    ax.legend(title=lgd_title, loc='center left', bbox_to_anchor=(0.9, 0.9), frameon=False, fontsize=13)
    leg = ax.get_legend()
    leg.legendHandles[0].set_edgecolor('black')
    leg.legendHandles[1].set_color('black')
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")
    fig.supxlabel('Local spatial frequency (cpd)')
    fig.supylabel('Response\n(% BOLD signal change)', ha='center')
    fig.subplots_adjust(left=0.3, bottom=0.17)


    utils.save_fig(save_path)
    return fig, ax


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

def plot_preferred_period(df, preferred_period, precision, hue, hue_order, fit_df,
                          pal=sns.color_palette("tab10"),
                          lgd_title=None, height=2.5,
                          col=None, col_order=None,
                          suptitle=None, width=3.4, errorbar=("ci", 68),
                          save_path=None):
    rc.update({'axes.titlepad': 10})
    sns.set_theme("notebook", style='ticks', rc=rc)

    new_df = df.copy()
    new_df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', subset=hue, jitter_scale=0.08)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', 'ecc', jitter_scale=0.03)


    new_df['value_and_weight'] = [v + w * 1j for v, w in zip(new_df[preferred_period], new_df[precision])]
    grid = sns.FacetGrid(new_df,
                         col=col,
                         col_order=col_order,
                         height=height,
                         aspect=width/height,
                         sharex=False, sharey=False)
    g = grid.map(sns.lineplot, 'ecc', 'value_and_weight',
                 hue, hue_order=hue_order, marker='o', palette=pal,
                 linestyle='', markersize=5, mew=0.1, mec='white',
                 estimator=utils.weighted_mean, errorbar=errorbar,
                 err_style='bars', err_kws={'elinewidth': 1.5}, alpha=0.85, zorder=10)

    if lgd_title is not None:
        if col is not None:
            handles, labels = g.axes[0,-1].get_legend_handles_labels()
        else:
            handles, labels = g.ax.get_legend_handles_labels()

        # Modifying the properties of the handles, e.g., increasing the line width
        for handle in handles:
            if hasattr(handle, 'set_linewidth'):  # Checking if the handle is a line
                handle.set_linewidth(4)  # Set line width to 3
                handle.set_alpha(1)
        if col is not None:
            grid.axes[0,-1].legend(handles=handles, labels=labels, loc=(1.1, 0.4), title=lgd_title, frameon=False)
        else:
            grid.ax.legend(handles=handles, labels=labels, loc=(1.02, 0.5), title=lgd_title, frameon=False)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(None)

    if fit_df is not None:
        for ax, col_name in zip(g.axes.flatten(), col_order):
            for i, cur_hue in enumerate(hue_order):
                tmp_fit_df = fit_df.copy()
                if col is not None:
                    tmp_fit_df = tmp_fit_df[tmp_fit_df[col] == col_name]
                tmp_fit_df = tmp_fit_df[tmp_fit_df[hue] == cur_hue]
                ax.plot(tmp_fit_df['ecc'], tmp_fit_df['fitted'], alpha=1,
                        color=pal[i], linestyle='-', linewidth=1.5, zorder=0)

    grid.axes[0,0].set(xlim=(0,10), xticks=[0,2,4,6,8,10], ylim=(0,2), yticks=[0, 1, 2])
    grid.axes[0,1].set(xlim=(0,4), xticks=[0,1,2,3,4], ylim=(0,1), yticks=[0, 0.5, 1])
    grid.set_axis_labels('Eccentricity (deg)', 'Deg per cycle')
    grid.fig.text(0.55, 0.95, suptitle, fontweight='bold', ha='center', fontsize=rc['figure.titlesize'])
    grid.fig.subplots_adjust(wspace=0.4)
    utils.save_fig(save_path)

    return g

def plot_bandwidth_in_octaves(df, bandwidth, precision, hue, hue_order, fit_df,
                              pal=sns.color_palette("tab10"),
                              lgd_title=None,
                              col=None, col_order=None,
                              suptitle=None, height=2.5, width=3.4, errorbar=("ci", 68),
                              save_path=None):

    rc.update({'axes.titlepad': 10})
    sns.set_theme("notebook", style='ticks', rc=rc)

    new_df = df.copy()
    new_df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', subset=hue, jitter_scale=0.08)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', 'ecc', jitter_scale=0.03)
    new_df['value_and_weight'] = [v + w * 1j for v, w in zip(new_df[bandwidth], new_df[precision])]
    grid = sns.FacetGrid(new_df,
                         col=col,
                         col_order=col_order,
                         height=height,
                         aspect=width/height,
                         sharex=False, sharey=False)
    g = grid.map(sns.lineplot, 'ecc', 'value_and_weight',
                 hue, hue_order=hue_order, marker='o', palette=pal,
                 linestyle='', markersize=5, mew=0.1, mec='white',
                 estimator=utils.weighted_mean, errorbar=errorbar,
                 err_style='bars', err_kws={'elinewidth': 1.5}, alpha=0.85, zorder=10)

    if lgd_title is not None:
        if col is not None:
            handles, labels = g.axes[0,-1].get_legend_handles_labels()
        else:
            handles, labels = g.ax.get_legend_handles_labels()

        # Modifying the properties of the handles, e.g., increasing the line width
        for handle in handles:
            if hasattr(handle, 'set_linewidth'):  # Checking if the handle is a line
                handle.set_linewidth(4)  # Set line width to 3
                handle.set_alpha(1)
        if col is not None:
            grid.axes[0,-1].legend(handles=handles, labels=labels, loc=(1.02, 0.55), title=lgd_title, frameon=False)
        else:
            grid.ax.legend(handles=handles, labels=labels, loc=(1.02, 0.55), title=lgd_title, frameon=False)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(None)

    if fit_df is not None:
        for ax, col_name in zip(g.axes.flatten(), col_order):
            for i, cur_hue in enumerate(hue_order):
                tmp_fit_df = fit_df.copy()
                if col is not None:
                    tmp_fit_df = tmp_fit_df[fit_df[col] == col_name]
                tmp_fit_df = tmp_fit_df[tmp_fit_df[hue] == cur_hue]
                ax.plot(tmp_fit_df['ecc'], tmp_fit_df['fitted'], alpha=1,
                        color=pal[i], linestyle='-', linewidth=1.5, zorder=0)

    grid.set_axis_labels('Eccentricity (deg)', 'FWHM (in octaves)')
    grid.axes[0,0].set(xlim=(0,10), xticks=[0,2,4,6,8,10],ylim=(4,10), yticks=[4,6,8,10])
    grid.axes[0,1].set(xlim=(0,4), xticks=[0,1,2,3,4], ylim=(4,10), yticks=[4,6,8,10])
    grid.fig.text(0.55, 0.95, suptitle, weight='bold', ha='center', fontsize=rc['figure.titlesize'])
    grid.fig.subplots_adjust(wspace=0.4)
    utils.save_fig(save_path)
    return g


def calculate_weighted_mean(df, value, precision, groupby=['vroinames']):
    df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df = df.groupby(groupby+['ecc']).apply(lambda x: (x[value] * x[precision]).sum() / x[precision].sum())
    new_df = new_df.reset_index().rename(columns={0: 'weighted_mean'})
    return new_df


def calculate_weighted_mean2(df, value, precision, groupby=['vroinames']):
    new_df = df.groupby(groupby).apply(lambda x: (x[value] * x[precision]).sum() / x[precision].sum())
    new_df = new_df.reset_index().rename(columns={0: 'weighted_mean'})
    return new_df

def _add_ecc_0(df, groupby=['vroinames']):
    unique_df = df.drop_duplicates(subset=groupby, inplace=False)
    unique_df['ecc'] = 0
    return pd.concat((df, unique_df), axis=0)


def fit_line_to_weighted_mean(df, values, precision, groupby=['vroinames']):
    weighted_mean_df = calculate_weighted_mean(df, values, precision, groupby)
    if 'ecc' not in weighted_mean_df.columns:
        weighted_mean_df['ecc'] = weighted_mean_df.apply(_get_middle_ecc, axis=1)
    weighted_mean_df = weighted_mean_df.sort_values(by='ecc')
    coeff_df = weighted_mean_df.groupby(groupby).apply(lambda x: np.polyfit(x['ecc'], x['weighted_mean'], 1))
    coeff_df = coeff_df.reset_index().rename(columns={0: 'coefficient'})
    fit_df = pd.merge(weighted_mean_df, coeff_df, on=groupby)
    fit_df = _add_ecc_0(fit_df, groupby)
    fit_df['fitted'] = fit_df.apply(lambda x: x['coefficient'][0] * x['ecc'] + x['coefficient'][1], axis=1)
    return fit_df

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

def plot_histogram_for_each_sub_and_roi(df, x, y, x_order, y_label=None,
                                     hue=None, hue_order=None,
                                     lgd_title=None,
                                     height=5, col=None, col_wrap=None,
                                     save_path=None, ylim=None,
                                     **kwargs):
    # Plot
    rc = {'axes.labelpad': 15}
    rc.update({'font.family': 'sans-serif',
          'font.sans-serif': ['HelveticaNeue', 'Helvetica', 'Arial'],
          'lines.linewidth': 2})
    utils.set_rcParams(rc)
    sns.set_context("notebook", font_scale=1.5, rc=rc)
    grid = sns.FacetGrid(df,
                         col=col, col_wrap=col_wrap,
                         height=height, hue=hue, hue_order=hue_order,
                         aspect=1.6,
                         sharex=True, sharey=True, **kwargs)
    g = grid.map(sns.stripplot, x, y,
                 order=x_order,
                 jitter=True, dodge=True,
                 marker='o', size=8, edgecolor='gray', linewidth=0.5)
    y_label = y_label if y_label else y
    grid.set_axis_labels('Regions of Interest (ROI)', y_label)
    if ylim is not None:
        g.set(ylim=ylim)
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    utils.save_fig(save_path)
    return g

def plot_median_for_each_sub_and_roi(df, x, y, x_order, y_label=None,
                                     hue=None, hue_order=None,
                                     lgd_title=None, aspect=1.6,
                                     height=5, col=None, col_wrap=None,
                                     save_path=None, to_logscale=False, ylim=None,
                                     **kwargs):
    # Plot
    rc = {'axes.labelpad': 15}
    rc.update({'font.family': 'sans-serif',
          'font.sans-serif': ['HelveticaNeue', 'Helvetica', 'Arial'],
          'lines.linewidth': 2})
    utils.set_rcParams(rc)
    sns.set_context("notebook", font_scale=1.5, rc=rc)
    grid = sns.FacetGrid(df,
                         col=col, col_wrap=col_wrap,
                         height=height, hue=hue, hue_order=hue_order,
                         aspect=aspect,
                         sharex=True, sharey=True, **kwargs)
    g = grid.map(sns.stripplot, x, y,
                 order=x_order,
                 jitter=True, dodge=True, alpha=0.95,
                 marker='o', size=8, edgecolor='gray', linewidth=0.5)
    y_label = y_label if y_label else y
    grid.set_axis_labels('Regions of Interest (ROI)', y_label)
    if to_logscale:
        grid.set(yscale='log')
    if ylim is not None:
        g.set(ylim=ylim)
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    utils.save_fig(save_path)
    return grid, g

def plot_datapoints(df, x, y, hue, hue_order=None, lgd_title=None,
                    col='names', col_order=None, suptitle=None,
                    height=5,
                    save_path=None, **kwargs):
    rc = {'text.color': 'black',
          'axes.labelcolor': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'axes.labelpad': 20,
          'axes.linewidth': 3,
          'axes.titlepad': 40,
          'axes.titleweight': "bold",
          "axes.spines.right": False,
          "axes.spines.top": False,
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
    sns.set_theme(style="ticks", context='notebook', rc=rc, font_scale=2)

    grid = sns.FacetGrid(df,
                         col=col,
                         col_order=col_order,
                         hue=hue,
                         height=height,
                         hue_order=hue_order,
                         sharex=True, sharey=True, **kwargs)
    grid.fig.suptitle(suptitle, fontweight="bold")
    g = grid.map(sns.scatterplot, x, y, s=90, alpha=0.9, edgecolor='gray')
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    grid.set(xscale='log')
    utils.save_fig(save_path)

    return g
