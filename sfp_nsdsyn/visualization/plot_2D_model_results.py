import os
import seaborn as sns
from sfp_nsdsyn import utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sfp_nsdsyn.two_dimensional_model import group_params
from sfp_nsdsyn.make_dataframes import calculate_local_orientation
from sfp_nsdsyn.two_dimensional_model import get_Pv_row
from sfp_nsdsyn.visualization.plot_1D_model_results import _get_x_and_y_prediction

mpl.rcParams.update(mpl.rcParamsDefault)
rc = {'text.color': 'black',
      'axes.labelcolor': 'black',
      'xtick.color': 'black',
      'ytick.color': 'black',
      'axes.edgecolor': 'black',
      'font.family': 'Helvetica',
      'axes.linewidth': 1,
      'axes.labelpad': 3,
      'axes.spines.right': False,
      'axes.spines.top': False,
      'xtick.major.pad': 5,
      'xtick.major.width': 1,
      'ytick.major.width': 1,
      'lines.linewidth': 1,
      'font.size': 11,
      'axes.titlesize': 11,
      'axes.labelsize': 11,
      'xtick.labelsize': 9,
      'ytick.labelsize': 9,
      'legend.title_fontsize': 11,
      'legend.fontsize': 11,
      'figure.titlesize': 14,
      'figure.dpi': 72 * 3,
      'savefig.dpi': 72 * 4
      }
mpl.rcParams.update(rc)


def get_y_values_for_summary_fig(goal):
    if goal in ['replication', 'Replication']:
        ylim_list = [(1, 3), (0, 0.42), (-0.15, 0.15), (-0.07, 0.07), (-0.11, 0.11)]
        ytick_list = [[1, 2, 3], [0, 0.2, 0.4], [-0.15, 0, 0.15], [-0.05, 0, 0.05], [-0.1, 0, 0.1]]
    elif goal in ['extension', 'Extension']:
        ylim_list = [(1.6, 5), (0, 0.4), (-0.2, 0.205), (-0.05, 0.07), (-0.8, 0.4)]
        ytick_list = [[2, 3, 4, 5], [0, 0.2, 0.4], [-0.2, 0, 0.2], [-0.05, 0, 0.05], [-0.8, -0.4, 0, 0.4]]
    return ylim_list, ytick_list

def get_params(param):
    if param == 'ecc_effect':
        return ['slope', 'intercept']
    else:
        return [param]

def get_param_y_values(param, goal):
    if goal in ['replication','Replication']:
        switcher = {'sigma': [(1,3), [1,2,3]],
                     'ecc_effect': [(0, 0.43), [0, 0.2, 0.4]],
                     'p_1': [(-0.2, 0.2),[-0.2,0,0.2]],
                     'p_2': [(-0.2, 0.2), [-0.2,0,0.2]],
                     'p_3': [(-0.2, 0.2), [-0.2,0,0.2]],
                     'p_4':  [(-0.2, 0.2), [-0.2,0,0.2]],
                     'A_1': [(-0.1,0.1), [-0.1,-0.05,0,0.05,0.1]],
                     'A_2': [(-0.05,0.05), [-0.05,0,0.05]]}
    elif goal in ['extension', 'Extension']:
        switcher = {'sigma': [(1.8, 5), [2,3,4,5]],
                     'ecc_effect': [(0, 0.4), [0, 0.2, 0.4]],
                     'p_1': [(-0.05, 0.22),[0,0.1,0.2]],
                     'p_2': [(-0.2, 0.02), [-0.2,-0.1,0]],
                     'p_3': [(-1, 0.2), [-1,-0.5,0]],
                     'p_4':  [(-0.2, 0.2), [-0.2,0,0.2]],
                     'A_1': [(-0.1,0.1), [-0.1,-0.05,0,0.05,0.1]],
                     'A_2': [(-0.1,0.1), [-0.1,-0.05,0,0.05,0.1]]}
    else:
        raise ValueError('the goal should be either replication or extension')
    return switcher.get(param, None)


def get_prediction_y_values(param, goal):
    if goal in ['replication','Replication']:
        switcher = {'sigma': [(0,1), [0, 0.5, 1]],
                     'ecc_effect': [(0,2), [0,1,2],],
                     'p_1': [(-0.5, 0.5),[-0.5,0,0.5]],
                     'p_2': [(-0.5, 0.5),[-0.5,0,0.5]],
                     'p_3': [(-0.5, 0.5),[-0.5,0,0.5]],
                     'p_4':  [(-0.5, 0.5),[-0.5,0,0.5]]}
    elif goal in ['extension', 'Extension']:
        switcher = {'sigma': [(0,1), [0, 0.5, 1]],
                    'ecc_effect': [(0,1), [0, 0.5, 1]],
                    'p_1': [(-0.1, 0.4),[0,0.2,0.4]],
                    'p_2': [(-0.4, 0.025),[-0.4,-0.2,0]],
                    'p_3': [(-2, 0.4), [-2,-1,0]],
                    'p_4': [(-0.4, 0.4), [-0.4,0,0.4]]}
    else:
        raise ValueError('the goal should be either replication or extension')
    return switcher.get(param, None)


def merge_model_and_precision(pt_path, precision_path, *ARGS_2D):
    from sfp_nsdsyn.two_dimensional_model import load_all_models
    if type(pt_path) is not list:
        pt_path = [pt_path]

    pt = load_all_models(pt_path, *ARGS_2D)
    precision_s = pd.read_csv(precision_path)
    df = pd.merge(pt, precision_s[['sub', 'vroinames', 'precision']], on=['sub', 'vroinames'])
    return df


def weighted_mean(x, **kws):
    """store weights as imaginery number"""
    numerator = np.sum(np.real(x) * np.imag(x))
    denominator = np.sum(np.imag(x))
    return numerator / denominator


def _change_params_to_math_symbols(params_col):
    params_col = params_col.replace({'sigma': r"$\sigma$",
                                     'slope': r"$Slope$" "\n" r"$m$",
                                     'intercept': r"$Intercept$" "\n" r"$b$",
                                     'p_1': r"$p_1$",
                                     'p_2': r"$p_2$",
                                     'p_3': r"$p_3$",
                                     'p_4': r"$p_4$",
                                     'A_1': r"$A_1$",
                                     'A_2': r"$A_2$"})
    return params_col



def filter_for_goal(params_df, goal):
    roi_pal = [sns.color_palette('dark', 10)[:][k] for k in [3, 2, 0]]
    roi_pal.insert(0, (0.3, 0.3, 0.3))
    goals = [goal.lower(), goal.lower().title()]
    #df = params_df.query('goal in @goals')
    if goal.lower() == 'replication':
        hue_order = ['Broderick et al. V1', 'NSD V1']
        pal = roi_pal[:2]
    elif goal.lower() == 'extension':
        hue_order = ['NSD V1','NSD V2','NSD V3']
        pal = roi_pal[1:]
    else:
        df = params_df
        hue_order = ['Broderick et al. V1', 'NSD V1', 'NSD V2', 'NSD V3']
        pal = roi_pal
    df = params_df.query('dset_type in @hue_order')

    return df, hue_order, pal

def plot_param_and_prediction(params_df, params,
                              prediction_df,
                              pal, hue, hue_order,
                              prediction_y=None,
                              params_ylim=None, params_yticks=None,
                              prediction_ylim=None, prediction_yticks=None,
                              xlim=(0,10), xticks=[0,5,10],
                              prediction_ylabel='Preferred period (deg)', title=None,
                              figsize=(3.5, 1.5), width_ratios=[1.5, 4], save_path=None, **kwargs):

    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': width_ratios},
                             sharey=False, sharex=False)

    g = plot_precision_weighted_avg_parameter(params_df, params,
                                              hue, hue_order,
                                              ax=axes[0], pal=pal,
                                              ylim=params_ylim, yticks=params_yticks, **kwargs)
    g.legend_.remove()
    if len(params) > 1:
        g.margins(x=0.1)
    if 'sigma' in params:
        g = plot_bandwidth_prediction(prediction_df, hue=hue, hue_order=hue_order,
                                      ax=axes[1], pal=pal)
    else:
        g = plot_preferred_period_in_axes(prediction_df, x='eccentricity', y=prediction_y, precision='precision',
                                          hline=True, ylim=prediction_ylim, yticks=prediction_yticks,
                                          ylabel=prediction_ylabel, xlim=xlim, xticks=xticks, hue=hue, hue_order=hue_order,
                                          pal=pal, ax=axes[1])
    g.legend(bbox_to_anchor=(1.05, 1), loc='best', frameon=False)
    if title is not None:
        fig.suptitle(title, fontweight="bold")
        fig.subplots_adjust(top=0.7)
    fig.subplots_adjust(wspace=1)
    utils.save_fig(save_path)
    return fig, axes


def plot_param_hierarchy_and_prediction(params_df, params,
                                  prediction_df,
                                  pal, hue, hue_order,
                                  prediction_y=None,
                                  params_ylim=None, params_yticks=None,
                                  prediction_ylim=None, prediction_yticks=None,
                                  hierarchy_ylim=None, hierarchy_yticks=None,
                                  xlim=(0,10), xticks=[0,5,10],
                                  prediction_ylabel='Preferred period (deg)', title=None,
                                  figsize=(3.5, 1.5), width_ratios=[1.5, 4], save_path=None, **kwargs):

    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={'width_ratios': [width_ratios[0], width_ratios[1], width_ratios[2]]},
                             sharey=False, sharex=False)

    g = plot_precision_weighted_avg_parameter(params_df, params,
                                              hue, hue_order,
                                              ax=axes[0], pal=pal,
                                              ylim=params_ylim, yticks=params_yticks, **kwargs)
    g.legend_.remove()
    if len(params) > 1:
        g.margins(x=0.1)
    if 'sigma' in params:
        g = plot_bandwidth_prediction(prediction_df, hue=hue, hue_order=hue_order,
                                      ax=axes[-1], pal=pal)
    else:
        g = plot_preferred_period_in_axes(prediction_df, x='eccentricity', y=prediction_y, precision='precision',
                                          hline=True, ylim=prediction_ylim, yticks=prediction_yticks,
                                          ylabel=prediction_ylabel, xlim=xlim, xticks=xticks, hue=hue, hue_order=hue_order,
                                          pal=pal, ax=axes[-1])
    g.legend(bbox_to_anchor=(1.05, 1), loc='best', frameon=False)
    plot_within_subject_error_for_V123(params_df, to_plot=params[0], precision='precision',
                                       ax=axes[1], ylim=hierarchy_ylim, yticks=hierarchy_yticks)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=rc['xtick.labelsize']-1)
    axes[1].margins(x=0.1)

    if title is not None:
        fig.suptitle(title, fontweight="bold")
        fig.subplots_adjust(top=0.7)
    fig.subplots_adjust(wspace=1.2)
    utils.save_fig(save_path)
    return fig, axes

def plot_precision_weighted_avg_parameter(df, params, hue, hue_order, ax, ylim=None, yticks=None, pal=None, **kwargs):
    sns.set_theme("paper", style='ticks', rc=rc)

    tmp = group_params(df, params, [1]*len(params))
    tmp = tmp.query('params in @params')
    tmp['value_and_weights'] = tmp.apply(lambda row: row.value + row.precision * 1j, axis=1)
    tmp['params'] = _change_params_to_math_symbols(tmp['params'])
    g = sns.pointplot(data=tmp,
                      x='params', y='value_and_weights',
                      hue=hue, hue_order=hue_order,
                      palette=pal, linestyles='',
                      estimator=weighted_mean, errorbar=("ci", 68),
                      dodge=0.23,
                      ax=ax, **kwargs)
    g.set(ylabel='Parameter estimates', xlabel=None)
    if 'p_' in params[0] or 'A_' in params[0] or 'sigma' in params[0]:
        g.tick_params(axis='x', labelsize=rc['axes.labelsize'], pad=5)
    else:
        g.tick_params(axis='x', pad=5)
    if ylim is not None:
        g.set(ylim=ylim)
    if yticks is not None:
        g.set(yticks=yticks)

    ticks = [t.get_text() for t in g.get_xticklabels()]
    if any('p_' in s for s in ticks) or any('A_' in s for s in ticks):
        g.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)

    return g

def make_param_summary_fig(params_df, hue, hue_order, pal,
                           params_list, ylim_list=None, yticks_list=None,
                           title_list=None, 
                           width_ratios=(0.8,1.8,1.3,1.3,1.3), fig_size=(7, 1.5),
                           save_path=None, **kwargs):
    rc.update({
          'axes.titlesize': 11,
          'axes.titlepad': 20,
          'axes.labelsize': 9.5,
          'ytick.labelsize': 9,
           'legend.title_fontsize': 9.5,
        'legend.fontsize': 9.5,
          })
    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, len(params_list), figsize=fig_size,
                             gridspec_kw={'width_ratios': width_ratios},
                             sharey=False, sharex=False)

    for i, ax in enumerate(axes.flatten()):
        g = plot_precision_weighted_avg_parameter(params_df, params_list[i],
                                                  hue, hue_order,
                                                  ylim=ylim_list[i] if ylim_list is not None else None,
                                                  yticks=yticks_list[i] if yticks_list is not None else None,
                                                  ax=ax,
                                                  pal=pal, **kwargs)
        g.legend_.remove()
        if title_list is not None:
            ax.set_title(title_list[i], fontweight="bold")
        if len(params_list[i]) > 1:
            g.margins(x=0.15)
        if 'p_' in params_list[i][0] or 'A_' in params_list[i][0] or 'sigma' in params_list[i][0]:
            g.tick_params(axis='x', labelsize=rc['axes.labelsize']+1, pad=5)
        else:
            g.margins(x=0.1)
            g.tick_params(axis='x', labelsize=rc['axes.labelsize']-0.6, pad=5)
    g.legend(bbox_to_anchor=(1.05, 1), loc='best', frameon=False)

    for i in np.arange(1, len(axes)):
        axes[i].set(ylabel='')

    fig.subplots_adjust(wspace=0.8, top=0.9)
    utils.save_fig(save_path)
    return fig, axes

def make_dset_palettes(dset):
    c_list = sns.diverging_palette(130, 300, s=100, l=30, n=2, as_cmap=False)
    hex_color = c_list.as_hex()
    if dset == 'broderick':
        pal = utils.color_husl_palette_different_shades(12, hex_color[1])
        pal.reverse()
    elif dset == 'nsdsyn':
        pal = utils.color_husl_palette_different_shades(8, hex_color[0])
        pal.reverse()
    else:
        palette = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
               (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
               (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
               (189, 189, 189)]
        pal = utils.convert_rgb_to_seaborn_color_palette(palette)
    return pal

def plot_individual_parameters(df, params, subplot_group, height=7, hue='subj', roi=None,
                               palette=None, row=None, lgd_title='Subjects'):
    sns.set_context("notebook", font_scale=2)
    hue_order = df.sort_values(by='precision', ignore_index=True, ascending=False).subj.unique()
    df = group_params(df, params, subplot_group)
    df['params'] = _change_params_to_math_symbols(df['params'])
    y_label = "Value"
    groups, counts = np.unique(subplot_group, return_counts=True)
    if palette is None:
        palette = make_dset_palettes('default')
    grid = sns.FacetGrid(df,
                         col="group",
                         row=row,
                         hue=hue,
                         hue_order=hue_order,
                         palette=palette,
                         height=height,
                         legend_out=True,
                         sharex=False, sharey=False, gridspec_kws={'width_ratios': counts})
    grid.map(sns.stripplot, "params", "value", dodge=True, jitter=True, size=20, alpha=0.82, edgecolor="gray", linewidth=1)
    grid.add_legend(title=lgd_title)
    for ax in range(len(groups)):
        grid.axes[0, ax].set_ylim(_find_ylim(ax, roi, avg=False))
        if counts[ax] > 1 and len(groups) < 4:
            grid.axes[0, ax].margins(x=1 - 0.45*ax)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f" ")
    grid.set_axis_labels("", y_label)
    return grid


def plot_preferred_period_difference(df,
                                      x, y='Pv_diff', precision='precision',
                                      hue=None, hue_order=None,
                                      col=None, col_wrap=None, ylabel=r'$P_v$ horizontal - $P_v$ vertical',
                                      lgd_title=None, width=3, height=1.2,
                                      xlim=(0,10), ylim=(-0.4,0.4), yticks=[-0.4,0,0.4],
                                      projection=None, save_path=None,
                                      **kwarg):
    if projection == 'polar':
        despine = False
        xticks = [0, np.pi/4, 2*np.pi/4, 3*np.pi/4, np.pi, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 2*np.pi]
        xticklabels = []
        xlim=(0, 2*np.pi)
        rc.update({'polaraxes.grid': True,
                   'axes.grid': True,
                   'axes.linewidth': 2,
                   'grid.alpha':0.8,
                   'axes.labelpad': 4,
                   'figure.subplot.left': 0.4})
    else:
        rc.update({
            'axes.linewidth': 1,
            'axes.labelpad': 3,
            'xtick.major.pad': 2,
            'ytick.major.pad': 2,
            'xtick.major.size': 3,
            'ytick.major.size': 2.5,
            'xtick.minor.size': 1.8,
            'xtick.major.width': 1,
            'xtick.minor.width': 0.8,
            'ytick.major.width': 1,
            'lines.linewidth': 1,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelsize': 10,
            'legend.title_fontsize': 10 * 0.8,
            'legend.fontsize': 10 * 0.8,
        })
        utils.set_rcParams(rc)
        despine = True
        xticks=[0, 5, 10]
        xticklabels = xticks
    x_label = x.title()
    sns.set_theme(style='ticks', rc=rc, font_scale=1)
    utils.scale_fonts(0.8)
    df['value_and_weights'] = [v + w * 1j for v, w in zip(df[y], df[precision])]
    # plotting average of prediction, not the prediction of average
    grid = sns.FacetGrid(df,
                         hue=hue,
                         hue_order=hue_order,
                         height=height,
                         col=col, col_wrap=col_wrap,
                         aspect=width/height,
                         subplot_kws={'projection': projection},
                         legend_out=True, despine=despine,
                         sharex=True, sharey=False,
                         **kwarg)
    grid = grid.map(sns.lineplot, x, "value_and_weights",
                    linewidth=1.5, estimator=weighted_mean,
                    n_boot=100, err_style='band', errorbar=('ci',68))
    grid.set_axis_labels(x.title(), ylabel)
    grid.set(xlim=xlim, xticks=xticks, xticklabels=xticklabels)
    if ylim is not None:
        grid.set(ylim=ylim)
    if yticks is not None:
        grid.set(yticks=yticks)

    for ax in grid.axes.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.9, zorder=0)
    if col is not None:
        for subplot_title, ax in grid.axes_dict.items():
            ax.set_title(None)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title, bbox_to_anchor=(1, 0.7))
    grid.fig.subplots_adjust(top=0.9, wspace=0.6)
    grid.axes[0, 1].set_ylabel(ylabel)
    utils.save_fig(save_path)
    return grid




def plot_preferred_period_in_axes(df, x, y, ax, hline=False,
                                  ylim=None, yticks=None, xlim=(0,10), xticks=[0,5,10], ylabel='Preferred period (deg)',
                                  hue=None, hue_order=None, pal=None, precision='precision'):
    sns.set_theme("notebook", style='ticks', rc=rc, font_scale=1)
    df['value_and_weights'] = [v + w * 1j for v, w in zip(df[y], df[precision])]
    g = sns.lineplot(df, x=x, y="value_and_weights",
                     hue=hue, hue_order=hue_order,
                     linewidth=1.5, estimator=weighted_mean, palette=pal,
                     err_style='band', errorbar=('ci', 68), ax=ax)
    g.legend_.remove()

    if ylim is not None:
        g.set(ylim=ylim)
    if yticks is not None:
        g.set(yticks=yticks)
    if hline is True:
        g.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)
    g.set(ylabel=ylabel)

    g.set(xlim=xlim, xticks=xticks, xlabel='Eccentricity (deg)')
    g.tick_params(axis='x', which='major', pad=3)
    return g


def plot_preferred_period(df,
                          x, y='Pv', precision='precision',
                          hue=None, hue_order=None,
                          col=None, col_wrap=None, ylabel='Preferred period',
                          lgd_title=None, width=3, height=3,
                          xlim=(0,10), ylim=(0,2), yticks=[0, 0.5, 1, 1.5, 2],
                          projection=None, save_path=None,
                          **kwarg):
    sns.set_theme("notebook", style='ticks', rc=rc, font_scale=1)
    if projection == 'polar':
        despine = False
        xticks = [0, np.pi/4, 2*np.pi/4, 3*np.pi/4, np.pi, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 2*np.pi]
        xticklabels = []
        xlim=(0, 2*np.pi)
        rc.update({'polaraxes.grid': True,
                   'axes.grid': True,
                   'grid.alpha':0.8,
                   'xtick.labelsize': 8,
                   'ytick.labelsize': 8,
                   'axes.labelpad': 4,
                   'figure.subplot.left': 0.4})
        utils.set_rcParams(rc)
    else:
        rc.update({
            'axes.linewidth': 1,
            'axes.labelpad': 3,
            'xtick.major.pad': 2,
            'ytick.major.pad': 2,
            'xtick.major.size': 3,
            'ytick.major.size': 2.5,
            'xtick.minor.size': 1.8,
            'xtick.major.width': 1,
            'xtick.minor.width': 0.8,
            'ytick.major.width': 1,
            'lines.linewidth': 1,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelsize': 10,
            'legend.title_fontsize': 10 * 0.8,
            'legend.fontsize': 10 * 0.8,
        })
        utils.set_rcParams(rc)
        despine = True
        xticks=[0, 5, 10]
        xticklabels = xticks
        sns.set_theme(style='ticks', rc=rc)
        utils.scale_fonts(0.8)
    x_label = x.title()
    df['value_and_weights'] = [v + w * 1j for v, w in zip(df[y], df[precision])]
    # plotting average of prediction, not the prediction of average
    grid = sns.FacetGrid(df,
                         hue=hue,
                         hue_order=hue_order,
                         height=height,
                         col=col, col_wrap=col_wrap,
                         aspect=width/height,
                         subplot_kws={'projection': projection},
                         legend_out=True, despine=despine,
                         sharex=True, sharey=False,
                         **kwarg)
    grid = grid.map(sns.lineplot, x, "value_and_weights",
                    linewidth=1.5, estimator=weighted_mean,
                    n_boot=100, err_style='band', errorbar=('ci',68))
    grid.set_axis_labels(x.title(), ylabel)
    grid.set(xlim=xlim, xticks=xticks, xticklabels=xticklabels, ylim=ylim, yticks=yticks)

    if col is not None:
        for subplot_title, ax in grid.axes_dict.items():
            ax.set_title(None)
            if y == 'Pv_diff':
                ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)
    else:
        if y == 'Pv_diff':
            grid.ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)

    if lgd_title is not None:
        grid.add_legend(title=lgd_title, bbox_to_anchor=(1, 0.8))
    grid.fig.subplots_adjust(top=0.9, wspace=0.5)
    grid.axes[0,1].set_ylabel('Preferred period')
    utils.save_fig(save_path)
    return grid


def plot_grouped_parameters_subj(df, params, col_group,
                            to_label="study_type", lgd_title="Study", label_order=None,
                            height=7,
                            save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/params.png'):
    df = group_params(df, params, col_group)
    sns.set_context("notebook", font_scale=1.5)
    x_label = "Parameter"
    y_label = "Value"
    pal = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
           (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
           (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
           (189, 189, 189)]
    n_labels = df[to_label].nunique()
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    pal = sns.color_palette(np.array(pal) / 255, n_labels)
    grid = sns.FacetGrid(df,
                         col="group",
                         col_wrap=3,
                         palette=pal,
                         hue=to_label,
                         height=height,
                         hue_order=label_order,
                         legend_out=True,
                         sharex=False, sharey=False)
    grid.map(sns.pointplot, "params", "value", estimator=np.median, alpha=0.9, orient="v", ci=68, dodge=True, linestyle=None, scale=1.5)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f" ")
    grid.fig.legend(title=lgd_title, labels=label_order)
    grid.set_axis_labels("", y_label)
    utils.save_fig(save_fig, save_path)



def plot_final_params(df, comb, params_list, params_group, save_fig=True, save_path='/Volumes/server/Projects/sfp_nsd/figures/pic.png'):
    mpl.rcParams['axes.linewidth'] = 2  # set the value globally
    params_order = np.arange(0, len(params_list))
    n_comb = len(comb)
    colors = mpl.cm.RdPu(np.linspace(0, 1, n_comb+2))[0:n_comb+1]
    colors[0] = [0, 0, 0, 1]
    colors[1] = [0.5, 0.5, 0.5, 1]
    n_subplots = len(set(params_group))
    fig, axes = plt.subplots(1, n_subplots, figsize=(22, 6), dpi=300,
                             gridspec_kw={'width_ratios': [params_group.count(x) for x in set(params_group)]})
    control_fontsize(14, 20, 15)
    for g in range(n_subplots):
        tmp_params_list = [i for (i, v) in zip(params_list, params_group) if v == g]
        tmp_params_order = [i for (i, v) in zip(params_order, params_group) if v == g]
        for p, c in zip(tmp_params_list, tmp_params_order):
            for x_color in np.arange(0, n_comb):
                roi = comb[x_color][0]
                dset = comb[x_color][1]
                tmp_df = df.query('params == @p & vroinames == @roi & dset == @dset')
                x = tmp_df.params
                y = tmp_df.mean_value
                yerr = tmp_df.std_value
                axes[g].errorbar(x, y, yerr=yerr, fmt="o", ms=16,
                                 color=colors[x_color],
                                 elinewidth=3, ecolor=colors[x_color],
                                 label=f'{dset} {roi}')

                axes[g].spines['top'].set_visible(False)
                axes[g].spines['right'].set_visible(False)
                axes[g].tick_params(axis='both', labelsize=22)
        if g !=0 and g != 1:
            axes[g].axhline(y=0, color='gray', linestyle='--', linewidth=2)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[n_subplots-1].legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(w_pad=2)
    fig.supylabel('Parameter Value', fontsize=20)
    fig.subplots_adjust(left=.09, bottom=0.15)
    utils.save_fig(save_fig, save_path)


def plot_final_params_2(df, comb, params_list, params_group, save_fig=True, save_path='/Volumes/server/Projects/sfp_nsd/figures/pic.png'):
    mpl.rcParams['axes.linewidth'] = 2  # set the value globally
    params_order = np.arange(0, len(params_list))
    n_comb = len(comb)
    colors = mpl.cm.RdPu(np.linspace(0, 1, n_comb+4))[2:n_comb+2]
    colors[0] = [0, 0, 0, 1]
    colors[1] = [0.7, 0.7, 0.7, 1]
    colors[4:] = [0, 0, 0, 0]
    n_subplots = len(set(params_group))
    fig, axes = plt.subplots(1, n_subplots, figsize=(22, 6), dpi=300,
                             gridspec_kw={'width_ratios': [params_group.count(x) for x in set(params_group)]})
    control_fontsize(14, 20, 15)
    for g in range(n_subplots):
        tmp_params_list = [i for (i, v) in zip(params_list, params_group) if v == g]
        tmp_params_order = [i for (i, v) in zip(params_order, params_group) if v == g]
        for p, c in zip(tmp_params_list, tmp_params_order):
            for x_color in range(n_comb):
                roi = comb[x_color][0]
                dset = comb[x_color][1]
                tmp_df = df.query('params == @p & vroinames == @roi & dset == @dset')
                x = tmp_df.params
                y = tmp_df.mean_value
                yerr = tmp_df.std_value
                axes[g].errorbar(x, y, yerr=yerr, fmt="o", ms=16,
                                 color=colors[x_color],
                                 elinewidth=3, ecolor=colors[x_color],
                                 label=f'{dset} {roi}')

                axes[g].spines['top'].set_visible(False)
                axes[g].spines['right'].set_visible(False)
                axes[g].tick_params(axis='both', labelsize=22)
        if g !=0 and g != 1:
            axes[g].axhline(y=0, color='gray', linestyle='--', linewidth=2)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #axes[n_subplots-1].legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(w_pad=2)
    #fig.supylabel('Parameter Value', fontsize=20)
    fig.subplots_adjust(left=.09, bottom=0.15)
    utils.save_fig(save_fig, save_path)



def plot_sd_with_different_shades(df):
    sns.set_context('notebook', font_scale=3)
    sns.catplot(data=df, x='dset', y="sigma_v_squared", kind="point", hue='subj', hue_order=df.subj.unique(),
                palette=broderick_pal, dodge=True, size=20, alpha=0.8, edgecolor="gray", linewidth=1)


def _get_common_lim(axes, round=False):
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    if round is True:
        new_lim = [utils.decimal_floor(min(xlim[0], ylim[0]), precision=2),
                   utils.decimal_ceil(max(xlim[1], ylim[1]), precision=2)]
    else:
        new_lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    new_lim[0] = max(new_lim[0], 0)
    new_ticks = np.linspace(new_lim[0], new_lim[1], 5)
    return new_lim, new_ticks

def plot_vareas(df, x, y, hue, style,
                hue_order=None,
                col=None,
                new_ticks_list=None,
                save_path=None,
                height=5, **kwargs):
    sns.set_context('notebook', font_scale=2)
    rc = {'axes.titlepad': 40,
          'axes.labelpad': 20,
          'axes.linewidth': 2,
          'xtick.major.pad': 10,
          'xtick.major.width': 2,
          'ytick.major.width': 2,
          'xtick.major.size': 10,
          'ytick.major.size': 10,
          'grid.linewidth': 2,
          'font.family': 'Helvetica',
          'lines.linewidth': 2}
    utils.set_rcParams(rc)
    grid = sns.relplot(data=df,
                       x=x, y=y,
                       col=col,
                       hue=hue, hue_order=hue_order,
                       style=style,
                       height=height,
                       facet_kws={'sharey': False, 'sharex': False},
                       **kwargs)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")

    for i in range(len(grid.axes.flatten())):
        if new_ticks_list is None:
            new_lim, new_ticks = _get_common_lim(grid.axes[0,i], True)
        else:
            new_ticks = new_ticks_list[i]
            new_lim = (new_ticks[0], new_ticks[-1])
        grid.axes[0,i].set(xlim=new_lim, ylim=new_lim,
                           xticks=new_ticks, yticks=new_ticks)
        grid.axes[0,i].plot(np.linspace(new_lim[0], new_lim[1], 10),
                            np.linspace(new_lim[0], new_lim[1], 10),
                            linestyle='--', color='gray', linewidth=1)
        grid.axes[0,i].set_aspect('equal')
        grid.axes[0,i].tick_params(right=True, top=True,
                                   labelleft=True, labelbottom=True,
                                   labelright=True, labeltop=True,
                                   labelrotation=0)
        grid.axes[0,i].spines['right'].set_visible(True)
        grid.axes[0,i].spines['top'].set_visible(True)
        grid.fig.subplots_adjust(wspace=.4, right=0.86)
        grid.set_axis_labels('V1', 'V2')
        grid.axes[0,-1].set_ylabel('V3', rotation=270)
        grid.axes[0,-1].yaxis.set_label_position("right")
        grid.axes[0,-1].set(xticks=new_ticks[1:], yticks=new_ticks[1:])
        utils.save_fig(save_path)
    return grid

def plot_varea(df, x, y, hue, style,
                hue_order=None,
                col=None,
                new_ticks_list=None,
                save_path=None,
                height=5, **kwargs):
    sns.set_context('notebook', font_scale=2)
    rc = {'axes.titlepad': 40,
          'axes.labelpad': 20,
          'axes.linewidth': 2,
          'xtick.major.pad': 10,
          'xtick.major.width': 2,
          'ytick.major.width': 2,
          'xtick.major.size': 10,
          'ytick.major.size': 10,
          'grid.linewidth': 2,
          'font.family': 'Helvetica',
          'lines.linewidth': 2}
    utils.set_rcParams(rc)
    grid = sns.relplot(data=df,
                       x=x, y=y,
                       col=col,
                       hue=hue, hue_order=hue_order,
                       style=style,
                       height=height,
                       facet_kws={'sharey': False, 'sharex': False},
                       **kwargs)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")

    for i in range(len(grid.axes.flatten())):
        if new_ticks_list is None:
            new_lim, new_ticks = _get_common_lim(grid.axes[0,i], True)
        else:
            new_ticks = new_ticks_list[i]
            new_lim = (new_ticks[0], new_ticks[-1])
        grid.axes[0,i].set(xlim=new_lim, ylim=new_lim,
                           xticks=new_ticks, yticks=new_ticks)
        grid.axes[0,i].plot(np.linspace(new_lim[0], new_lim[1], 10),
                            np.linspace(new_lim[0], new_lim[1], 10),
                            linestyle='--', color='gray', linewidth=1)
        grid.axes[0,i].set_aspect('equal')
        grid.axes[0,i].tick_params(right=False, top=False,
                                   labelleft=True, labelbottom=True,
                                   labelright=False, labeltop=False,
                                   labelrotation=0)
        grid.axes[0,i].spines['right'].set_visible(False)
        grid.axes[0,i].spines['top'].set_visible(False)
        grid.fig.subplots_adjust(wspace=.4, right=0.86)
        grid.set_axis_labels('V1', 'V2')
        grid.axes[0,-1].set(xticks=new_ticks[1:], yticks=new_ticks[1:])
        utils.save_fig(save_path)
    return grid


def make_multiple_xy(df, id_var, to_var, to_val):
    pivot_df = df.pivot(id_var, to_var, to_val).reset_index()
    var_list = df[to_var].unique()

    multiple_xy_df = pd.DataFrame({})
    for k, v in zip(var_list[0:], var_list[1:]):
        tmp = pd.DataFrame({})
        tmp[id_var] = pivot_df[id_var]
        tmp['x'] = pivot_df[k]
        tmp['y'] = pivot_df[v]
        tmp[to_var] = f'{k}-{v}'
        multiple_xy_df = multiple_xy_df.append(tmp)
    return multiple_xy_df

def make_multiple_xy_with_vars(df, id_var, to_var, to_vals, val_name='params'):
    multiple_xy_dfs = pd.DataFrame({})
    for to_val in to_vals:
        tmp = make_multiple_xy(df, id_var, to_var, to_val)
        tmp[val_name] = to_val
        multiple_xy_dfs = multiple_xy_dfs.append(tmp)
    return multiple_xy_dfs


def get_w_a_and_w_r_for_each_stim_class(stim_description_path,
                                        stim_class=['pinwheel','annulus','forward spiral', 'reverse spiral']):
    stim_info = pd.read_csv(stim_description_path)
    stim_info = stim_info.query('names in @stim_class')
    stim_info = stim_info.drop_duplicates('names')
    stim_info = stim_info[['names', 'w_r', 'w_a']]
    return stim_info


def merge_continuous_values_to_the_df(df, val_range=(0,6), n=100, col_name='eccentricity', endpoint=True):

    val_range = np.linspace(val_range[0], val_range[-1], n, endpoint=endpoint)
    all_ecc_df = pd.DataFrame({})
    for val in val_range:
        df[col_name] = val
        all_ecc_df = all_ecc_df.append(df, ignore_index=True)
    return all_ecc_df

def make_synthetic_dataframe_for_2D(stim_info,
                                    ecc_range, n_ecc,
                                    angle_range, n_angle,
                                    ecc_col='eccentricity', angle_col='angle'):
    merged_df = stim_info.copy()
    for val, n, col in zip([ecc_range, angle_range], [n_ecc, n_angle], [ecc_col, angle_col]):
        merged_df = merge_continuous_values_to_the_df(merged_df,
                                                      val_range=val,
                                                      n=n,
                                                      col_name=col)
    return merged_df


def calculate_preferred_period_for_synthetic_df(stim_info, final_params,
                                                ecc_range, n_ecc,
                                                angle_range, n_angle,
                                                ecc_col='eccentricity', angle_col='angle',
                                                angle_in_radians=True,
                                                sfstimuli='scaled'):
    if sfstimuli not in ['scaled', 'constant']:
        raise ValueError('reference_frame should be either scaled or constant')
    merged_df = make_synthetic_dataframe_for_2D(stim_info,
                                                ecc_range, n_ecc,
                                                angle_range, n_angle,
                                                ecc_col, angle_col)
    merged_df['sfstimuli'] = sfstimuli
    merged_df['local_ori'] = calculate_local_orientation(merged_df['w_a'], merged_df['w_r'],
                                                         retinotopic_angle=merged_df[angle_col],
                                                         angle_in_radians=angle_in_radians,
                                                         sfstimuli=sfstimuli)
    merged_df['Pv'] = merged_df.apply(get_Pv_row, params=final_params, axis=1)
    if sfstimuli == 'constant':
        rename_cols = {'forward spiral': 'right oblique',
                       'reverse spiral': 'left oblique',
                       'annulus': 'vertical',
                       'pinwheel': 'horizontal'}
        merged_df = merged_df.replace({'names': rename_cols})
    return merged_df

def calculate_preferred_period_for_all_subjects(subj_list, synthetic_df, final_params):
    all_subj_df = pd.DataFrame({})
    for s in subj_list:
        tmp = synthetic_df.copy()
        tmp_params = final_params.query('sub == @s')
        tmp['sub'] = s
        tmp['Pv'] = tmp.apply(get_Pv_row, params=tmp_params, axis=1)
        all_subj_df = all_subj_df.append(tmp, ignore_index=True)
    return all_subj_df

def get_Av(row, local_ori):
    return 1 + row['A_1'] * np.cos(2*local_ori) + row['A_2'] * np.cos(4*local_ori)

def get_peak(row, local_ori, ecc, angle):
    ecc_dependency = row['slope']*ecc + row['intercept']
    pv = ecc_dependency* (1 + row['p_1'] * np.cos(2*local_ori) +
                            row['p_2'] * np.cos(4*local_ori) +
                            row['p_3'] * np.cos(2*(local_ori - angle)) +
                            row['p_4'] * np.cos(4*(local_ori - angle)))
    return 1/pv

def get_weighted_average_of_params(params_df, groupby, ecc, angle, local_ori):

    params_df['Av'] = params_df.apply(get_Av, local_ori=local_ori, axis=1)
    params_df['Pv'] = params_df.apply(get_peak, local_ori=local_ori, ecc=ecc, angle=angle, axis=1)

    weighted_mean_df = params_df[groupby].drop_duplicates(subset=groupby).copy()
    for param in ['Av', 'Pv', 'sigma']:
        tmp = params_df.groupby(groupby).apply(
            lambda x: (x[param] * x['precision']).sum() / x['precision'].sum())
        tmp = tmp.reset_index().rename(columns={0: f'{param}'})
        weighted_mean_df = pd.merge(weighted_mean_df, tmp, on=groupby)
    return weighted_mean_df


def plot_bandwidth_prediction(weighted_mean_df, hue, hue_order, pal, ax, save_path=None):
    sns.set_theme("paper", style='ticks', rc=rc, font_scale=1)

    for i, dset in enumerate(hue_order):
        tmp = weighted_mean_df[weighted_mean_df[hue] == dset]
        pred_x, pred_y = _get_x_and_y_prediction(0.1, 20,
                                                 tmp['Av'].item(),
                                                 tmp['Pv'].item(),
                                                 tmp['sigma'].item(), n_points=1000)
        ax.plot(pred_x, pred_y, linewidth=1.5, color=pal[i], label=dset)

    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='x', which='major', labelsize=7, pad=3)
    ax.set_xlabel('Spatial frequency (cpd)')
    ax.set_ylabel(' ' '\nPredicted\nBOLD Response')
    return ax


def get_Pv_difference(df, orientation_1, orientation_2, to_group=['sub','dset_type','eccentricity','vroinames'],
                      orientation_col='names'):
    all_cols = to_group + ['Pv']
    if 'cardinal' == orientation_1 or 'cardinal' == orientation_2:
        df = df.replace({orientation_col: {'horizontal': 'cardinal',
                                           'vertical': 'cardinal',
                                           'left oblique': 'oblique',
                                           'right oblique': 'oblique'}})
        df = df.groupby(to_group+[orientation_col]).mean().reset_index()
    elif orientation_1 == 'spirals' or orientation_2 == 'spirals':
        df = df.replace({orientation_col: {'forward spiral': 'spirals',
                                           'reverse spiral': 'spirals',
                                           'pinwheel': 'non-spirals',
                                           'annulus': 'non-spirals'}})
        df = df.groupby(to_group+[orientation_col]).mean().reset_index()
    else:
        df = df

    ori_df_1 = df[df[orientation_col] == orientation_1]
    ori_df_2 = df[df[orientation_col] == orientation_2]

    p_df = pd.merge(ori_df_1, ori_df_2[all_cols],
                    on=to_group, suffixes=('_1', '_2'))

    p_df['Pv_diff'] = p_df[f'Pv_1'] - p_df[f'Pv_2']
    p_df = p_df.groupby(to_group).mean().reset_index()
    return p_df

def calculate_within_subject_error_for_V123(df, value, subject='sub', roi='vroinames'):
    # Pivoting the dataframe
    new_df = df.pivot(index=subject, columns=roi, values=value)

    # Renaming the columns to 'sigma_V1', 'sigma_V2', 'sigma_V3'
    new_df.columns = [f'{value}_{roi}' for roi in new_df.columns]

    # Resetting the index if needed
    new_df = new_df.reset_index()
    new_df[f'V3_minus_V2'] = new_df[f'{value}_V3'] - new_df[f'{value}_V2']
    new_df[f'V2_minus_V1'] = new_df[f'{value}_V2'] - new_df[f'{value}_V1']
    new_df = new_df.drop(columns=[f'{value}_V1', f'{value}_V2', f'{value}_V3'])
    new_df = new_df.melt(id_vars=[subject], value_vars=[f'V3_minus_V2', f'V2_minus_V1'], value_name=value)

    return new_df

def plot_within_subject_error_for_V123(df, to_plot, precision, ylim=None, yticks=None,
                                       ax=None, title=None, save_path=None):
    sns.set_theme("paper", style='ticks', rc=rc, font_scale=1)
    new_df = calculate_within_subject_error_for_V123(df, to_plot, subject='sub', roi='vroinames')
    new_df = pd.merge(new_df, df[['sub', precision]], on='sub')

    new_df['value_and_weights'] = new_df.apply(lambda row: row[to_plot] + row[precision] * 1j, axis=1)
    new_names = ['V2-V1','V3-V2']
    #new_names = ['V2—V1','V3—V2']
    new_df = new_df.replace({'variable': {'V2_minus_V1': new_names[0],
                                  'V3_minus_V2': new_names[1]}})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.8))
    ax = sns.barplot(data=new_df, x='variable', y='value_and_weights', order=new_names,
                     width=0.5, errorbar=('ci', 68),
                     estimator=weighted_mean, color='gray', ax=ax)
    ax.set(xlabel='', ylabel='Parameter difference')

    if ylim is not None:
        ax.set(ylim=ylim)
    if yticks is not None:
        ax.set(yticks=yticks)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=1)
    ax.set_title(title)
    utils.save_fig(save_path)
    return ax

def rand_jitter(arr, jitter_amount=0.05):
    arr = np.asarray(arr)
    if len(arr) == 1:
        stdev = 0.05
    else:
        stdev = jitter_amount * (arr.max() - arr.min()) 
    return arr + np.random.randn(len(arr)) * stdev


def plot_individual_parameters(params_df, x, y, hue, hue_order, 
                                params_list, ylim_list=None, yticks_list=None,
                                title_list=None, 
                                figsize=(7, 6),
                                save_path=None, **kwargs):
    rc.update({
          'axes.titlesize': 11,
          'axes.titlepad': 20,
          'axes.labelsize': 11,
          'ytick.labelsize': 11,
          'legend.title_fontsize': 11,
          'legend.fontsize': 11})
    
    sns.set_theme("paper", style='ticks', rc=rc)
    pal = sns.color_palette('tab10', len(hue_order))

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)

    # Define a 2x6 grid for flexibility
    gs = GridSpec(nrows=2, ncols=6, figure=fig)

    # First row
    ax1 = fig.add_subplot(gs[0, 0:2])        # 1 column = smallest
    ax2 = fig.add_subplot(gs[0, 2:5])      # 3 columns = largest
    ax3 = fig.add_subplot(gs[1, 0:2])      # 2 columns = medium
    # Second row
    ax4 = fig.add_subplot(gs[1, 2:4])      # 2 columns = medium
    ax5 = fig.add_subplot(gs[1, 4:6])      # 2 columns = medium

    # Optionally turn off unused cells (gs[1, 4:6])
    # Or you can assign ax6 = fig.add_subplot(gs[1, 4:6]) and hide it

    # Add simple demo content
    axes = [ax1, ax2, ax3, ax4, ax5]
    for i, ax in enumerate(axes):
        param = params_list[i]
        for j, sub in enumerate(hue_order):
            tmp = params_df.query('parameter in @param & sub == @sub')
            x_jittered = rand_jitter(np.arange(len(param)))
            ax.errorbar(x=x_jittered, 
                        y=tmp[y], 
                        yerr=[tmp['yerr_lower'], tmp['yerr_upper']], 
                        fmt='o', 
                        color=pal[j], 
                        label=sub,
                        linewidth=1,
                        capsize=3,
                        markersize=3,
                        )
            ax.set_ylabel('')
            ax.set_xlabel('')
        if len(param) > 1:
            ax.margins(x=0.2)
        else:
            ax.margins(x=0.1)
        if i == 0:
            ax.set(xlim=[-1,1], xticks=[0])
        else:
            ax.set(xlim=[-0.4,1.4], xticks=np.arange(len(param)))
        ax.set_xticklabels(param)

        if ylim_list is not None:
                ax.set_ylim(ylim_list[i])
        if yticks_list is not None:
            ax.set_yticks(yticks_list[i])

    axes[0].set_ylabel('Parameter estimates')
    axes[2].set_ylabel('Parameter estimates')
    axes[3].set_xlabel('Parameter')

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.8, hspace=0.3)

    utils.save_fig(save_path)
    #return fig, axes

def _replace_param_names_with_latex(params_list):
    """
    Replace entries in params_list with LaTeX-formatted names for plotting.
    """
    new_list = {
        'sigma': r"$\sigma$",
        'slope': r"$Slope$" "\n" r"$m$",
        'intercept': r"$Intercept$" "\n" r"$b$",
        'p_1': r"$p_1$",
        'p_2': r"$p_2$",
        'p_3': r"$p_3$",
        'p_4': r"$p_4$",
        'A_1': r"$A_1$",
        'A_2': r"$A_2$"
    }
    return [[new_list.get(param, param) for param in sublist] for sublist in params_list]

def plot_model_comparison_params(model_df, 
                                 params_list,
                                 hue='sub', 
                                 ax=None, 
                                 save_path=None, 
                                 weighted_average=False, 
                                 ylim=None, yticks=None,
                                 **kwargs):
    """
    Plot model parameters
    """ 
    rc.update({
        'axes.titlesize': 11,
        'axes.titlepad': 20,
        'axes.labelsize': 11,
        'ytick.labelsize': 9,
        'xtick.labelsize': 11,
        'legend.title_fontsize': 11,
        'legend.fontsize': 11,
        })
    sns.set_theme("paper", style='ticks', rc=rc)
    # Get columns that are not in params_list
    flat_params_list = [param for sublist in params_list for param in sublist]
    non_param_columns = [col for col in model_df.columns if col not in flat_params_list]
    model_long_df = model_df.melt(id_vars=non_param_columns,
                                  var_name='param', 
                                  value_name='value')
    if weighted_average:
        model_long_df['value_and_weights'] = model_long_df.apply(lambda row: row.value + row.precision * 1j, axis=1)
        kwargs['estimator'] = weighted_mean
        y = 'value_and_weights'
    else:
        y = 'value'
    model_long_df['param'] = _change_params_to_math_symbols(model_long_df['param'])
    params_list = _replace_param_names_with_latex(params_list)
    if ax is None:
        fig, axes = plt.subplots(1,len(params_list), figsize=(8.5, 2.5), 
                                 gridspec_kw={'width_ratios': [1,2,1.7,1.5,1.5]})
    for i, ax in enumerate(axes.flatten()):
        tmp_param = params_list[i]
        tmp = model_long_df.query(f'param in @tmp_param')
        
        sns.pointplot(ax=ax, data=tmp, linestyles='',
                      x='param', y=y, scale=1, 
                      errorbar=('ci', 68),
                      order=params_list[i], 
                      hue=hue, dodge=0.5,
                      **kwargs)
        
        ax.set_xlabel('')
        ax.get_legend().remove()
        if i >= 2:
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)
            ax.margins(x=0.05)
        if i == 1:    
            ax.margins(x=0.04)
            ax.tick_params(axis='x', labelsize=rc['xtick.labelsize']-0.5, pad=5)
        if i == 0:
            ax.set_ylabel('Parameter estimates')
        else:
            ax.set_ylabel('')

    if ylim is not None:
        for i,ax in enumerate(axes.flatten()):
            ax.set(ylim=ylim[i])
    if yticks is not None:
        for i,ax in enumerate(axes.flatten()):
            ax.set(yticks=yticks[i])
    #ax.legend(loc='center left', title='Model type', bbox_to_anchor=(1.02, 0.7), frameon=False)
    fig.subplots_adjust(wspace=0.8, top=0.9)
    if save_path:
        utils.save_fig(save_path)
