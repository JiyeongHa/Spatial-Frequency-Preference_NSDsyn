import os
import seaborn as sns
from sfp_nsdsyn import utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sfp_nsdsyn.two_dimensional_model import group_params
from sfp_nsdsyn.preprocessing import calculate_local_orientation
from sfp_nsdsyn.two_dimensional_model import get_Pv_row

mpl.rcParams.update(mpl.rcParamsDefault)

rc = {'text.color': 'black',
      'axes.labelcolor': 'black',
      'xtick.color': 'black',
      'ytick.color': 'black',
      'axes.edgecolor': 'black',
      'font.family': 'Helvetica',
      'font.size': 12,
      'axes.titlesize': 13,
      'axes.labelsize': 15,
      'xtick.labelsize': 12,
      'ytick.labelsize': 12,
      'legend.title_fontsize': 15,
      'legend.fontsize': 15,
      'figure.titlesize': 15,
      'figure.dpi': 72 * 3,
      'savefig.dpi': 72 * 4
      }
mpl.rcParams.update(rc)

def weighted_mean(x, **kws):
    """store weights as imaginery number"""
    return np.sum(np.real(x) * np.imag(x)) / np.sum(np.imag(x))

def violinplot_parameters(df, params, subplot_group,
                                                 save_fig=False,
                                                 save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/params.png'):
    sns.set_context("notebook", font_scale=2.5)
    groups, counts = np.unique(subplot_group, return_counts=True)
    df = group_params(df, params, subplot_group)
    grid = sns.FacetGrid(df,
                         col="group",
                         height=8,
                         legend_out=True,
                         sharex=False, sharey=False, gridspec_kws={'width_ratios': counts})
    pal = sns.diverging_palette(145,300,s=100, l=65, n=2, as_cmap=False)
    grid = grid.map(sns.violinplot, "params", "value", 'dset', hue_order=['nsdsyn','broderick'], split=True,
                    palette=pal, inner='stick', linewidth=3, saturation=0.5)
    axes = grid.axes
    axes[0, 0].margins(x=0.2)
    axes[0, 1].margins(x=0.2)
    axes[0, 2].margins(x=0.05)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f" ")
    grid.add_legend()
    grid.set_axis_labels('', 'Value')
    utils.save_fig(save_fig, save_path)


def _change_params_to_math_symbols(params_col):
    params_col = params_col.replace({'sigma': r"$Bandwidth$" "\n" r"$\sigma$",
                                     'slope': r"$Slope$" "\n" r"$a$",
                                     'intercept': r"$Intercept$" "\n" r"$b$",
                                     'p_1': r"$Horizontal$" "\n" r"$p_1$",
                                     'p_2': r"$Oblique$" "\n"r"$p_2$",
                                     'p_3': r"$Radial$" "\n"r"$p_3$",
                                     'p_4': r"$Spiral$" "\n"r"$p_4$",
                                     'A_1': r"$Horizontal$" "\n"r"$A_1$",
                                     'A_2': r"$Oblique$" "\n"r"$A_2$"})
    return params_col

def _find_ylim(ax, roi, avg=True):
    if avg is True:
        if roi == "V1":
            switcher = {0: [0, 2.6],
                        1: [0, 0.5],
                        2: [-0.2, 0.2]}
        elif roi == "V2":
            switcher = {0: [1, 4],
                        1: [0, 0.5],
                        2: [-0.4, 0.2]}
        elif roi == "V3":
            switcher = {0: [2, 5],
                        1: [0, 0.5],
                        2: [-0.6, 0.2]}
        else:
            switcher = {0: [2, 5],
                        1: [0, 0.3],
                        2: [-0.15, 0.1],
                        3: [-0.7, 0.3],
                        4: [-0.05, 0.05]}
        return switcher.get(ax)
    elif avg is False:
        if roi == "V1":
            switcher = {0: [1.5, 3.1],
                        1: [0, 0.3],
                        2: [0.0, 0.5],
                        3: [-0.4, 0.2]}
        elif roi == "V2":
            switcher = {0: [2.5, 4.5],
                        1: [0.05, 0.35],
                        2: [0.0, 0.5],
                        3: [-0.4, 0.4]}
        elif roi == "V3":
            switcher = {0: [1.5, 6.5],
                        1: [0, 0.5],
                        2: [0.0, 0.5],
                        3: [-0.4, 0.7]}
        else:
            switcher = {0: [1.8, 7],
                        1: [0, 0.5],
                        2: [-0.03, 0.5],
                        3: [-1.5, 0.5]}
        return switcher.get(ax)


def plot_precision_weighted_avg_parameters(df, params, subplot_group,
                                           hue, hue_order=None, lgd_title=None,
                                           weight='precision', dodge=0.14,
                                           save_path=None, pal=None, dot_scale=1,
                                           width=7, suptitle=None, ylim_list=None, ytick_list=None, **kwargs):
    rc.update({'axes.labelpad': 8,
               'xtick.labelsize': 13})
    sns.set_theme("notebook", style='ticks', rc=rc, font_scale=1)
    df = group_params(df, params, subplot_group)
    df['params'] = _change_params_to_math_symbols(df['params'])
    df['value_and_weights'] = [v + w*1j for v, w in zip(df.value, df[weight])]
    groups, counts = np.unique(subplot_group, return_counts=True)
    counts[0] = 1.5
    counts[1] = 2.8
    counts[2] = 2.6
    counts[3] = 2
    counts[4] = 2.6
    if pal is None:
        pal = sns.cubehelix_palette(n_colors=df[hue].nunique()+1, as_cmap=False, reverse=True)
    grid = sns.FacetGrid(df,
                         col="group",
                         height=utils.get_height_based_on_width(7, 0.59),
                         legend_out=True,
                         sharex=False, sharey=False,
                         aspect=0.59,
                         gridspec_kws={'width_ratios': counts}, **kwargs)

    g = grid.map(sns.pointplot, "params", "value_and_weights", hue, hue_order=hue_order,
                 dodge=dodge, palette=pal, estimator=weighted_mean, linestyles='', scale=dot_scale,
                 orient="v", errorbar=("ci", 68))
    for ax in grid.axes.flatten():
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        if any('p_' in s for s in ticks) or any('A_' in s for s in ticks):
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.9, zorder=0)
        if len(ticks) == 2:
            ax.margins(x=0.22)
    grid.axes[0, 2].margins(x=0.1)
    grid.axes[0, 4].margins(x=0.1)
    if ylim_list is not None:
        for ax in range(len(groups)):
            grid.axes[0, ax].set_ylim(ylim_list[ax])
    if ytick_list is not None:
        for ax in range(len(groups)):
            grid.axes[0, ax].set_yticks(ytick_list[ax])
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f" ")
    grid.set_axis_labels("", 'Value')
    if lgd_title is not None:
        g.add_legend(title=lgd_title, bbox_to_anchor=(1,0.725))
    if suptitle is not None:
        g.fig.suptitle(suptitle, fontweight="bold")
    grid.set_axis_labels("", "Parameter value")
    utils.save_fig(save_path)
    return grid


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

def plot_preferred_period(df,
                          x, y='Pv', precision='precision',
                          hue=None, hue_order=None,
                          col=None, col_wrap=None,
                          lgd_title=None, height=7,
                          xlim=(0,10),
                          projection=None, save_path=None,
                          **kwarg):
    sns.set_context("notebook", font_scale=2)
    if projection == 'polar':
        despine = False
        xticks = []
    else:
        despine = True
        xticks=[0, 5, 10]
    yticks = [0, 0.5, 1, 1.5, 2]
    rc = {'axes.labelpad': 15,
          'axes.linewidth': 2,
          'axes.titlepad': 40,
          'axes.titleweight': "bold",
          'xtick.major.pad': 8,
          'ytick.major.pad': 8,
          'xtick.major.width': 2,
          'xtick.minor.width': 2,
          'ytick.major.width': 2,
          'xtick.major.size': 7,
          'ytick.major.size': 7,
          'grid.linewidth': 2,
          'font.family': 'Helvetica',
          'lines.linewidth': 2}
    utils.set_rcParams(rc)
    x_label = x.title()
    y_label = "Preferred period"
    df['value_and_weights'] = [v + w * 1j for v, w in zip(df[y], df[precision])]
    # plotting average of prediction, not the prediction of average
    if hue_order is None:
        pal = utils.get_colors(hue, df[hue].unique())
    else:
        pal = utils.get_colors(to_color=hue, to_plot=hue_order)
    grid = sns.FacetGrid(df,
                         hue=hue, palette=pal,
                         hue_order=hue_order,
                         height=height,
                         col=col, col_wrap=col_wrap,
                         aspect=1.2,
                         subplot_kws={'projection': projection},
                         legend_out=True, despine=despine,
                         sharex=True, sharey=True,
                         **kwarg)
    grid = grid.map(sns.lineplot, x, "value_and_weights",
                    linewidth=2, estimator=weighted_mean,
                    n_boot=100, err_style='band', ci=68)
    grid.set_axis_labels(x.title(), y_label)
    grid.set(xlim=xlim, xticks=xticks, yticks=yticks)
    utils.set_rcParams(rc)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    utils.save_fig(save_path)
    return grid


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


def scatter_comparison(df, x, y, col, col_order,
                       to_label="study_type", lgd_title="Study", label_order=None,
                       height=7, x_label="Broderick et al.(2022)", y_label="My_value",
                       save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/params.png'):
    sns.set_context("notebook", font_scale=1.5)
    pal = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
           (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
           (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
           (189, 189, 189)]
    n_labels = df[to_label].nunique()
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    pal = sns.color_palette(np.array(pal) / 255, n_labels)
    grid = sns.relplot(data=df, x=x, y=y, kind="scatter",
                       col=col, col_wrap=3, col_order=col_order,
                       palette=pal, facet_kws={'sharey': False, 'sharex': False},
                       hue=to_label, hue_order=label_order,
                       height=height, s=100)
    for _, ax in grid.axes_dict.items():
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        newlim = [min(x0, y0), max(x1, y1)]
        lims = [newlim[0], newlim[1]]
        ax.plot(lims, lims, '--k', linewidth=2)
        ax.set_xlim(xmin=newlim[0], xmax=newlim[1])
        ax.set_ylim(ymin=newlim[0], ymax=newlim[1])
        ax.set_xticks(np.round(np.linspace(newlim[0], newlim[1], 4), 1))
        ax.set_yticks(np.round(np.linspace(newlim[0], newlim[1], 4), 1))
        ax.axis('scaled')
    grid.set_axis_labels(x_label, y_label)
    #grid.fig.legend(title=lgd_title, labels=label_order)
    utils.save_fig(save_fig, save_path)
    return grid

def scatterplot_two_avg_params(x_df, y_df, params_list, params_group, x_label='Broderick et al.(2022) values',
                               y_label = 'My values',
                               save_fig=False, save_path='/Volumes/server/Projects/sfp_nsd/derivatives/figures/sfp_model/results_2D/scatterplot.png'):
    params_order = np.arange(0, len(params_list))
    colors = mpl.cm.tab10(np.linspace(0, 1, len(params_list)))
    n_subplots = len(set(params_group))
    fig, axes = plt.subplots(1, n_subplots, figsize=(20, 5), dpi=300)
    axes[2].axvline(x=0, color='gray', linestyle='--')
    axes[2].axhline(y=0, color='gray', linestyle='--')
    axes[3].axvline(x=0, color='gray', linestyle='--')
    axes[3].axhline(y=0, color='gray', linestyle='--')
    for g in range(n_subplots):
        tmp_params_list = [i for (i, v) in zip(params_list, params_group) if v == g]
        tmp_params_order = [i for (i, v) in zip(params_order, params_group) if v == g]
        for p, c in zip(tmp_params_list, tmp_params_order):
            x = x_df.query('params == @p')['mean_value']
            xerr = x_df.query('params == @p')['std_value']
            y = y_df.query('params == @p')['mean_value']
            yerr = y_df.query('params == @p')['std_value']
            axes[g].errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=colors[c], ecolor=colors[c], label=p)
            axes[g].legend(loc='best', ncol=1)
        axes[g].axis('scaled')
        newlim = _get_common_lim(axes[g])
        if g == 0:
            newlim = [1.5, 2.5]
        elif g == 1:
            newlim = [0, 0.41]
        elif g == 2:
            newlim = [-0.17, 0.10]
        elif g == 3:
            newlim = [-0.05, 0.05]
        axes[g].set_xlim(newlim[0], newlim[1])
        axes[g].set_ylim(newlim[0], newlim[1])
        if (g == 2):
            axes[g].set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
            axes[g].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
        elif (g == 3):
            axes[g].set_xticks([-0.05 , -0.025,  0,  0.025,  0.05 ])
            axes[g].set_yticks([-0.05 , -0.025,  0,  0.025,  0.05 ])
        else:
            axes[g].set_xticks(np.round(np.linspace(newlim[0], newlim[1], 5), 2))
            axes[g].set_yticks(np.round(np.linspace(newlim[0], newlim[1], 5), 2))
        axes[g].plot(newlim, newlim, '--k', linewidth=2)
        control_fontsize(14, 20, 15)

    # common axis labels
    fig.supxlabel(x_label, fontsize=20)
    fig.supylabel(y_label, fontsize=20)
    plt.tight_layout(w_pad=2)
    fig.subplots_adjust(left=.09, bottom=0.15)
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


def SD_histogram(df, col='dset', save_fig=False, save_path='/',
                 col_wrap=None, **kwargs):
    grid = sns.FacetGrid(df,
                         height=7,
                         col=col,
                         col_order=['NSD','Broderick'],
                         hue="subj",
                         col_wrap=col_wrap,
                         palette=sns.color_palette("husl"),
                         sharex=False, sharey=False)
    grid = grid.map(sns.histplot, "sigma_v_squared", edgecolor='gray', linewidth=1,
                    stat="density", alpha=0.7, **kwargs)
    grid.set_axis_labels(r'$\sigma^2_v$', 'Density')
    axes = grid.axes
    axes[0, 0].set_xlim([0, 55])
    axes[0, 1].set_xlim([0, 2])
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(subplot_title)
    utils.save_fig(save_fig, save_path)
    return grid


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


def merge_continuous_values_to_the_df(df, val_range=(0,6), n=100, col_name='eccentricity', endpoint=True):

    val_range = np.linspace(val_range[0], val_range[-1], n, endpoint=endpoint)
    all_ecc_df = pd.DataFrame({})
    for val in val_range:
        df[col_name] = val
        all_ecc_df = all_ecc_df.append(df, ignore_index=True)
    return all_ecc_df

def get_w_a_and_w_r_for_each_stim_class(stim_description_path,
                                        stim_class=['pinwheel','annulus','forward spiral', 'reverse spiral']):
    stim_info = pd.read_csv(stim_description_path)
    stim_info = stim_info.query('names in @stim_class')
    stim_info = stim_info.drop_duplicates('names')
    stim_info = stim_info[['names', 'w_r', 'w_a']]
    return stim_info

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
                                                reference_frame='relative'):
    merged_df = make_synthetic_dataframe_for_2D(stim_info,
                                                ecc_range, n_ecc,
                                                angle_range, n_angle,
                                                ecc_col, angle_col)
    merged_df['local_ori'] = calculate_local_orientation(merged_df['w_a'],
                                                         merged_df['w_r'],
                                                         retinotopic_angle=merged_df[angle_col],
                                                         angle_in_radians=angle_in_radians,
                                                         reference_frame=reference_frame)
    merged_df['Pv'] = merged_df.apply(get_Pv_row, params=final_params, axis=1)
    if reference_frame == 'absolute':
        rename_cols = {'forward spiral': 'left oblique',
                       'reverse spiral': 'right oblique',
                       'annulus': 'horizontal',
                       'pinwheel': 'vertical'}
        merged_df = merged_df.replace({'names': rename_cols})
    merged_df['frame'] = reference_frame
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