import os
import seaborn as sns
from sfp_nsdsyn import utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sfp_nsdsyn.two_dimensional_model import group_params


def plot_avg_parameters(df, params, subplot_group, height=6,
                        save_fig=False,
                        save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/params.png'):
    sns.set_context("notebook", font_scale=2)
    df = group_params(df, params, subplot_group)
    x_label = "Parameter"
    y_label = "Value"
    groups, counts = np.unique(subplot_group, return_counts=True)
    grid = sns.FacetGrid(df,
                         col="group",
                         height=height,
                         legend_out=True,
                         sharex=False, sharey=False, gridspec_kws={'width_ratios': counts})
    grid.map(sns.pointplot, "params", "value", estimator=np.mean, linestyles='',
             scale=1.5, alpha=0.9, color='k', orient="v", errorbar=("ci", 68))
    for ax in range(len(groups)):
        plt.setp(grid.axes[0, ax].collections, sizes=[200])
    axes = grid.axes
    axes[0, 1].margins(x=0.5)
    axes[0, 2].margins(x=0.1)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f" ")
    grid.set_axis_labels("", y_label)
    utils.save_fig(save_fig, save_path)
    return grid


def plot_individual_parameters(df, params, subplot_group, height=6, hue='subj',
                               save_fig=False,
                               save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/params.png'):
    sns.set_context("notebook", font_scale=2)
    df = group_params(df, params, subplot_group)
    x_label = "Parameter"
    y_label = "Value"
    groups, counts = np.unique(subplot_group, return_counts=True)
    grid = sns.FacetGrid(df,
                         col="group",
                         hue=hue,
                         hue_order=df[hue].unique(),
                         palette=sns.color_palette("husl", df[hue].nunique()),
                         height=height,
                         legend_out=True,
                         sharex=False, sharey=False, gridspec_kws={'width_ratios': counts})
    grid.map(sns.pointplot, "params", "value", estimator=np.mean, linestyles='',
             scale=1.5, alpha=0.9, orient="v", errorbar=("ci", 68))
    for ax in range(len(groups)):
        plt.setp(grid.axes[0, ax].collections, sizes=[200])
    axes = grid.axes
    axes[0, 1].margins(x=0.5)
    axes[0, 2].margins(x=0.1)
    grid.add_legend()
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f" ")
    grid.set_axis_labels("", y_label)
    utils.save_fig(save_fig, save_path)


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


def plot_loss_history(loss_history_df,
                      to_label=None, label_order=None, to_row=None, to_col=None, height=5,
                      lgd_title=None, save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/loss.png',
                       ci=68, n_boot=100, log_y=True, sharey=False):
    sns.set_context("notebook", font_scale=1.5)
    #sns.set(font_scale=1.5)
    x_label = 'Epoch'
    y_label = 'Loss'
    grid = sns.FacetGrid(loss_history_df,
                         hue=to_label,
                         hue_order=label_order,
                         row=to_row,
                         col=to_col,
                         height=height,
                         palette=sns.color_palette("rocket", loss_history_df[to_label].nunique()),
                         legend_out=True,
                         sharex=True, sharey=sharey)
    g = grid.map(sns.lineplot, 'epoch', 'loss', linewidth=2, ci=ci, n_boot=n_boot)
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    #grid.fig.legend(title=legend_title, bbox_to_anchor=(1, 1), labels=labels, fontsize=18)
    #grid.fig.suptitle(f'{title}', fontweight="bold")
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_fig, save_path)




def plot_param_history(df, params, group,
                       to_label=None, label_order=None, ground_truth=True, to_col=None,
                       lgd_title=None,
                       save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/.png',
                       ci=68, n_boot=100, log_y=True, sharey=True):
    df = group_params(df, params, group)
    sns.set_context("notebook", font_scale=1.5)
    x_label = "Epoch"
    y_label = "Parameter value"
    grid = sns.FacetGrid(df.query('lr_rate != "ground_truth"'),
                         hue=to_label,
                         hue_order=label_order,
                         row="params",
                         col=to_col,
                         height=7,
                         palette=sns.color_palette("rocket"),
                         legend_out=True,
                         sharex=True, sharey=sharey)
    g = grid.map(sns.lineplot, 'epoch', "value", linewidth=2, ci=ci, n_boot=n_boot)
    if ground_truth is True:
        for x_param, ax in g.axes_dict.items():
            #ax.set_aspect('auto')
            g_value = df.query('params == @x_param[0] & lr_rate == "ground_truth"').value.item()
            ax.axhline(g_value, ls="--", linewidth=3, c="black")
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    #grid.fig.suptitle(f'{title}', fontweight="bold")
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_fig, save_path)


def plot_param_history_horizontal(df, params, group,
                                  to_label=None, label_order=None, ground_truth=True,
                                  lgd_title=None, height=5, col_wrap=3,
                                  save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/.png',
                                  ci=68, n_boot=100, log_y=True):
    df = group_params(df, params, group)
    sns.set_context("notebook", font_scale=1.5)
    to_x = "epoch"
    to_y = "value"
    x_label = "Epoch"
    y_label = "Parameter value"
    pal = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
           (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
           (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
           (189, 189, 189)]
    n_labels = df[to_label].nunique()
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    pal = sns.color_palette(np.array(pal) / 255, n_labels)
    grid = sns.FacetGrid(df.query('lr_rate != "ground_truth"'),
                         hue=to_label,
                         hue_order=label_order,
                         col="params",
                         col_wrap=col_wrap,
                         height=height,
                         palette=pal,
                         legend_out=True,
                         sharex=True, sharey=False)
    g = grid.map(sns.lineplot, to_x, to_y, linewidth=2, ci=ci, n_boot=n_boot)
    if ground_truth is True:
        for x_param, ax in g.axes_dict.items():
            # ax.set_aspect('auto')
            g_value = df.query('params == @x_param & lr_rate == "ground_truth"').value.item()
            ax.axhline(g_value, ls="--", linewidth=2, c="black")
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    # grid.fig.suptitle(f'{title}', fontweight="bold")
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_fig, save_path)


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
    #grid.fig.subplots_adjust(top=0.85, right=0.75)  # adjust the Figure in rp
    #grid.fig.suptitle(f'{title}', fontweight="bold")
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


def _get_common_lim(axes):
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    return [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]


def control_fontsize(small, medium, large):
    plt.rc('font', size=small)  # controls default text sizes
    plt.rc('axes', titlesize=small, labelsize=medium)
    plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title


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