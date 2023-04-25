import sys
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
import pandas as pd




def plot_loss_history(loss_history_df,
                      to_label=None, lgd_title=None, label_order=None,
                      to_col=None, height=5, save_path=None, log_y=True, sharey=False):
    sns.set_context("notebook", font_scale=1.5)
    x_label = 'Epoch'
    y_label = 'Loss'
    grid = sns.FacetGrid(loss_history_df,
                         hue=to_label,
                         hue_order=label_order,
                         col=to_col,
                         height=height,
                         palette=sns.color_palette("rocket", loss_history_df[to_label].nunique()),
                         legend_out=True,
                         sharex=True, sharey=sharey)
    g = grid.map(sns.lineplot, 'epoch', 'loss', linewidth=2, ci=68)
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_path)




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
