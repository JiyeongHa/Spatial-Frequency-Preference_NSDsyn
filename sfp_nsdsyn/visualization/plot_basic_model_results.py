import sys
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
import pandas as pd
from sfp_nsdsyn.two_dimensional_model import group_params
import matplotlib as mpl

rc = {'text.color': 'black',
      'axes.labelcolor': 'black',
      'xtick.color': 'black',
      'ytick.color': 'black',
      'axes.edgecolor': 'black',
      'font.family': 'Helvetica',
      'figure.dpi': 72 * 2,
      'savefig.dpi': 72 * 4
      }
mpl.rcParams.update(rc)


def plot_loss_history(loss_history_df,
                      hue=None, lgd_title=None, hue_order=None,
                      height=3, save_path=None, row=None,
                      log_y=True, sharey=False, errorbar=None,
                      suptitle=None, **kwargs):

    sns.set_theme(style="ticks", context='notebook', rc=rc, font_scale=1)

    grid = sns.FacetGrid(loss_history_df,
                         hue=hue, row=row,
                         hue_order=hue_order,
                         height=height,
                         legend_out=True,
                         aspect=2.5,
                         sharey=sharey,
                         **kwargs)
    g = grid.map(sns.lineplot, 'epoch', 'loss', linewidth=2, errorbar=errorbar)
    if log_y is True:
        g.set(yscale='log')
    grid.set_axis_labels('Epoch', 'Loss')
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    if row is not None:
        for subplot_title, ax in grid.axes_dict.items():
            ax.set_title(f"{subplot_title.title()}")
    if suptitle is not None:
        grid.fig.suptitle(suptitle, fontweight="bold")
        grid.fig.subplots_adjust(top=0.85)
    utils.save_fig(save_path)


def plot_param_history(model_history_df, params,
                       hue=None, hue_order=None, lgd_title=None,
                       suptitle=None, errorbar=None, height=5,
                       save_path=None, sharey=False,
                       log_y=False, **kwargs):
    rc = {'text.color': 'black',
          'axes.titleweight': "bold",
          "axes.spines.right": False,
          "axes.spines.top": False,
          'font.family': 'Helvetica'}
    sns.set_theme(style="ticks", context='notebook', rc=rc, font_scale=2)
    id_cols = [a for a in model_history_df.columns.tolist() if a not in params]
    melted_df = pd.melt(model_history_df, id_vars=id_cols, value_vars=params, var_name='parameter', value_name='value')
    grid = sns.FacetGrid(melted_df,
                         hue=hue, row='parameter',
                         hue_order=hue_order,
                         height=height,
                         legend_out=True,
                         aspect=2.5,
                         sharex=True,
                         sharey=sharey,
                         **kwargs)
    g = grid.map(sns.lineplot, 'epoch', "value", errorbar=errorbar, linewidth=2)
    if log_y is True:
        g.set(yscale='log')
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    grid.set_axis_labels('Epoch', 'Parameter value')
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    if suptitle is not None:
        grid.fig.suptitle(suptitle, fontweight="bold")
        grid.fig.subplots_adjust(top=0.85)
    utils.save_fig(save_path)
