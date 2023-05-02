import sys
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
import pandas as pd
from sfp_nsdsyn.two_dimensional_model import group_params



def plot_loss_history(loss_history_df,
                      hue=None, lgd_title=None, hue_order=None,
                      col=None, height=4, save_path=None,
                      log_y=True, sharey=False,
                      suptitle=None, **kwargs):
    sns.set_context("notebook", font_scale=1.5)
    grid = sns.FacetGrid(loss_history_df,
                         hue=hue,
                         hue_order=hue_order,
                         row=col,
                         height=height,
                         legend_out=True,
                         aspect=2.5,
                         sharey=sharey,
                         **kwargs)
    g = grid.map(sns.lineplot, 'epoch', 'loss', linewidth=2, ci=None)
    if log_y is True:
        g.set(yscale='log')
    grid.set_axis_labels('Epoch', 'Loss')
    if lgd_title is not None:
        g.add_legend(title=lgd_title)
    if col is not None:
        for subplot_title, ax in grid.axes_dict.items():
            ax.set_title(f"{subplot_title.title()}")
    if suptitle is not None:
        grid.fig.suptitle(suptitle, fontweight="bold")
        grid.fig.subplots_adjust(top=0.85)
    utils.save_fig(save_path)


def plot_param_history(df, params, ground_truth=False,
                       hue=None, hue_order=None, lgd_title=None,
                       col=None, suptitle=None,
                       save_path=None, height=5,
                       log_y=False, **kwargs):
    sns.set_context("notebook", font_scale=1.5)
    x_label = "Epoch"
    y_label = "Parameter value"
    grid = sns.FacetGrid(group_params(df, params, np.arange(0, len(params))),
                         hue=hue,
                         hue_order=hue_order,
                         col=col,
                         row='params',
                         height=height,
                         legend_out=True,
                         aspect=2.5,
                         sharex=True, sharey=False, **kwargs)
    g = grid.map(sns.lineplot, 'epoch', "value", linewidth=2, ci=None)
    if log_y is True:
        g.set(yscale='log')
    if col is None:
        for subplot_title, ax in grid.axes_dict.items():
            ax.set_title(f"{subplot_title.title()}")
    if ground_truth is True:
        #TODO: ground truth is for simulation
        pass
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        g.add_legend(title=lgd_title)
    if suptitle is not None:
        grid.fig.suptitle(suptitle, fontweight="bold")
        grid.fig.subplots_adjust(top=0.95)
    utils.save_fig(save_path)
