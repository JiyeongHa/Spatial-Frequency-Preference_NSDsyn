import sys
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
import pandas as pd
from sfp_nsdsyn.two_dimensional_model import group_params



def plot_loss_history(loss_history_df,
                      hue=None, lgd_title=None, hue_order=None,
                      col=None, height=5, save_path=None,
                      log_y=True, sharey=False, **kwargs):
    sns.set_context("notebook", font_scale=1.5)
    x_label = 'Epoch'
    y_label = 'Loss'
    grid = sns.FacetGrid(loss_history_df,
                         hue=hue,
                         hue_order=hue_order,
                         row=col,
                         height=height,
                         legend_out=True,
                         aspect=3,
                         sharey=sharey,
                         **kwargs)
    g = grid.map(sns.lineplot, 'epoch', 'loss', linewidth=2, ci=None)
    if log_y is True:
        g.set(yscale='log')
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        g.add_legend(title=lgd_title)

    utils.save_fig(save_path)


def plot_param_history(df, params, group, ground_truth=False,
                       hue=None, hue_order=None, lgd_title=None,
                       save_path=None, height=5,
                       log_y=False):
    sns.set_context("notebook", font_scale=1.5)
    x_label = "Epoch"
    y_label = "Parameter value"
    grid = sns.FacetGrid(group_params(df, params, group),
                         hue=hue,
                         hue_order=hue_order,
                         row="params",
                         height=height,
                         palette=utils.color_palettes(hue, df[hue].nunique()),
                         legend_out=True,
                         aspect=3,
                         sharex=True, sharey=False)
    g = grid.map(sns.lineplot, 'epoch', "value", linewidth=2, ci=None)
    if log_y is True:
        grid.set(yscale='log')
    if ground_truth is True:
        #TODO: ground truth is for simulation
        pass
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    utils.save_fig(save_path)
