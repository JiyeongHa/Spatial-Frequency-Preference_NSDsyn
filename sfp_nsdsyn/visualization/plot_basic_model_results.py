import sys
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
import pandas as pd
from sfp_nsdsyn.two_dimensional_model import group_params



def plot_loss_history(loss_history_df,
                      hue=None, lgd_title=None, hue_order=None,
                      col=None, height=5, save_path=None, log_y=True, sharey=False):
    sns.set_context("notebook", font_scale=1.5)
    x_label = 'Epoch'
    y_label = 'Loss'
    grid = sns.FacetGrid(loss_history_df,
                         hue=hue,
                         hue_order=hue_order,
                         col=col,
                         height=height,
                         palette=sns.color_palette("rocket", loss_history_df[hue].nunique()),
                         legend_out=True,
                         sharex=True, sharey=sharey)
    g = grid.map(sns.lineplot, 'epoch', 'loss', linewidth=2, ci=68)
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_path)


def plot_param_history(df, ground_truth=False,
                       hue=None, hue_order=None, lgd_title=None,
                       col=None,
                       save_path=None,
                       log_y=True, sharey=False):
    sns.set_context("notebook", font_scale=1.5)
    x_label = "Epoch"
    y_label = "Parameter value"
    grid = sns.FacetGrid(df,
                         hue=hue,
                         hue_order=hue_order,
                         row="params",
                         col=col,
                         height=7,
                         palette=sns.color_palette("rocket"),
                         legend_out=True,
                         sharex=True, sharey=sharey)
    g = grid.map(sns.lineplot, 'epoch', "value", linewidth=2)
    if ground_truth is True:
        #TODO: ground truth is for simulation
        pass
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_path)
