import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import os
import numpy as np
import pandas as pd
import seaborn as sns
import sfp_nsd_utils as utils
from matplotlib import pyplot as plt

def plot_sfp(subj, df, to_subplot="vroinames", err=68,
             n_sp_low=4, legend_out=True, x_axis_label="Eccentricity",
             y_axis_label="Preferred Period (deg)",
             save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
             save_file_name='.png'):
    if len(subj) > 1:
        fig_title = f'N = {len(subj)}'
        sn = f'avg_n_{len(subj)}'
    elif len(subj) == 1:
        fig_title = utils.sub_number_to_string(subj)
        sn = utils.sub_number_to_string(subj)
    subplot_labels = utils.sort_a_df_column(df[to_subplot])
    #x_subplot = "V1"
    cur_df = df.copy()
    ecc_labels = ['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°']
    replace_values = {1.0: 0.5, 2.0: 0.75, 3.0: 1.5, 4.0: 3, 5.0: 4.5}
    cur_df.loc[:, "ecc_x"] = cur_df.loc[:, "eccrois"]
    cur_df = cur_df.replace({"ecc_x":replace_values})
    cur_df.loc[:, "sfp"] = np.divide(1, cur_df.loc[:, "mode"])
    grid = sns.FacetGrid(cur_df,
                         hue=to_subplot,
                         col=to_subplot,
                         col_order=subplot_labels,
                         col_wrap=n_sp_low,
                         palette=sns.color_palette("husl"),
                         legend_out=legend_out,
                         xlim=[0, 5],
                         sharey=False)
    grid.map(sns.lineplot, "ecc_x", "sfp",
             ci=err, marker='o', err_style="bars", color=None)
    broderick_findings = np.arange(0,11)*(1/6.2) + 0.3
    grid.axes[0].plot(broderick_findings, color="black", label="Broderick et al V1 findings")
    lgd = grid.fig.legend(bbox_to_anchor=(1.04,0.8), loc="upper left")
    grid.axes[1].plot(broderick_findings, color="black")
    grid.axes[2].plot(broderick_findings, color="black")
    grid.axes[3].plot(broderick_findings, color="black")
    plt.xticks([0.5, 0.75, 1.5, 3, 4.5], ['0.5°', '0.75°', '1.5°', '3°', '4.5°'])
    grid.set_axis_labels(x_axis_label, y_axis_label)
    for i in range(0, len(grid.axes)):
        grid.axes[i].tick_params(labelrotation=45)
    for roi, ax in grid.axes_dict.items():
        ax.set_title(f"{roi.title()}")
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    grid.fig.suptitle(f'{fig_title}', fontsize=18, fontweight="bold")
    #grid.tight_layout()
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, sn + save_file_name)
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return grid