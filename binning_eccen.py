import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import os
import numpy as np
import pandas as pd
import seaborn as sns
import sfp_nsd_utils as utils
from matplotlib import pyplot as plt

def label_eccband(row):
    if row.eccentricity <= 6:
        return np.floor(row.eccentricity)
def _load_and_copy_df(df_path, create_vroinames_col, selected_cols, remove_cols=False):
    """load a dataframe of one subject and return a copy of part of the dataframe with selected columns.
     the selected_cols has to include frequency level, beta, local orientation, and local frequency."""
    # import dataframe from make_df.py
    df = pd.read_csv(df_path)
    if create_vroinames_col:
        # label visual areas regardless of dorsal and ventral streams.
        df['vroinames'] = df.apply(label_Vareas, axis=1)
        selected_cols.insert(1, 'vroinames')
    if remove_cols:
        tmp_selected_cols = [i for i in list(df.keys()) if i not in selected_cols]
        selected_cols = tmp_selected_cols

    # copy df with needed columns
    selected_df = df[selected_cols].copy()
    return selected_df
def _get_df_for_each_ROI(selected_df, roi_list, full_roi=True):
    """ split the dataframe by visual ROIs.
    full_roi means a whole visual area such as V1, V2, .. instead of partial of them (e.g. dorsal V1, ventral V2)
    check with roi_df.items() if all visual areas are stored as an individual dataframe."""
    if not roi_list:
        roi_list = selected_df.vroinames.unique()
    roi_df = {}
    if full_roi:
        for cur_roi in roi_list:
            exist = cur_roi in selected_df.vroinames.unique()
            if not exist:
                raise ValueError(f'{cur_roi} does not exist in the dataframe.\n')
            roi_df[cur_roi] = selected_df.groupby('vroinames').get_group(cur_roi)

    return roi_df

def _summary_stat_for_each_ecc_bin(roi_df, bin_group, central_tendency):
    """get mean betas for each eccen values"""

    mean_df = {}
    roi_list = roi_df.keys()
    for cur_roi in roi_list:
        mean_df[cur_roi] = roi_df[cur_roi].groupby(bin_group).agg(central_tendency).reset_index()
        # this should be fixed for cases when I want to get more than two central tendencies.
        mean_df[cur_roi].columns = mean_df[cur_roi].columns.get_level_values(0)

    mean_df = pd.concat(mean_df).reset_index().drop(columns=['level_1']).rename(columns={"level_0": "vroinames"})

    return mean_df
def _sort_vroinames(df_vroinames):
    roi_list = df_vroinames.unique().tolist()
    if all(isinstance(item, str) for item in roi_list):
        roi_list.sort(key=lambda x: int(x[1]))
    if all(isinstance(item, float) for item in roi_list):
        roi_list.sort(key=lambda x: int(x))

    return roi_list
def bin_subject(sn,
                df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes',
                df_file_name='stim_voxel_info_df.csv',
                create_vroinames_col=False,
                cols_to_select=["names", "avg_betas", "vroinames", "eccrois", "local_ori", "local_sf", "freq_lvl"],
                roi_to_bin=None,
                dv_to_group=["eccrois", "freq_lvl"],
                central_tendency=["mean"]):
    subj = utils.sub_number_to_string(sn)
    df_path = os.path.join(df_dir, subj + '_' + df_file_name)
    selected_df = _load_and_copy_df(df_path=df_path,
                                    create_vroinames_col=create_vroinames_col,
                                    selected_cols=cols_to_select)
    roi_df = _get_df_for_each_ROI(selected_df, roi_list=roi_to_bin, full_roi=True)
    mean_df = _summary_stat_for_each_ecc_bin(roi_df, bin_group=dv_to_group, central_tendency=central_tendency)
    mean_df['subj'] = subj

    return mean_df

def get_all_subj_df(subjects_to_run=np.arange(1, 9),
                    central_tendency=["mean"],
                    dv_to_group=["subj", "vroinames", "eccrois", "freq_lvl"],
                    save_avg_df=False,
                    df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes',
                    df_file_name='stim_voxel_info_all_subj_mean_df.csv'):
    """load each subject's dataframe and bin according to eccentricity,  """
    all_subj_df = []
    for sn in subjects_to_run:
        mean_df = bin_subject(sn=sn, central_tendency=central_tendency)
        all_subj_df.append(mean_df)
    all_subj_df = pd.concat(all_subj_df, ignore_index=True)
    all_subj_df = all_subj_df.groupby(dv_to_group).mean().reset_index()
    if ("subj" in dv_to_group) == False:
        all_subj_df['subj'] = 'avg'
    if save_avg_df:
        all_subj_path = os.path.join(df_dir, df_file_name)
        all_subj_df.to_csv(all_subj_path, index=False)

    return all_subj_df

def get_avgerage_all_subj_df(all_subj_df,
                             save_avg_df=False,
                             df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes',
                             df_file_name='stim_voxel_info_all_subj_mean_df.csv'):
    all_subj_df = all_subj_df.groupby(["vroinames", "eccrois", "freq_lvl"]).mean().reset_index()
    all_subj_df['subj'] = 'avg'
    if save_avg_df:
        all_subj_path = os.path.join(df_dir, df_file_name)
        all_subj_df.to_csv(all_subj_path, index=False)
    return all_subj_df
def scatterplot_2D(mean_df,
            subj,
            labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
            x_axis_col="local_sf",
            y_axis_col="avg_betas",
            x_axis_rename="Spatial Frequency",
            y_axis_rename="Beta values",
            col_to_subplot='vroinames',
            hue_to_label="eccrois",
            n_of_subplots_in_row=4,
            legend_out=True,
            x_log_scale=True,
            save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
            save_file_name='.png',
            save_fig=True):
    roi_order = _sort_vroinames(mean_df[col_to_subplot])
    grid = sns.FacetGrid(mean_df,
                         col=col_to_subplot,
                         col_order=roi_order,
                         hue=hue_to_label,
                         palette=sns.color_palette("husl"),
                         col_wrap=n_of_subplots_in_row,
                         legend_out=legend_out,
                         xlim=[10 ** -1, 10 ** 2],
                         sharex=True, sharey=True)
    grid.map(sns.scatterplot, x_axis_col, y_axis_col)
    grid.set_axis_labels(x_axis_rename, y_axis_rename)
    grid.fig.legend(title='Eccentricity', bbox_to_anchor=(1, 1), labels=labels)
    for roi, ax in grid.axes_dict.items():
        ax.set_title(f"{roi.title()}")
    grid.tight_layout()
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    grid.fig.suptitle(f'{subj}', fontweight="bold")
    if x_log_scale:
        plt.xscale('log')
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_rename + '_afo_' + x_axis_rename)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, subj + save_file_name)
        plt.savefig(save_path)
    plt.show()

    return grid
def barplot_2D(mean_df,
               vroi_list=["V1"],
               ecc_list=None,
               ecc_labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
               x_axis_col="freq_lvl",
               y_axis_col="avg_betas",
               x_axis_rename="Spatial Frequency Level",
               y_axis_rename="Beta values",
               col_to_subplot='eccrois',
               legend_out=True,
               err='sd',
               capsize=.2,
               save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
               save_file_name='.png',
               save_fig=False):

    # change frequency level type to string, because it's not plotting zero level
    subj = mean_df['subj'].unique()
    if len(subj) > 1 or subj == ["avg"]:
        subj = "averaged (N = %d)" % len(subj)
    elif len(subj) == 1:
        subj = subj[0]

    if ecc_list is None:
        ecc_list =_sort_vroinames(mean_df['eccrois'])
    if vroi_list is None:
        vroi_list = _sort_vroinames(mean_df['vroinames'])

    mean_df = mean_df[mean_df['eccrois'].isin(ecc_list) & mean_df['vroinames'].isin(vroi_list)]
    freq_levels = _sort_vroinames(mean_df['freq_lvl'])
    n_of_subplots_in_row = len(ecc_list)
    # plot on figure per visual ROI.
    # for one figure, we will have 5 subplots for each eccentricity roi.
    for cur_roi in vroi_list:
        plt.close()
        roi_filtered_df = mean_df.query('vroinames == @cur_roi')
        grid = sns.FacetGrid(roi_filtered_df,
                             col=col_to_subplot,
                             col_order=ecc_list,
                             hue=col_to_subplot,
                             palette=sns.color_palette("husl"),
                             col_wrap=n_of_subplots_in_row,
                             legend_out=legend_out,
                             sharey=True)
        grid.map(sns.barplot, x_axis_col, y_axis_col, order=freq_levels, ci=err, capsize=capsize)
        grid.set_axis_labels(x_axis_rename, y_axis_rename)
        for ecc, ax in grid.axes_dict.items():
            ax.set_title(f"{ecc_labels[int(ecc-1)]}")
        grid.tight_layout()
        grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
        grid.fig.suptitle(f'{subj}, {cur_roi.title()}', fontweight="bold")
        if save_fig:
            if not save_dir:
                raise Exception("Output directory is not defined!")
            fig_dir = os.path.join(save_dir, y_axis_rename.replace(" ", "-") + '_afo_' + x_axis_rename.replace(" ", "-"))
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            save_path = os.path.join(fig_dir, cur_roi + save_file_name)
            plt.savefig(save_path)
        plt.show()

    return grid
def lineplot_2D(mean_df,
            subj,
            labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
            x_axis_col="local_sf",
            y_axis_col="avg_betas",
            x_axis_rename="Spatial Frequency",
            y_axis_rename="Beta values",
            col_to_subplot='vroinames',
            hue_to_label="eccrois",
            n_of_subplots_in_row=4,
            estimator='mean',
            marker='o',
            legend_out=True,
            x_log_scale=True,
            save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
            save_file_name='.png',
            save_fig=True):
    roi_order = _sort_vroinames(mean_df[col_to_subplot])
    grid = sns.FacetGrid(mean_df,
                         col=col_to_subplot,
                         col_order=roi_order,
                         hue=hue_to_label,
                         palette=sns.color_palette("husl"),
                         col_wrap=n_of_subplots_in_row,
                         legend_out=legend_out,
                         xlim=[10 ** -1, 10 ** 2],
                         sharex=True, sharey=True)
    grid.map(sns.lineplot, x_axis_col, y_axis_col, marker=marker)
    grid.set_axis_labels(x_axis_rename, y_axis_rename)
    grid.fig.legend(title='Eccentricity', bbox_to_anchor=(1, 1), labels=labels)
    for roi, ax in grid.axes_dict.items():
        ax.set_title(f"{roi.title()}")
    grid.tight_layout()
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    grid.fig.suptitle(f'{subj}', fontweight="bold")
    if x_log_scale:
        plt.xscale('log')
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_rename + '_afo_' + x_axis_rename)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, subj + save_file_name)
        plt.savefig(save_path)
    plt.show()

    return grid
