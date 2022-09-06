import sys
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

def sub_number_to_string(sub_number, dataset="nsd"):
    """ Return number (1,2,3,..) to "subj0x" form """
    if dataset == "nsdsyn":
        return "subj%02d" % sub_number
    elif dataset == "broderick":
        return "sub-wlsubj{:03d}".format(sub_number)


def remove_subj_strings(subj_list):
    """ Remove 'subj' from the list and change the list type into integers """
    if not isinstance(subj_list, list):
        subj_list = subj_list.unique().tolist()
    num_list = [int(i.replace('subj', '')) for i in subj_list]
    return num_list

def sort_a_df_column(df_vroinames):
    """ Input should be the whole column of a dataframe.
    Sort a column that contains either strings or numbers in a descending order"""

    roi_list = df_vroinames.unique().tolist()
    if df_vroinames.name == 'vroinames':
        roi_list.sort(key=lambda x: int(x[1]))
    elif df_vroinames.name != 'vroinames':
        if all(isinstance(item, str) for item in roi_list):
            roi_list.sort()
        elif all(isinstance(item, float) for item in roi_list):
            roi_list.sort(key=lambda x: int(x))

    return roi_list

def load_df(sn, df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
            df_name='results_1D_model.csv', dataset="nsd"):
    subj = sub_number_to_string(sn, dataset)
    df_path = os.path.join(df_dir, subj + '_' + df_name)
    df = pd.read_csv(df_path)
    return df

def load_all_subj_df(subj_to_run,
                     df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                     df_name='results_1D_model.csv', dataset="nsd"):
    all_subj_df = []
    for sn in subj_to_run:
        tmp_df = load_df(sn, df_dir=df_dir, df_name=df_name, dataset=dataset)
        if not 'subj' in tmp_df.columns:
            tmp_df['subj'] = sub_number_to_string(sn)
        all_subj_df.append(tmp_df)
    all_subj_df = pd.concat(all_subj_df, ignore_index=True)
    return all_subj_df

def create_empty_df(col_list=None):
    empty_df = pd.DataFrame(columns=col_list)
    return empty_df

def save_df_to_csv(df, output_path, indexing=False):
    """Save dataframe to .csv files under the designated path. Make a directory if it's needed."""
    parent_path = Path(output_path)
    if not os.path.exists(parent_path.parent.absolute()):
        os.makedirs(parent_path.parent.absolute())
    df.to_csv(output_path, index=indexing)

def count_voxels(df, to_group=['subj', 'vroinames']):
    n_voxel_df = df.groupby(to_group, as_index=False)['voxel'].nunique()
    n_voxel_df = n_voxel_df.rename(columns={"voxel": "n_voxel"})
    return n_voxel_df

def check_28cond(df, print_msg=True):
    rand_subj = sub_number_to_string(random.randint(1, 9))
    rand_voxel = np.random.choice(df.query('subj == @rand_subj').voxel.unique())
    new_df = df.query('subj == @rand_subj & voxel == @rand_voxel')
    if print_msg:
        print(f'Voxel no.{rand_voxel} for {rand_subj} has {new_df.shape[0]} conditions.')

    return new_df.shape[0]

def complete_path(dir):
    """returns absolute path of the directory"""
    return os.path.abspath(dir)


def old_save_fig(save_fig, save_dir, y_label, x_label, f_name):
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir, y_label + '_vs_' + x_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f_name)
        plt.savefig(save_path, bbox_inches='tight')

def save_fig(save_fig, save_path):
    if save_fig:
        if not save_path:
            raise Exception("Output directory is not defined!")
        parent_path = Path(save_path)
        if not os.path.exists(parent_path.parent.absolute()):
            os.makedirs(parent_path.parent.absolute())
        plt.savefig(save_path, bbox_inches='tight')

def plot_voxels(df, n_voxels=1, to_plot="normed_beta", save_fig=False, save_path=None):
    x = np.arange(df[to_plot].shape[0])
    fig = plt.figure()
    color = plt.cm.rainbow(np.linspace(0,1,5))
    plt.plot(x, df[to_plot], color='k', label="betas", linewidth=2, linstyle='dashed', markersize=12, marker='o')
    plt.legend(title='Noise SD')
    plt.ylabel('Synthetic BOLD')
    plt.title('1 Synthetic voxel with noise')
    plt.tight_layout()
    save_fig(save_fig, save_path)


def melt_df(df, value_vars, var_name="type", value_name="value"):
    """This function uses pd.melt to melt df while maintaining all columns"""
    id_cols = df.drop(columns=value_vars).columns.tolist()
    long_df = pd.melt(df, id_vars=id_cols, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return long_df
