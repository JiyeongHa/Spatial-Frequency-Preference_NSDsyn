import sys
import os
import numpy as np
import pandas as pd
import random
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def sub_number_to_string(sub_number, dataset="nsdsyn"):
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
                     df_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes/',
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

def save_fig(save_path):
    if save_path is not None:
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

def melt_params(df, value_name='value', params=None):
    if params == None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
    id_cols = df.drop(columns=params).columns.tolist()
    long_df = pd.melt(df, id_vars=id_cols,
                      value_vars=params,
                      var_name='params', value_name=value_name)
    return long_df

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def load_R2(sn, R2_path='func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/R2_nsdsynthetic.nii.gz'):
    """load a variance explained file (nii.gz 3D) for a subject"""
    subj = sub_number_to_string(sn)
    R2_dir=f'/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/{subj}/'
    R2_path = os.path.join(R2_dir, R2_path)
    R2_file = nib.load(f'{R2_path}').get_fdata()
    return R2_file

def load_R2_all_subj(sn_list, R2_path='func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/R2_nsdsynthetic.nii.gz'):
    all_subj_R2 = {}
    for sn in sn_list:
        all_subj_R2[utils.sub_number_to_string(sn)] = load_R2(sn, R2_path=R2_path)
    return all_subj_R2

def R2_histogram(sn_list, all_subj_R2, n_col=2, n_row=4, xlimit=100, n_bins=200,
                 save_fig=True, save_file_name='R2_hist.png',
                 save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/'):

    subj_list = [utils.sub_number_to_string(i) for i in sn_list]
    kwargs = dict(alpha=0.5, bins=n_bins, density=True, stacked=True)
    color_list = sns.color_palette("hls", len(sn_list))
    fig, axes = plt.subplots(n_col, n_row, figsize=(12,6), sharex=True, sharey=True)
    max_list = []
    if xlimit == 'final_max':
        for xSN in subj_list:
            x = all_subj_R2[xSN]
            max_list.append(x[~np.isnan(x)].max())
            xlimit = max(max_list)

    for i, (ax, xSN) in enumerate(zip(axes.flatten(), subj_list)):
        x = all_subj_R2[xSN]
        ax.hist(x[~np.isnan(x)], **kwargs, label=xSN, color=color_list[i])
        ax.set_xlim([0, xlimit])
        #ax.set_xlabel('Variance Explained (%)', fontsize=20)
        #ax.set_ylabel('Density', fontsize=20)
        ax.legend(fontsize=15)

    fig = axes[0, 0].figure
    plt.suptitle('Probability Histogram of Variance Explained', y=1.05, size=16)
    fig.text(0.5, 0.04, "Variance Explained (%)", ha="center", va="center", fontsize=20)
    fig.text(0.05, 0.5, "Density", ha="center", va="center", rotation=90, fontsize=20)
    #plt.tight_layout()
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + 'density_histogram' + '_vs_' + 'R2')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f'{save_file_name}')
        plt.savefig(save_path)
    plt.show()
    return

def convert_rgb_to_seaborn_color_palette(rgb_list, n_colors):
    return sns.color_palette(np.array(rgb_list) / 255, n_colors)

def color_husl_palette_different_shades(n_colors, hex_hue):
    """hue must be seaborn palettes._ColorPalette"""
    pal = sns.color_palette(f'light:{hex_hue}', n_colors=n_colors)
    return pal

def subject_color_palettes(dset, sub_list):
    if dset == 'nsdsyn':
        subj_list = [sub_number_to_string(sn, dset) for sn in np.arange(1,9)]
        pal = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
               (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0)]
    elif dset == 'broderick':
        broderick_sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
        subj_list = [sub_number_to_string(sn, dset) for sn in  broderick_sn_list]
        pal = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
               (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
               (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
               (189, 189, 189)]
    sub_dict = dict(zip(subj_list, pal))
    sub_list_pal = [c for k,c in sub_dict.items() if k in sub_list]
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    return sns.color_palette(np.array(sub_list_pal) / 255, len(sub_list))


def weighted_mean(x, **kws):
    """store weights as imaginery number"""
    return np.sum(np.real(x) * np.imag(x)) / np.sum(np.imag(x))