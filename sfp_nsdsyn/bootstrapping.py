import sys
from . import utils as utils
import pandas as pd
import numpy as np
from tqdm import tqdm

def bootstrap_sample(data, stat=np.mean, n_select=8, n_bootstrap=100):
    """ Bootstrap sample from data"""
    bootstrap = []
    for i in range(n_bootstrap):
        samples = np.random.choice(data, size=n_select, replace=True)
        i_bootstrap = stat(samples)
        bootstrap.append(i_bootstrap)
    return bootstrap


def bootstrap_dataframe(df, n_bootstrap=100,
                        to_sample='avg_betas',
                        to_group=['voxel', 'names', 'freq_lvl'], replace=True):
    """ Bootstrap using a dataframe. Progress bar will be displayed according to the
    number of the voxels for each subject."""

    selected_cols = to_group + [to_sample]
    all_df = pd.DataFrame(columns=selected_cols)
    for i_v in tqdm(df.voxel.unique()):
        sample_df = df.query('voxel == @i_v')
        for i in range(n_bootstrap):
            tmp = sample_df[selected_cols].groupby(to_group).sample(n=8, replace=replace)
            tmp = tmp.groupby(to_group).mean().reset_index()
            tmp['bootstrap'] = i
            tmp['bootstrap'] = tmp['bootstrap'].astype(int)
            all_df = pd.concat([all_df, tmp], ignore_index=True)

    return all_df

def bootstrap_dataframe_all_subj(sn_list, df, n_bootstrap=100,
                        to_sample='betas',
                        to_group=['subj', 'voxel', 'names', 'freq_lvl'], replace=True):
    """ Bootstrap for each subject's dataframe. Message will be displayed for each subject."""

    selected_cols = to_group + [to_sample]
    all_df = pd.DataFrame(columns=selected_cols)
    for sn in sn_list:
        subj = utils.sub_number_to_string(sn)
        tmp = df.query('subj == @subj')
        print(f'***{subj} bootstrapping start!***')
        tmp = bootstrap_dataframe(tmp,
                                  n_bootstrap=n_bootstrap,
                                  to_sample=to_sample,
                                  to_group=to_group,
                                  replace=replace)
        all_df = pd.concat([all_df, tmp], ignore_index=True)

    return all_df


def sigma_vi(bts_df, power, to_sd='normed_betas', to_group=['subj', 'voxel', 'class_idx']):
    sigma_vi_df = bts_df.groupby(to_group)[to_sd].apply(lambda x: (abs(np.percentile(x, 84)-np.percentile(x, 16))/2)**power)
    sigma_vi_df = sigma_vi_df.reset_index().rename(columns={to_sd: 'sigma_vi'})
    return sigma_vi_df

def sigma_v(bts_df, power, to_sd='normed_betas', to_group=['voxel', 'subj']):
    selected_cols = to_group + ['class_idx']
    sigma_vi_df = sigma_vi(bts_df, power, to_sd=to_sd, to_group=selected_cols)
    sigma_v_df = sigma_vi_df.groupby(to_group)['sigma_vi'].mean().reset_index()
    sigma_v_df = sigma_v_df.rename(columns={'sigma_vi': 'sigma_v'})
    return sigma_v_df

def get_multiple_sigma_vs(df, power, columns, to_sd='normed_betas', to_group=['voxel','subj']):
    """Generate multiple sigma_v_squared using different powers. power argument must be passed as a list."""
    sigma_v_df = sigma_v(df, power=power, to_sd=to_sd, to_group=to_group)
    sigma_v_df = sigma_v_df.rename(columns={'sigma_v': 'tmp'})
    sigma_v_df[columns] = pd.DataFrame(sigma_v_df['tmp'].to_list(), columns=columns)
    sigma_v_df = sigma_v_df.drop(columns=['tmp'])
    return sigma_v_df

def normalize_betas_by_frequency_magnitude(betas_df, betas='betas', freq_lvl='freq_lvl'):
    tmp = betas_df.groupby(['voxel', freq_lvl])[betas].mean().reset_index()
    tmp = tmp.pivot('voxel', freq_lvl, betas)
    index_col = tmp.index.to_numpy().reshape(-1, 1)
    tmp = np.linalg.norm(tmp, axis=1, keepdims=True)
    length = np.concatenate((index_col, tmp), axis=1)
    length = pd.DataFrame(length, columns=['voxel','length'])
    new_df = pd.merge(betas_df, length, on='voxel')
    new_df['normed_betas'] = np.divide(new_df['betas'], new_df['length'])
    return new_df

def get_sigma_v_for_whole_brain(betas_df, betas, class_list=None, sigma_power=2):
    """This function has the same purpose as the functions above, but is designed to perform faster
    to decrease the processing time, usually for whole brain voxels.
    precision_vi contains a matrix (voxel X 8 phases) for each class i.
    Then this matrix is normalized for each voxel.
    For all the classes, we average these normalized matrices and take a mean to get a single value for each voxel."""
    sigma_squared_v = []
    if class_list is None:
        class_list = betas_df.class_idx.unique()
    for class_i in class_list:
        sigma_vi = betas_df.query('class_idx == @class_i')[betas].to_numpy().reshape((betas_df.voxel.nunique(), -1))
        sigma_squared_v.append(np.std(sigma_vi, axis=1) ** sigma_power)
    return np.mean(sigma_squared_v, axis=0)

def merge_sigma_v_to_main_df(bts_v_df, subj_df, on=['subj', 'voxel']):
    return subj_df.merge(bts_v_df, on=on)

def average_sigma_v_across_voxels(df, subset=['subj']):

    if all(df.groupby(['voxel']+subset)['sigma_v_squared'].count() == 1) == False:
        df = df.drop_duplicates(['voxel']+subset)
    avg_sigma_v_df = df[subset+['sigma_v_squared']].groupby(subset).mean().reset_index()
    avg_sigma_v_df = avg_sigma_v_df.rename(columns={'sigma_v_squared': 'sigma_squared_s'})
    return avg_sigma_v_df

def get_precision_s(df, subset):
    avg_sigma_v_df = average_sigma_v_across_voxels(df, subset)
    avg_sigma_v_df['precision'] = 1 / avg_sigma_v_df['sigma_squared_s']
    return avg_sigma_v_df[subset + ['precision']]

