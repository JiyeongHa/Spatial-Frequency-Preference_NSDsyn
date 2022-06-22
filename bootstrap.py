import sys
sys.path.append('../../')
import os
import seaborn as sns
from matplotlib import pyplot as plt
import sfp_nsd_utils as utils
import pandas as pd
import numpy as np
from first_level_analysis import np_log_norm_pdf
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

def sigma_vi(bts_df, to_sample='avg_betas', to_group=['voxel', 'names', 'freq_lvl']):
    bts_vi_df = bts_df.groupby(to_group)['avg_betas'].apply(lambda x: (abs(np.percentile(x, 84)-np.percentile(x, 16))/2)**2)
    bts_vi_df = bts_vi_df.reset_index().rename(columns={to_sample: 'sigma_vi'})
    return bts_vi_df

def sigma_v(bts_df, to_sample='avg_betas', to_group=['voxel', 'subj']):
    selected_cols = to_group + ['names', 'freq_lvl']
    bts_vi_df = sigma_vi(bts_df, to_sample=to_sample, to_group=selected_cols)
    bts_v_df = bts_vi_df.groupby(to_group)['sigma_vi'].mean().reset_index()
    return bts_v_df




