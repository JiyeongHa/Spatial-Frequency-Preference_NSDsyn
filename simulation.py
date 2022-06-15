import sys
# sys.path.append('../../')
import os
import seaborn as sns
import sfp_nsd_utils as utils
import numpy as np
import pandas as pd
import make_df
from  itertools import combinations


def get_w_a_w_r(stim_description_path='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'):
    stim_info = make_df._load_stim_info(stim_description_path=stim_description_path)
    stim_info = pd.DataFrame(stim_info)
    stim_info = stim_info.query('phase_idx == 0')
    stim_info = stim_info.drop(columns=['phase', 'phase_idx'])
    return stim_info

def replicate_df(stim_info, n_replicate=100):
    stim_info['voxel'] = 0
    tmp_df = stim_info.copy()
    for i in np.arange(1, n_replicate):
        tmp_df['voxel'] = i
        stim_info = pd.concat([stim_info, tmp_df], ignore_index=True)
    return stim_info

def generate_synthesized_data(n_voxels=100, stim_description_path='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'):
    """Generate synthesized data for n voxels.
    Each voxel's polar angle and eccentricity will be drawn from uniform distribution.
    The polar angle is in the unit of degree and eccentricity is in the unit of visual angle."""
    stim_info = get_w_a_w_r(stim_description_path)
    stim_info = replicate_df(stim_info, n_replicate=n_voxels)

    df = pd.DataFrame()
    df['voxel'] = np.arange(0, n_voxels)
    df['angle'] = np.random.uniform(0, 360, size=n_voxels)
    df['eccentricity'] = np.random.uniform(0, 4.2, size=n_voxels)
    syn_df = stim_info.merge(df, on='voxel')
    syn_df = make_df._calculate_local_orientation(syn_df)
    syn_df = make_df._calculate_local_sf(syn_df)
    return syn_df

def add_noise(betas, noise_mean=0, noise_sd=0.05):
    return betas + np.random.normal(noise_mean, noise_sd, len(betas))

def melt_beta_task_type(df, id_cols=None):

    tasks = ['fixation_task_betas', 'memory_task_betas', 'avg_betas']
    new_tasks = [x.replace('_task_betas', '') for x in tasks]
    df = df.rename(columns=dict(zip(tasks, new_tasks)))
    if id_cols == None:
        id_cols = df.drop(columns=new_tasks).columns.tolist()
    df = pd.melt(df, id_vars=id_cols, value_vars=new_tasks, var_name='task', value_name='betas')
    return df


def measure_sd_each_stim(df, to_sd, dv_to_group=['names', 'voxel', 'subj', 'freq_lvl']):
    """Measure each voxel's sd across 8 conditions (2 tasks x 4 phases)"""
    std_df = df.groupby(dv_to_group)[to_sd].agg(np.std, ddof=0).reset_index()
    std_df = std_df.rename(columns={to_sd: 'sd_' + to_sd})

    return std_df
