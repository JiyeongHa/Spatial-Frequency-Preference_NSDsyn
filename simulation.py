import sys
# sys.path.append('../../')
import os
import seaborn as sns
import sfp_nsd_utils as utils
import numpy as np
import pandas as pd
import make_df

stim_description_dir = '/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives',
stim_description_file = 'nsdsynthetic_sf_stim_description.csv',

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

