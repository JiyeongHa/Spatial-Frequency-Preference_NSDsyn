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

    return stim_info


def generate_synthetized_data(n_voxels=100):
    df = pd.DataFrame()
    df['voxel'] = np.arange(0, n_voxels)
    df['angle'] = np.random.uniform(0, 360, size=n_voxels)
    df['eccentricity'] = np.random.uniform(0, 4.2, size=n_voxels)
    return df
