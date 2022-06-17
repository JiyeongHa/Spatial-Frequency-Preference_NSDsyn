import sys
# sys.path.append('../../')
import os
import seaborn as sns
import sfp_nsd_utils as utils
import numpy as np
import pandas as pd
import make_df
import two_dimensional_model as model
from itertools import combinations



class SynthesizeData():
    """Synthesize data for 1D model and 2D model simulations. This class consists of three parts:
    1. Load stimulus information (stim class, frequency level, w_a, w_r, etc) without phase information
    2. Generate synthetic voxels. Eccentricity and polar angle will be drawn from the uniform distribution.
    3. Generate BOLD predictions, with or without noise. """

    def __init__(self, n_voxels=100, stim_info_path='/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'):
        self.n_voxels = n_voxels
        self.stim_info = self.get_stim_info_for_n_voxels(stim_info_path)
        self.syn_df = self.generate_synthetic_voxels()

    def get_stim_info_for_n_voxels(self, stim_info_path):
        stim_info = make_df._load_stim_info(stim_info_path, drop_phase=True)
        stim_info['voxel'] = 0
        tmp_df = stim_info.copy()
        for i in np.arange(1, self.n_voxels):
            tmp_df['voxel'] = i
            stim_info = pd.concat([stim_info, tmp_df], ignore_index=True)
        return stim_info

    def _sample_from_data(self):
        if self.df is None:
            df_dir = '/Volumes/derivatives/subj_dataframes'
            tmp_df = utils.load_all_subj_df(np.arange(1, 2), df_dir=df_dir, df_name='df_LITE_after_vs.csv')
        else:
            tmp_df = self.df
        polar_angles = np.random.choice(tmp_df['angle'], size=(self.n_voxels,), replace=self.replace)
        eccentricity = np.random.choice(tmp_df['eccentricity'], size=(self.n_voxels,), replace = self.replace)
        return polar_angles, eccentricity

    def generate_synthetic_voxels(self):
        """Generate synthesized data for n voxels.
        Each voxel's polar angle and eccentricity will be drawn from uniform distribution.
        The polar angle is in the unit of degree and eccentricity is in the unit of visual angle."""
        #TODO: add another distribution
        df = pd.DataFrame()
        df['voxel'] = np.arange(0, self.n_voxels)
        if self.p_dist is "uniform":
            df['angle'] = np.random.uniform(0, 360, size=self.n_voxels)
            df['eccentricity'] = np.random.uniform(0, 4.2, size=self.n_voxels)
        elif self.p_dist is "data":
            df['angle'], df['eccentricity'] = self._sample_from_data()
        syn_df = self.stim_info.merge(df, on='voxel')
        syn_df = make_df._calculate_local_orientation(syn_df)
        syn_df = make_df._calculate_local_sf(syn_df)
        return syn_df


    def synthesize_BOLD_1d(self, params):
        #TODO: write 1D model forward class?
        pass

    def synthesize_BOLD_2d(self, params, beta_col='betas', full_ver=True):
        syn_model = model.Forward(params, 0, self.syn_df)
        return syn_model.two_dim_prediction(full_ver=full_ver)


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

def forward_1D(df):

    return predicted_betas