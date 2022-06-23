import sys
# sys.path.append('../../')
import os
import seaborn as sns
import sfp_nsd_utils as utils
import numpy as np
import pandas as pd
import make_df
import two_dimensional_model as model
import binning_eccen as binning
import first_level_analysis as fitting
from itertools import combinations



class SynthesizeData():
    """Synthesize data for 1D model and 2D model simulations. This class consists of three parts:
    1. Load stimulus information (stim class, frequency level, w_a, w_r, etc) without phase information
    2. Generate synthetic voxels. Eccentricity and polar angle will be drawn from the uniform distribution.
    3. Generate BOLD predictions, with or without noise. """

    def __init__(self, n_voxels=100, df=None, replace=True, p_dist="uniform", stim_info_path='/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'):
        self.n_voxels = n_voxels
        self.df = df
        self.replace = replace,
        self.p_dist = p_dist
        self.stim_info = self.get_stim_info_for_n_voxels(stim_info_path)
        self.syn_voxels = self.generate_synthetic_voxels()

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
            #TODO: set df_dir to the /derivatives/subj_dataframes and then complete the parent path
            df_dir = '/Volumes/derivatives/subj_dataframes'
            random_sn = np.random.randint(1, 9, size=1)
            tmp_df = utils.load_all_subj_df(random_sn, df_dir=df_dir, df_name='df_LITE_after_vs.csv')
        else:
            tmp_df = self.df
        polar_angles = np.random.choice(tmp_df['angle'], size=(self.n_voxels,), replace=self.replace)
        eccentricity = np.random.choice(tmp_df['eccentricity'], size=(self.n_voxels,), replace = self.replace)
        return polar_angles, eccentricity

    def generate_synthetic_voxels(self):
        """Generate synthesized data for n voxels. if p_dist is set to "uniform",
        Each voxel's polar angle and eccentricity will be drawn from uniform distribution.
        In case p_dist == "data", the probability distribution will be from the actual data.
        The polar angle is in the unit of degree and eccentricity is in the unit of visual angle."""
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

    def synthesize_BOLD_1d(self, bin_list, bin_labels, params):
        #TODO: write 1D model forward class?
        # binning
        syn_df = self.syn_voxels.copy()
        syn_df['bins'] = binning.bin_ecc(self.syn_voxels, bin_list=bin_list, to_bin='eccentricity', bin_labels=bin_labels)
        syn_df = binning.summary_stat_for_ecc_bin(syn_df,
                                                  to_bin=['eccentricity', 'local_sf'],
                                                  bin_group=['bins', 'names', 'freq_lvl'],
                                                  central_tendency="mean")
        # forward
        amp = params['amp']
        slope = params['slope']
        intercept = params['intercept']
        sigma = params['sigma']
        syn_df['betas'] = fitting.np_log_norm_pdf(syn_df['local_sf'],
                                                  amp=amp,
                                                  mode=1 / (slope * syn_df['eccentricity'] + intercept),
                                                  sigma=sigma)

        return syn_df

    def synthesize_BOLD_2d(self, params, full_ver=True):
        syn_df = self.syn_voxels.copy()
        syn_model = model.Forward(params, 0, self.syn_voxels)
        syn_df['betas'] = syn_model.two_dim_prediction(full_ver=full_ver)
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

def change_voxel_info_in_df(df):
    voxel_list = df.voxel.unique()
    df['voxel'] = df['voxel'].replace(voxel_list, range(voxel_list.shape[0]))
    return df
