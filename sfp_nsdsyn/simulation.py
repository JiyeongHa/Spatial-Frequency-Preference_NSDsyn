import os
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import product
from . import utils as utils
from . import two_dimensional_model as model2d
from . import bootstrapping as bts
from . import binning as binning
from . import one_dimensional_model as fitting
from . import make_dataframes as prep
from gsn.perform_gsn import perform_gsn
import re

class SynthesizeData():
    """Synthesize data for 1D model and 2D model simulations. This class consists of three parts:
    1. Load stimulus information (stim class, frequency level, w_a, w_r, etc) without phase information
    2. Generate synthetic voxels. Eccentricity and polar angle will be drawn from the uniform distribution.
    3. Generate BOLD predictions, with or without noise. """

    def __init__(self, roi, n_voxels=100, 
                 precision_weight=True, p_dist="data",
                 sample_subj_list='all',
                 df = None,
                 stim_info_path='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsyn_stim_description_corrected.csv',
                 dataframe_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes',
                 random_state=42):
        """
        grating_type: 'scaled' or 'constant'
        """
        self.roi = roi
        self.precision_weight = precision_weight
        self.random_state = random_state
        self.stim_info_path = stim_info_path
        self.dataframe_dir = dataframe_dir
        self.p_dist = p_dist
        if df is not None:
            self.syn_df = df[['sub','w_r','w_a','class_idx','names','freq_lvl','voxel','angle','eccentricity','noise_SD','sigma_v_squared']]
            self.syn_df['trial'] = self.syn_df.groupby(['sub','voxel','class_idx']).cumcount()
            self.n_voxels = self.syn_df.voxel.nunique()
        else:
            self.n_voxels = n_voxels
            self.sample_subj_list = self._resolve_subj_list(sample_subj_list)
            self.syn_df = self._synthesize_data()

    def _resolve_subj_list(self, sample_subj_list):
        if sample_subj_list == 'all':
            self.sample_subj_list = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']
        else:
            self.sample_subj_list = sample_subj_list
        return self.sample_subj_list
    
    def _synthesize_data(self):
        stim_info = self._load_and_expand_stim_info()
        syn_df = self._generate_pRF_data(stim_info)
        syn_df = self._generate_sigma_v(syn_df)
        return syn_df
    
    def _add_local_stim_properties(self, grating_type):
        syn_df = self.syn_df.copy()
        syn_df['local_sf'], syn_df['local_ori'] = prep.calculate_local_stim_properties(w_a=syn_df['w_a'], 
                                                                                      w_r=syn_df['w_r'],
                                                                                      eccentricity=syn_df['eccentricity'],
                                                                                      angle=syn_df['angle'],
                                                                                      angle_in_radians=True,
                                                                                      stimulus=grating_type)
        return syn_df
    
    def synthesize_BOLD_2d(self, params, grating_type='scaled', model=7, phase_info=False):
        sim_df = self._add_local_stim_properties(grating_type)
        my_model = model2d.SpatialFrequencyModel(params=params, model=model)
        sim_df['betas'] = my_model.forward(theta_l=sim_df['local_ori'], 
                                           theta_v=sim_df['angle'],
                                           r_v=sim_df['eccentricity'], 
                                           w_l=sim_df['local_sf'], to_numpy=True)
        sim_df['normed_betas'] = model2d.normalize(sim_df, to_norm="betas", to_group=['voxel'], phase_info=phase_info)
        return sim_df
        
    def add_noise(self, df, beta_col, noise_mean=0, noise_sd=0.03995):
        sim_noise = np.random.normal(noise_mean, noise_sd, len(df[beta_col]))
        return df[beta_col] + sim_noise
    
    def add_noise_from_covariance(self, df, beta_col, n_trials, noise_cov, noise_level=1):
        sim_noise = sample_noise_from_covariance(noise_cov, n_samples=n_trials)
        sim_noise_df = pd.DataFrame(sim_noise.T).reset_index().replace(np.arange(self.n_voxels), df.voxel.unique())
        sim_noise_df.rename(columns={'index':'voxel'}, inplace=True)
        sim_noise_df = sim_noise_df.melt(id_vars='voxel', var_name='trial', value_name='noise')
        df = df.merge(sim_noise_df, on=['voxel', 'trial'])
        df['noise'] = df['noise']*noise_level
        df[f'noisy_{beta_col}'] = df[beta_col] + df['noise']
        return df


    def _load_and_expand_stim_info(self):
        stim_info = prep.load_stim_info_as_df(self.stim_info_path, 
                                              drop_phase=False, 
                                              force_download=False)
        stim_info = stim_info.drop_duplicates(subset=['class_idx','names'])
        stim_info = stim_info.drop(columns=['image_idx','stim_idx','phase','phase_idx']).reset_index(drop=True)
        stim_info['voxel'] = 0
        if self.n_voxels > 1:
            tmp_df = stim_info.copy()
            for i in np.arange(1, self.n_voxels):
                tmp_df['voxel'] = i
                stim_info = pd.concat((stim_info, tmp_df), ignore_index=True)
        return stim_info
    
    def _generate_pRF_data(self, stim_info):
        """Generate synthesized data for n voxels. if p_dist is set to "uniform",
        Each voxel's polar angle and eccentricity will be drawn from uniform distribution.
        For  p_dist == "data", the probability distribution will be from the actual data.
        The polar angle is in the unit of degree and eccentricity is in the unit of visual angle."""
        if self.p_dist == "uniform":
            prf_df = pd.DataFrame({'voxel': range(self.n_voxels), 
                                   'angle': np.random.uniform(0, 360, size=self.n_voxels), 
                                   'eccentricity': np.random.uniform(0, 4.2, size=self.n_voxels)})
        elif self.p_dist == "data":
            prf_df = self._sample_pRF_from_data()
        syn_df = stim_info.merge(prf_df, on='voxel')
        return syn_df
    
    def _generate_sigma_v(self, df):
        """Sample sigma_v_squared from the NSD synthetic data.
        The final output is a dataframe that contains two columns: noise_SD (sigma_v_squared) & sigma_v_squared.
        It will first attempt to read in an any saved sigma_v_squared.csv files.
        If none exists, it will automatically sample from the actual data after calculating sigma_v_squared for each  voxel and for each
        subject. """

        if self.precision_weight is False:
            measured_noise_sd = 0.03995
            sigma_v_df = pd.DataFrame({'voxel': range(self.n_voxels), 
                                       'noise_SD': measured_noise_sd, 
                                       'sigma_v_squared': 1.0})
        else:
            sigma_v_df = pd.DataFrame({})
            for subj in self.sample_subj_list:
                tmp = pd.read_csv(os.path.join(self.dataframe_dir, 'nsdsyn', 'precision', f'precision-v_sub-{subj}_roi-{self.roi}_vs-pRFsize.csv'))
                sigma_v_df = pd.concat((sigma_v_df, tmp), ignore_index=True)
            sigma_v_df = sigma_v_df.drop_duplicates(subset=['voxel'])
            sigma_v_df = sigma_v_df.sample(n=self.n_voxels, replace=False, random_state=self.random_state)
            sigma_v_df['voxel'] = range(self.n_voxels)
            sigma_v_df = sigma_v_df.reset_index(drop=True)
        return df.merge(sigma_v_df[['voxel', 'noise_SD', 'sigma_v_squared']], on='voxel').reset_index(drop=True)
        
    def _sample_pRF_from_data(self):
        
        all_subj_df = []
        for subj in self.sample_subj_list:
            subj_df = pd.read_csv(os.path.join(self.dataframe_dir, 'nsdsyn', 'model', 
                                               f'dset-nsdsyn_sub-{subj}_roi-{self.roi}_vs-pRFsize_tavg-False.csv'))
            subj_df = subj_df.drop_duplicates(subset=['voxel'])
            all_subj_df.append(subj_df)
        all_subj_df = pd.concat(all_subj_df, ignore_index=True)

        sampled_voxels = all_subj_df.sample(n=self.n_voxels, replace=False, random_state=self.random_state)
        polar_angles = sampled_voxels['angle'].values
        eccentricity = sampled_voxels['eccentricity'].values
        prf_df = pd.DataFrame({'voxel': range(self.n_voxels), 'angle': polar_angles, 'eccentricity': eccentricity})
        return prf_df

    def synthesize_BOLD_1d(self, bin_list, bin_labels, params):
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


def _reshape_betas_to_numpy(df, beta_col, new_shape=['voxel', 'class_idx', 'trial']):
    df = df.sort_values(new_shape)
    # Pivot to get shape: (n_voxels, n_class_idx, n_trials)
    pivoted = df.pivot_table(
        index=new_shape[0], 
        columns=new_shape[1:], 
        values=beta_col
    )
    # Convert to numpy and reshape
    np_array = pivoted.to_numpy().reshape(
        df[new_shape[0]].nunique(),
        df[new_shape[1]].nunique(),
        df[new_shape[2]].nunique()
    )
    return np_array

def measure_noise_covariance(df, beta_col, new_shape=['voxel', 'class_idx', 'trial'], return_all=False):
    np_array = _reshape_betas_to_numpy(df, beta_col, new_shape)
    results = perform_gsn(np_array, {'wantshrinkage': True})
    if return_all:
        return results
    else:
        return results['cNb']

def sample_noise_from_covariance(cov_matrix, mean=None, n_samples=100):
    if mean is None:
        mean = np.zeros(cov_matrix.shape[0])
    return np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=n_samples,)




def copy_df_and_add_noise(df, beta_col, noise_mean=0, noise_sd=0):

    noisy_df = df.copy()
    noisy_df['noise_SD'] = noise_sd
    noisy_df[beta_col] = add_noise(df[beta_col], noise_mean=noise_mean, noise_sd=noise_sd)
    return noisy_df


def melt_beta_task_type(df, include_avg=False, id_cols=None):
    tasks = ['fixation_task_betas', 'memory_task_betas']
    if include_avg is True:
        tasks = tasks + ['avg_betas']
    new_tasks = [x.replace('_task_betas', '') for x in tasks]
    df = df.rename(columns=dict(zip(tasks, new_tasks)))
    if id_cols == None:
        id_cols = df.drop(columns=new_tasks).columns.tolist()
    df = pd.melt(df, id_vars=id_cols, value_vars=new_tasks, var_name='task', value_name='betas')
    return df

def _check_all_sigma_v_df_in_path(betas, all_sigma_v_path, subj_df_dir):
    if os.path.exists(all_sigma_v_path):
        all_sigma_v_df = pd.read_csv(all_sigma_v_path)
    else:
        all_subj_df = utils.load_all_subj_df(np.arange(1, 9), df_dir=subj_df_dir, df_name='stim_voxel_info_df_vs.csv')
        all_sigma_v_df = bts.get_multiple_sigma_vs(all_subj_df, power=[1,2], to_group=['voxel','subj'],
                                                   columns=['noise_SD', 'sigma_v_squared'], to_sd=betas)
        utils.save_df_to_csv(all_sigma_v_df, all_sigma_v_path, indexing=False)
    return all_sigma_v_df

def measure_sd_each_cond(df, to_sd, dv_to_group=['subj', 'voxel', 'names', 'freq_lvl'], ddof=0, normalize=True):
    """Measure each voxel's sd across 8 trials in a condition (2 tasks x 4 phases)"""
    if normalize:
        df[f'{to_sd}'] = model.normalize(df, to_norm=to_sd)
    std_df = df.groupby(dv_to_group)[f'{to_sd}'].agg(np.std, ddof=ddof).reset_index()
    std_df = std_df.rename(columns={f'{to_sd}': 'sigma_vi'})

    return std_df

def measure_sd_each_voxel(df, to_sd, dv_to_group=['subj', 'voxel'], ddof=0, normalize=True):
    tmp_dv_to_group = dv_to_group + ['names', 'freq_lvl']
    std_df = measure_sd_each_cond(df, to_sd, dv_to_group=tmp_dv_to_group, ddof=ddof, normalize=normalize)
    std_df = std_df.groupby(dv_to_group)['sigma_vi'].mean().reset_index()
    std_df = std_df.rename(columns={'sigma_vi': 'sigma_v_squared'})
    return std_df

def plot_sd_histogram(std_df, to_x="sd_normed_betas", to_label="freq_lvl",
                      x_label="SD for each voxel across 8 trials",
                      stat="probability",
                      lgd_title="Frequency level",
                      height=3,
                      save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/', f_name="sd_histogram.png"):
    sns.set_context("notebook")
    grid = sns.FacetGrid(std_df,
                         hue=to_label,
                         palette=sns.color_palette("hls", std_df[to_label].nunique()),
                         legend_out=True,
                         height=height,
                         sharex=True, sharey=True)
    grid.map(sns.histplot, to_x, stat=stat, alpha=0.7)
    grid.set_axis_labels(x_label, stat.title())
    if lgd_title is not None:
        grid.fig.legend(title=lgd_title)
    save_path = os.path.join(save_dir, f'{x_label}_vs_{stat.title()}', f_name)
    utils.save_fig(save_fig, save_path)


def plot_sd_histogram_with_cols(std_df, to_x="sd_normed_betas", to_label="freq_lvl",
                      x_label="SD for each voxel across 8 trials",
                      stat="probability", col=None, col_wrap=1,
                      lgd_title="Frequency level",
                      height=3,
                      save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/', f_name="sd_histogram.png"):
    sns.set_context("notebook")
    grid = sns.FacetGrid(std_df,
                         hue=to_label,
                         palette=sns.color_palette("hls", std_df[to_label].nunique()),
                         legend_out=True,
                         height=height,
                         col=col,
                         col_warp=col_wrap,
                         sharex=True, sharey=True)
    grid.map(sns.histplot, to_x, stat=stat, alpha=0.7)
    grid.set_axis_labels(x_label, stat.title())
    if lgd_title is not None:
        grid.fig.legend(title=lgd_title)
    save_path = os.path.join(save_dir, f'{x_label}_vs_{stat.title()}', f_name)
    utils.save_fig(save_fig, save_path)

def change_voxel_info_in_df(df):
    voxel_list = df.voxel.unique()
    df['voxel'] = df['voxel'].replace(voxel_list, range(voxel_list.shape[0]))
    return df

def load_losses_history_df(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch):
    losses_history_path = os.path.join(output_dir, f'losses_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{noise_sd}_n_vox-{n_voxels}_lr-{lr_rate}_eph-{max_epoch}.csv')
    losses_history = pd.read_csv(losses_history_path)
    return losses_history

def load_history_df(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch, df_type):
    all_history_df = pd.DataFrame()
    for cur_ver, cur_pw, cur_noise, cur_lr, cur_epoch  in product(full_ver, pw, noise_sd, lr_rate, max_epoch):
        model_history_path = os.path.join(output_dir, f'{df_type}_history_full_ver-{cur_ver}_pw-{cur_pw}_noise_mtpl-{cur_noise}_n_vox-{n_voxels}_lr-{cur_lr}_eph-{cur_epoch}.csv')
        tmp = pd.read_csv(model_history_path)
        if {'lr_rate', 'noise_sd', 'max_epoch', 'full_ver', 'pw'}.issubset(tmp.columns) is False:
            tmp['lr_rate'] = cur_lr
            tmp['noise_sd'] = cur_noise
            tmp['max_epoch'] = cur_epoch
            tmp['full_ver'] = cur_ver
            tmp['pw'] = cur_pw #precision weight
        all_history_df = pd.concat((all_history_df, tmp), axis=0, ignore_index=True)
    return all_history_df

def add_ground_truth_to_df(ground_truth, df, id_val='ground_truth'):
    """Add ground truth information to a dataframe.
    The ground truth should be either a dict or a pd dataframe that contains each param value. """
    if type(ground_truth) is dict:
        ground_truth = pd.DataFrame(ground_truth)
    unused_params = ground_truth.columns.difference(df.columns)
    ground_truth = ground_truth.drop(columns=unused_params)
    ground_truth['epoch'] = df.epoch.max()
    common_param_columns = np.intersect1d(ground_truth.columns, df.columns)
    id_cols = df.drop(columns=common_param_columns).columns.tolist()
    ground_truth[id_cols] = id_val
    df = pd.concat((df, ground_truth), axis=0, ignore_index=True)
    return df


def load_model_history_df_with_ground_truth(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch,
                                            ground_truth, id_val='ground_truth'):
    df = load_history_df(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch, df_type="model")
    df = add_ground_truth_to_df(ground_truth, df, id_val=id_val)
    return df

def load_all_model_fitting_results(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch,
                                   ground_truth, id_val='ground_truth'):
    model_history = load_model_history_df_with_ground_truth(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch,
                                                            ground_truth, id_val)
    loss_history = load_history_df(output_dir, full_ver, pw, noise_sd, n_voxels, lr_rate, max_epoch, df_type="loss")

    return model_history, loss_history


def get_params_name_and_group(params, full_ver):
    if full_ver:
        params_col = params.columns.tolist()[:-2]
    else:
        params_col = params.columns.tolist()[:3]
    group = list(range(len(params_col)))
    return params_col, group
