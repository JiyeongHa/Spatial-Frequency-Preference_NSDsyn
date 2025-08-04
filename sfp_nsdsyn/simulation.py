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
import re

class SynthesizeData():
    """Synthesize data for 1D model and 2D model simulations. This class consists of three parts:
    1. Load stimulus information (stim class, frequency level, w_a, w_r, etc) without phase information
    2. Generate synthetic voxels. Eccentricity and polar angle will be drawn from the uniform distribution.
    3. Generate BOLD predictions, with or without noise. """

    def __init__(self, roi, n_voxels=100, grating_type='scaled', 
                 precision_weight=True, p_dist="data",
                 sample_subj_list='all',
                 stim_info_path='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsyn_stim_description_corrected.csv',
                 dataframe_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes',
                 random_state=42):
        """
        grating_type: 'scaled' or 'constant'
        """
        self.roi = roi
        self.n_voxels = n_voxels
        self.grating_type = grating_type
        self.stim_info_path = stim_info_path
        self.p_dist = p_dist
        self.precision_weight = precision_weight
        self.dataframe_dir = dataframe_dir
        if sample_subj_list == 'all':
            self.sample_subj_list = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']
        else:
            self.sample_subj_list = sample_subj_list
        self.random_state = random_state


    def get_stim_info_for_n_voxels(self):
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
    
    def add_pRF_info(self, stim_info):
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
        syn_df['local_sf'], syn_df['local_ori'] = prep.calculate_local_stim_properties(w_a=syn_df['w_a'], 
                                                                                       w_r=syn_df['w_r'],
                                                                                       eccentricity=syn_df['eccentricity'],
                                                                                       angle=syn_df['angle'],
                                                                                       angle_in_radians=True,
                                                                                       stimulus=self.grating_type)
        return syn_df
    
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

    def add_sigma_v(self, df):
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
        
    def synthesize_BOLD_2d(self, syn_df, params, model=7):
        my_model = model2d.SpatialFrequencyModel(params=params, model=model)
        syn_df['betas'] = my_model.forward(theta_l=syn_df['local_ori'], theta_v=syn_df['angle'],
                                           r_v=syn_df['eccentricity'], w_l=syn_df['local_sf'], to_numpy=True)
        syn_df['normed_betas'] = model2d.normalize(syn_df, to_norm="betas", to_group=['voxel'], phase_info=False)
        return syn_df
        
    def add_noise(self, df, beta_col, noise_mean=0, noise_sd=0.03995):
        df[beta_col] = df[beta_col] + np.random.normal(noise_mean, noise_sd, len(df[beta_col]))
        return df

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





class SynthesizeRealData():
    """Synthesize using the real dataset information as much as possible for 2D model simulations.
    This class consists of three parts:
    1. Load stimulus information (stim class, frequency level, w_a, w_r, etc) without phase information
    2. Choose one subject and load in all prf information and sigma_v in V1. The relationship between voxel info and
    prf values should not be changed.
    3. Generate BOLD predictions, with or without noise. """

    def __init__(self, sn, pw=True,
                 subj_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes'):
        self.sn = sn
        self.subj_df_dir = subj_df_dir
        self.pw = pw
        self.subj_voxels = self.load_prf_stim_info_from_data()
        self.n_voxels = self.subj_voxels.voxel.nunique()

    def _get_sigma_v(self, subj_df, voxel_list):
        sigma_v_dir = os.path.join(self.subj_df_dir, 'sigma_v')
        if self.pw is False:
            measured_noise_sd = 0.03995
            sigma_v_df = pd.DataFrame({'voxel': voxel_list})
            sigma_v_df['noise_SD'] = np.ones((voxel_list.shape[0]), dtype=np.float64) * measured_noise_sd
            sigma_v_df['sigma_v_squared'] = np.ones((voxel_list.shape[0]), dtype=np.float64)
        else:
            betas = 'normed_betas'
            all_sigma_v_path = os.path.join(sigma_v_dir, f'sigma_v_{betas}.csv')
            if os.path.exists(all_sigma_v_path):
                all_sigma_v_df = pd.read_csv(all_sigma_v_path)
                subj = utils.sub_number_to_string(self.sn)
                sigma_v_df = all_sigma_v_df.query('subj == @subj & voxel in @voxel_list')
                sigma_v_df = sigma_v_df[['voxel', 'noise_SD', 'sigma_v_squared']]
                print('used existing dir')
            else:
                sigma_v_df = bts.get_multiple_sigma_vs(subj_df, power=[1,2], to_group=['voxel'],
                                                   columns=['noise_SD', 'sigma_v_squared'], to_sd=betas)
                print('made new sigma_v')
        return sigma_v_df

    def _drop_phase_info(self, subj_df):
        tmp_df = subj_df.groupby(['voxel', 'freq_lvl', 'names']).mean().reset_index()
        subj_df_no_phase = tmp_df.drop(columns=['betas', 'normed_betas', 'phase', 'phase_idx', 'avg_betas'])
        return subj_df_no_phase

    def load_prf_stim_info_from_data(self):
        tmp_df = utils.load_df(self.sn, df_dir=self.subj_df_dir, df_name='stim_voxel_info_df_vs.csv')
        subj_df = tmp_df.query('vroinames == "V1"')
        sigma_v_df = self._get_sigma_v(subj_df, tmp_df.voxel.unique())
        subj_df = subj_df.merge(sigma_v_df, on='voxel')
        subj_df = self._drop_phase_info(subj_df)
        return subj_df

    def synthesize_BOLD_2d(self, params, full_ver=True):
        syn_df = self.subj_voxels.copy()
        syn_model = model.PredictBOLD2d(params, 0, self.subj_voxels)
        syn_df['betas'] = syn_model.forward(full_ver=full_ver)
        syn_df['normed_betas'] = model.normalize(syn_df, to_norm="betas", to_group=['voxel'], phase_info=False)
        return syn_df

def merge_sigma_v_squared(syn_df):
    sigma_v_df = bts.sigma_v(syn_df, power=2, to_sd='normed_betas', to_group=['voxel'])
    sigma_v_df = sigma_v_df.rename(columns={'sigma_v': 'sigma_v_squared'})
    new_df = syn_df.merge(sigma_v_df, on='voxel')
    return new_df









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
