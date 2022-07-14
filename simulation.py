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
import matplotlib.pyplot as plt
from itertools import product
import bootstrap as bts



class SynthesizeData():
    """Synthesize data for 1D model and 2D model simulations. This class consists of three parts:
    1. Load stimulus information (stim class, frequency level, w_a, w_r, etc) without phase information
    2. Generate synthetic voxels. Eccentricity and polar angle will be drawn from the uniform distribution.
    3. Generate BOLD predictions, with or without noise. """

    def __init__(self, n_voxels=100, to_noise_sample='normed_betas', p_dist="uniform",
                 stim_info_path='/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv',
                 subj_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes'):
        self.n_voxels = n_voxels
        self.p_dist = p_dist
        self.subj_df_dir = subj_df_dir
        self.to_sd = to_noise_sample
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

    def _sample_pRF_from_data(self):
        random_sn = np.random.randint(1, 9, size=1)
        tmp_df = utils.load_all_subj_df(random_sn, df_dir=self.subj_df_dir, df_name='stim_voxel_info_df_vs.csv')
        polar_angles = np.random.choice(tmp_df['angle'], size=(self.n_voxels,), replace=False)
        eccentricity = np.random.choice(tmp_df['eccentricity'], size=(self.n_voxels,), replace=False)
        return polar_angles, eccentricity
    #
    # def _sample_noise_from_data(self):
    #     if self.df is None:
    #         random_sn = np.random.randint(1, 9, size=1)
    #         tmp_df = utils.load_all_subj_df(random_sn, df_dir=self.subj_df_dir, df_name='df_LITE_after_vs.csv')
    #     else:
    #         tmp_df = self.df
    #     #measure_sd_each_cond(to_sd='betas', dv_to_group=['subj', 'voxel'])
    #     return noise

    def generate_synthetic_voxels(self):
        """Generate synthesized data for n voxels. if p_dist is set to "uniform",
        Each voxel's polar angle and eccentricity will be drawn from uniform distribution.
        For  p_dist == "data", the probability distribution will be from the actual data.
        The polar angle is in the unit of degree and eccentricity is in the unit of visual angle."""
        df = pd.DataFrame()
        df['voxel'] = np.arange(0, self.n_voxels)
        if self.p_dist is "uniform":
            df['angle'] = np.random.uniform(0, 360, size=self.n_voxels)
            df['eccentricity'] = np.random.uniform(0, 4.2, size=self.n_voxels)
        elif self.p_dist is "data":
            df['angle'], df['eccentricity'] = self._sample_pRF_from_data()
        sigma_v = generate_sigma_v(self.n_voxels, to_sd=self.to_sd, df_dir=self.subj_df_dir)
        df = df.merge(sigma_v, on='voxel')
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
        syn_model = model.PredictBOLD2d(params, 0, self.syn_voxels)
        syn_df['noise_SD'] = 0
        syn_df['betas'] = syn_model.forward(full_ver=full_ver)
        syn_df['normed_betas'] = model.normalize(syn_df, to_norm="betas", to_group=['voxel'], phase_info=False)
        return syn_df


def add_noise(betas, noise_mean=0, noise_sd=0.03995):
    return betas + np.random.normal(noise_mean, noise_sd, len(betas))

def copy_df_and_add_noise(df, beta_col, noise_mean=0, noise_sd=0):
    # if noise_sd == 0:
    #     raise Exception('noise sd == 0 is the same as original data!\n')
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

def generate_sigma_v(n_voxels, to_sd='normed_betas',
                   df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes/'):
    if to_sd is None:
        sample_sigma_v = np.ones((n_voxels,), dtype=np.float64)
    else:
        sample_sigma_v_path = os.path.join(df_dir, f'sigma_v_to_sd-{to_sd}_n_vox-{n_voxels}.csv')
        if os.path.exists(sample_sigma_v_path):
            sample_sigma_v = pd.read_csv(sample_sigma_v_path)
        else:
            all_subj_df = utils.load_all_subj_df(np.arange(1, 9), df_dir=df_dir,
                                                 df_name='stim_voxel_info_df_vs.csv')
            sigma_v_df = bts.sigma_v(all_subj_df, to_sd=to_sd)
            sample_sigma_v = np.random.choice(sigma_v_df['sigma_v'], size=(n_voxels,), replace=False)
            sample_sigma_v = pd.DataFrame(sample_sigma_v).reset_index().rename(columns={'index': 'voxel', 0: 'sigma_v'})
            sample_sigma_v.to_csv(sample_sigma_v_path, index=False)
    return sample_sigma_v

def measure_sd_each_cond(df, to_sd, dv_to_group=['subj', 'voxel', 'names', 'freq_lvl'], normalize=True):
    """Measure each voxel's sd across 8 trials in a condition (2 tasks x 4 phases)"""
    if normalize:
        df[f'{to_sd}'] = model.normalize(df, to_norm=to_sd)
    std_df = df.groupby(dv_to_group)[f'{to_sd}'].agg(np.std, ddof=0).reset_index()
    std_df = std_df.rename(columns={f'{to_sd}': 'sigma_vi'})

    return std_df

def measure_sd_each_voxel(df, to_sd, dv_to_group=['subj', 'voxel'], normalize=True):
    tmp_dv_to_group = dv_to_group + ['names', 'freq_lvl']
    std_df = measure_sd_each_cond(df, to_sd, dv_to_group=tmp_dv_to_group, normalize=normalize)
    std_df = std_df.groupby(dv_to_group)['sigma_vi'].mean().reset_index()
    std_df = std_df.rename(columns={'sigma_vi': 'sigma_v'})
    return std_df

def plot_sd_histogram(std_df, to_x="sd_normed_betas", to_label="freq_lvl",
                      x_label="SD for each voxel across 8 trials",
                      stat="probability",
                      lgd_title="Frequency level",
                      save_fig=False, save_dir='/Users/auna/Dropbox/NYU/Projects/SF/MyResults/', f_name="sd_histogram.png"):
    grid = sns.FacetGrid(std_df,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         legend_out=True,
                         sharex=True, sharey=True)
    grid.map(sns.histplot, to_x, stat=stat)
    grid.set_axis_labels(x_label, stat.title())
    if to_label is not None:
        grid.fig.legend(title=lgd_title)
    utils.save_fig(save_fig=save_fig, save_dir=save_dir, y_label=stat.title(), x_label=x_label, f_name=f_name)
    plt.tight_layout()
    plt.show()


def change_voxel_info_in_df(df):
    voxel_list = df.voxel.unique()
    df['voxel'] = df['voxel'].replace(voxel_list, range(voxel_list.shape[0]))
    return df



def load_history_df(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels, df_type):
    all_history_df = pd.DataFrame()
    for cur_noise, cur_lr, cur_epoch, cur_ver, in product(noise_sd, lr_rate, max_epoch, full_ver):
        model_history_path = os.path.join(output_dir, f'{df_type}_history_full_ver-{cur_ver}_sd-{cur_noise}_n_vox-{n_voxels}_lr-{cur_lr}_eph-{cur_epoch}.csv')
        tmp = pd.read_csv(model_history_path)
        if {'lr_rate', 'noise_sd', 'max_epoch', 'full_ver'}.issubset(tmp.columns) is False:
            tmp['lr_rate'] = cur_lr
            tmp['noise_sd'] = cur_noise
            tmp['max_epoch'] = cur_epoch
            tmp['full_ver'] = cur_ver
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


def load_model_history_df_with_ground_truth(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels,
                                            ground_truth, id_val='ground_truth'):
    df = load_history_df(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels, df_type="model")
    df = add_ground_truth_to_df(ground_truth, df, id_val=id_val)
    return df

def load_all_model_fitting_results(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels,
                                   ground_truth, id_val='ground_truth'):
    model_history = load_model_history_df_with_ground_truth(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels,
                                                            ground_truth, id_val)
    loss_history = load_history_df(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels, df_type="loss")

    return model_history, loss_history


def get_params_name_and_group(params, full_ver):
    if full_ver:
        params_col = params.columns.tolist()[:-2]
    else:
        params_col = params.columns.tolist()[:3]
    group = list(range(len(params_col)))
    return params_col, group
