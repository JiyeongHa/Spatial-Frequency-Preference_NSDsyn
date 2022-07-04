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

    def _sample_pRF_from_data(self):
        if self.df is None:
            #TODO: set df_dir to the /derivatives/subj_dataframes and then complete the parent path
            random_sn = np.random.randint(1, 9, size=1)
            tmp_df = utils.load_all_subj_df(random_sn, df_dir=self.subj_df_dir, df_name='df_LITE_after_vs.csv')
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
            df['angle'], df['eccentricity'] = self._sample_pRF_from_data()
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

    def synthesize_BOLD_2d(self, params, full_ver=True, noise_m=0, noise_sd=0):
        syn_df = self.syn_voxels.copy()
        syn_model = model.Forward(params, 0, self.syn_voxels)
        syn_df['betas'] = syn_model.two_dim_prediction(full_ver=full_ver)
        syn_df['betas'] = add_noise(syn_df['betas'], noise_mean=noise_m, noise_sd=noise_sd)

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

def measure_sd_each_cond(df, to_sd, dv_to_group=['subj', 'voxel', 'names', 'freq_lvl']):
    """Measure each voxel's sd across 8 trials in a condition (2 tasks x 4 phases)"""
    df[f'normed_{to_sd}'] = model.normalize(df, to_norm=to_sd, group_by=['voxel', 'subj'],
                                            for_two_dim_model=False)
    std_df = df.groupby(dv_to_group)[f'normed_{to_sd}'].agg(np.std, ddof=0).reset_index()
    std_df = std_df.rename(columns={f'normed_{to_sd}': 'sd_' + f'normed_{to_sd}'})

    return std_df

def measure_sd_each_voxel(df, to_sd, dv_to_group=['subj', 'voxel']):
    tmp_dv_to_group = dv_to_group + ['names', 'freq_lvl']
    std_df = measure_sd_each_cond(df, to_sd, dv_to_group=tmp_dv_to_group)
    std_df = std_df.groupby(dv_to_group)[f'sd_normed_{to_sd}'].mean().reset_index()
    return std_df

def plot_sd_histogram(std_df, to_x="sd_normed_betas", to_label="freq_lvl",
                      x_label="SD across 8 trials",
                      stat="probability",
                      l_title="Frequency level",
                      save_fig=False, save_dir='/Users/auna/Dropbox/NYU/Projects/SF/MyResults/', f_name="sd_histogram.png"):
    grid = sns.FacetGrid(std_df,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         legend_out=True,
                         sharex=True, sharey=True)
    grid.map(sns.histplot, to_x, stat=stat)
    grid.set_axis_labels(x_label, stat.title())
    grid.fig.legend(title=l_title)
    utils.save_fig(save_fig=save_fig, save_dir=save_dir, y_label=stat.title(), x_label=x_label, f_name=f_name)
    plt.show()


def change_voxel_info_in_df(df):
    voxel_list = df.voxel.unique()
    df['voxel'] = df['voxel'].replace(voxel_list, range(voxel_list.shape[0]))
    return df



def load_history_df(output_dir, noise_sd, lr_rate, max_epoch, n_voxels, df_type):
    all_history_df = pd.DataFrame()
    for cur_noise, cur_lr, cur_epoch in product(noise_sd, lr_rate, max_epoch):
        model_history_path = os.path.join(output_dir, f'{df_type}_history_noise-{cur_noise}_lr-{cur_lr}_eph-{cur_epoch}_n_vox-{n_voxels}.csv')
        tmp = pd.read_csv(model_history_path)
        #TODO: remove adding lr_rate and noise columns part and edit fit_model() to make columns in the first place
        tmp['lr_rate'] = cur_lr
        tmp['noise_sd'] = cur_noise
        tmp['max_epoch'] = cur_epoch
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


def load_model_history_df_with_ground_truth(output_dir, noise_sd, lr_rate, max_epoch, n_voxels,
                                            ground_truth, id_val='ground_truth'):
    df = load_history_df(output_dir, noise_sd, lr_rate, max_epoch, n_voxels, df_type="model")
    df = add_ground_truth_to_df(ground_truth, df, id_val=id_val)
    return df

def load_all_model_fitting_results(output_dir, noise_sd, lr_rate, max_epoch, n_voxels,
                                            ground_truth, id_val='ground_truth'):
    model_history = load_model_history_df_with_ground_truth(output_dir, noise_sd, lr_rate, max_epoch, n_voxels,
                                                            ground_truth, id_val)
    loss_history = load_history_df(output_dir, noise_sd, lr_rate, max_epoch, n_voxels, df_type="loss")

    return model_history, loss_history


