import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('macosx') #plt.show(block=True)
import sfp_nsd_utils as utils
import pandas as pd
import two_dimensional_model as model
import simulation as sim
from importlib import reload
from itertools import product
from inspect import getfullargspec

params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                       'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                       'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

stim_info_path = '/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'
subj_df_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes'
output_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/derivatives_HPC/simulation/results_2D'
fig_dir = '/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/'

measured_noise_sd =0.03995  # unnormalized 1.502063
noise_mptl = [1,3,5,7]
noise_sd = [1,3,5,7] #[np.round(measured_noise_sd*x, 2) for x in [1]]
max_epoch = [25000]
lr_rate = [0.0005, 0.001, 0.005, 0.01]
n_voxel = 100
full_ver = [True]
pw = [True]

np.random.seed(0)
syn_data = sim.SynthesizeData(n_voxels=100, pw=True, p_dist="data",
                              stim_info_path=stim_info_path, subj_df_dir=subj_df_dir)
syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=True)

noisy_syn_df_2d = sim.copy_df_and_add_noise(syn_df_2d, beta_col="normed_betas", noise_mean=0, noise_sd=syn_df_2d['noise_SD'])

cur_lr = 0.0005
syn_df = noisy_syn_df_2d.copy()
syn_SFdataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
syn_model = model.SpatialFrequencyModel(syn_SFdataset.my_tensor, full_ver=True)


##
utils.save_df_to_csv(syn_model_history, os.path.join(output_dir, model_f_name), indexing=False)
utils.save_df_to_csv(syn_loss_history, os.path.join(output_dir, loss_f_name), indexing=False)

# load and make a figure?

model_history, loss_history = sim.load_all_model_fitting_results(output_dir, full_ver, pw, noise_sd, n_voxel, lr_rate, max_epoch,
                                                                 ground_truth=params, id_val='ground_truth')


f_name = f'loss_plot_full_ver-True_pw-True_n_vox-{n_voxel}_lr-0.0005to0.01_eph-25000.png'

model.plot_loss_history(loss_history, to_x="epoch", to_y="loss",
                        to_label='noise_sd', to_col='lr_rate', lgd_title="Multiples of noise SD", to_row=None,
                        save_fig=True, save_path=os.path.join(fig_dir, "simulation", "results_2D", 'Epoch_vs_Loss', f_name),
                        ci="sd", n_boot=100, log_y=True, sharey=True)
plt.show()


to_label = 'lr_rate'
params_col, params_group = sim.get_params_name_and_group(params, True)
label_order = model_history[to_label].unique().tolist()
label_order.insert(0, label_order.pop())

df = model_history.query('(lr_rate in [0.0005, "ground_truth"])')


f_name = f'param_history_plot_p_full_ver-True_pw-True_n_vox-{n_voxel}_lr-0.0005_eph-25000_Aterms.png'

model.plot_param_history_horizontal(df, params=params_col[-2:], group=params_group[-2:],
                         to_label='noise_sd', label_order=None, ground_truth=True,
                         lgd_title=None, save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name),
                         ci="sd", n_boot=100, log_y=False)
plt.show()

df = model_history.query('(noise_sd in [1, "ground_truth"])')
f_name = f'final_params_plot_p_full_ver-True_pw-False_n_mptl-1_n_vox-{n_voxel}_lr-0.01to0.0005_eph-25000.png'
model.plot_grouped_parameters(df.query('epoch == 24999'), params_col, [1,2,3,4,4,4,4,5,5],
                              to_label="lr_rate", lgd_title="Learning rate", label_order=label_order,
                              save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name))
plt.tight_layout()
plt.show()

final_params = model_history.query('epoch == 24999 & lr_rate == 0.0005 & noise_sd == 1')[params_col]
final_params['A_3'] = [0]
final_params['A_4'] = [0]

def group(row):
    if row['voxel'] in small_sigma_voxels:
        return 'Smallest_10%_sigma'
    elif row['voxel'] in large_sigma_voxels:
        return 'Largest_10%_sigma'
    else:
        return 'middle'


losses_tmp = losses.query('epoch == 24999')
melt_df = melt_df.merge(losses_tmp, on='voxel')

melt_df.query('loss in [0.571709, 0.590803]')

min_group = melt_df.groupby('group')['loss'].min().reset_index()
melt_df.loc('loss', 0.571709)
min_group = melt_df.query('voxel in [96,22]')
min_group['loss_type'] = 'min'
max_group = melt_df.query('voxel in [52,53]')
max_group['loss_type'] = 'max'

new_df = pd.concat((min_group, max_group), axis=0)

voxel_order = losses.sort_values(by='sigma_v_squared').voxel.unique()
small_sigma_voxels = voxel_order[:10]
large_sigma_voxels = voxel_order[-10:]
small_sigma = losses.query('voxel in @small_sigma_voxels')
small_sigma['group'] = 'small_sigma'
large_sigma = losses.query('voxel in @large_sigma_voxels')
large_sigma['group'] = 'large_sigma'

orig_df = pd.read_csv('/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-True_noise_mtpl-1_n_vox-100.csv')
orig_df['group'] = orig_df.apply(lambda row: group(row), axis=1)
model_pred = model.PredictBOLD2d(params=final_params, params_idx=0, subj_df=orig_df)
orig_df['pred'] = model_pred.forward(full_ver=True)
orig_df['pred'] = model.normalize(orig_df, to_norm='pred', to_group='voxel', phase_info=False)

orig_df['condition'] = np.tile(np.arange(0,28),100)
ten_df = orig_df.query('group != "middle"')
ten_df = ten_df.drop(columns='betas')

melt_df = pd.melt(ten_df, id_vars=['voxel','group','names','names_idx','freq_lvl','image_idx','sigma_v_squared', 'condition'],
        value_vars=['normed_betas','pred'], var_name='beta_type', value_name='betas')


new_list = small_sigma_voxels + large_sigma_voxels
df = orig_df.query('voxel in @new_list')

new_sigma = pd.concat((small_sigma, large_sigma), axis=0)
new_sigma.merge(df, on='voxel')

f_name = f'loss_plot_full_ver-True_pw-True_n_vox-{n_voxel}_lr-0.0005to0.01_eph-25000.png'

model.plot_loss_history(new_sigma, to_x="epoch", to_y="loss",
                        to_label='group', to_col=None, lgd_title="voxel", to_row=None,
                        save_fig=True, save_path=os.path.join(fig_dir, "simulation", "results_2D", 'Epoch_vs_Loss', 'pw_group.png'),
                        ci=68, n_boot=100, log_y=True, sharey=True)
plt.show()

subj_data = sim.SynthesizeRealData(sn=1, pw=True, subj_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes')
subj_syn_df_2d = subj_data.synthesize_BOLD_2d(params, full_ver=True)
subj_syn_df_2d.to_csv('/Users/jh7685/Documents/Projects/test.csv')
subj_syn_df = pd.read_csv('/Users/jh7685/Documents/Projects/test.csv')
noisy_df_2d = sim.copy_df_and_add_noise(subj_syn_df, beta_col="normed_betas", noise_mean=0, noise_sd=subj_syn_df['noise_SD']*1)
noisy_df_2d.to_csv('/Users/jh7685/Documents/Projects/test_2.csv')
# add noise
output_losses_history = '/Users/jh7685/Documents/Projects/test_3.csv'
output_model_history = '/Users/jh7685/Documents/Projects/test_4.csv'
output_loss_history = '/Users/jh7685/Documents/Projects/test_5.csv'

syn_df = pd.read_csv('/Users/jh7685/Documents/Projects/test_2.csv')
syn_dataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor, full_ver=True)
syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
    learning_rate=0.001, max_epoch=3, print_every=2000, anomaly_detection=False, amsgrad=False, eps=1e-8)
losses_history = model.shape_losses_history(losses, syn_df)
utils.save_df_to_csv(losses_history, output_losses_history, indexing=False)
utils.save_df_to_csv(syn_model_history, output_model_history, indexing=False)
utils.save_df_to_csv(syn_loss_history, output_loss_history, indexing=False)

#broderick
df = pd.read_csv('')

subj_df = pd.read_csv('/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/sub-wlsubj007_stim_voxel_info_df_vs_md.csv')
subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
subj_model = model.SpatialFrequencyModel(subj_dataset.my_tensor, full_ver=True)
syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(subj_model, subj_dataset,
                                                                                learning_rate=0.0005,
                                                                                max_epoch=2,
                                                                                print_every=2000,
                                                                                anomaly_detection=False, amsgrad=False,
                                                                                eps=1e-8)
losses_history = model.shape_losses_history(losses, subj_df)
utils.save_df_to_csv(losses_history, output.losses_history, indexing=False)
utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)