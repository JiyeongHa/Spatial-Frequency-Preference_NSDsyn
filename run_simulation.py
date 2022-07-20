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
noise_sd = [np.round(measured_noise_sd*x, 2) for x in [1]]
max_epoch = [1000]
lr_rate = [0.0005]
n_voxel = 100
full_ver = [True]
pw = [True, False]

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


f_name = f'loss_plot_full_ver-True_pw-FalseTrue_n_vox-{n_voxel}_lr-0.0005_eph-1000.png'

model.plot_loss_history(loss_history, to_x="epoch", to_y="loss",
                        to_label='pw', to_col=None, lgd_title="Precision weight", to_row=None,
                        save_fig=True, save_path=os.path.join(fig_dir, "simulation", "results_2D", 'Epoch_vs_Loss', f_name),
                        ci="sd", n_boot=100, log_y=True, sharey=False)
plt.show()


to_label = 'pw'
params_col, params_group = sim.get_params_name_and_group(params, True)
label_order = model_history[to_label].unique().tolist()
label_order.insert(0, label_order.pop())

df = model_history.query('(noise_sd != 0.0)')


f_name = f'param_history_plot_p_full_ver-True_pw-FalseTrue_sd-0.04_n_vox-{n_voxel}_lr-0.0005_eph-1000.png'
model.plot_param_history(df, params=params_col, group=params_group,
                         to_label='noise_sd', to_col='lr_rate', label_order=None, ground_truth=True,
                         lgd_title="Precision weight", save_fig=False, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name),
                         ci="sd", n_boot=100, log_y=False, sharey=False)
plt.tight_layout()
plt.show()


to_label = 'noise_sd'
params_col, params_group = sim.get_params_name_and_group(params, True)
label_order = model_history[to_label].unique().tolist()
label_order.insert(0, label_order.pop())

df = model_history.query('(lr_rate in [0.0005, "ground_truth"])')


f_name = f'param_history_plot_p_full_ver-True_pw-FalseTrue_sd-0.04_n_vox-{n_voxel}_lr-0.0005_eph-1000.png'
model.plot_param_history_horizontal(model_history, params=params_col, group=params_group,
                         to_label='pw', label_order=None, ground_truth=True,
                         lgd_title='Precision weight', save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name),
                         ci="sd", n_boot=100, log_y=False)
plt.tight_layout()
plt.show()

df = model_history.query('(lr_rate in [0.0005, "ground_truth"])')
f_name = f'final_params_plot_p_full_ver-True_pw-False_sd-0.00to0.32_n_vox-{n_voxel}_lr-0.0005_eph-25000.png'
model.plot_grouped_parameters(df.query('epoch == 24999'), params_col, [1,2,3,4,4,4,4,5,5],
                              to_label="noise_sd", lgd_title="Noise SD", label_order=label_order,
                              save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name))
plt.tight_layout()
plt.show()
