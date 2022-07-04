import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
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

stim_info_path = '/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'
subj_df_dir='/Volumes/derivatives/subj_dataframes'
output_dir = '/Users/auna/Desktop'
save_dir = '/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/'
noise_sd = [0]
max_epoch = [35000]
lr_rate = [0.001]
n_voxel = 3
full_ver = False


syn_data = sim.SynthesizeData(n_voxels=n_voxel, df=None, replace=True, p_dist="data",
                              stim_info_path=stim_info_path, subj_df_dir=subj_df_dir)
syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=full_ver)

# add noise
for cur_noise, cur_lr, cur_epoch in product(noise_sd, lr_rate, max_epoch):
    syn_df = syn_df_2d.copy()
    syn_df['betas'] = sim.add_noise(syn_df['betas'], noise_mean=0, noise_sd=cur_noise)
    syn_df['sigma_v'] = np.ones(syn_df.shape[0], dtype=np.float64)
    syn_SFdataset = model.SpatialFrequencyDataset(syn_df, beta_col='betas')
    syn_model = model.SpatialFrequencyModel(syn_SFdataset.my_tensor, full_ver=False)
    loss_history, model_history, elapsed_time, losses = model.fit_model(syn_model, syn_SFdataset, print_every=2000,
                                                                                    max_epoch=int(cur_epoch),
                                                                 anomaly_detection=False)
    model_f_name = f'model_history_noise-{cur_noise}_lr-{cur_lr}_eph-{cur_epoch}.csv'
    loss_f_name = f'loss_history_noise-{cur_noise}_lr-{cur_lr}_eph-{cur_epoch}.csv'


    ##
    utils.save_df_to_csv(syn_model_history, os.path.join(output_dir, model_f_name), indexing=False)
    utils.save_df_to_csv(syn_loss_history, os.path.join(output_dir, loss_f_name), indexing=False)

# load and make a figure?

model_history, loss_history = sim.load_all_model_fitting_results(output_dir, noise_sd, lr_rate, max_epoch, n_voxels=n_voxel,
                                                                 ground_truth=params, id_val='ground_truth')

for cur_noise, cur_lr, cur_epoch in product(noise_sd, lr_rate, max_epoch):
    f_name = f'loss_noise-{cur_noise}_lr-{cur_lr}_eph-{cur_epoch}.png'
    model.plot_loss_history(loss_history, to_x="epoch", to_y="loss",
                            to_label=None, lgd_title='Learning rate',
                            title=f'learning rate={cur_lr}, without noise',
                            save_fig=False, save_dir=save_dir, save_file_name=f_name,
                            ci="sd", n_boot=100, log_y=True)

to_label = 'lr_rate'
if full_ver is True:
    param_names = params.columns.tolist()
else:
    param_names = [p for p in params.columns if '_' not in p]

label_order = model_history[to_label].unique().tolist()
label_order.insert(0, label_order.pop())

for cur_epoch in [20000, 25000, 30000, 35000]:
    f_name = f'final_params_eph-{cur_epoch}_label-{to_label}_n_params-{len(param_names)}.png'
    model.plot_grouped_parameters(model_history.query('epoch == @cur_epoch-1'), params=param_names, col_group=[1, 2, 3], to_x="params", to_y="value",
                                  to_label=to_label, lgd_title="Learning rate", label_order=label_order,
                                  title=f'Parameters at epoch = {cur_epoch} (100 synthetic voxels)',
                                  save_fig=False, save_dir=save_dir, f_name=f_name)
cur_noise=0
cur_epoch=35000
f_name = f'param_history_noise-{cur_noise}_eph-{cur_epoch}_after6999.png'
model.plot_param_history(model_history.query('epoch > 6999 & lr_rate != 0.0001'), params=param_names, col_group=[1, 2, 3],
                         to_x="epoch", to_y="value", to_row='params',
                        to_label=to_label, label_order=label_order[1:], lgd_title='Learning rate',
                        title=f'Parameter history, without noise',
                        save_fig=False, save_dir=save_dir, save_file_name=f_name,
                        ci="sd", n_boot=100, log_y=True)