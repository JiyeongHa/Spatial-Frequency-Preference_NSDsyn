import os
import sys
import numpy as np
import make_df as mdf
import sfp_nsd_utils as utils
import pandas as pd
import plot_1D_model_results as plotting
import seaborn as sns
import matplotlib.pyplot as plt
import variance_explained as R2
import voxel_selection as vs
import two_dimensional_model as model
import torch
import simulation as sim
from importlib import reload
import binning_eccen as binning
import first_level_analysis as fitting
import bootstrap as bts


params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                       'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                       'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

stim_info_path = '/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'
subj_df_dir='/Volumes/derivatives/subj_dataframes'
output_dir = '/Users/auna/Desktop'
noise_sd = np.linspace(1, 3, 5)
max_epoch = 5
lr_rate = 1e-2

syn_data = sim.SynthesizeData(n_voxels=100, df=None, replace=True, p_dist="data",
                              stim_info_path=stim_info_path, subj_df_dir=subj_df_dir)
syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=False)

# add noise
for i in noise_sd:
    syn_df = syn_df_2d.copy()
    syn_df['betas'] = sim.add_noise(syn_df['betas'], noise_mean=0, noise_sd=i)
    syn_df['sigma_v'] = np.ones(syn_df.shape[0], dtype=np.float64)
    syn_SFdataset = model.SpatialFrequencyDataset(syn_df, beta_col='betas')
    syn_model = model.SpatialFrequencyModel(syn_SFdataset.my_tensor, full_ver=False)
    syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model,
                                                                                    syn_SFdataset,
                                                                                    learning_rate=lr_rate,
                                                                                    max_epoch=max_epoch,
                                                                                    print_every=1,
                                                                                    loss_all_voxels=True,
                                                                                    anomaly_detection=False)
    str_noise_sd = str(i).replace('.', 'p')
    str_lr_rate = str(lr_rate).replace('.', 'p')
    model_f_name = f'model_history_w_noise_{str_noise_sd}_lr_{str_lr_rate}_eph_{max_epoch}.csv'
    loss_f_name = f'loss_history_w_noise_{str_noise_sd}_lr_{str_lr_rate}_eph_{max_epoch}.csv'


    ##
    utils.save_df_to_csv(syn_model_history, output_dir, model_f_name, indexing=False)
    utils.save_df_to_csv(syn_loss_history, output_dir, loss_f_name, indexing=False)

