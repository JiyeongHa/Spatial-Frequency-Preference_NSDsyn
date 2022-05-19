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

df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'
# load subjects df.
subj_list = np.arange(1,9)
all_subj_df = utils.load_all_subj_df(subj_list,
                                     df_dir=df_dir,
                                     df_name='stim_voxel_info_df_LITE.csv')

# break down phase
dv_to_group = ['subj', 'freq_lvl', 'names', 'voxel', 'hemi', 'vroinames']
df = all_subj_df.groupby(dv_to_group).mean().reset_index()
df.drop(['phase'], axis=1, inplace=True)

dv_to_group = all_subj_df.drop(columns=['phase', 'avg_betas', 'stim_idx']).columns.tolist()
df = all_subj_df.groupby(dv_to_group).mean().reset_index()

params =pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                      'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                      'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})
#normalize
normed_betas = model.normalize(filtered_df, to_norm='avg_betas', group_by=["subj", "voxel"])
filtered_df['norm_betas'] = normed_betas

# forward
filtered_df['pred'] = model.Forward(params, 0, filtered_df).two_dim_prediction()
filtered_df['norm_pred'] = model.normalize(filtered_df, to_norm='pred', group_by=["subj", "voxel"])

df[df.columns & colnames]

# plot

dv_to_group = ['subj', 'freq_lvl', 'vroinames', "eccrois"]
labels=filtered_df.names.unique()
avg_df_3 = filtered_df.groupby(dv_to_group).median().reset_index()
my_list = ["annulus", "forward spiral", "reverse spiral", "pinwheel"]
avg_df_3 = avg_df_3.query('names.isin(@my_list)', engine='python')

for sn in np.arange(1,9):
    beta_comp(sn, avg_df_3.query('vroinames == "V1"'), to_subplot='names', to_label="names",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred', set_max=False,
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title=None, labels=None,
              n_row=4, legend_out=True, alpha=0.7,
              save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_stim_class_inV1.png')

for sn in np.arange(1,9):
    model.beta_comp(sn, avg_df_3, to_subplot="vroinames", to_label="eccrois",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred', set_max=True,
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
              n_row=4, legend_out=True, alpha=0.7,
              save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_filtered_df_median.png')

for sn in np.arange(1, 9):
    beta_2Dhist(sn, df, to_subplot="vroinames", to_label='vroinames',
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred',
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title=None, labels=None, bins=200, set_max=False,
              n_row=4, legend_out=True, alpha=0.9,
              save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_2Dhist.png')


for sn in np.arange(1, 9):
    beta_1Dhist(sn, df, save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                save_file_name='1Dhist_comp.png')


# R2
all_subj_R2 = R2.load_R2_all_subj(np.arange(1,9))
R2.R2_histogram(sn_list, all_subj_R2, n_bins=300, save_fig=True, xlimit=30, save_file_name='R2_xlimit_30.png')
R2.R2_histogram(sn_list, all_subj_R2, n_bins=300, save_fig=True, xlimit=100, save_file_name='R2_xlimit_100.png')

# voxel selection
# compare num of voxels before & after applying voxel_selection()
n_voxel_df = vs.count_voxels(all_subj_df, dv_to_group=["subj", "vroinames"], count_beta_sign=True)
vs.plot_num_of_voxels(n_voxel_df, save_fig=True,
                      save_file_name='n_voxels_ROI_beta_sign.png',
                      super_title='All voxels')

pos_df = vs.drop_voxels_with_mean_negative_amplitudes(df)
n_voxel_pos_df = vs.count_voxels(pos_df, dv_to_group=["subj", "vroinames"], count_beta_sign=True)
vs.plot_num_of_voxels(n_voxel_pos_df, save_fig=True,
                      save_file_name='n_voxels_ROI_beta_sign_after_drop.png',
                      super_title='Voxels with positive mean betas only')

# drop voxels
filtered_df = vs.drop_voxels_outside_stim_range(pos_df)
n_voxel_df_0 = vs.count_voxels(df, dv_to_group=["subj", "vroinames"], count_beta_sign=False)
n_voxel_df_1 = vs.count_voxels(pos_df, dv_to_group=["subj", "vroinames"], count_beta_sign=False)
n_voxel_df_2 = vs.count_voxels(filtered_df, dv_to_group=["subj", "vroinames"], count_beta_sign=False)

n_voxel_df_0 = n_voxel_df_0.rename(columns={'n_voxel': 'n_voxel_all'})
n_voxel_df_1 = n_voxel_df_1.rename(columns={'n_voxel': 'n_voxel_positive_mean_beta_only'})
n_voxel_df_2 = n_voxel_df_2.rename(columns={'n_voxel': 'n_voxel_positive_and_stim_range'})
new_n_voxel_df = n_voxel_df_0.merge(n_voxel_df_1, on=['subj', 'vroinames'])
new_n_voxel_df = new_n_voxel_df.merge(n_voxel_df_2, on=['subj', 'vroinames'])

new_n_voxel_df = pd.melt(new_n_voxel_df, id_vars=['subj', 'vroinames'], value_vars=['n_voxel_all', 'n_voxel_positive_mean_beta_only'],
        var_name='voxel_selection', value_name='n_voxel')
vs.plot_num_of_voxels(new_n_voxel_df, new_legend=['All voxels', 'Voxels with positive mean betas'],
                      to_hue='voxel_selection', legend_title=None,
                      super_title='Voxel selection', save_fig=True,
                      save_file_name='change_in_n_voxel_2.png')

subj01_df = filtered_df.query('subj == "subj01" & vroinames == "V1"')

# 2D model fitting
test_model = model.SpatialFrequencyModel()
optimizer = torch.optim.Adam(test_model.parameters(), lr=5e-3)

theta_l = subj01_df['local_ori']
w_l = subj01_df['local_sf']
theta_v = subj01_df['angle']
r_v = subj01_df['eccentricity']

losses = []
param_vals = []
pbar = range(100)
for i in pbar:
    # these next four lines are the core of the loop:
    # 1. generate the prediction
    y_pred = test_model.forward(theta_l, theta_v, r_v, w_l)
    # 2. compute the loss
    loss = model.objective_func(y_pred, subj01_df['avg_betas'].values)
    # 3. compute all the gradients ("run the backwards pass")
    loss.backward()
    # 4. update all the parameters ("step the optimizer")
    optimizer.step()
    # these next lines are just to keep track of the loss and parameter values over time
    # .item() and .clone().detach() do the same thing -- make sure that we just get the *value*,
    # rather than the *object* itself (which will update on each loop) (.item() only works for scalars)
    losses.append(loss.item())
    param_vals.append(test_model.sigma.clone().detach())
    #pbar.set_postfix(loss=losses[-1], params=param_vals[-1])
# turn this list of 1d tensors into one 2d tensor
param_vals = torch.stack(param_vals)