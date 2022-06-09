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
import plot_1D_model_results as plotting


df_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'
# load subjects df.
subj_list = np.arange(1, 9)
all_subj_df = utils.load_all_subj_df(subj_list,
                                     df_dir=df_dir,
                                     df_name='stim_voxel_info_df_LITE.csv')

# break down phase
dv_to_group = ['subj', 'freq_lvl', 'names', 'voxel', 'hemi', 'vroinames']
df = all_subj_df.groupby(dv_to_group).mean().reset_index()
df.drop(['phase'], axis=1, inplace=True)

dv_to_group = all_subj_df.drop(columns=['phase', 'avg_betas', 'stim_idx']).columns.tolist()
df = all_subj_df.groupby(dv_to_group).mean().reset_index()

params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                       'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                       'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})
# normalize
normed_betas = model.normalize(filtered_df, to_norm='avg_betas', group_by=["subj", "voxel"])
filtered_df['norm_betas'] = normed_betas

# forward
filtered_df['pred'] = model.Forward(params, 0, filtered_df).two_dim_prediction()
filtered_df['norm_pred'] = model.normalize(filtered_df, to_norm='pred', group_by=["subj", "voxel"])

df[df.columns & colnames]

# plot

dv_to_group = ['subj', 'freq_lvl', 'vroinames', "eccrois"]
labels = filtered_df.names.unique()
avg_df_3 = filtered_df.groupby(dv_to_group).median().reset_index()
my_list = ["annulus", "forward spiral", "reverse spiral", "pinwheel"]
avg_df_3 = avg_df_3.query('names.isin(@my_list)', engine='python')

for sn in np.arange(1, 9):
    beta_comp(sn, avg_df_3.query('vroinames == "V1"'), to_subplot='names', to_label="names",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred', set_max=False,
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title=None, labels=None,
              n_row=4, legend_out=True, alpha=0.7,
              save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_stim_class_inV1.png')

for sn in np.arange(1, 9):
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
all_subj_R2 = R2.load_R2_all_subj(np.arange(1, 9))
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

new_n_voxel_df = pd.melt(new_n_voxel_df, id_vars=['subj', 'vroinames'],
                         value_vars=['n_voxel_all', 'n_voxel_positive_mean_beta_only'],
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
    # pbar.set_postfix(loss=losses[-1], params=param_vals[-1])
# turn this list of 1d tensors into one 2d tensor
param_vals = torch.stack(param_vals)

#
subj = "subj01"
subj_df = filtered_df.query('subj == @subj')

ww = subj_df[['voxel', 'local_ori', 'angle', 'eccentricity', 'local_sf', 'avg_betas']].copy()
www = subj_df[['eccentricity', 'voxel', 'local_ori', 'angle', 'local_sf', 'avg_betas']].copy()
ww_np = ww.groupby('voxel').apply(np.asarray).values

# dataset
subj_data = model.SpatialFrequencyDataset(subj_df)
# model
my_model = model.SpatialFrequencyModel(subj_data.my_tensor)

# fitting
# model.fit_model(model=my_model, dataset=subj_data, learning_rate=1e-3, max_epoch=1000)
#
# my_parameters = [p for p in my_model.parameters() if p.requires_grad]
#
# optimizer = torch.optim.Adam(my_parameters, lr=learning_rate)
# loss_history = []
# model_history = []
# for t in range(10):
#     pred = my_model.forward()  # predictions should be put in here
#     loss = model.loss_fn(subj_data.voxel_info, prediction=pred, target=subj_data.target)  # loss should be returned here
#     model_values = [p.detach().numpy().item() for p in my_model.parameters() if p.requires_grad]# output needs to be put in there
#     loss_history.append(loss.item())
#     model_history.append(model_values)  # more than one item here
#     print(f'loss at {t}: {loss.item()}')
#     print(f'param values at {t}: {model_values}')
#
#     optimizer.zero_grad()  # clear previous gradients
#     loss.backward()  # compute gradients of all variables wrt loss
#     optimizer.step()  # perform updates using calculated gradients
#     my_model.eval()
param_cols = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

filtered_V1_df = filtered_df.query('vroinames == "V1"')
loss_history_df = {}
model_history_df = {}
time_subj = []
subj_list = filtered_V1_df['subj'].unique()
subj_list = ['subj02', 'subj06']
for subj in subj_list:
    subj_df = filtered_V1_df.query('subj == @subj')
    print(f'##### {subj} #####\n')
    # dataset
    subj_data = model.SpatialFrequencyDataset(subj_df)
    # model
    my_model = model.SpatialFrequencyModel(subj_data.my_tensor)
    loss_history, model_history, elapsed_time = model.fit_model(my_model, subj_data)
    time_subj.append(elapsed_time)
    loss_history_df[subj] = pd.DataFrame(loss_history, columns=['loss'])
    loss_history_df[subj]['subj'] = subj
    model_history_df[subj] = pd.DataFrame(model_history, columns=param_cols)
    model_history_df[subj]['subj'] = subj
    print(f'##### {subj} has finished! #####\n')


loss_history_df = pd.concat(loss_history_df).reset_index().drop(columns={'level_0'}).rename(
    columns={'level_1': 'epoch'})
model_history_df = pd.concat(model_history_df).reset_index().drop(columns={'level_0'}).rename(
    columns={'level_1': 'epoch'})
model.plot_loss_history(loss_history_df, save_fig=True,
                        save_file_name='loss_change.png')

val_vars = model_history_df.drop(columns=['subj', 'epoch']).columns.tolist()
model_history_df = model_history_df.melt(id_vars=['subj', 'epoch'],
                                         value_vars=val_vars,
                                         var_name='param',
                                         value_name='value')
model_history_df['study_type'] = 'NSD synthetic'
params['subj'] = 'broderick'
params['study_type'] = 'Broderick et al.(2022)'
params['epoch'] = 999
params.drop(columns=['A_3', 'A_4'])
broderick_params = params.melt(id_vars=['subj', 'epoch', 'study_type'],
                               value_vars=val_vars,
                               var_name='param',
                               value_name='value')
model_history_df = model_history_df.append(broderick_params, ignore_index=True)
model.plot_parameters(model_history_df.query('epoch == 999'), to_x_axis='param', save_fig=True,
                      save_file_name='final_param_v1.png')

hue_order = [utils.sub_number_to_string(p) for p in np.arange(1,9)]
model.plot_parameters(model_history_df.query('epoch == 999'), to_x_axis='param',
                      to_label="subj", hue_order=None, legend_title="subjects", save_fig=True,
                      save_file_name='final_param_v1_individual.png')

#save filtered df
df_save_path = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes/filtered_df_all_subj.csv'
filtered_df.to_csv(df_save_path, index=False)

#save model_history_df, loss_history_df
df_save_path = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/model_results/results_2D/model_history_V1.csv'
model_history_df.to_csv(df_save_path, index=False)
df_save_path = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/model_results/results_2D/loss_history_V1.csv'
loss_history_df.to_csv(df_save_path, index=False)

# model recovery

syn_df = sim.generate_synthesized_data()
params_new = pd.concat([params]*2, ignore_index=True)
params_new.iloc[1, 3:9] = 0
syn_model = model.Forward(params_new, 0, syn_df)
syn_df['avg_betas'] = syn_model.two_dim_prediction(full_ver=False)

syn_SFdataset = model.SpatialFrequencyDataset(syn_df)
# model

syn_loss_history_df = {}
syn_model_history_df = {}
syn_time_subj = []
noise_sd_levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
for i in noise_sd_levels:
    print(f'##### Noise level: {str(i)} #####\n')
    syn_noise_df = syn_df.copy()
    syn_noise_df['avg_betas'] = sim.add_noise(syn_noise_df.avg_betas, noise_mean=0, noise_sd=i)
    syn_SFdataset = model.SpatialFrequencyDataset(syn_noise_df)
    syn_model = model.SpatialFrequencyModel(syn_SFdataset.my_tensor, full_ver=False)
    syn_loss_history, syn_model_history, syn_elapsed_time = model.fit_model(syn_model, syn_SFdataset, learning_rate=5e-3, max_epoch=3000, anomaly_detection=False)
    syn_time_subj.append(syn_elapsed_time)
    syn_loss_history_df[f'noise sd = {i}'] = pd.DataFrame(syn_loss_history, columns=['loss']).reset_index().rename(columns={'index': 'epoch'})
    syn_model_history_df[f'noise sd = {i}'] = pd.DataFrame(syn_model_history, columns=param_cols[0:3])


model.plot_loss_history(syn_loss_history, save_fig=True, save_file_name='no_noise_simulation.png',
                        title="Loss change: 100 voxel simulation without noise")


syn_loss_history_df = pd.concat(syn_loss_history_df).reset_index().rename(
    columns={'level_0': 'Noise', 'level_1': 'epoch'})
syn_model_history_df = pd.concat(syn_model_history_df).reset_index().rename(
    columns={'level_0': 'Noise', 'level_1': 'epoch'})
ground_truth= pd.DataFrame({'Noise': ['Ground truth'], 'epoch': [2999], 'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35]})
syn_model_history_df = syn_model_history_df.append(ground_truth, ignore_index=True)
val_vars = syn_model_history_df.drop(columns=['Noise', 'epoch']).columns.tolist()
syn_model_history_df = syn_model_history_df.melt(id_vars=['Noise', 'epoch'],
                                         value_vars=val_vars,
                                         var_name='param',
                                         value_name='value')
syn_model_history_df = syn_model_history_df.append(ground_truth, ignore_index=True)
syn_loss_history_df['Noise'].unique()
syn_model_history_df = syn_model_history_df.replace({'noise sd = 0': '0', 'noise sd = 0.25': '0.25',
                             'noise sd = 0.5': '0.5', 'noise sd = 0.75':'0.75',
                             'noise sd = 1': '1', 'noise sd = 1.25':'1.25',
                             'noise sd = 1.5': '1.5'})
np.linspace()
model.plot_loss_history(syn_loss_history_df[syn_loss_history_df.epoch % 2 == 0], to_label="Noise",
                        save_fig=True, save_file_name='simulation.png',
                        legend_title='Noise sd',
                       title="Loss change: 100 voxel simulation", ci='sd')
id_list = ["Ground truth", "0", "0.5", "1", "1.5"]
id_list = ["Ground truth", "0", "0.25", "0.5", "0.75", "1", "1.25", "1.5"]

syn_model_history_df_fil = syn_model_history_df.query('Noise in @id_list')
model.plot_parameters(syn_model_history_df.query('epoch == 2999 & param == "sigma"'),
                      to_label="Noise", hue_order=id_list, legend_title="Noise sd", rotate_ticks=False,
                      title="Model recovery with different noise levels",
                      save_fig=True, save_file_name="simulation_sigma.png")

model.plot_parameters(syn_model_history_df.query('epoch == 2999 & param != "sigma"'),
                      to_label="Noise", hue_order=id_list, legend_title="Noise sd", rotate_ticks=False,
                      title="Model recovery with different noise levels",
                      save_fig=True, save_file_name="simulation_slope_else.png")

model_history_df['study_type'] = 'NSD synthetic'
params['subj'] = 'broderick'
params['study_type'] = 'Broderick et al.(2022)'
params['epoch'] = 999
params.drop(columns=['A_3', 'A_4'])
broderick_params = params.melt(id_vars=['subj', 'epoch', 'study_type'],
                               value_vars=val_vars,
                               var_name='param',
                               value_name='value')
model_history_df = model_history_df.append(broderick_params, ignore_index=True)
model.plot_parameters(model_history_df.query('epoch == 999'), to_x_axis='param', save_fig=True,
                      save_file_name='final_param_v1.png')

hue_order = [utils.sub_number_to_string(p) for p in np.arange(1,9)]
model.plot_parameters(model_history_df.query('epoch == 999'), to_x_axis='param',
                      to_label="subj", hue_order=None, legend_title="subjects", save_fig=True,
                      save_file_name='final_param_v1_individual.png')


import binning_eccen as binning
import first_level_analysis as fitting

bin_list = np.round(np.linspace(filtered_df.eccentricity.min(), filtered_df.eccentricity.max(), 6),2).tolist()
bin_labels = [f'{str(a)}-{str(b)}' for a, b in zip(bin_list[:-1], bin_list[1:])]

filtered_b_df = binning.bin_ecc(filtered_df, bin_list, bin_labels)
for sn in np.arange(1, 9):
    binning.plot_bin_histogram(sn, filtered_b_df, labels=bin_labels, to_x_axis='local_sf', to_subplot="freq_lvl",
                               normalize=False, x_axis_label='Local SF', save_fig=True,
                               save_file_name='histogram_for_bins.png', n_rows=6, top_m=0.8, right_m=0.86)


mode_filtered_b_df = binning.summary_stat_for_ecc_bin(filtered_b_df, to_bin=["avg_betas", "local_sf", "eccentricity"], central_tendency="mode")
mean_filtered_b_df = binning.summary_stat_for_ecc_bin(filtered_b_df, to_bin=["avg_betas", "local_sf", "eccentricity"], central_tendency="mean")

stim_class_list = ['annulus', 'forward spiral', 'pinwheel', 'reverse spiral']
varea_list = mode_filtered_b_df.vroinames.unique()

model_history_df = {}
loss_history_df = {}
for sn in np.arange(1, 9):

    subj = utils.sub_number_to_string(sn)
    m_m_df = {}
    l_l_df = {}
    for stim_class in stim_class_list:
        m_df = {}
        l_df = {}
        for varea in varea_list:
            m_df[varea], l_df[varea], e_time = fitting.fit_1D_model(mean_filtered_b_df, sn=sn, stim_class=stim_class,
                                                            varea=varea, ecc_bins="bins", n_print=10000,
                                                            initial_val=[1,1,1], epoch=50000, lr=1e-3)

        m_m_df[stim_class] = pd.concat(m_df)
        l_l_df[stim_class] = pd.concat(l_df)
    model_history_df[subj] = pd.concat(m_m_df)
    loss_history_df[subj] = pd.concat(l_l_df)

col_replaced = {'level_0': 'subj', 'level_1': 'names', 'level_2': 'vroinames'}
model_history_df = pd.concat(model_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)
loss_history_df = pd.concat(loss_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)

# ["bins", "vroinames", "names", "freq_lvl"]
merged_df = plotting.merge_pdf_values(model_history_df.query('epoch == 49999'), subj_df=mean_filtered_b_df,
                                      merge_on_cols=['subj', 'vroinames', 'bins', 'names'])

merged_df = merged_df.merge(mean_filtered_b_df[['subj', 'vroinames', 'bins', 'names', 'eccentricity']],
                            on=['subj', 'vroinames', 'bins', 'names'])

bin_middle = [np.round((a+b)/2,2) for a, b in zip(bin_list[1:], bin_list[:-1])]

def label_eccen(row, bin_label, bin_list):
   if row['bins'] == bin_label[0]:
      return bin_list[0]
   if row['bins'] == bin_label[1]:
      return bin_list[1]
   if row['bins'] == bin_label[2]:
      return bin_list[2]
   if row['bins'] == bin_label[3]:
      return bin_list[3]
   if row['bins'] == bin_label[4]:
      return bin_list[4]

merged_df['eccentricity'] = merged_df.apply(lambda row: label_eccen(row, bin_label=bin_labels, bin_list=bin_middle), axis=1)

bin_list[]

model.plot_loss_history(loss_history_df[loss_history_df.epoch % 100 == 0], title="Loss change during 1D model fitting (N = 9)",
                        to_label="ecc_bins", to_subplot="names", n_rows=4, labels=bin_labels, ci=68, n_boot=100, save_fig=True, save_file_name="loss_for_1D.png")

for varea in ["V2", "V3", "V4v"]:
    plotting.plot_beta_all_subj(subj_to_run=np.arange(1,9),
                                merged_df = merged_df.query('vroinames == @varea'),
                                to_subplot="names", n_sp_low=4, labels=bin_labels,
                                to_label="bins", save_fig=True, save_file_name=f'_{varea}.png')


plotting.plot_preferred_period(merged_df, save_fig=True, save_file_name='pf_period_all_rois_median.png',
                               labels=merged_df.names.unique(), title="Spatial Frequency Tuning (N = 9)", ci=68, estimator=np.median, legend=True)


plotting.plot_preferred_period(merged_df.query('preferred_period < 10'), save_fig=True, save_file_name='pf_period_all_rois_median_less_than_10.png',
                               labels=merged_df.names.unique(), title="Spatial Frequency Tuning (N = 9)", ci=68, estimator=np.median, legend=True)


merged_df.query('vroinames == "V1" & names == "annulus" & eccentricity == ")
df = merged_df
df['preferred_period'] = 1 / df[dp_to_y_axis]
col_order = utils.sort_a_df_column(df[to_subplot])

w = merged_df.groupby(['names', 'vroinames', 'bins']).median()

grid = sns.FacetGrid(df,
                     hue=to_label,
                     hue_order=labels,
                     palette=sns.color_palette("husl"),
                     legend_out=True,
                     sharex=True, sharey=False)
grid.map(sns.lineplot, dp_to_x_axis, 'preferred_period', estimator=np.mean, ci="sd", marker='o', err_style='bars')
sns.lineplot(data=df.query('vroinames == "V1"'), x="eccentricity", y="preferred_period", hue='names', linestyle='',  ci=68)
plt.show()