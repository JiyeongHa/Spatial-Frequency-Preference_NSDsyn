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
import voxel_selection as vs
import seaborn as sns


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


f_name = f'final_params_plot_p_full_ver-True_pw-False_n_mptl-1_n_vox-{n_voxel}_lr-0.01to0.0005_eph-25000.png'
model.plot_grouped_parameters(df, params_col, [1,2,3,4,4,4,4,5,5],
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


sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
all_subj_df = utils.load_all_subj_df(sn_list, '/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes', 'stim_voxel_info_df_vs_md.csv', dataset="broderick")
df = utils.load_all_subj_df(np.arange(1,9), df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes',
                            df_name='stim_voxel_info_df_vs.csv')
df = df.query('vroinames == "V1"')
sigma_v_df = bts.get_multiple_sigma_vs(df, power=[1,2], columns=['normed_noise_SD', 'normed_sigma_v_squared'], to_group=['voxel','subj'])
nsd_sigma_v_df = pd.read_csv('/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes/sigma_v/sigma_v_normed_betas_V1.csv')
bd_sigma_v_df = pd.read_csv('/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/sigma_v/sigma_v_normed_betas_V1.csv')
nsd_sigma_v_df['dataset'] = "nsd_syn"
bd_sigma_v_df['dataset'] = "Broderick_et_al"

all_sigma_df = pd.concat((nsd_sigma_v_df, bd_sigma_v_df), axis=0)
sim.plot_sd_histogram(bd_sigma_v_df, to_x="normed_noise_SD", x_label="SD for each voxel across 100 bootstraps", to_label='subj', lgd_title='Subjects',
                      f_name="Broderick_sd_histogram_normed.png", height=5, save_fig=True)
plt.tight_layout()
plt.show()

sim.plot_sd_histogram(all_sigma_df, to_x="normed_noise_SD", x_label="SD for each voxel", to_label='dataset', lgd_title='Dataset',
                      f_name="Broderick_sd_histogram_vs_nsd_syn.png", height=5, save_fig=True)
plt.tight_layout()
plt.show()

n_voxel_df = all_sigma_df.groupby(['subj','dataset'])['voxel'].nunique().reset_index()
n_voxel_df = n_voxel_df.rename(columns={'voxel': 'n_voxel'})
vs.plot_num_of_voxels(n_voxel_df, to_hue='dataset', legend_title="Dataset",
                      x_axis='dataset', y_axis='n_voxel', x_axis_label='Dataset',
                      save_file_name='n_voxels_diff_dataset.png', save_fig=True)
plt.show()


model_history, loss_history = sim.load_all_model_fitting_results(output_dir, full_ver, pw, noise_sd, n_voxel, lr_rate, max_epoch,
                                                                 ground_truth=params, id_val='ground_truth')

output_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/derivatives_HPC/Broderick_dataset/sfp_model/results_2D'
loss_history, model_history, _ = model.load_loss_and_model_history_Broderick_subj(output_dir, full_ver=[True],
                                                                                 sn_list=sn_list, lr_rate=[0.0005, 0.001],
                                                                                       max_epoch=[20000], losses=False)

f_name = 'broderick_all_subj_0.0005_vs_0.001.png'
model.plot_loss_history(loss_history, to_x="epoch", to_y="loss",
                        to_label='lr_rate', to_col=None, lgd_title='Learning rate', to_row=None,
                        save_fig=True, save_path=os.path.join(fig_dir, "simulation", "results_2D", 'Epoch_vs_Loss',
                                                              f_name), ci=68, n_boot=100, log_y=True, sharey=True)
plt.show()

model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')

f_name = 'Broderick_all_subj_model_param_history_sab.png'
model.plot_param_history_horizontal(model_history, params=params_col[0:3], group=params_group[0:3],
                         to_label='params', label_order=None, ground_truth=True,
                         lgd_title=None, save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name),
                         ci=68, n_boot=100, log_y=False)
plt.show()


f_name = 'Broderick_all_subj_model_param_history_pterms.png'
model.plot_param_history_horizontal(model_history, params=params_col[3:7], group=params_group[3:7],
                         to_label='params', label_order=None, ground_truth=True,
                         lgd_title=None, save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name),
                         ci=68, n_boot=100, log_y=False)
plt.show()


f_name = 'Broderick_all_subj_model_param_history_lr_rate_0.0005_vs_0.001.png'
model.plot_param_history_horizontal(model_history, params=params_col, group=np.arange(0,9),
                         to_label='lr_rate', label_order=label_order, ground_truth=False,
                         lgd_title=None, save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name),
                         ci=68, n_boot=100, log_y=False)
plt.show()
ori_subj = model_history.subj.unique().tolist()
new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
subj_replace_dict = dict(zip(ori_subj, new_subj))
model_history = model_history.replace({'subj': subj_replace_dict})
final_df = model_history.query('epoch == 19999 & subj != "ground_truth"')
to_label = 'lr_rate'
label_order = final_df[to_label].unique().tolist()
f_name = 'Broderick_all_subj_final_params_lr_0.0005_vs_0.001.png'



model.plot_grouped_parameters(final_df, params_col, [1,2,3,4,4,5,5,6,6],
                              to_label="lr_rate", lgd_title="Learning rate", label_order=label_order, height=9,
                              save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name))
plt.show()

model.plot_grouped_parameters_subj(final_df, params_col, np.arange(0,9),
                              to_label="lr_rate", lgd_title="Learning rate", label_order=label_order, height=9,
                              save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name))
plt.show()

df.to_csv(os.path.join(df_dir, 'tmp', f"{subj}_stim_voxel_info_df_vs.csv"))
fnl_df = df.groupby(['voxel', 'subj', 'names', 'freq_lvl', 'class_idx']).median().reset_index()

bd_df = pd.read_csv('/Users/jh7685/Documents/Github/spatial-frequency-preferences/data/tuning_2d_model/individual_subject_params.csv')
bd_df = bd_df.rename(columns={'model_parameter': 'params', 'subject':'subj', 'fit_value':'value'})
orig_col = ['sigma', 'abs_amplitude_cardinals', 'abs_amplitude_obliques',
       'rel_amplitude_cardinals', 'rel_amplitude_obliques',
       'abs_mode_cardinals', 'abs_mode_obliques', 'rel_mode_cardinals',
       'rel_mode_obliques', 'sf_ecc_slope', 'sf_ecc_intercept']
new_col = ['sigma','A_1','A_2','A_3','A_4','p_1','p_2','p_3','p_4','slope','intercept']
param_replace_col = dict(zip(orig_col, new_col))
bd_df = bd_df.replace({'params': param_replace_col, 'subj': subj_replace_dict})
bd_df = bd_df.drop(columns=['fit_model_type'])
bd_df_wide = bd_df.pivot(index=['subj','bootstrap_num'], columns='params', values='value').reset_index()

f_name='broderick_data_median.png'
model.plot_grouped_parameters_subj(bd_df_wide, params_col, np.arange(0,9),
                              to_label="subj", lgd_title="Subj", label_order=label_order, height=9,
                              save_fig=True, save_path=os.path.join(fig_dir, 'Epoch_vs_ParamValues', f_name))
plt.show()

final_bd_df = bd_df.groupby(['subj', 'params']).median().reset_index()
final_bd_df = final_bd_df.drop(columns='bootstrap_num')
final_bd_df = final_bd_df.rename(columns={'value': 'Broderick_value'})
final_bd_df = final_bd_df.query('params in @params_col')

final_long_df = pd.melt(final_df, id_vars=['subj'], value_vars=params_col, var_name='params', value_name='My_value')
df = final_bd_df.merge(final_long_df, on=['subj','params'])

param_list = params_col
f_name='scatter_comparison.png'
grid = model.scatter_comparison(df.query('params in @param_list'),
                                x="Broderick_value", y="My_value", col="params",
                                col_order=param_list, label_order=label_order,
                                to_label='subj', lgd_title="Subjects", height=5,
                                save_fig=True, save_path=os.path.join(fig_dir, 'Broderick_value_vs_My_value', f_name))
plt.show()
grid = sns.FacetGrid(df,
                     col="params",
                     col_wrap=3,
                     palette=pal,
                     hue="subj",
                     height=7,
                     hue_order=label_order,
                     legend_out=True,
                     sharex=False, sharey=False)
grid.map(sns.relplot, x="Broderick_value", y="My_value", kind="scatter")
plt.show()

syn_df = pd.read_csv('/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-True_noise_mtpl-1_n_vox-100.csv')
syn_df['class_idx'] = np.tile(np.arange(0, 28), 100)
syn_df = syn_df.query('voxel < 5')
#
# syn_SFdataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
# syn_model = model.SpatialFrequencyModel(syn_SFdataset.my_tensor, full_ver=True)
# loss_history, model_history, el_time, losses = model.fit_model(syn_model, syn_SFdataset, max_epoch=10)

from torch.utils import data as torchdata

syn_SFdatset_new = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
syn_model_new = model.SpatialFrequencyModel(full_ver=True)
dataloader = torchdata.DataLoader(syn_SFdatset_new, 10)
# my_parameters = [p for p in syn_model_new.parameters() if p.requires_grad]
# optimizer = torch.optim.Adam(my_parameters, lr=0.005)
# pred = syn_model_new.forward()
log_file='/Users/jh7685/Documents/test_log.txt'
loss_history, model_history, el_time, losses = model.fit_model(syn_model_new, syn_SFdatset_new, log_file, max_epoch=10, print_every=2)


#
sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
all_subj_df = utils.load_all_subj_df(sn_list, df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
                                 df_name='stim_voxel_info_df_vs_md.csv', dataset="broderick")
sns.histplot(data = all_subj_df, x = "angle", stat = "probability", hue="subj")
utils.save_fig(True, os.path.join(fig_dir, 'theta_v_histogram.png'))
plt.show()

sns.histplot(data = all_subj_df, x = "local_ori", stat = "probability", hue="subj")
utils.save_fig(True, os.path.join(fig_dir, 'theta_l_histogram.png'))
plt.show()


subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
subj_model = model.SpatialFrequencyModel(full_ver=True)
ori_plot = subj_dataset.ori.numpy()
angle_plot = subj_dataset.angle.numpy()

theta_l_theta_v= ori_plot-angle_plot

n_bins = 100
x = theta_l_theta_v
plt.hist(x, n_bins, density=True,
         histtype='bar')
plt.title('l-v after torch\n\n',
          fontweight="bold")
plt.tight_layout()
plt.show()

torch.autograd.set_detect_anomaly(False)
# [sigma, slope, intercept, p_1, p_2, p_3, p_4, A_1, A_2]
my_parameters = [p for p in subj_model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(my_parameters, lr=0.0005, amsgrad=False, eps=1e-8)
losses_history = []
loss_history = []
model_history = []
start = timer()

for t in range(max_epoch):
    subj_model.get_Pv()
    pred = subj_model.forward(theta_l=subj_dataset.ori, theta_v=subj_dataset.angle, r_v=subj_dataset.eccen, w_l=subj_dataset.sf)  # predictions should be put in here
    losses = model.loss_fn(subj_dataset.sigma_v_squared, pred, subj_dataset.target) # loss should be returned here
    loss = torch.mean(losses)
    model_values = [p.detach().numpy().item() for p in model.parameters() if p.requires_grad]  # output needs to be put in there
    loss_history.append(loss.item())
    model_history.append(model_values)  # more than one item here
    if (t + 1) % print_every == 0 or t == 0:
        with open(log_file, "a") as file:
            content = f'**epoch no.{t} loss: {np.round(loss.item(), 3)} \n'
            file.write(content)
            file.close()

    optimizer.zero_grad()  # clear previous gradients
    loss.backward()  # compute gradients of all variables wrt loss
    optimizer.step()  # perform updates using calculated gradients
    model.eval()
end = timer()
elapsed_time = end - start
params_col = [name for name, param in model.named_parameters() if param.requires_grad]
with open(log_file, "a") as file:
    file.write(f'**epoch no.{max_epoch}: Finished! final model params...\n {dict(zip(params_col, model_values))}\n')
    file.write(f'Elapsed time: {np.round(end - start, 2)} sec \n')
    file.close()
voxel_list = dataset.voxel_info
loss_history = pd.DataFrame(loss_history, columns=['loss']).reset_index().rename(columns={'index': 'epoch'})
model_history = pd.DataFrame(model_history, columns=params_col).reset_index().rename(columns={'index': 'epoch'})

loss_history, model_history, elapsed_time, losses = model.fit_model(subj_model, subj_dataset, output.log_file,
    learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch),
    print_every=100, loss_all_voxels=False,
    anomaly_detection=False, amsgrad=False, eps=1e-8)

orig_n_voxels = vs.count_voxels(all_subj_df, dv_to_group=['subj'], count_beta_sign=False)
new_df = vs.drop_voxels_near_border(all_subj_df, groupby_col=['subj','voxel'])
new_n_voxels = vs.count_voxels(new_df, dv_to_group=['subj'], count_beta_sign=False)


orig_n_voxels['type'] = 'before border removal'
new_n_voxels['type'] = 'after border removal'
voxel_df = pd.concat((orig_n_voxels, new_n_voxels), axis=0)

vs.plot_num_of_voxels(voxel_df, to_hue='type', legend_title='type',
                      x_axis='type', x_axis_label='Type',
                      save_fig=True, save_file_name='n_voxels_border_removal.png')
plt.show()
