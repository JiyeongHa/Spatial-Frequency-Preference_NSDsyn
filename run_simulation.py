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
subj_df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/dataframes'
output_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/simulation/results_2D'
fig_dir = '/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/'
measured_noise_sd =0.03995  # unnormalized 1.502063
noise_sd = [1,2]
max_epoch = [3]
lr_rate = [0.01]
n_voxel = 100
full_ver = [True]
full_ver=True


syn_data = sim.SynthesizeData(n_voxels=n_voxel, p_dist="data", to_noise_sample='normed_betas',
                              stim_info_path=stim_info_path, subj_df_dir=subj_df_dir)
syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=True)

sim.copy_df_and_add_noise(syn_df_2d, beta_col="normed_beta", noise_mean=0, noise_sd=cur_noise)


for cur_noise, cur_lr, cur_epoch in product(noise_sd, lr_rate, max_epoch):
    syn_df = syn_df_2d.copy()
    syn_df['normed_betas'] = sim.add_noise(syn_df['normed_betas'], noise_mean=0, noise_sd=cur_noise)
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

model_history, loss_history = sim.load_all_model_fitting_results(output_dir, full_ver, noise_sd, lr_rate, max_epoch, n_voxels=n_voxel,
                                                                 ground_truth=params, id_val='ground_truth')

for cur_ver, cur_noise, cur_lr, cur_epoch in product(full_ver, noise_sd, lr_rate, max_epoch):
    f_name = f'loss_plot_full_ver-{cur_ver}_sd-{cur_noise}_n_vox-{n_voxel}_lr-{cur_lr}_eph-{cur_epoch}.png'
    model.plot_loss_history(loss_history, to_x="epoch", to_y="loss",
                            to_label=None, lgd_title='Learning rate', to_row=None,
                            title=f'100 synthetic voxel simulation with noise',
                            save_fig=False, save_path=os.path.join(fig_dir, 'Epoch_vs_Loss', f_name),
                            ci="sd", n_boot=100, log_y=True, adjust="tight")
    plt.show()


to_label = 'noise_sd'
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

for cur_ver, cur_noise, cur_lr, cur_epoch in product(full_ver, noise_sd, lr_rate, max_epoch):
    params_col, group = sim.get_params_name_and_group(params, cur_ver)
    f_name = f'param_history_plot_full_ver-{cur_ver}_sd-{cur_noise}_n_vox-{n_voxel}_lr-{cur_lr}_eph-{cur_epoch}.png'
    model.plot_param_history(model_history, params=params_col, group=group, to_label=None,
                             to_col=None, lgd_title=None,
                             title=f'100 synthetic voxel simulation with noise', save_fig=False,
                             save_path=os.path.join(fig_dir, 'Epoch_vs_PramValues', f_name), ci="sd", n_boot=100,
                             log_y=False, adjust=[0.9, 0.85])
    plt.show()


f_name = 'ww'
params_col = params.columns.tolist()[:3]
model.plot_param_history(model_history, params=params_col, group=list(range(len(params_col))),
                         to_label=None, label_order=None, ground_truth=True, to_col=None,
                         lgd_title=None, title='wow',
                         save_fig=False, save_path=os.path.join(fig_dir, 'Epoch_vs_PramValues', f_name),
                         ci=68, n_boot=100, log_y=True, adjust="tight", sharey=False)

df = model._group_params(model_history, params_col, list(range(len(params_col))))
sns.set(font_scale=1.3)
to_x = "epoch"
to_y = "value"
x_label = "Epoch"
y_label = "Parameter value"
grid = sns.FacetGrid(df.query('lr_rate != "ground_truth"'),
                     hue=None,
                     hue_order=None,
                     row='params',
                     col=None,
                     palette=sns.color_palette("rocket"),
                     legend_out=True,
                     sharex=True, sharey=True)
g = grid.map(sns.lineplot, to_x, to_y, linewidth=2, ci="sd")
for x_param, ax in g.axes_dict.items():
    ax.set_aspect(2 / 3)
    g_value = df.query('params == @x_param & lr_rate == "ground_truth"').value.item()
    ax.axhline(g_value, ls="--", linewidth=3, c="black")
# grid.fig.set_figwidth(10)
# grid.fig.set_figheight(13)
#plt.tight_layout()
plt.show()


syn_df_orig = pd.read_csv(
    '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes/syn_data_2d_100.csv')
syn_df = syn_df_orig.query('voxel == 1')

for cur_noise in noise_sd:

    syn_df[f'{cur_noise}'] = sim.add_noise(syn_df['normed_betas'], noise_mean=0, noise_sd=cur_noise)

x = np.arange(0, syn_df['normed_betas'].shape[0])
color = plt.cm.rainbow(np.linspace(0, 1, 5))
fig = plt.figure()
for e in np.flip(np.arange(0,5)):
    plt.plot(x, syn_df[str(noise_sd[e])], color=color[e], label=str(noise_sd[e]), linewidth=2)
plt.plot(x, syn_df['normed_betas'], label='Synthetic beta (normalized)', linewidth=2, linestyle='dashed', markersize=12, color='k', marker='o')
plt.legend(title='Noise SD')
plt.ylabel('Synthetic BOLD')
plt.title('1 synthetic voxel with noise')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '1_voxel.png'))
plt.show()
