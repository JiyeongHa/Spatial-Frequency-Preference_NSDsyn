import sys
import numpy as np
import make_df as mdf
import sfp_nsd_utils as utils
import pandas as pd
import two_dimensional_model as model
import plot_1D_model_results as plotting
import seaborn as sns
import matplotlib.pyplot as plt
import variance_explained as R2

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
df_2 = all_subj_df.groupby(dv_to_group).mean().reset_index()

params =pd.DataFrame({'sigma': [2.2], 'amp': [0.12], 'intercept': [0.35],
                      'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                      'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})
#normalize
normed_betas = model.normalize(df, to_norm='avg_betas', group_by=["subj", "voxel"])
df['norm_betas'] = normed_betas

# forward
df['pred'] = model.Forward(params, 0, df).two_dim_prediction()
df['norm_pred'] = model.normalize(df, to_norm='pred', group_by=["subj", "voxel"])

df[df.columns & colnames]

# plot

dv_to_group = ['subj', 'freq_lvl', 'vroinames', 'names']
labels=df.names.unique()
avg_df_3 = df.groupby(dv_to_group).median().reset_index()
my_list = ["annulus", "forward spiral", "reverse spiral", "pinwheel"]
avg_df_3 = avg_df_3.query('names.isin(@my_list)', engine='python')

for sn in np.arange(1,9):
    beta_comp(sn, avg_df_3.query('vroinames == "V1"'), to_subplot='names', to_label="names",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred', set_max=False,
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title=None, labels=None,
              n_row=4, legend_out=True, alpha=0.7,
              save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_stim_class_inV1.png')

for sn in np.arange(1,9):
    beta_comp(sn, avg_df_2, to_subplot="vroinames", to_label="eccrois",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred', set_max=True,
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
              n_row=4, legend_out=True, alpha=0.7,
              save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_median.png')

for sn in np.arange(1, 9):
    beta_2Dhist(sn, df, to_subplot="vroinames", to_label='vroinames',
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred',
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title=None, labels=None, bins=200, set_max=False,
              n_row=4, legend_out=True, alpha=0.9,
              save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred_2Dhist.png')


n_voxel_df = count_voxels(df)
sns.barplot(x="vroinames", y="n_voxel",  ci=68, capsize=0.1, palette='hls', data=n_voxel_df)
plt.show()

grid = sns.FacetGrid(n_voxel_df,
                     hue='vroinames',
                     palette=sns.color_palette("hls"),
                     sharex=True, sharey=True)
grid.map(sns.barplot, 'vroinames', 'n_voxel', order=utils.sort_a_df_column(n_voxel_df['vroinames']), ci=68, capsize=0.1)
grid.set_axis_labels('ROI', 'Number of Voxels')
grid.tight_layout()
fig_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/vroinames_vs_n_voxel'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
save_file_name = 'n_voxels'
save_path = os.path.join(fig_dir, f'{save_file_name}')
plt.savefig(save_path)
plt.show()

for sn in np.arange(1, 9):
    beta_1Dhist(sn, df, save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                save_file_name='1Dhist_comp.png')


# R2
all_subj_R2 = R2.load_R2_all_subj(np.arange(1,9))
R2.R2_histogram(sn_list, all_subj_R2, n_bins=300, save_fig=True, xlimit=30, save_file_name='R2_xlimit_30.png')
R2.R2_histogram(sn_list, all_subj_R2, n_bins=300, save_fig=True, xlimit=100, save_file_name='R2_xlimit_100.png')