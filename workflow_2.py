import sys
import numpy as np
import make_df as mdf
import sfp_nsd_utils as utils
import pandas as pd
import two_dimensional_model as model
import plot_1D_model_results as plotting
import seaborn as sns
import matplotlib.pyplot as plt

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

params =pd.DataFrame({'sigma': [2.2], 'amp': [0.12], 'intercept': [0.35],
                      'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                      'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})
#normalize
normed_betas = model.normalize(df, to_norm='avg_betas', group_by=["subj", "voxel"])
df['norm_betas'] = normed_betas

# forward
df['pred'] = model.Forward(params, 0, df).two_dim_prediction()
df['norm_pred'] = model.normalize(df, to_norm='pred', group_by=["subj", "voxel"])


# plot
dv_to_group = ['subj', 'freq_lvl', 'eccrois', 'vroinames']
avg_df = df.groupby(dv_to_group).mean().reset_index()

for sn in np.arange(1,9):
    model.beta_comp(sn, avg_df, to_subplot="vroinames", to_label="eccrois",
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred',
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
              n_row=4, legend_out=True, alpha=0.7,
              save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
              save_file_name='model_pred.png')

for sn in np.arange(1, 9):
    beta_2Dhist(sn, df, to_subplot="vroinames", to_label='vroinames',
              dp_to_x_axis='norm_betas', dp_to_y_axis='norm_pred',
              x_axis_label='Measured Betas', y_axis_label="Model estimation",
              legend_title=None, labels=None, bins=200, set_max=True,
              n_row=4, legend_out=True, alpha=0.9,
              save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
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

subj='subj01'
cur_df = df.query('subj == @subj')
col_order = utils.sort_a_df_column(cur_df['vroinames'])
grid = sns.FacetGrid(cur_df,
                         col='vroinames',
                         col_order=col_order,
                         hue='vroinames',
                         palette=sns.color_palette("husl"),
                         col_wrap=4,
                         legend_out=True,
                         sharex=False, sharey=False)
g = grid.map(sns.histplot, 'norm_betas', 'norm_pred', alpha=0.5)
plt.show()

sns.histplot(x='norm_betas', y='norm_pred', data=cur_df)
plt.show()

subj = 'subj01'
cur_df = df.query('subj == @subj')

melt_df = pd.melt(cur_df, id_vars=['subj', 'voxel', 'vroinames'], value_vars=['norm_betas', 'norm_pred'],
        var_name='beta_type', value_name='beta_value')

sns.histplot(data=melt_df, x="beta_value", hue="beta_type", stat="density")
