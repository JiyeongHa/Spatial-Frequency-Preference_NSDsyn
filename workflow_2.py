import sys
import numpy as np
import make_df as mdf
import sfp_nsd_utils as utils
import pandas as pd
import two_dimensional_model as model
import plot_1D_model_results as plotting


# load subjects df.
subj_list = np.arange(1,3)
all_subj_df = utils.load_all_subj_df(subj_list,
                                     df_dir='/Volumes/derivatives/subj_dataframes',
                                     df_name='stim_voxel_info_df_LITE.csv')

# break down phase
dv_to_group = ['subj', 'freq_lvl', 'names_idx', 'voxel', 'hemi']
df = all_subj_df.groupby(dv_to_group).mean().reset_index()

params =pd.DataFrame({'sigma': [2.2], 'amp': [0.12], 'intercept': [0.35],
                      'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                      'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})
df.drop(['phase', 'phase_idx'], axis=1, inplace=True)

forward = Forward(params, 0, df).two_dim_prediction()
df['pred'] = forward



avg_df = df.groupby(["subj", "vroinames", "eccrois", "freq_lvl"]).mean().reset_index()

for subj in subj_list:
    beta_vs_sf_scatterplot(subj, avg_df, to_subplot="vroinames", n_sp_low=4, log_x=False,
                                    legend_out=True, to_label="eccrois",
                                    dp_to_x_axis='avg_betas', dp_to_y_axis='pred', plot_pdf=False,
                                    ln_y_axis="y_lg_pdf", x_axis_label="Betas", y_axis_label="Beta Prediction",
                                    legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                                    save_fig=True, save_dir='/Users/auna/Dropbox/NYU/Projects/SF/MyResults/',
                                    save_file_name='.png')

