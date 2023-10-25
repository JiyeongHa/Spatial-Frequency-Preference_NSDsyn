import os
from scipy.stats import f_oneway
from sfp_nsdsyn import preprocessing as prep

def get_whole_brain_betas(betas_path, design_mat_path,
                          stim_info_path,
                          task_keys, task_average,
                          x_axis='voxel', y_axis='stim_idx', long_format=True):
    stim_df = prep.load_stim_info_as_df(stim_info_path, drop_phase=False)
    betas_dict = prep.load_betas_as_dict(betas_path, design_mat_path,
                                         stim_df['image_idx'], None,
                                         task_keys, task_average)
    betas_dict = prep.melt_2D_betas_dict_into_df(betas_dict, x_axis, y_axis, long_format)
    betas_df = prep.merge_stim_df_and_betas_df(stim_df, betas_dict, on='stim_idx')
    return betas_df

def sf_one_way_anova(df, to_test, test_unique=None):
    if test_unique is None:
        test_unique = df[to_test].unique().tolist()
    test = [df[df[to_test] == k].betas for k in test_unique]
    F, p =f_oneway(*test)
    return F, p