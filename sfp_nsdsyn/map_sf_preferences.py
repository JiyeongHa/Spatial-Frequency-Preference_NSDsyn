import os
import sys
import numpy as np
import pandas as pd
sys.path.append('/Users/jh7685/Documents/Projects/pysurfer')
from pysurfer import mgz_helper as fs
from scipy.stats import f_oneway
from sfp_nsdsyn import preprocessing as prep

def get_whole_brain_betas(betas_path, design_mat_path,
                          stim_info_path,
                          task_keys, task_average, eccentricity_path=None,
                          x_axis='voxel', y_axis='stim_idx', long_format=True,
                          reference_frame='absolute'):
    stim_df = prep.load_stim_info_as_df(stim_info_path, drop_phase=False)
    betas_dict = prep.load_betas_as_dict(betas_path, design_mat_path,
                                         stim_df['image_idx'], None,
                                         task_keys, task_average)
    betas_dict = prep.melt_2D_betas_dict_into_df(betas_dict, x_axis, y_axis, long_format)
    betas_df = prep.merge_stim_df_and_betas_df(stim_df, betas_dict, on='stim_idx')
    if eccentricity_path is not None:
        prf_dict = prep.load_prf_properties_as_dict([eccentricity_path], mask=None, angle_to_radians=False)
        betas_df = prep.add_1D_prf_dict_to_df(prf_dict, betas_df, on='voxel')
        betas_df['local_sf'] = prep.calculate_local_sf(w_a=betas_df['w_a'],
                                                       w_r=betas_df['w_r'],
                                                       eccentricity=betas_df['eccentricity'],
                                                       reference_frame=reference_frame)
    return betas_df

def divide_df_into_n_bins(df, to_bin, n_bins, return_step=False):
    assert df[to_bin].min() == 0
    step = np.floor(df[to_bin].nunique()/n_bins).astype(int)
    bin_list = np.arange(0, df[to_bin].nunique(), step).astype(int)
    bin_labels = np.arange(0, len(bin_list)-1)
    bins = pd.cut(df[to_bin], bins=bin_list, include_lowest=True, labels=bin_labels)
    if return_step:
        return bins, step
    else:
        return bins


def sf_one_way_anova(df, to_test, values, test_unique=None):
    if test_unique is None:
        test_unique = df[to_test].unique().tolist()
    test = [df[df[to_test] == k][values] for k in test_unique]
    F, p =f_oneway(*test)
    return F, p

def  _organize_df_into_wide_format(sub_df, identifier_list, columns, values):
    sub_df['identifier'] = sub_df[identifier_list].apply(lambda x: '_'.join(map(str, x)), axis=1)
    return sub_df.pivot(columns=columns, index='identifier', values=values)


def sf_multiple_one_way_anova(df, to_test, values, on, identifier_list,
                              test_unique=None, return_identifiers=True):
    if test_unique is None:
        test_unique = df[to_test].unique().tolist()
    test = []
    for i in test_unique:
        sub_df = df[df[to_test] == i]
        tmp = _organize_df_into_wide_format(sub_df, identifier_list, columns=on, values=values)
        test.append(tmp)
    ref = test[0].index.tolist()
    same_identifiers = [t.index.tolist() for t in test]
    if all([ref==t for t in same_identifiers]) is False:
        raise Exception('Identifier lists are different between to_test variables!\n')
    F, p = f_oneway(*test)
    if return_identifiers is True:
        identifiers = ref
        return F, p, identifiers
    else:
        return F, p


def map_stats_as_mgz(template, data, save_path=None):
    template_mgz = fs.load_mgzs(template, fdata_only=False)
    stat_mgz = fs.make_mgzs(data=data,
                       header=template_mgz.header,
                       affine=template_mgz.affine,
                       save_path=save_path)
    return stat_mgz
