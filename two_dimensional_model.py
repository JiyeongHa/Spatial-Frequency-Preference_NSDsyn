import sys
sys.path.append('../../')
import os
import seaborn as sns
import sfp_nsd_utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import warnings
import argparse
import itertools
import re
import functools
from scipy import stats
from torch.utils import data as torchdata
from hessian import hessian
import binning_eccen as binning


df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'

for sn in np.arange(1,2):
    subj = utils.sub_number_to_string(sn)
    df_path = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'
    df = utils.load_df(sn, df_dir=df_path,
            df_name='stim_voxel_info_df.csv')

    # creating a dict file
    df['hemi'] = df['hemi'].replace(['lh','rh'], [1,2])
    dv_to_group = ['freq_lvl', 'names_idx', 'voxel', 'hemi']
    df = df.groupby(dv_to_group).mean().reset_index()
    selected_cols = ['voxel',
                     'avg_betas', 'hemi', 'visualrois', 'eccrois',
                     'eccentricity', 'angle', 'size', 'names_idx',
                     'w_r', 'w_a', 'freq_lvl', 'local_ori', 'local_sf']
    df = df[selected_cols].copy()
    # save the final output
    df_save_name = "%s_%s" % (subj, "mean_df_across_phase.csv")
    df_save_dir = f'{df_path}/for_MATLAB'

    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir)
    df_save_path = os.path.join(df_save_dir, df_save_name)
    df.to_csv(df_save_path, index=False, header=True)
    print(f'... {subj} dataframe saved.')


subj = utils.sub_number_to_string(1)
df_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'
df_file_name = 'stim_voxel_info_df.csv'
df_path = os.path.join(df_dir, subj + '_' + df_file_name)
df = binning._load_and_copy_df(df_path=df_path, create_vroinames_col=False, selected_cols=["fixation_task_betas", "memory_task_betas", "fixation_task", "memory_task"], remove_cols=True)


f

