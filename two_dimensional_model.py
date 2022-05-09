import sys
#sys.path.append('../../')
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


def break_down_phase(df):
    dv_to_group = ['subj', 'freq_lvl', 'names_idx', 'voxel', 'hemi']
    df = df.groupby(dv_to_group).mean().reset_index()

    return df


df_dir='/Volumes/derivatives/subj_dataframes/for_MATLAB'


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
df_file_name = 'stim_voxel_info_df.csv'
df_path = os.path.join(df_dir, subj + '_' + df_file_name)
df = binning._load_and_copy_df(df_path=df_path, create_vroinames_col=False, selected_cols=["fixation_task_betas", "memory_task_betas", "fixation_task", "memory_task"], remove_cols=True)

df = utils.load_df(1, df_dir=df_dir, df_name='mean_df_across_phase.csv')

params =pd.DataFrame({'sigma': [2.2], 'amp': [0.12], 'intercept': [0.35],
          'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
          'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

class Forward():
    """ Define parameters used in forward model"""
    def __init__(self, params, params_idx, subj_df):
        self.params_df = params.iloc[params_idx]
        self.sigma = self.params_df['sigma']
        self.amp = self.params_df['amp']
        self.intercept = self.params_df['intercept']
        self.p_1 = self.params_df['p_1']
        self.p_2 = self.params_df['p_2']
        self.p_3 = self.params_df['p_3']
        self.p_4 = self.params_df['p_4']
        self.A_1 = self.params_df['A_1']
        self.A_2 = self.params_df['A_2']
        self.A_3 = self.params_df['A_3']
        self.A_4 = self.params_df['A_4']
        self.subj_df = subj_df.copy()
        self.theta_l = self.subj_df['local_ori']
        self.theta_v = self.subj_df['angle']
        self.r_v = self.subj_df['eccentricity']
        self.w_l = self.subj_df['local_sf']


    def get_Av(self):
        """ Calculate A_v (formula no. 7 in Broderick et al. (2022)) """

        Av = 1 + self.A_1*np.cos(2*self.theta_l) + \
             self.A_2*np.cos(4*self.theta_l) + \
             self.A_3*np.cos(2*(self.theta_l - self.theta_v)) + \
             self.A_4*np.cos(4*(self.theta_l-self.theta_v))
        return Av

    def get_Pv(self):
        """ Calculate p_v (formula no. 6 in Broderick et al. (2022)) """
        ecc_dependency = self.amp*self.r_v + self.intercept
        Pv = ecc_dependency*(1 + self.A_1*np.cos(2*self.theta_l) +
                             self.A_2*np.cos(4*self.theta_l) +
                             self.A_3*np.cos(2*(self.theta_l - self.theta_v)) +
                             self.A_4*np.cos(4*(self.theta_l-self.theta_v)))
        return Pv

    def two_dim_prediction(self):
        """ Return predicted BOLD response in eccentricity (formula no. 5 in Broderick et al. (2022)) """
        Av = self.get_Av()
        Pv = self.get_Pv()
        return Av*np.exp(-(np.log2(self.w_l)+np.log2(Pv))**2/(2*self.sigma**2))




class TwoDimensionalAnalysis(df, model_params):
    """Dataset for first level results"""
    def __init__(self, ):
        self.df = df.reset_index()
        self.device = device
        self.df_path = df_path
        self.stimulus_class = df.stimulus_class.unique()

    def forward(self, idx):