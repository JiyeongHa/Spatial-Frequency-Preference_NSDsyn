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
        self.r_v = self.subj_df['eccentricity'] # voxel eccentricity (in degrees)
        self.w_l = self.subj_df['local_sf'] # in cycles per degree


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
        return Av*np.exp(-(np.log2(self.w_l)-np.log2(Pv))**2/(2*self.sigma**2))

def normalize(df, to_norm, group_by=["voxel"]):
    """calculate L2 norm for each voxel """

    if all(df.groupby(group_by).size() == 28) is False:
        raise Exception('There are more than 28 conditions for one voxel!\n')
    l2_norm = df.groupby(group_by)[to_norm].apply(lambda x: x / np.linalg.norm(x))

    return l2_norm


class TwoDimensionalAnalysis(df, model_params):
    """Dataset for first level results"""
    def __init__(self, ):
        self.df = df.reset_index()
        self.device = device
        self.df_path = df_path
        self.stimulus_class = df.stimulus_class.unique()

    def forward(self, idx):