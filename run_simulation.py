import os
import sys
import numpy as np
import make_df as mdf
import sfp_nsd_utils as utils
import pandas as pd
import plot_1D_model_results as plotting
import seaborn as sns
import matplotlib.pyplot as plt
import variance_explained as R2
import voxel_selection as vs
import two_dimensional_model as model
import torch
import simulation as sim
from importlib import reload
import binning_eccen as binning
import first_level_analysis as fitting
import bootstrap as bts


params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                       'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                       'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

stim_info_path = '/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv'
syn_data = sim.SynthesizeData(n_voxels=100, df=None, replace=True, p_dist="data",
                              stim_info_path=stim_info_path)
syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=False)