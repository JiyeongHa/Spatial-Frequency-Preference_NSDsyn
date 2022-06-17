import sys
sys.path.append('../../')
import os
import seaborn as sns
from matplotlib import pyplot as plt
import sfp_nsd_utils as utils
import pandas as pd
import numpy as np
from first_level_analysis import np_log_norm_pdf

def bootstrap_sample(data, stat=np.mean, n_select=8, n_bootstrap=100):
    """ Bootstrap sample from data"""
    bootstrap = []
    for i in range(n_bootstrap):
        samples = np.random.choice(data, size=n_select, replace=True)
        i_bootstrap = stat(samples)
        bootstrap.append(i_bootstrap)
    return bootstrap

def get_all_trials_for_each_stim(df):
    """ Get all trials for each stimulus"""

    return trials

