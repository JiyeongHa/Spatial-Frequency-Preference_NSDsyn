import sys
import os
import numpy as np
import make_df
import binning_eccen as binning
import sfp_nsd_utils as utils

## load all data
subj_list = np.arange(1,9)

all_subj_df = binning.get_all_subj_df(subjects_to_run=subj_list, dv_to_group=["vroinames", "eccrois", "freq_lvl"])
first_level_output_df = utils.load_all_subj_df(subj_to_run=subj_list,
                                               df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
                                               df_name='results_1D_model.csv')
