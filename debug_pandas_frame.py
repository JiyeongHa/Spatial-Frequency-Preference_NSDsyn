import sys
import os
import itertools
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import make_df

sn_list=[1]
freesurfer_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/freesurfer/'
betas_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/'
beta_version = 3
stim_description_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives'
stim_description_file='nsdsynthetic_sf_stim_description.csv'
prf_label_list = ["prf-visualrois", "prf-eccrois", "prfeccentricity", "prfangle", "prfsize"]
vroi_range = [1, 2, 3, 4, 5, 6, 7]
eroi_range = [1, 2, 3, 4, 5]
mask_type = ['visroi', 'eccroi']
df_save_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'
save_df=False

print(f'*** subject list: {sn_list} ***')
if save_df:
    print(f'save mode: on')
    print(f'.... Each dataframe will be saved in dir=\n{df_save_dir}.')
print(f'\n')
for sn in sn_list:
    subj = "%s%s" % ('subj', str(sn).zfill(2))
    print(f'*** creating a dataframe for subject no.{sn} ***')
    mask = make_df._masking(freesurfer_dir=freesurfer_dir, subj=subj,
                    visroi_range=vroi_range, eccroi_range=eroi_range, mask_type=mask_type)
    mgzs_prf = make_df._load_prf_properties(freesurfer_dir=freesurfer_dir, subj=subj, prf_label_names=prf_label_list, mask=mask,
                                apply_mask=True)
    mgzs = mgzs_prf.copy()
    stim_df = make_df._load_stim_info(stim_description_dir=stim_description_dir, stim_description_file=stim_description_file)
    mgzs_betas = make_df._load_betas(mgzs=mgzs, beta_dir=betas_dir, subj=subj, beta_version=beta_version, sf_stim_df=stim_df,
                       mask=mask)
    df_betas = make_df._melt_2D_beta_mgzs_into_df(mgzs=mgzs_betas)
    df_hemi = make_df._add_prf_columns_to_df(mgzs=mgzs_prf, df=df_betas, prf_label_names=prf_label_list)
    df_after_stim_info = df_hemi.copy()
    df_after_stim_info = make_df._concat_lh_rh_df(df=df_after_stim_info)
    df_after_stim_info['vroinames'] = df_after_stim_info.apply(make_df.label_Vareas, axis=1)
    df_after_stim_info = make_df._add_stim_info_to_df(df=df_after_stim_info, stim_description_df=stim_df)
    df = make_df._calculate_local_orientation(df=df)
    df = _calculate_local_sf(df=df)
    if save_df:
        # save the final output
        df_save_name = "%s_%s" % (subj, "stim_voxel_info_df.csv")
        df_save_dir = df_save_dir
        if not os.path.exists(df_save_dir):
            os.makedirs(df_save_dir)
        df_save_path = os.path.join(df_save_dir, df_save_name)
        df.to_csv(df_save_path, index=False)
        print(f'... {subj} dataframe saved.')
