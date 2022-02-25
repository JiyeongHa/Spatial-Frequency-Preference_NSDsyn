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
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
#import sfp


def _load_prf_properties(freesurfer_dir, subj, prf_label_names, mask=None, apply_mask=True):
    """ Output format will be mgzs[hemi-property]. """

    # load pRF labels & save masked voxels
    mgzs = {}
    for hemi, prf_properties in itertools.product(['lh', 'rh'], prf_label_names):
        prf_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.' + prf_properties + '.mgz')
        #load pRF labels
        tmp_prf = nib.load(prf_path).get_fdata().squeeze()
        if (apply_mask):
            if not mask:
                raise Exception("Mask is not defined!")
            # throw away voxels that are not included in visual rois & eccen rois
            tmp_prf = tmp_prf[mask[hemi]]
        k = "%s-%s" % (hemi, prf_properties.replace("prf", "").replace("-", ""))
        # save ROIs, ecc rois, & pRF labels into a dict
        mgzs[k] = tmp_prf

    return mgzs
def _load_stim_info(stim_description_dir, stim_description_file='nsdsynthetic_sf_stim_description.csv'):
    """stimulus description file will be loaded as a dataframe."""

    # stimuli information
    stim_description_path = os.path.join(stim_description_dir, stim_description_file)
    stim_df = pd.read_csv(stim_description_path)
    return stim_df
def __get_beta_folder_name(beta_version):
    # load GLMdenoise file
    # f.keys() -> shows betas
    switcher = {
        2: 'nsdsyntheticbetas_fithrf',
        3: 'nsdsyntheticbetas_fithrf_GLMdenoise_RR',
    }

    return switcher.get(beta_version, "Not available beta type")

def _load_betas(mgzs, beta_dir, subj, sf_stim_df, beta_version=3, mask=None, apply_mask=True):
    """Check the directory carefully. There are three different types of beta in NSD synthetic dataset: b1, b2, or b3.
    b2 (folder: betas_fithrf) is results of GLM in which the HRF is estimated for each voxel.
    b3 (folder: betas_fithrf_GLMdenoise_RR) is GLM in which the HRF is estimated for each voxel,
    the GLMdenoise technique is used for denoising,
    and ridge regression is used to better estimate the single-trial betas."""

    beta_version_dir = __get_beta_folder_name(beta_version)
    betas = {}
    for hemi in ['lh', 'rh']:
        betas_file_name = hemi + '.betas_nsdsynthetic.hdf5'
        betas_path = os.path.join(beta_dir, subj, 'nativesurface', beta_version_dir, betas_file_name)
        f = h5py.File(betas_path, 'r')
        # betas[hemi].shape shows 784 x # of voxels
        tmp_betas = f.get('betas')
        # extract only sf-related trials
        tmp_betas = np.float32(tmp_betas[sf_stim_df.image_idx, :]) / 300
        k = "%s-%s" % (hemi, 'betas')
        if apply_mask:
            if not mask:
                raise Exception("Mask is not defined!")
            # mask betas
            betas[k] = tmp_betas[:, mask[hemi]].T
        elif not apply_mask:
            betas[k] = tmp_betas.T

    # add beta values to mgzs
    mgzs.update(betas)

    return mgzs
def label_Vareas (row):
    result = np.remainder(row.visualrois, 7)
    if result == 1 or result == 2:
        return 'V1'
    elif result == 3 or result == 4:
        return 'V2'
    elif result == 5 or result == 6:
        return 'V3'
    elif result == 0:
        return 'V4v'
# put mgzs into a dataframe
def _melt_2D_beta_mgzs_into_df(mgzs):
    """mgz[hemi-betas] has a shape of (voxel_num, 112). each of element out of 112 represents the spatial frequency images.
     What we want to do is that we break down this shape into X by 3 columns, (voxel, stim index, and beta values).
     In other words, we will change the wide shape of data to long shape.
      So far for this function, the output of each hemisphere will be still seperated, as df['lh'] and df['rh']. """

    # initialize df
    df = {}
    # unfold 2D mgzs - betas
    for hemi in ['lh', 'rh']:
        mgz_key = '%s-%s' % (hemi, 'betas')
        tmp_df = pd.DataFrame(mgzs[mgz_key])
        tmp_df = pd.melt(tmp_df.reset_index(), id_vars='index')
        tmp_df = tmp_df.rename(columns={'index': 'voxel', 'variable': 'stim_idx', 'value': 'beta'})
        # image_idx range starts from 104
        #tmp_df.image_idx = tmp_df.image_idx + 104
        tmp_df['hemi'] = hemi
        df[hemi] = tmp_df

    return df
def _add_prf_columns_to_df(mgzs, df, prf_label_names):
    """This function has to be used after applying _melt_2D_beta_mgzs_into_df(),
     since it does not include melting df part."""

    # add pRF properties as columns to df
    for hemi, prf_full_name in itertools.product(['lh', 'rh'], prf_label_names):
        prf_name = prf_full_name.replace("prf", "").replace("-", "")
        mgz_key = "%s-%s" % (hemi, prf_name)
        test_df = pd.DataFrame(mgzs[mgz_key]) # organized in a voxel order
        # To combine test_df to the existing df, we have to set a common column, which is 'voxel'
        test_df = test_df.reset_index().rename(columns={'index': 'voxel', 0: prf_name})
        df[hemi] = df[hemi].merge(test_df)

    return df
def _concat_lh_rh_df(df):

    # concat df['lh'] and df['rh']
    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max() + 1
    #df = pd.concat(df).reset_index(0, drop=True)
    df = pd.concat(df).reset_index().drop(columns=['level_0', 'level_1'])

    return df

def label_Vareas (row):
    result = np.remainder(row.visualrois, 7)
    if result == 1 or result == 2:
        return 'V1'
    elif result == 3 or result == 4:
        return 'V2'
    elif result == 5 or result == 6:
        return 'V3'
    elif result == 0:
        return 'V4v'
def _add_stim_info_to_df(df, stim_description_df):

    # add stim information to the df
    # dataframes are merged based on index
    # Therefore, we first set df's index to stim_idx (same idx as stim_df)
    # combine them and set df's index back to 0,1,2,... n rows.
    df = df.set_index('stim_idx')
    df = df.join(stim_description_df)
    df = df.reset_index().rename(columns={'index': 'stim_idx'})
    return df
def _calculate_local_orientation(df):
    # calculate distance
    ang = np.arctan2(df.w_a, df.w_r)
    df['local_ori'] = ang  # this should be added to theta
    df['local_ori'] = np.deg2rad(df['angle']) + df['local_ori']  # prf angle is the same as orientation
    df['local_ori'] = np.remainder(df['local_ori'], np.pi)

    return df
def _calculate_local_sf(df):
    # calculate local frequency
    df['local_sf'] = np.sqrt((df.w_r ** 2 + df.w_a ** 2))  # this should be divided by R
    df['local_sf'] = df['local_sf'] / df['eccentricity']  # prf eccentricity is the same as R
    df['local_sf'] = np.divide(df['local_sf'], 2 * np.pi)

    return df


def main(sn_list,
         freesurfer_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/freesurfer/',
         betas_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/',
         beta_version=3,
         stim_description_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives', # change spelling
         stim_description_file='nsdsynthetic_sf_stim_description.csv',
         prf_label_list = ["prf-visualrois", "prf-eccrois", "prfeccentricity", "prfangle", "prfsize"],
         vroi_range = [1, 2, 3, 4, 5, 6, 7],
         eroi_range = [1, 2, 3, 4, 5],
         mask_type = ['visroi', 'eccroi'],
         df_save_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes',
         save_df=False):


    print(f'*** subject list: {sn_list} ***')
    if save_df:
        print(f'save mode: on')
        print(f'.... Each dataframe will be saved in dir=\n{df_save_dir}.')
    print(f'\n')
    for sn in sn_list:
        subj = "%s%s" % ('subj', str(sn).zfill(2))
        print(f'*** creating a dataframe for subject no.{sn} ***')
        mgzs = _load_betas(mgzs=mgzs, beta_dir=betas_dir, subj=subj, beta_version=beta_version, sf_stim_df=stim_df, mask=mask)
        df = _melt_2D_beta_mgzs_into_df(mgzs=mgzs)
        df = _add_prf_columns_to_df(mgzs=mgzs, df=df, prf_label_names=prf_label_list)
        df = _concat_lh_rh_df(df=df)
        df['vroinames'] = df.apply(label_Vareas, axis=1)
        df = _add_stim_info_to_df(df=df, stim_description_df=stim_df)
        df = _calculate_local_orientation(df=df)
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

    return df

freesurfer_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/freesurfer/'
betas_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/'
prf_label_names = ["prf-visualrois", "prf-eccrois", "prfeccentricity", "prfangle", "prfsize"]
stim_description_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives'
stim_description_file='nsdsynthetic_sf_stim_description.csv'
subj='subj02'
visroi_range=[1, 2, 3, 4, 5, 6, 7]
eccroi_range=[1, 2, 3, 4, 5]
mask_type= ['visroi', 'eccroi']
df_save_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes'
apply_mask=True

mask = {}
# make masks
for hemi in ['lh', 'rh']:

    if ('visroi' in mask_type) & ('eccroi' in mask_type):
        # make vare roi mask
        visroi_label_name = 'prf-visualrois.mgz'
        visroi_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.' + 'prf-visualrois.mgz')
        visroi = nib.load(visroi_path).get_fdata().squeeze()
        visroi_mask = np.isin(visroi, visroi_range)

        # make eccen roi mask
        eccroi_label_name = 'prf-eccrois.mgz'
        eccroi_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.' + 'prf-eccrois.mgz')
        eccroi = nib.load(eccroi_path).get_fdata().squeeze()
        eccroi_mask = np.isin(eccroi, eccroi_range)

        # make a combined mask
        mask[hemi] = visroi_mask & eccroi_mask

#mask['lh'].shape =>  (239633,)
#mask['rh'].shape => (239309,)

# load pRF labels & save masked voxels
mgzs = {}
for hemi, prf_properties in itertools.product(['lh', 'rh'], prf_label_names):
    prf_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.' + prf_properties + '.mgz')
    #load pRF labels
    tmp_prf = nib.load(prf_path).get_fdata().squeeze()
    print(f'{prf_properties}: {tmp_prf.shape}')
    if (apply_mask):
        if not mask:
            raise Exception("Mask is not defined!")
        # throw away voxels that are not included in visual rois & eccen rois
        tmp_prf = tmp_prf[mask[hemi]]
    k = "%s-%s" % (hemi, prf_properties.replace("prf", "").replace("-", ""))
    # save ROIs, ecc rois, & pRF labels into a dict
    mgzs[k] = tmp_prf
    print(f'{k}: {mgzs[k].shape}')

#lh shape all 239633 -> 11226
#rh shape all 239309 -> 10312

# stimuli information
stim_description_path = os.path.join(stim_description_dir, stim_description_file)
stim_df = pd.read_csv(stim_description_path)

# load betas
beta_version_dir = __get_beta_folder_name(beta_version)
betas = {}
for hemi in ['lh', 'rh']:
    betas_file_name = hemi + '.betas_nsdsynthetic.hdf5'
    betas_path = os.path.join(betas_dir, subj, 'nativesurface', beta_version_dir, betas_file_name)
    f = h5py.File(betas_path, 'r')
    # betas[hemi].shape shows 784 x # of voxels
    tmp_betas = f.get('betas')
    # extract only sf-related trials
    tmp_betas = np.float32(tmp_betas[sf_stim_df.image_idx, :]) / 300
    k = "%s-%s" % (hemi, 'betas')
    if apply_mask:
        if not mask:
            raise Exception("Mask is not defined!")
        # mask betas
        betas[k] = tmp_betas[:, mask[hemi]].T
    elif not apply_mask:
        betas[k] = tmp_betas.T

# add beta values to mgzs
mgzs.update(betas)