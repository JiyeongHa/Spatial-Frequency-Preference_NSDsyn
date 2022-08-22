import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import os
import itertools
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import itertools
import pandas as pd
from scipy.io import loadmat


def cart2pol(xramp, yramp):
    R = np.sqrt(xramp ** 2 + yramp ** 2)
    TH = np.arctan2(yramp, xramp)
    return (R, TH)
def _masking(freesurfer_dir, subj, visroi_range, eccroi_range, mask_type):
    """ Make a mask. Mask_type input should be an array."""
    mask = {}
    # make masks
    for hemi in ['lh', 'rh']:
        # get_fdata() = (x, 1, 1) -> squeeze -> (x,)
        # make visual roi mask
        if len(mask_type) == 1:
            if ('visroi' in mask_type):
                label_name = 'prf-visualrois.mgz'
                roi_range = visroi_range
            elif ('eccroi' in mask_type):
                label_name = 'prf-eccrois.mgz'
                roi_range = eccroi_range

            roi_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.' + label_name)
            roi = nib.load(roi_path).get_fdata().squeeze()
            roi_mask = np.isin(roi, roi_range)
            mask[hemi] = roi_mask

        elif ('visroi' in mask_type) & ('eccroi' in mask_type):
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

    return mask


def _load_prf_properties(freesurfer_dir, subj, prf_label_names, mask=None, apply_mask=True):
    """ Output format will be mgzs[hemi-property]. """

    # load pRF labels & save masked voxels
    mgzs = {}
    for hemi, prf_properties in itertools.product(['lh', 'rh'], prf_label_names):
        prf_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.' + prf_properties + '.mgz')
        # load pRF labels
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


def _load_stim_info(stim_description_path, drop_phase=False):
    """stimulus description file will be loaded as a dataframe.
       drop_phase arg will remove phase information in the output.
       For example, each unique combination of stim classes and frequency levels
       (total of 28) will have one row."""

    # stimuli information
    stim_df = pd.read_csv(stim_description_path)
    if drop_phase is True:
        stim_df = stim_df.query('phase_idx == 0')
        stim_df = stim_df.drop(columns=['phase', 'phase_idx'])
    return stim_df


def _get_beta_folder_name(beta_version):
    # load GLMdenoise file
    # f.keys() -> shows betas
    switcher = {
        2: 'nsdsyntheticbetas_fithrf',
        3: 'nsdsyntheticbetas_fithrf_GLMdenoise_RR',
    }

    return switcher.get(beta_version, "Not available beta type")


def _load_exp_design_mat(design_mat_dir, design_mat_file):
    mat_file = os.path.join(design_mat_dir, design_mat_file)
    mat_file = loadmat(mat_file)
    trial_orders = mat_file['masterordering'].reshape(-1)

    return trial_orders


def _find_beta_index_for_spiral_stimuli(design_mat_dir, design_mat_file, stim_df):
    trial_orders = _load_exp_design_mat(design_mat_dir, design_mat_file)
    spiral_index = stim_df[['image_idx']].copy()
    spiral_index['fixation_task'] = np.nan
    spiral_index['memory_task'] = np.nan
    for x_trial in np.arange(0, trial_orders.shape[0]):
        task_number = np.ceil(x_trial + 1 / 93) % 2  # 1 is fixation task, 0 is memory task
        if task_number == 1:
            task_name = "fixation_task"
        elif task_number == 0:
            task_name = "memory_task"
        # if it's the first trial and if it's not a picture repeated
        if x_trial == 0 or trial_orders[x_trial] != trial_orders[x_trial - 1]:
            # if a spiral image was presented at that trial
            if np.isin(trial_orders[x_trial], spiral_index['image_idx']):
                # add that trial number (for extracting beta) to that image index
                cur_loc = np.where(trial_orders[x_trial] == spiral_index['image_idx'])
                if len(cur_loc) != 1:
                    raise Exception(f'cur_loc length is more than 1!\n')
                spiral_index.loc[cur_loc[0], task_name] = x_trial
    spiral_index["fixation_task"] = spiral_index["fixation_task"].astype(int)
    spiral_index["memory_task"] = spiral_index["memory_task"].astype(int)
    stim_df = stim_df.set_index('image_idx')
    spiral_index = spiral_index.set_index('image_idx')
    stim_df = stim_df.join(spiral_index).reset_index()

    return stim_df


def _average_two_task_betas(betas, hemi):
    """ put in a beta dict and average them voxel-wise """

    if len([v for v in betas.keys() if hemi in v]) != 2:
        raise Exception('The beta values are from less than two tasks!\n')
    avg_betas = (betas[f"{hemi}-fixation_task_betas"] + betas[f"{hemi}-memory_task_betas"]) / 2

    return avg_betas


def _load_betas(beta_dir, subj, sf_stim_df, beta_version=3, task_from="both", beta_average=True, mask=None,
                apply_mask=True):
    """Check the directory carefully. There are three different types of beta in NSD synthetic dataset: b1, b2, or b3.
    b2 (folder: betas_fithrf) is results of GLM in which the HRF is estimated for each voxel.
    b3 (folder: betas_fithrf_GLMdenoise_RR) is the result from GLM ridge regression,
    the GLMdenoise technique is used for denoising,
    and ridge regression is used to better estimate the single-trial betas.
    task_from should be either 'fix', 'memory', or 'both'. """

    beta_version_dir = _get_beta_folder_name(beta_version)
    betas = {}
    for hemi in ['lh', 'rh']:
        betas_file_name = hemi + '.betas_nsdsynthetic.hdf5'
        betas_path = os.path.join(beta_dir, subj, 'nativesurface', beta_version_dir, betas_file_name)
        f = h5py.File(betas_path, 'r')
        # betas[hemi].shape shows 784 x # of voxels
        tmp_betas = f.get('betas')

        if apply_mask:
            if not mask:
                raise Exception("Mask is not defined!")
            # mask betas
            # betas[k] = tmp_betas[:, mask[hemi]].T
            tmp_betas = tmp_betas[:, mask[hemi]]
        # extract only sf-related trials
        if task_from == 'both':
            task_name_list = ['fixation_task', 'memory_task']
        else:
            task_name_list = [task_from + '_task']
        for task_name in task_name_list:
            task_betas = np.empty([sf_stim_df.shape[0], tmp_betas.shape[1]])
            for x_img_idx in np.arange(0, sf_stim_df.shape[0]):
                task_betas[x_img_idx] = tmp_betas[sf_stim_df[task_name][x_img_idx], :]
            task_betas = np.float32(task_betas) / 300
            k = "%s-%s" % (hemi, task_name + '_betas')
            betas[k] = task_betas.T
        if task_from == 'both' and beta_average:
            avg_k = "%s-%s" % (hemi, 'avg' + '_betas')
            betas[avg_k] = _average_two_task_betas(betas, hemi)

    # add beta values to mgzs
    # mgzs.update(betas)
    return betas


# put mgzs into a dataframe
def _melt_2D_beta_mgzs_into_df(beta_mgzs):
    """mgz[hemi-betas] has a shape of (voxel_num, 112). each of element out of 112 represents the spatial frequency images.
     What we want to do is that we break down this shape into X by 3 columns, (voxel, stim index, and beta values).
     In other words, we will change the wide shape of data to long shape.
      So far for this function, the output of each hemisphere will be still seperated, as df['lh'] and df['rh']. """

    # initialize df
    df = {}
    for hemi in ['lh', 'rh']:
        # unfold 2D mgzs - betas
        df[hemi] = pd.DataFrame(columns=['voxel'])
        hemi_beta_mgzs_keys = [v for v in beta_mgzs.keys() if hemi in v]
        for mgz_key in hemi_beta_mgzs_keys:
            # mgz_key = '%s-%s' % (hemi, 'betas')
            tmp_df = pd.DataFrame(beta_mgzs[mgz_key])
            tmp_df = pd.melt(tmp_df.reset_index(), id_vars='index')
            tmp_df = tmp_df.rename(
                columns={'index': 'voxel', 'variable': 'stim_idx', 'value': mgz_key.replace(hemi + "-", "")})
            df[hemi] = df[hemi].merge(tmp_df, how='outer')
        df[hemi]['hemi'] = hemi
    return df


def _label_Vareas(row):
    result = np.remainder(row.visualrois, 7)
    if result == 1 or result == 2:
        return 'V1'
    elif result == 3 or result == 4:
        return 'V2'
    elif result == 5 or result == 6:
        return 'V3'
    elif result == 0:
        return 'V4v'


def _add_prf_columns_to_df(prf_mgzs, df, prf_label_names):
    """This function has to be used after applying _melt_2D_beta_mgzs_into_df(),
     since it does not include melting df part."""

    # add pRF properties as columns to df
    for hemi, prf_full_name in itertools.product(['lh', 'rh'], prf_label_names):
        prf_name = prf_full_name.replace("prf", "").replace("-", "")
        mgz_key = "%s-%s" % (hemi, prf_name)
        test_df = pd.DataFrame(prf_mgzs[mgz_key])  # organized in a voxel order
        # To combine test_df to the existing df, we have to set a common column, which is 'voxel'
        test_df = test_df.reset_index().rename(columns={'index': 'voxel', 0: prf_name})
        if prf_name == 'visualrois':
            test_df['vroinames'] = test_df.apply(_label_Vareas, axis=1)
        df[hemi] = df[hemi].merge(test_df, on='voxel')

    return df


def _concat_lh_rh_df(df):
    # concat df['lh'] and df['rh']
    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max() + 1
    # df = pd.concat(df).reset_index(0, drop=True)
    df = pd.concat(df).reset_index().drop(columns=['level_0', 'level_1'])

    return df


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


def sub_main(sn,
             freesurfer_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/freesurfer/',
             betas_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/',
             beta_version=3,
             stim_description_path='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives/nsdsynthetic_sf_stim_description.csv',
             design_mat_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/experiments/nsdsynthetic',
             design_mat_file='nsdsynthetic_expdesign.mat',
             task_from="both",
             beta_average=True,
             prf_label_list=["prf-visualrois", "prf-eccrois", "prfeccentricity", "prfangle", "prfsize"],
             vroi_range=[1, 2, 3, 4, 5, 6, 7],
             eroi_range=[1, 2, 3, 4, 5],
             mask_type=['visroi', 'eccroi'],
             df_save_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes',
             save_df=False):
    subj = f"subj{str(sn).zfill(2)}"
    print(f'*** creating a dataframe for subject no.{sn} ***')
    mask = _masking(freesurfer_dir=freesurfer_dir, subj=subj,
                    visroi_range=vroi_range, eccroi_range=eroi_range, mask_type=mask_type)
    mgzs = _load_prf_properties(freesurfer_dir=freesurfer_dir, subj=subj, prf_label_names=prf_label_list, mask=mask,
                                apply_mask=True)
    stim_df = _load_stim_info(stim_description_path=stim_description_path)
    stim_df = _find_beta_index_for_spiral_stimuli(design_mat_dir, design_mat_file, stim_df=stim_df)
    beta_mgzs = _load_betas(beta_dir=betas_dir, subj=subj, beta_version=beta_version, task_from=task_from,
                            beta_average=beta_average, sf_stim_df=stim_df, mask=mask)
    df = _melt_2D_beta_mgzs_into_df(beta_mgzs=beta_mgzs)
    df = _add_prf_columns_to_df(prf_mgzs=mgzs, df=df, prf_label_names=prf_label_list)
    df = _concat_lh_rh_df(df=df)
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


def main(sn_list,
         df_save_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/subj_dataframes',
         save_df=True):
    print(f'*** subject list: {sn_list} ***')
    if save_df:
        print(f'save mode: on')
        print(f'.... Each dataframe will be saved in dir=\n{df_save_dir}.')
    print(f'\n')
    for sn in sn_list:
        df = sub_main(sn=sn, df_save_dir=df_save_dir, save_df=save_df)
    return df


if __name__ == '__main__':
    main(sn_list)
