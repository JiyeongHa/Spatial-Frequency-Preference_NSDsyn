import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import os
import nibabel as nib
import numpy as np
import h5py
import itertools
import pandas as pd
from scipy.io import loadmat
from . import voxel_selection as vs
from . import utils as utils
from . import bootstrapping as bts
from . import two_dimensional_model as model


def _load_exp_design_mat(design_mat_path):
    mat_file = loadmat(design_mat_path)
    trial_orders = mat_file['masterordering'].reshape(-1)

    return trial_orders

def load_stim_info(stim_description_path, drop_phase=False):
    """stimulus description file will be loaded as a dataframe.
       drop_phase arg will remove phase information in the output.
       For example, each unique combination of stim classes and frequency levels
       (total of 28) will have one row."""

    # stimuli information
    stim_df = pd.read_csv(stim_description_path)
    stim_df = stim_df.drop(columns=['phase_idx'])
    if drop_phase is True:
        stim_df = stim_df.query('phase == 0')
    return stim_df

def cart2pol(xramp, yramp):
    R = np.sqrt(xramp ** 2 + yramp ** 2)
    TH = np.arctan2(yramp, xramp)
    return (R, TH)

def load_mask_and_roi(roi_path, roi_vals):
    mask = {}
    roi = {}
    tmp = nib.load(roi_path).get_fdata().squeeze()
    mask = np.isin(tmp, roi_vals)
    roi = tmp[mask]
    return mask, roi

def _label_vareas(roi_val):
    result = roi_val // 2
    if result == 1:
        return 'V1'
    elif result == 2:
        return 'V2'
    elif result == 3:
        return 'V3'
    elif result == 4:
        return 'h4v'

def vrois_num_to_names(vrois):
    return [_label_vareas(i) for i in vrois]

def load_common_mask_and_rois(rois_path, rois_vals):
    roi_dict, all_masks = {}, []
    roi_names = [k.split('/')[-1].replace('prf-', '').replace('.mgz','') for k in rois_path]
    for roi_name, roi_path, roi_val in zip(roi_names, rois_path, rois_vals):
        tmp, roi_dict[roi_name] = load_mask_and_roi(roi_path, roi_val)
        if 'visualrois' in roi_name:
            roi_dict[roi_name] = vrois_num_to_names(roi_dict[roi_name])
        all_masks.append(tmp)
    mask = np.all(all_masks, axis=0)
    return mask, roi_dict

def load_prf_properties_as_dict(prf_path_list, mask):
    # load pRF labels & save masked voxels
    prf_dict = {}
    for prf_path in prf_path_list:
        tmp_prf = nib.load(prf_path).get_fdata().squeeze()
        if mask is not None:
            # throw away voxels that are not included in visual rois & eccen rois
            tmp_prf = tmp_prf[mask]
        k = prf_path.split('/')[-1].replace('prf', '').replace('.mgz', '')
        # save ROIs, ecc rois, & pRF labels into a dict
        prf_dict[k] = tmp_prf
    return prf_dict

def _get_beta_folder_name(beta_version):
    # load GLMdenoise file
    # f.keys() -> shows betas
    switcher = {
        2: 'nsdsyntheticbetas_fithrf',
        3: 'nsdsyntheticbetas_fithrf_GLMdenoise_RR',
    }

    return switcher.get(beta_version, "Not available beta type")

def _find_beta_index_for_spiral_stimuli(design_mat_path, image_idx):
    trial_orders = _load_exp_design_mat(design_mat_path)
    spiral_index = pd.DataFrame({})
    spiral_index['image_idx'] = image_idx
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
    return spiral_index

def load_betas_as_dict(betas_path, design_mat, image_idx, mask, task_keys=['fixation_task','memory_task'], average=True):
    """Check the directory carefully. There are three different types of beta in NSD synthetic dataset: b1, b2, or b3.
    b2 (folder: betas_fithrf) is results of GLM in which the HRF is estimated for each voxel.
    b3 (folder: betas_fithrf_GLMdenoise_RR) is the result from GLM ridge regression,
    the GLMdenoise technique is used for denoising,
    and ridge regression is used to better estimate the single-trial betas.
    task_from should be either 'fix', 'memory', or 'both'. """
    betas_dict = {}
    design_df = _find_beta_index_for_spiral_stimuli(design_mat, image_idx)
    with h5py.File(betas_path, 'r') as f:
        # betas[hemi].shape shows 784 x # of voxels
        tmp_betas = f.get('betas')
        tmp_betas = tmp_betas[:, mask]
        n_image = design_df.shape[0]
        for task_name in task_keys:
            task_betas = np.empty([n_image, tmp_betas.shape[1]])
            for x_img_idx in np.arange(0, n_image):
                task_betas[x_img_idx] = tmp_betas[design_df[task_name][x_img_idx], :]
            task_betas = np.float32(task_betas) / 300
            k = f'{task_name}_betas'
            betas[k] = task_betas.T
    if average is True:
        if len(task_keys) < 2:
            raise Exception('The task list length is less than 2!\n')
        betas_dict['avg_betas'] = np.mean([a for a in betas_dict.values()], axis=0)
    return betas_dict

def melt_2D_betas_dict_into_df(betas_dict, x_axis='voxel', y_axis='stim_idx', long_format=True):
    """convert beta mgzs to dataframe and melt it into a long dataframe based on x and y axes.
    In general x axis should be the voxel index and y axis is stim index."""
    shapes = [betas_dict[k].shape for k in betas_dict.keys()]
    if all(shapes) is False:
        raise Exception('The sizes of the arrays are different between entries!')
    # initialize df
    x, y = range(shapes[0][0]), range(shapes[0][1])
    df = pd.DataFrame(list(itertools.product(x, y)), columns=[x_axis, y_axis])
    for k in betas_dict.keys():
        tmp = pd.DataFrame(betas_dict[k])
        y_list = tmp.columns.to_list()
        tmp = tmp.reset_index().rename(columns={'index': x_axis})
        tmp = tmp.melt(id_vars=x_axis,
                       value_vars=y_list,
                       var_name=y_axis,
                       value_name=k)
        df = pd.merge(df, tmp, on=[x_axis, y_axis])
    if long_format:
        new_names = [k.replace('_task', '').replace('_betas', '') for k in betas_dict.keys()]
        df = df.rename(columns=dict(zip(betas_dict.keys(), new_names)))
        df = df.melt(id_vars=[x_axis, y_axis], var_name='task', value_name='betas')
    return df

def add_1D_prf_dict_to_df(prf_dict, df, roi_mgzs=None, on='voxel'):
    """This function has to be used after applying _melt_2D_beta_mgzs_into_df(),
     since it does not include melting df part."""
    if roi_mgzs is not None:
        prf_dict = {**prf_dict, **roi_mgzs}
    prf_df = pd.DataFrame(prf_dict).reset_index().rename(columns={'index': on})
    return df.merge(prf_df, on=on)

def _add_stim_info_to_df(df, stim_description_df):
    # add stim information to the df
    # dataframes are merged based on index
    # Therefore, we first set df's index to stim_idx (same idx as stim_df)
    # combine them and set df's index back to 0,1,2,... n rows.
    df = df.set_index('stim_idx')
    df = df.join(stim_description_df)
    df = df.reset_index().rename(columns={'index': 'stim_idx'})
    return df


def calculate_local_orientation(w_a, w_r, retinotopic_angle, radians=False):
    # calculate distance
    frequency_ratio = np.arctan2(w_a, w_r)
    if radians is False:
        retinotopic_angle = np.deg2rad(retinotopic_angle)
    local_ori = retinotopic_angle + frequency_ratio  # prf angle is the same as orientation
    return np.remainder(local_ori, np.pi)


def calculate_local_sf(df):
    # calculate local frequency
    df['local_sf'] = np.sqrt((df.w_r ** 2 + df.w_a ** 2))  # this should be divided by R
    df['local_sf'] = df['local_sf'] / df['eccentricity']  # prf eccentricity is the same as R
    df['local_sf'] = np.divide(df['local_sf'], 2 * np.pi)

    return df

def _concat_lh_rh_df(df):
    # concat df['lh'] and df['rh']
    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max() + 1
    # df = pd.concat(df).reset_index(0, drop=True)
    df = pd.concat(df).reset_index().drop(columns=['level_0', 'level_1'])

    return df



def sub_main(sn,
             freesurfer_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/freesurfer/',
             betas_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/',
             beta_version=3,
             stim_description_path='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsynthetic_sf_stim_description.csv',
             design_mat_path='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat',
             task_from="both",
             prf_label_list=["prf-visualrois", "prf-eccrois", "prfeccentricity", "prfangle", "prfsize"],
             vroi_range=[1, 2, 3, 4, 5, 6, 7],
             eroi_range=[1, 2, 3, 4, 5],
             mask_type=['visroi', 'eccroi'],
             df_save_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes/nsdsyn',
             save_df=True, voxel_criteria='pRFcenter'):
    subj = utils.sub_number_to_string(sn, dataset="nsdsyn")
    print(f'*** creating a dataframe for subject no.{sn} ***')
    mask = _masking(freesurfer_dir=freesurfer_dir, subj=subj,
                    visroi_range=vroi_range, eccroi_range=eroi_range, mask_type=mask_type)
    mgzs = _load_prf_properties(freesurfer_dir=freesurfer_dir, subj=subj, prf_label_names=prf_label_list, mask=mask,
                                apply_mask=True)
    stim_df = load_stim_info(stim_description_path=stim_description_path)
    stim_df = _find_beta_index_for_spiral_stimuli(design_mat_path, stim_df=stim_df)
    beta_mgzs = _load_betas(beta_dir=betas_dir, subj=subj, beta_version=beta_version, task_from=task_from, sf_stim_df=stim_df, mask=mask)
    df = _melt_2D_beta_mgzs_into_df(beta_mgzs=beta_mgzs)
    df = _add_prf_columns_to_df(prf_mgzs=mgzs, df=df, prf_label_names=prf_label_list)
    df = _concat_lh_rh_df(df=df)
    df['angle'] = np.deg2rad(df['angle'])
    df = df.rename(columns={'size': 'sigma'})
    df = _add_stim_info_to_df(df=df, stim_description_df=stim_df)
    df = vs.select_voxels(df, vs.pix_to_deg(42.878), vs.pix_to_deg(714/2), ['voxel'], 'betas', near_border=True)
    df['subj'] = subj
    sigma_v_df = bts.get_multiple_sigma_vs(df, power=[1, 2], columns=['noise_SD', 'sigma_v_squared'], to_sd='betas', to_group=['voxel'])
    df = df.merge(sigma_v_df, on=['voxel'])
    df = calculate_local_orientation(df=df)
    df = calculate_local_sf(df=df)
    fnl_df = df.groupby(['voxel', 'names', 'class_idx', 'vroinames']).mean().reset_index()
    fnl_df = fnl_df.drop(['phase', 'phase_idx', 'stim_idx', 'image_idx', 'fixation_task', 'memory_task'], axis=1)
    fnl_df['normed_betas'] = model.normalize(fnl_df, 'betas', ['voxel'], phase_info=False)
    if save_df:
        if not os.path.exists(df_save_dir):
            os.makedirs(df_save_dir)
        for roi in df.vroinames.unique():
            roi_df = df.query('vroinames == @roi')
            roi_fnl_df = fnl_df.query('vroinames == @roi')
            df_save_path = os.path.join(df_save_dir, f'{subj}_stim_voxel_info_df_vs-{voxel_criteria}_{roi}.csv')
            roi_df.to_csv(df_save_path, index=False)
            print(f'... {subj} {roi} dataframe saved.')
            fnl_df_save_path = os.path.join(df_save_dir, f"{subj}_stim_voxel_info_df_vs-{voxel_criteria}_{roi}_mean.csv")
            roi_fnl_df.to_csv(fnl_df_save_path, index=False)
            print(f'... {subj} {roi} mean dataframe dataframe saved.')
    return fnl_df


def main(sn_list, freesurfer_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/freesurfer/',
         betas_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/',
         beta_version=3,
         stim_description_path='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsynthetic_sf_stim_description.csv',
         design_mat_path='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat',
         task_from="both",
         prf_label_list=["prf-visualrois", "prf-eccrois", "prfeccentricity", "prfangle", "prfsize"],
         vroi_range=[1, 2, 3, 4, 5, 6, 7],
         eroi_range=[1, 2, 3, 4, 5],
         mask_type=['visroi', 'eccroi'],
         df_save_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes/nsdsyn',
         save_df=True):
    print(f'*** subject list: {sn_list} ***')
    if save_df:
        print(f'save mode: on')
        print(f'.... Each dataframe will be saved in dir=\n{df_save_dir}.')
    print(f'\n')
    df = {}
    for sn in sn_list:
        df[sn] = sub_main(sn, freesurfer_dir, betas_dir, beta_version, stim_description_path, design_mat_path,
                          task_from, prf_label_list, vroi_range, eroi_range, mask_type, df_save_dir, save_df)
    all_subj = pd.concat(df, ignore_index=True)
    return all_subj

def add_class_idx_to_stim_df(save=True):
    nsd_stim_df = pd.read_csv('/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/nsddata/stimuli/nsdsynthetic/nsdsynthetic_sf_stim_description.csv')
    tmp_nsd_stim_df = nsd_stim_df.query('phase == 0')
    tmp_nsd_stim_df['class_idx'] = np.arange(0, 28)
    tmp_nsd_stim_df = tmp_nsd_stim_df[['names', 'w_a', 'w_r', 'class_idx']]
    tmp = np.tile(np.arange(0,6), 4)
    tmp_nsd_stim_df['freq_lvl'] =np.concatenate((tmp, np.array([3, 3, 3, 3])))
    nsd_stim_df = nsd_stim_df.merge(tmp_nsd_stim_df, on=['names', 'w_a', 'w_r'])
    if save:
        nsd_stim_df.to_csv('/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsynthetic_sf_stim_description.csv', index=False)
    return nsd_stim_df


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

def _average_two_task_betas(betas, hemi):
    """ put in a beta dict and average them voxel-wise """

    if len([v for v in betas.keys() if hemi in v]) != 2:
        raise Exception('The beta values are from less than two tasks!\n')
    avg_betas = (betas[f"{hemi}-fixation_task_betas"] + betas[f"{hemi}-memory_task_betas"]) / 2

    return avg_betas

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
            tmp_df = tmp_df.rename(columns={'index': 'voxel', 'variable': 'stim_idx', 'value': mgz_key.replace(hemi + "-", "")})
            df[hemi] = df[hemi].merge(tmp_df, how='outer')
        df[hemi] = pd.melt(df[hemi], id_vars=['voxel','stim_idx'], value_vars=['fixation_task_betas','memory_task_betas'], var_name='task', value_name='betas')
        df[hemi] = df[hemi].replace({'task': {'fixation_task_betas': 'fixation', 'memory_task_betas': 'memory'}})
        df[hemi]['hemi'] = hemi
    return df


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
