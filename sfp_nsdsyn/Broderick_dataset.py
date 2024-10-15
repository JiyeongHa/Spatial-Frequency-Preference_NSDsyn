import sys
import os
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import itertools
from . import make_dataframes as prep
from . import utils as utils
from . import voxel_selection as vs
from . import two_dimensional_model as model
from . import bootstrapping as bts


def _label_stim_names(row):
    if row.w_r == 0 and row.w_a != 0:
        return 'pinwheel'
    elif row.w_r != 0 and row.w_a == 0:
        return 'annulus'
    elif row.w_r == row.w_a:
        return 'reverse spiral'
    elif row.w_r == -1*row.w_a:
        return 'forward spiral'
    else:
        return 'mixtures'

def _label_freq_lvl(freq_lvl=10, phi_repeat=8, main_classes=4, mixture_freq_lvl=5, n_mixtures=8):
    freq_lvl_main_classes = np.tile(np.repeat(np.arange(0, freq_lvl), phi_repeat), main_classes)
    freq_lvl_mixtures = np.repeat(mixture_freq_lvl, phi_repeat * n_mixtures)
    freq_lvl = np.concatenate((freq_lvl_main_classes, freq_lvl_mixtures))
    return freq_lvl

def load_broderick_stim_info(stim_description_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/stimuli/task-sfprescaled_stim_description.csv',
                             drop_phase=True):
    """ Broderick et al (2022) had different convention for naming conditions.
    This function will load in the original broderick stimulus .csv file and convert its format to match the sfp_nsd convention."""

    if not os.path.exists(stim_description_path):
        raise Exception('The original stim description file does not exist!\n')
    stim_df = pd.read_csv(stim_description_path)
    stim_df = stim_df.dropna(axis=0)
    stim_df = stim_df.astype({'class_idx': int})
    stim_df = stim_df.drop(columns='res')
    stim_df = stim_df.rename(columns={'phi': 'phase', 'index': 'image_idx'})
    stim_df['names'] = stim_df.apply(_label_stim_names, axis=1)
    stim_df['freq_lvl'] = _label_freq_lvl()
    if drop_phase:
        stim_df = stim_df.drop_duplicates(subset=['class_idx'])
        stim_df = stim_df.drop(columns='phase')
    return stim_df

def _get_benson_atlas_rois(roi_index):
    """ switch num to roi names or the other way around. see https://osf.io/knb5g/wiki/Usage/"""

    num_key = {1: "V1", 2: "V2", 3: "V3", 4: "hV4", 5: "VO1", 6: "VO2",
        7: "LO1", 8: "LO2", 9: "TO1", 10: "TO2", 11: "V3b", 12: "V3a"}
    name_key = {y: x for x, y in num_key.items()}
    switcher = {**num_key, **name_key}
    return switcher.get(roi_index, "No Visual area")



def make_a_mask(varea_path, roi, eccentricity_path, eccen_range):
    """create a mask using visual rois and eccentricity range.
    Broderick dataset doesn't have eccentricity-based ROIs. Therefore, the possible input for the rois_path is inferred_varea and inferred_eccen."""
    vroi_mask, _ = prep.load_mask_and_roi(varea_path, [_get_benson_atlas_rois(roi)])
    eccen = nib.load(eccentricity_path).get_fdata().squeeze()
    eccen_mask = (eccen_range[0] < eccen) & (eccen < eccen_range[-1])
    mask = vroi_mask & eccen_mask
    return mask

def make_lh_rh_masks(lh_varea_path, rh_varea_path, lh_eccen_path, rh_eccen_path, roi, eccen_range):
    mask={}
    lh_mask = make_a_mask(lh_varea_path, roi, lh_eccen_path, eccen_range)
    rh_mask = make_a_mask(rh_varea_path, roi, rh_eccen_path, eccen_range)
    mask['lh'] = lh_mask
    mask['rh'] = rh_mask
    return mask

def _get_prf_property_name(prf_path):
    """ switch num to roi names or the other way around. see https://osf.io/knb5g/wiki/Usage/"""
    k = prf_path.split('/')[-1].replace('.mgz', '')
    hemi = k.split('.')[0]
    k = k.split('.')[-1]  # remove hemi
    switcher = {'inferred_angle': f'{hemi}_angle',
               'inferred_sigma': f'{hemi}_size',
               'inferred_eccen': f'{hemi}_eccentricity',
               'inferred_varea': f'{hemi}_vroinames'}
    return switcher.get(k, None)

def load_prf_properties_as_dict(prf_path_list, mask=None):
    # load pRF labels & save masked voxels
    prf_dict = {}
    for prf_path in prf_path_list:
        k = _get_prf_property_name(prf_path)
        tmp_prf = nib.load(prf_path).get_fdata().squeeze()
        if mask is not None:
            # throw away voxels that are not included in visual rois & eccen rois
            tmp_prf = tmp_prf[mask]
        # save ROIs, ecc rois, & pRF labels into a dict
        prf_dict[k] = tmp_prf
    return prf_dict
def load_lh_rh_prf_properties_as_dict(lh_prf_path_list,
                                      rh_prf_path_list, lh_mask=None, rh_mask=None):
    lh_dict = load_prf_properties_as_dict(lh_prf_path_list, lh_mask)
    rh_dict = load_prf_properties_as_dict(rh_prf_path_list, rh_mask)
    return {**lh_dict, **rh_dict}

def _transform_angle(row):
    """this function is a modified ver. of the script from sfp (Broderick et al.2022).
     transform angle from Benson14 convention to our convention
    The Benson atlases' convention for angle in visual field is: zero is the upper vertical
    meridian, angle is in degrees, the left and right hemisphere both run from 0 to 180 from the
    upper to lower meridian (so they increase as you go clockwise and counter-clockwise,
    respectively). For our calculations, we need the following convention: zero is the right
    horizontal meridian, angle is in radians (and lie between 0 and 360, rather than -pi and pi),
    angle increases as you go clockwise, and each angle is unique (refers to one point on the
    visual field; we don't have the same number in the left and right hemispheres)
    """
    if row.hemi == 'rh':
        # we want to remap the right hemisphere angles to negative. Noah says this is the
        # convention, but I have seen positive values there, so maybe it changed at one point.
        if row.angle > 0:
            row.angle = -row.angle
    return np.mod((row.angle - 90), 360) #TODO: maybe should be + 90


def _transform_angle_corrected(row):
    """This function is to correct the above function _transform_angle to match the sfp_nsd convention. First, the right visual field should be negative, and then add 90.
    The right visual field is from left hemisphere's angle file. So in case the hemisphere is 'lh', we multiply -1.

    """
    if row.hemi == 'lh':
        if row.angle > 0:
            row.angle = -row.angle
    return np.mod((row.angle + 90), 360)

def _transform_angle_baseline(row):
    """this function is a modified ver. of the script from sfp (Broderick et al.2022).
     transform angle from Benson14 convention to our convention
    The Benson atlases' convention for angle in visual field is: zero is the upper vertical
    meridian, angle is in degrees, the left and right hemisphere both run from 0 to 180 from the
    upper to lower meridian (so they increase as you go clockwise and counter-clockwise,
    respectively). For our calculations, we need the following convention: zero is the right
    horizontal meridian, angle is in radians (and lie between 0 and 360, rather than -pi and pi),
    angle increases as you go clockwise, and each angle is unique (refers to one point on the
    visual field; we don't have the same number in the left and right hemispheres)
    """
    return row.angle

def prf_mgzs_to_df(prf_dict, angle_to_radians=True, transform_func=_transform_angle):

    prf_df = {}
    for hemi in ['lh','rh']:
        prf_df[hemi] = pd.DataFrame({})
        hemi_key_list = [k for k in prf_dict.keys() if hemi in k]
        for k in hemi_key_list:
            prf_name = k.split('_')[-1]
            tmp = pd.DataFrame(prf_dict[k], columns=[prf_name])
            if prf_name == 'vroinames':
                tmp['vroinames'] = tmp['vroinames'].apply(lambda x: _get_benson_atlas_rois(x))
            prf_df[hemi] = pd.concat((prf_df[hemi], tmp), axis=1)
        prf_df[hemi]['hemi'] = hemi
        prf_df[hemi] = prf_df[hemi].reset_index().rename(columns={'index': 'voxel'})
        prf_df[hemi]['angle'] = prf_df[hemi].apply(transform_func, axis=1)
    for hemi in ['lh', 'rh']:
        if angle_to_radians is True:
            prf_df[hemi]['angle'] = np.deg2rad(prf_df[hemi]['angle'])
    return prf_df

def load_lh_rh_prf_proporties_as_df(lh_prf_path_list, rh_prf_path_list, lh_mask=None, rh_mask=None, angle_to_radians=True, transform_func=_transform_angle):
    prf_dict = load_lh_rh_prf_properties_as_dict(lh_prf_path_list, rh_prf_path_list, lh_mask, rh_mask)
    prf_df = prf_mgzs_to_df(prf_dict, angle_to_radians, transform_func=transform_func)
    return prf_df



def load_betas(betas_path, mask, results_names=['modelmd']):
    """Load beta files. This is shaped as .mat files and lh and rh files are concatenated.
    See _load_mat_files() in one_dimensional_model.py of the Broderick et al (2022) Github."""
    # this is the case for all the models fields of the .mat file (modelse, modelmd,
    # models). [0, 0] contains the hrf, and [1, 0] contains the actual results
    betas = {}

    f = h5py.File(betas_path, 'r')
    for var in results_names:
        tmp_ref = f['results'][var]
        res = f[tmp_ref[1, 0]][:]  # actual data
        tmp = res.squeeze().transpose()  # voxel x conditions
        if var == "modelmd":
            for hemi in ['lh', 'rh']:
                k = f"{hemi}_betas"
                if hemi == 'lh':
                    tmper = tmp[:mask['lh'].shape[0]]
                else:
                    tmper = tmp[-mask['rh'].shape[0]:]
                tmper = tmper[mask[hemi], :]
                betas[k] = tmper
        elif var == "models":
            for hemi in ['lh', 'rh']:
                k = f"{hemi}_betas"
                if hemi == 'lh':
                    tmper = tmp[:mask['lh'].shape[0], :, :] # voxel x conditions x bootstrap
                else:
                    tmper = tmp[-mask['rh'].shape[0]:, :, :]
                tmper = tmper[mask[hemi], :, :]
                betas[k] = tmper

    return betas

def melt_2D_betas_into_df(betas):
    """Melt 2D shaped betas dict into a wide form of a dataframe."""
    betas_df = {}
    for hemi in ['lh', 'rh']:
        k = f'{hemi}_betas'
        if betas[k].ndim < 3:
            tmp = pd.DataFrame(betas[k]).reset_index()
            tmp = pd.melt(tmp, id_vars='index', var_name='class_idx', value_name='betas', ignore_index=True)
            tmp = tmp.rename(columns={'index': 'voxel'})
            betas_df[hemi] = tmp
        else:
            names = ['voxel', 'class_idx', 'bootstraps']
            index = pd.MultiIndex.from_product([range(s) for s in betas[k].shape], names=names)
            tmp = pd.DataFrame({'betas': betas[k].flatten()}, index=index)['betas']
            betas_df[hemi] = tmp.reset_index()
    return betas_df

def load_betas_as_df(betas_path, mask, results_names=['modelmd']):
    betas = load_betas(betas_path, mask, results_names)
    betas_df = melt_2D_betas_into_df(betas)
    return betas_df

def merge_prf_and_betas(prf_df, betas_df):
    merged_df = {}
    for hemi in ['lh', 'rh']:
        merged_df[hemi] = betas_df[hemi].merge(prf_df[hemi], on='voxel')
    return merged_df

def concat_lh_and_rh_df(df):
    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max() + 1
    return pd.concat((df['lh'], df['rh']), ignore_index=True, axis=0)

def merge_stim_df_and_betas_df(prf_betas_df, stim_df, on='class_idx'):
    if 'phase' in stim_df.columns:
        raise Exception('The stimulus info df should not contain phase column.')
    return prf_betas_df.merge(stim_df, on=on)

def make_broderick_sf_dataframe(stim_info_path,
                                lh_varea_path, rh_varea_path,
                                lh_eccentricity_path, rh_eccentricity_path,
                                lh_prf_path_list, rh_prf_path_list,
                                betas_path,
                                transform_func=_transform_angle_corrected,
                                eccen_range=[1,12], roi="V1",
                                angle_to_radians=True,
                                results_names=['modelmd']):
    stim_info = load_broderick_stim_info(stim_info_path)
    mask = make_lh_rh_masks(lh_varea_path, rh_varea_path, lh_eccentricity_path, rh_eccentricity_path, roi, eccen_range)
    prf_df = load_lh_rh_prf_proporties_as_df(lh_prf_path_list, rh_prf_path_list,
                                             mask['lh'], mask['rh'],
                                             angle_to_radians, transform_func=transform_func)
    betas_df = load_betas_as_df(betas_path, mask, results_names)
    prf_betas_df = merge_prf_and_betas(prf_df, betas_df)
    prf_betas_df = concat_lh_and_rh_df(prf_betas_df)
    sf_df = merge_stim_df_and_betas_df(prf_betas_df, stim_info, on='class_idx')
    sf_df['local_sf'], sf_df['local_ori'] = prep.calculate_local_stim_properties(w_a=sf_df['w_a'], w_r=sf_df['w_r'],
                                                                                 eccentricity=sf_df['eccentricity'], angle=sf_df['angle'],
                                                                                 angle_in_radians=angle_to_radians)
    return sf_df

def run_all_subj_main(sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121],
                      stim_description_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/stimuli/task-sfprescaled_stim_description_haji.csv',
             vroi_range=["V2","V3"], eroi_range=[1, 12],
             mask_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
             prf_label_names=['angle', 'eccen', 'sigma', 'varea'],
             prf_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
             beta_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/GLMdenoise/',
             df_save_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes/broderick',
             save_df=True, voxel_criteria='pRFcenter'):
    df = {}
    for sn in sn_list:
        df[sn] = sub_main(sn, stim_description_path,
                      vroi_range, eroi_range,
                      mask_path, prf_label_names, prf_dir, beta_dir, df_save_dir, save_df, voxel_criteria)
    all_subj_df = pd.concat(df, ignore_index=True)
    return all_subj_df

def save_sigma_v_df(sn, beta_col='betas', columns=['noise_SD', 'sigma_v_squared'],
                    df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/diff_vs_test',
                    df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/diff_vs_test/sigma_v'):
    subj = utils.sub_number_to_string(sn, dataset="broderick")
    df = pd.read_csv(os.path.join(df_dir, f"{subj}_stim_voxel_info_df_vs.csv"))
    sigma_v_df = bts.get_multiple_sigma_vs(df, power=[1, 2], columns=columns, to_sd=beta_col, to_group=['voxel','subj'])
    # save the final output
    df_save_name = f"{subj}_sigma_v_{beta_col}.csv"
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir)
    df_save_path = os.path.join(df_save_dir, df_save_name)
    sigma_v_df.to_csv(df_save_path, index=False)
    print(f'... {subj} sigma_v dataframe saved.')
    return sigma_v_df

def save_sigma_v_df_all_subj(sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121],
                             beta_col='betas', columns=['noise_SD', 'sigma_v_squared'],
                    df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/diff_vs_test',
                    df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/diff_vs_test/sigma_v'):
    df = {}
    for sn in sn_list:
        df[sn] = save_sigma_v_df(sn, beta_col, columns, df_dir, df_save_dir)
    sigma_v_df = pd.concat(df, ignore_index=True)
    return sigma_v_df

def modify_voxel_stim_info_df(sn, beta_col='betas',
                              df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/diff_vs_test',
                              df_name='stim_voxel_info_df_vs.csv',
                              df_save_name='stim_voxel_info_df_vs.csv'):
    subj = utils.sub_number_to_string(sn, dataset="broderick")
    df = pd.read_csv(os.path.join(df_dir, f"{subj}_{df_name}"))
    sigma_v_df = bts.get_multiple_sigma_vs(df, power=[1,2], columns=['noise_SD', 'sigma_v_squared'], to_sd='betas', to_group=['voxel','subj'])
    df = df.merge(sigma_v_df, on=['voxel','subj'])
    df.to_csv(os.path.join(df_dir, f"{subj}_{df_name}"))
    groupby_col = df.drop(columns=['bootstraps', 'betas', 'normed_betas']).columns.tolist()
    fnl_df = df.groupby(groupby_col).median().reset_index()
    fnl_df = fnl_df.drop(['bootstraps'], axis=1)
    fnl_df_save_path = os.path.join(df_dir, 'diff_vs_test_md', f"{subj}_{df_save_name}")
    fnl_df.to_csv(fnl_df_save_path, index=False)
    print(f'... {subj} df dataframe saved as {df_save_name}.')
    return fnl_df

