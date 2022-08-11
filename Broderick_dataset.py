import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import os
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import itertools
import sfp_nsd_utils as utils

def _label_stim_names(row):
    if row.w_r == 0 and row.w_a != 0:
        return 'pinwheel'
    elif row.w_r != 0 and row.w_a == 0:
        return 'annulus'
    elif row.w_r == row.w_a:
        return 'forward spiral'
    elif row.w_r == -1*row.w_a:
        return 'reverse spiral'
    else:
        return 'mixtures'

def _label_freq_lvl(freq_lvl=10, phi_repeat=8, main_classes=4, mixture_freq_lvl=5, n_mixtures=8):
    freq_lvl_main_classes = np.tile(np.repeat(np.arange(0, freq_lvl), phi_repeat), main_classes)
    freq_lvl_mixtures = np.repeat(mixture_freq_lvl, phi_repeat * n_mixtures)
    freq_lvl = np.concatenate((freq_lvl_main_classes, freq_lvl_mixtures))
    return freq_lvl

def load_stim_info(stim_description_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/stimuli/task-sfprescaled_stim_description_haji.csv',
                   save_copy=True):
    """stimulus description file will be loaded as a dataframe."""

    # stimuli information
    if os.path.exists(stim_description_path):
        stim_df = pd.read_csv(stim_description_path)
    else:
        if not os.path.exists(stim_description_path.replace('_haji.csv', '.csv')):
            raise Exception('The original stim description file does not exist!\n')
        stim_df = pd.read_csv(stim_description_path)
        stim_df = stim_df.dropna(axis=0)
        stim_df = stim_df.astype({'class_idx': int})
        stim_df = stim_df.drop(columns='res')
        stim_df = stim_df.rename(columns={'phi': 'phase', 'index': 'image_idx'})
        stim_df['names'] = stim_df.apply(_label_stim_names, axis=1)
        stim_df['freq_lvl'] = _label_freq_lvl()
        stim_df.to_csv(stim_description_path, index=False)
    return stim_df


def _get_benson_atlas_rois(roi_index):
    """ switch num to roi names or the other way around. see https://osf.io/knb5g/wiki/Usage/"""

    num_key = {1: "V1", 2: "V2", 3: "V3", 4: "hV4", 5: "VO1", 6: "VO2",
        7: "LO1", 8: "LO2", 9: "TO1", 10: "TO2", 11: "V3b", 12: "V3a"}
    name_key = {y: x for x, y in num_key.items()}
    switcher = {**num_key, **name_key}
    return switcher.get(roi_index, "No Visual area")


def masking(sn, vroi_range=["V1"], eroi_range=[0.98, 12], mask_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/'):
    """create a mask using visual rois and eccentricity range."""
    mask = {}
    vroi_num = [_get_benson_atlas_rois(x) for x in vroi_range]
    for hemi in ['lh', 'rh']:
        varea_name = f"{hemi}.inferred_varea.mgz"
        varea_path = os.path.join(mask_path, "sub-wlsubj{:03d}".format(sn), 'bayesian_posterior', varea_name)
        varea = nib.load(varea_path).get_fdata().squeeze()
        vroi_mask = np.isin(varea, vroi_num)
        eccen_name = f"{hemi}.inferred_eccen.mgz"
        eccen_path = os.path.join(mask_path, "sub-wlsubj{:03d}".format(sn), 'bayesian_posterior', eccen_name)
        eccen = nib.load(eccen_path).get_fdata().squeeze()
        eccen_mask = (eroi_range[0] < eccen) & (eccen < eroi_range[-1])
        roi_mask = vroi_mask & eccen_mask
        mask[hemi] = roi_mask
    return mask


def load_prf(sn, mask, prf_label_names=['angle', 'eccen', 'sigma', 'varea'],
             prf_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/'):
    mgzs = {}
    for hemi, prf_names in itertools.product(['lh', 'rh'], prf_label_names):
        k = f"{hemi}-{prf_names}"
        prf_file = f"{hemi}.inferred_{prf_names}.mgz"
        prf_path = os.path.join(prf_dir,
                                "sub-wlsubj{:03d}".format(sn), 'bayesian_posterior', prf_file)
        prf = nib.load(prf_path).get_fdata().squeeze()
        prf = prf[mask[hemi]]
        mgzs[k] = prf
    return mgzs


def load_betas(sn, mask, results_names=['modelmd'], beta_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/GLMdenoise/'):
    """Load beta files. This is shaped as .mat files and lh and rh files are concatenated.
    See _load_mat_files() in first_level_analysis.py of the Broderick et al (2022) Github."""
    betas = {}
    subj = "sub-wlsubj{:03d}".format(sn)
    betas_file = f'{subj}_ses-04_task-sfprescaled_results.mat'
    betas_path = os.path.join(beta_dir, betas_file)
    # this is the case for all the models fields of the .mat file (modelse, modelmd,
    # models). [0, 0] contains the hrf, and [1, 0] contains the actual results
    f = h5py.File(betas_path, 'r')
    for var in results_names:
        tmp_ref = f['results'][var]
        res = f[tmp_ref[1, 0]][:]  # actual data
        tmp = res.squeeze().transpose()  # voxel x conditions
        if var == "modelmd":
            for hemi in ['lh', 'rh']:
                k = f"{hemi}-betas"
                if hemi == 'lh':
                    tmper = tmp[:mask['lh'].shape[0]]
                else:
                    tmper = tmp[-mask['rh'].shape[0]:]
                tmper = tmper[mask[hemi], :]
                betas[k] = tmper
        elif var == "models": # voxel x conditions x bootstrap
            for hemi in ['lh', 'rh']:
                k = f"{hemi}-betas"
                if hemi == 'lh':
                    tmper = tmp[:mask['lh'].shape[0], :, :]
                else:
                    tmper = tmp[-mask['rh'].shape[0]:, :, :]
                tmper = tmper[mask[hemi], :, :]
                betas[k] = tmper

    return betas

def melt_2D_betas_into_df(betas):
    """Melt 2D shaped betas dict into a wide form of a dataframe."""
    betas_df = {}
    for hemi in ['lh', 'rh']:
        k = f'{hemi}-betas'
        tmp = pd.DataFrame(betas[k]).reset_index()
        tmp = pd.melt(tmp, id_vars='index', var_name='class_idx', value_name='betas', ignore_index=True)
        tmp = tmp.rename(columns={'index': 'voxel'})
        betas_df[hemi] = tmp
    return betas_df


def _label_vareas(row):
    return _get_benson_atlas_rois(row.varea)

def merge_prf_and_betas(betas_df, prf_mgzs, prf_label_names=['angle','eccen','sigma','varea']):

    for hemi, prf_name in itertools.product(['lh','rh'], prf_label_names):
        k = f'{hemi}-{prf_name}'
        tmp = pd.DataFrame(prf_mgzs[k]).reset_index()
        tmp = tmp.rename(columns={'index': 'voxel', 0: prf_name})
        if prf_name == 'vareas':
            tmp['vroinames'] = tmp.apply(_label_vareas, axis=1)
        betas_df[hemi] = betas_df[hemi].merge(tmp, on='voxel')
    return betas_df

def concat_lh_and_rh_df(df):
    df['rh'].voxel = df['rh'].voxel + df['lh'].voxel.max() + 1
    return pd.concat(df).reset_index(0).rename(columns={'level_0': 'hemi'})

def add_stim_info_to_df(df, stim_df):
    stim_df = stim_df[stim_df['phase'] == 0]
    return df.merge(stim_df, on='class_idx')

def calculate_local_orientation(df):
    ang = np.arctan2(df['w_a'], df['w_r'])
    df['local_ori'] = np.deg2rad(df['angle']) + ang
    df['local_ori'] = np.remainder(df['local_ori'], np.pi)
    return df

def calculate_local_sf(df):
    df['local_sf'] = np.sqrt((df.w_r ** 2 + df.w_a ** 2))
    df['local_sf'] = df['local_sf'] / df['eccen']
    df['local_sf'] = np.divide(df['local_sf'], 2 * np.pi)
    return df




