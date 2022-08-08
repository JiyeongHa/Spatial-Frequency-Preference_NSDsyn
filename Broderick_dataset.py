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


def _masking(sn, vroi_range=["V1"], eroi_range=[0.98, 12], mask_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/'):
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


def load_prf(sn, prf_label_names=['angle', 'eccen', 'sigma', 'varea'], vroi_range=["V1"], eroi_range=[0.98, 12],
             prf_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/'):
    mgzs = {}
    mask = _masking(sn, vroi_range=vroi_range, eroi_range=eroi_range, mask_path=prf_path)
    for hemi, prf_names in itertools.product(['lh', 'rh'], prf_label_names):
        k = f"{hemi}-{prf_names}"
        prf_file = f"{hemi}.inferred_{prf_names}.mgz"
        prf_path = os.path.join('/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
                                "sub-wlsubj{:03d}".format(sn), 'bayesian_posterior', prf_file)
        prf = nib.load(prf_path).get_fdata().squeeze()
        prf = prf[mask[hemi]]
        mgzs[k] = prf
    return mgzs
