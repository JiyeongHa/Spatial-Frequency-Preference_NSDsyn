import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import os
import nibabel as nib
import numpy as np
import pandas as pd
import h5py
import itertools
import sfp_nsd_utils as utils
import voxel_selection as vs
import two_dimensional_model as model
import bootstrap as bts


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


def _label_vareas(row):
    return _get_benson_atlas_rois(row.varea)


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
    return np.mod((row.angle - 90), 360)

def prf_mgzs_to_df(prf_mgzs, prf_label_names=['angle','eccen','sigma','varea']):

    prf_df = {}
    for hemi in ['lh', 'rh']:
        tmp = {}
        for prf_name in prf_label_names:
            k = f'{hemi}-{prf_name}'
            tmp[prf_name] = pd.DataFrame(prf_mgzs[k])
            tmp[prf_name] = tmp[prf_name].rename(columns={0: prf_name})
            if prf_name == 'varea':
                tmp[prf_name]['vroinames'] = tmp[prf_name].apply(_label_vareas, axis=1)
            if prf_name == 'angle':
                if hemi == 'rh':
                    tmp[prf_name]['angle'] = -tmp[prf_name]['angle']
                tmp[prf_name]['angle'] = np.mod((tmp[prf_name]['angle'] - 90), 360)
        prf_df[hemi] = pd.concat(tmp, axis=1).droplevel(0, axis=1)
        prf_df[hemi] = prf_df[hemi].reset_index().rename(columns={'index': 'voxel'})

    return prf_df

def merge_prf_and_betas(betas_df, prf_df):

    for hemi in ['lh', 'rh']:
        betas_df[hemi] = betas_df[hemi].merge(prf_df[hemi], on='voxel')
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
    df['local_ori'] = np.mod(df['local_ori'], np.pi)
    return df

def calculate_local_sf(df):
    df['local_sf'] = np.sqrt((df.w_r ** 2 + df.w_a ** 2))
    df['local_sf'] = df['local_sf'] / df['eccen']
    df['local_sf'] = np.divide(df['local_sf'], 2 * np.pi)
    return df


def sub_main(sn,
             stim_description_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/stimuli/task-sfprescaled_stim_description_haji.csv',
             vroi_range=["V1"], eroi_range=[1, 12],
             mask_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
             prf_label_names=['angle', 'eccen', 'sigma', 'varea'],
             prf_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
             results_names=['models'],
             beta_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/GLMdenoise/',
             df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
             save_df=False, vs=True):
    subj = utils.sub_number_to_string(sn, dataset="broderick")
    stim_df = load_stim_info(stim_description_path, save_copy=False)
    mask = masking(sn, vroi_range, eroi_range, mask_path)
    prf_mgzs = load_prf(sn, mask, prf_label_names, prf_dir)
    betas = load_betas(sn, mask, results_names=results_names, beta_dir=beta_dir)
    betas_df = melt_2D_betas_into_df(betas)
    prf_df = prf_mgzs_to_df(prf_mgzs)
    voxel_df = merge_prf_and_betas(betas_df, prf_df)
    df = concat_lh_and_rh_df(voxel_df)
    df = add_stim_info_to_df(df, stim_df)
    df = calculate_local_orientation(df)
    df = calculate_local_sf(df)
    df['subj'] = subj
    df = df.rename(columns={'eccen': 'eccentricity'})
    df['normed_betas'] = model.normalize(df, 'betas', ['subj', 'voxel', 'bootstraps'], phase_info=True)
    if save_df:
        # save the final output
        df_save_name = "%s_%s" % (subj, "stim_voxel_info_df.csv")
        if not os.path.exists(df_save_dir):
            os.makedirs(df_save_dir)
        df_save_path = os.path.join(df_save_dir, df_save_name)
        df.to_csv(df_save_path, index=False)
        print(f'... {subj} dataframe saved.')
    if vs:
        df = select_voxels(subj, df, dv_to_group=['subj', 'voxel'], beta_col='betas',
                              df_save_dir=df_save_dir, save_df=save_df)
    return df

def run_all_subj_main(sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121],
                      stim_description_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/stimuli/task-sfprescaled_stim_description_haji.csv',
             vroi_range=["V1"], eroi_range=[1, 12],
             mask_path='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
             prf_label_names=['angle', 'eccen', 'sigma', 'varea'],
             prf_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/prf_solutions/',
             results_names=['models'],
             beta_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/GLMdenoise/',
             df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
             save_df=False, vs=True):
    df = {}
    for sn in sn_list:
        df[sn] = sub_main(sn, stim_description_path,
                      vroi_range, eroi_range,
                      mask_path, prf_label_names, prf_dir, results_names, beta_dir, df_save_dir, save_df, vs=vs)
    all_subj_df = pd.concat(df, ignore_index=True)
    return all_subj_df

def select_voxels(subj, df, dv_to_group=['subj','voxel'], beta_col='betas',
                  df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
                  save_df=False):
    vs_df = vs.drop_voxels_with_mean_negative_amplitudes(df, dv_to_group, beta_col)
    if save_df:
        # save the final output
        df_save_name = f"{subj}_stim_voxel_info_df_vs.csv"
        if not os.path.exists(df_save_dir):
            os.makedirs(df_save_dir)
        df_save_path = os.path.join(df_save_dir, df_save_name)
        df.to_csv(df_save_path, index=False)
        print(f'... {subj} dataframe_vs saved.')
    return vs_df

def save_sigma_v_df(sn, beta_col='betas',
                    df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
                    df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/sigma_v'):
    subj = utils.sub_number_to_string(sn, dataset="broderick")
    df = pd.read_csv(os.path.join(df_dir, f"{subj}_stim_voxel_info_df.csv"))
    sigma_v_df = bts.get_multiple_sigma_vs(df, power=[1,2], columns=['noise_SD', 'sigma_v_squared'], to_sd='betas', to_group=['voxel','subj'])
    # save the final output
    df_save_name = f"{subj}_sigma_v_{beta_col}.csv"
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir)
    df_save_path = os.path.join(df_save_dir, df_save_name)
    sigma_v_df.to_csv(df_save_path, index=False)
    print(f'... {subj} sigma_v dataframe saved.')
    return sigma_v_df

def save_sigma_v_df_all_subj(sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121], beta_col='betas',
                    df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
                    df_save_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes/sigma_v'):
    df = {}
    for sn in sn_list:
        df[sn] = save_sigma_v_df(sn, beta_col, df_dir, df_save_dir)
    sigma_v_df = pd.concat(df, ignore_index=True)
    return sigma_v_df

def modify_voxel_stim_info_df(sn, beta_col='betas',
                              df_dir='/Volumes/server/Projects/sfp_nsd/Broderick_dataset/derivatives/dataframes',
                              df_name='stim_voxel_info_df.csv',
                              df_save_name='stim_voxel_info_df_vs.csv'):
    subj = utils.sub_number_to_string(sn, dataset="broderick")
    df = pd.read_csv(os.path.join(df_dir, f"{subj}_{df_name}"))
    sigma_v_df = bts.get_multiple_sigma_vs(df, power=[1,2], columns=['noise_SD', 'sigma_v_squared'], to_sd='betas', to_group=['voxel','subj'])
    df = df.merge(sigma_v_df, on=['voxel','subj'])
    df = vs.drop_voxels_with_mean_negative_amplitudes(df, dv_to_group=['subj', 'voxel'], beta_col=beta_col)
    df.to_csv(os.path.join(df_dir, 'tmp', f"{subj}_stim_voxel_info_df_vs.csv"))
    fnl_df = df.groupby(['voxel', 'subj', 'names', 'freq_lvl', 'class_idx']).median().reset_index()
    fnl_df = fnl_df.drop(['bootstraps','phase'], axis=1)
    fnl_df_save_path = os.path.join(df_dir, f"{subj}_stim_voxel_info_df_vs_md.csv")
    fnl_df.to_csv(fnl_df_save_path, index=False)
    print(f'... {subj} df dataframe saved as {df_save_name}.')
    return fnl_df

