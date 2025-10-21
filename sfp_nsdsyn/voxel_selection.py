import os
import numpy as np
import h5py
import pandas as pd
import seaborn as sns
from . import utils as utils
from math import atan2, degrees


def count_voxels(tmp, dv_to_group=["subj", "vroinames"], count_beta_sign=True):
    """count the number of voxels for each group specified in the dv_to_group variable.
     count_beta_sign arg will let you count voxels depending on the sign of mean avg betas
     across 28 conditions."""


    if count_beta_sign is True:
        tmp = tmp.groupby(dv_to_group + ["voxel"])['betas'].mean().reset_index()
        tmp['beta_sign'] = np.sign(tmp['betas'])
        tmp['beta_sign'] = tmp['beta_sign'].map({-1.0: "negative", 1.0: "positive"})
        dv_to_group += ["beta_sign"]

    n_voxel_df = tmp.groupby(dv_to_group, as_index=False)['voxel'].nunique()
    n_voxel_df = n_voxel_df.rename(columns={"voxel": "n_voxel"})
    return n_voxel_df


def plot_num_of_voxels(n_voxel_df, graph_type='bar',
                       to_hue='beta_sign', legend_title='Mean beta sign',
                       new_legend=None,
                       x_axis='vroinames', y_axis='n_voxel',
                       x_axis_label='ROI', y_axis_label='Number of Voxels',
                       save_fig=False, save_file_name='n_voxels_ROI_beta_sign.png',
                       save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                       super_title=None):
    sns.set_context("notebook", font_scale=1.5)
    grid = sns.catplot(data=n_voxel_df,
                       x=x_axis, y=y_axis,
                       hue=to_hue,
                       kind=graph_type,
                       ci=68, capsize=0.1,
                       order=utils.sort_a_df_column(n_voxel_df[x_axis]),
                       hue_order=utils.sort_a_df_column(n_voxel_df[to_hue])
                       )
    grid.set_axis_labels(x_axis_label, y_axis_label, fontsize=15)
    grid.fig.legend(title=legend_title)
    if new_legend is not None:
        for t, l in zip(legend.texts, new_legend):
            t.set_text(l)
    if super_title is not None:
        grid.fig.suptitle(f'{super_title}', fontsize=18, fontweight="bold")
    grid.fig.subplots_adjust(top=0.9, right=0.7)
    grid.tight_layout()
    fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
    save_path = os.path.join(fig_dir, f'{save_file_name}')
    utils.save_fig(save_fig, save_path)
    return grid



def pix_to_deg(pixels, screen_height=39.29, n_pixel_height=1080, visual_distance=176.5):
    """1920x1080 pixels, 69.84x39.29cm, 176.5 cm distance"""
    theta = degrees(atan2(0.5 * screen_height, visual_distance))
    deg_per_pix = theta / (0.5 * n_pixel_height)

    return deg_per_pix * pixels

def drop_voxels_outside_stim_range(df, in_pix=42.878, out_pix=714/2):
    #inner_radi = pix_to_deg(np.asarray([1.581139, 3.535534, 6.670832, 12.349089, 23.119256, 42.877733]))  # in pixels
    #outer_radi = pix_to_deg(714 / 2)
    inner_radi = pix_to_deg(in_pix)
    outer_radi = pix_to_deg(out_pix)
    df = df.query('@inner_radi < eccentricity < @outer_radi')
    return df


def _rgb2gray(rgb_image):
    return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])


def _get_distance(size=(714, 1360)):
    """Calculate distance from origin and save them as a matrix.
    This function is equivalent to mkR() in SF. """

    origin = ((size[0] + 1) / 2, (size[1] + 1) / 2)
    x, y = np.meshgrid(np.arange(1, size[1] + 1) - origin[1],
                       np.arange(1, size[0] + 1) - origin[0])
    return np.sqrt(x ** 2 + y ** 2)


# first function: min & max eccentricity of stimulus
def _find_ecc_range_of_stim(image='nsdsynthetic_stimuli.hdf5',
                            image_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_stimuli/stimuli/nsdsynthetic',
                            image_idx=(105, 216),
                            mid_val=114.543):
    """find min and max eccentricity of each stimulus where stimuli were not presented"""
    image_path = os.path.join(image_dir, image)
    f = h5py.File(image_path, 'r').get('imgBrick')
    f = f[image_idx[0] - 1:image_idx[1], :, :, :]
    f = _rgb2gray(f)
    f = np.round(f, 3)
    R = _get_distance(f.shape[1:3])
    img_min_max = np.empty((f.shape[0], 2))
    for k in np.arange(0, f.shape[0]):
        x, y = np.where(f[k, :, :] != mid_val)
        img_min_max[k, :] = [R[x, y].min(), R[x, y].max()]

    return img_min_max


def _find_min_and_max(img_min_max, stim_info_df):
    """ find minimum val of max eccentricity & max val of minimum ecc"""
    spiral_radii = [1.581139, 3.535534, 6.670832, 12.349089, 23.119256, 42.877733]
    from sfp_nsdsyn import prep
    img_df = pd.DataFrame(img_min_max).rename(columns={0: 'min_stim_ecc_R', 1: 'max_stim_ecc_R'})
    img_df = img_df.reset_index()
    img_df = stim_info_df.reset_index().merge(img_df)
    outer_boundary = img_min_max[:, 1].min()
    inner_boundary = img_min_max[:, 0]
    return inner_boundary, outer_boundary

def drop_voxels(df, dv_to_group=["subj", "voxel"], beta_col='avg_betas', in_pix=42.878, out_pix=712/4,
                mean_negative=True, stim_range=True):
    """Performs voxel selection all at once"""
    if mean_negative is True:
        df = drop_voxels_with_negative_mean_amplitudes(df, to_group=dv_to_group)
    if stim_range is True:
        df = drop_voxels_outside_stim_range(df, in_pix, out_pix, dv_to_group=['freq_lvl', 'subj'])

    return df

def drop_voxels_with_negative_mean_amplitudes(df, to_group=['voxel'], return_voxel_list=False):
    """drop all voxels that have an average negative amplitude across 28 conditions
        """
    tmp = df.copy()
    if 'bootstraps' in tmp.columns:
        tmp = tmp.groupby(to_group).median().reset_index()
    mean_tmp = tmp.groupby(to_group)['betas'].mean().reset_index()
    negative_mean_voxels = mean_tmp.groupby(to_group).filter(lambda x: (x['betas'] < 0)).voxel.unique()
    if return_voxel_list is True:
        return df.query('voxel not in @negative_mean_voxels'), negative_mean_voxels
    else:
        return df.query('voxel not in @negative_mean_voxels')

def drop_voxels_near_border(df, inner_border, outer_border):
    tmp = df.query('eccentricity + size <= @outer_border')
    vs_df = tmp.query('eccentricity - size >= @inner_border')
    # tmp = df.groupby(to_group).filter(lambda x: (x.eccentricity + x.size <= outer_border).all())
    # vs_df = tmp.groupby(to_group).filter(lambda x: (x.eccentricity - x.size >= inner_border).all())
    return vs_df


def select_voxels(df,
                  drop_by,
                  inner_border, outer_border,
                  to_group=['voxel'], return_voxel_list=False):
    vs_df = df.copy()
    if drop_by == 'pRFsize':
        vs_df = drop_voxels_near_border(vs_df, inner_border, outer_border)
    elif drop_by == 'pRFcenter':
        vs_df = vs_df.query('@inner_border < eccentricity < @outer_border')
    vs_df = drop_voxels_with_negative_mean_amplitudes(vs_df, to_group, return_voxel_list)
    return vs_df
