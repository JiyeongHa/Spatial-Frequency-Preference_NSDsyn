import sys
import os
import numpy as np
import h5py
import pandas as pd
from math import atan2, degrees


def drop_voxels_with_mean_negative_amplitudes(df):
    """drop all voxels that have an average negative amplitude across 28 conditions
        """
    dv_to_group = ["subj", "voxel"]
    tmp = df.groupby(dv_to_group)['avg_betas'].mean().reset_index()
    tmp.drop(tmp[tmp.avg_betas < 0].index, inplace=True)
    voxels = tmp.voxel.unique()
    df = df.query('voxel in @voxels')
    return df

def pix_to_deg(size_in_pix=100, screen_height=39.29, n_pixel_height=1080, visual_distance=176.5):
    """1920x1080 pixels, 69.84x39.29cm, 176.5 cm distance"""
    theta = degrees(atan2(0.5*screen_height, visual_distance))
    deg_per_pix = theta / (0.5*n_pixel_height)

    return deg_per_pix * size_in_pix

def drop_voxels_outside_stim_range(df, dv_to_group=['freq_lvl', 'subj']):
    inner_radi = pix_to_deg(np.asarray([1.581139, 3.535534, 6.670832, 12.349089, 23.119256, 42.877733])) # in pixels
    outer_radi = pix_to_deg(714/2)
    df = df.query('@inner_radi[-1] < eccentricity < @outer_radi')
    return df





def _rgb2gray(rgb_image):
        return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])

def _get_distance(size=(714, 1360)):
    """Calculate distance from origin and save them as a matrix.
    This function is equivalent to mkR() in SF. """

    origin = ((size[0]+1)/2, (size[1]+1)/2)
    x, y = np.meshgrid(np.arange(1, size[1]+1)-origin[1],
                       np.arange(1, size[0]+1)-origin[0])
    return np.sqrt(x ** 2 + y ** 2)

# first function: min & max eccentricity of stimulus
def _find_ecc_range_of_stim(image='nsdsynthetic_stimuli.hdf5',
                            image_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_stimuli/stimuli/nsdsynthetic',
                            image_idx=(105, 216),
                            mid_val=114.543):
    """find min and max eccentricity of each stimulus where stimuli were not presented"""
    image_path = os.path.join(image_dir, image)
    f = h5py.File(image_path, 'r').get('imgBrick')
    f = f[image_idx[0]-1:image_idx[1],:,:,:]
    f = _rgb2gray(f)
    f = np.round(f, 3)
    R = _get_distance(f.shape[1:3])
    img_min_max = np.empty((f.shape[0], 2))
    for k in np.arange(0, f.shape[0]):
        x, y = np.where(f[k,:,:] != mid_val)
        img_min_max[k, :] = [R[x, y].min(), R[x, y].max()]

    return img_min_max

def _find_min_and_max(img_min_max):
    """ find minimum val of max eccentricity & max val of minimum ecc"""
    spiral_radii=[1.581139, 3.535534, 6.670832, 12.349089, 23.119256, 42.877733]

    stim_info_df = make_df._load_stim_info(stim_description_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/derivatives')
    img_df = pd.DataFrame(img_min_max).rename(columns={0: 'min_stim_ecc_R', 1: 'max_stim_ecc_R'})
    img_df = img_df.reset_index()
    img_df = stim_info_df.reset_index().merge(img_df)
    outer_boundary = img_min_max[:,1].min()
    inner_boundary = img_min_max[:,0]
    return inner_boundary, outer_boundary
