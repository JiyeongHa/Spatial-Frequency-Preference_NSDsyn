import seaborn as sns

import argparse
import sys
sys.path.append('/Users/jh7685/Documents/GitHub/spatial-frequency-preferences')
import sfp
import os
import itertools
import scipy as sp
from matplotlib.colors import LinearSegmentedColormap
from sklearn import linear_model
from bids.layout import BIDSLayout

import sys
import os
import nibabel as nib
import numpy as np
import pandas as pd
import h5py

vareas=[1]
eccen_range=(1, 12)
stim_rad_deg=12

# directory info.
base_dir = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/'
os.chdir(base_dir)
freesurfer_dir = 'nsddata/freesurfer/'
betas_dir='nsddata_betas/ppdata/'

# scan info.
nTR = 264 # number of measurements


## Arrange pRF mgzs into dict
# pRF properties are stored in each subject's freesurfer folder.
subj = "%s%s" % ('subj', '01')
# vareas = [1, 2, ]
# np.unique(varea_mask['lh]) is [0:7]
# 0 Unknown
# 1 V1v
# 2 V1d
# 3 V2v
# 4 V2d
# 5 V3v
# 6 V3d
# 7 hV4

eccen_range = []
varea_mask = {}
eccen_mask = {}
ecroi_mask = {}
angle_mask = {}
sigma_mask = {}
# eccen_range?
eccen_range = np.unique(ecroi_mask['lh'])

np.isin(varea_mask[hemi], vareas)

for hemi in ['lh', 'rh']:
    varea_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.prf-visualrois.mgz')
    # get_fdata() = (x, 1, 1) -> squeeze -> (x,)
    varea_mask[hemi] = nib.load(varea_path).get_fdata().squeeze()
    #varea_mask[hemi] = np.isin(varea_mask[hemi], )

    eccen_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.prfeccentricity.mgz')
    eccen_mask[hemi] = nib.load(eccen_path).get_fdata().squeeze()
#    eccen_mask[hemi] = (eccen_mask[hemi] > eccen_range[0]) & (eccen_mask[hemi] < eccen_range[-1:])

    ecroi_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.prf-eccrois.mgz')
    ecroi_mask[hemi] = nib.load(ecroi_path).get_fdata().squeeze()

    angle_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.prfangle.mgz')
    angle_mask[hemi] = nib.load(angle_path).get_fdata().squeeze()

    sigma_path = os.path.join(freesurfer_dir, subj, 'label', hemi + '.prfsize.mgz')
    sigma_mask[hemi] = nib.load(sigma_path).get_fdata().squeeze()

# load GLMdenoise file
# f.keys() -> shows betas
betas = {}
for hemi in ['lh', 'rh']:
    betas_file_name = hemi + '.betas_nsdsynthetic.hdf5'
    betas_path = os.path.join(betas_dir, subj, 'nativesurface/nsdsyntheticbetas_fithrf_GLMdenoise_RR', betas_file_name)
    f = h5py.File(betas_path, 'r') # read hdf
    betas[hemi] = f.get('betas') # shape shows 744 x voxels



## stimuli information
stim_description_file = 'nsdsynthetic_sf_stim_description.csv'
stim_description_path = os.path.join('/Users/jh7685/Dropbox/NYU/Projects/SF/natural-scenes-dataset/nsddata/stimuli/nsdsynthetic',
                                     stim_description_file)
stim_df = pd.read_csv(stim_description_path)

