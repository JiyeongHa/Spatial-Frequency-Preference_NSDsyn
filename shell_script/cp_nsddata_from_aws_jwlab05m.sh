!bin/bash

SNs=(01 02 03 04 05 06 07 08)
run=(01 02 03 04 05 06 07 08)

# In this script, we are going to do:
# ** Make the same data structure in the lab server
# ** Download NSD data from s3://natural-scenes-dataset.
# 1. beta weight files (surface)
# 2. pRF files
# 3. ROIs
# 4. freesurfer
# 5. Stimuli

s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/nativesurface/nsdsyntheticbetas_fithrf/
#### Make folders first
base_dir=/Volumes/server/Projects/sfp_nsd
nsd_dir=${base_dir}/natural-scenes-dataset
betas_dir=${nsd_dir}/nsddata_betas
# pRF_dir=
# Not sure about this, I need to ask Jan.
#ROIs_dir=
# I think ROIs are in the freesurfer_dir.
freesurfer_dir=${nsd_dir}/nsddata/freesurfer
# for now, I'll skip other than native space (fsaverage_sym, fsaverage, fsaverage3, ...)
stimuli_dir_hdf5=${nsd_dir}/nsddata_stimuli/stimuli/nsdsynthetic
# stimuli_dir_hdf5 folder contains each subj's color stimuli & all stimuli in a hdf5 format.
stimuli_dir_png=${nsd_dir}/nsddata/stimuli/nsdsynthetic
# stimuli_dir_png foldr contains stimuli in a png format.

# Repeat for each subject
for xSN in "${SNs[@]}"
do
xSN_betas_dir_voxel=${betas_dir}/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR

xSN_betas_dir_surface=${betas_dir}/ppdata/subj${xSN}/nativesurface/nsdsyntheticbetas_fithrf_GLMdenoise_RR

xSN_freesurfer_dir=${freesurfer_dir}/subj${xSN}

mkdir -p ${xSN_betas_dir_surface}
mkdir -p ${xSN_betas_dir_voxel}
mkdir -p ${xSN_freesurfer_dir}

# One time
mkdir -p ${stimuli_dir_hdf5}
mkdir -p ${stimuli_dir_png}
done

##### Download

aws_nsd_dir=s3://natural-scenes-dataset
aws_betas_dir=${aws_nsd_dir}/nsddata_betas
# pRF_dir =
# Not sure about this, I need to ask Jan.
#ROIs_dir =
# I think ROIs are in the freesurfer_dir.
aws_freesurfer_dir=${aws_nsd_dir}/nsddata/freesurfer
# for now, I'll skip other than native space (fsaverage_sym, fsaverage, fsaverage3, ...)
aws_stimuli_dir_hdf5=${aws_nsd_dir}/nsddata_stimuli/stimuli/nsdsynthetic
# stimuli_dir_hdf5 folder contains each subj's color stimuli & all stimuli in a hdf5 format.
aws_stimuli_dir_png=${aws_nsd_dir}/nsddata/stimuli/nsdsynthetic/nsdsynthetic
# stimuli_dir_png foldr contains stimuli in a png format.
# We are going to download only the common stimuli (nsdsynthetic/) from the png directory.


# Repeat for each subject
for xSN in "${SNs[@]}"
do
#xSN_betas_dir_voxel=${betas_dir}/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR
#aws_xSN_betas_dir_voxel=${aws_betas_dir}/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR

xSN_betas_dir_surface=${betas_dir}/ppdata/subj${xSN}/nativesurface/nsdsyntheticbetas_fithrf
aws_xSN_betas_dir_surface=${aws_betas_dir}/ppdata/subj${xSN}/nativesurface/nsdsyntheticbetas_fithrf
#aws s3 sync ${aws_xSN_betas_dir_voxel} ${xSN_betas_dir_voxel}/

aws s3 sync ${aws_xSN_betas_dir_surface} ${xSN_betas_dir_surface}/
done

# One time
aws s3 sync ${aws_stimuli_dir_png} ${stimuli_dir_png}/nsdsynthetic
aws s3 sync --exclude "fsaverage_sym"  --exclude "fsaverage"  --exclude "fsaverage3"  --exclude "fsaverage4"  --exclude "fsaverage5"  --exclude "fsaverage6" ${aws_freesurfer_dir} ${freesurfer_dir}

SNs=(01 02 03 04 05 06 07 08)

# Repeat for each subject
for xSN in "${SNs[@]}"
do
base_dir=/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR
mkdir -p ${base_dir}

aws s3 sync s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/ ${base_dir}
done

SNs=(01 02 03 04 05 06 07)

for xSN in "${SNs[@]}"
do
base_dir=/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsddata_betas/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR
cd ${base_dir}
aws s3 sync s3://natural-scenes-dataset/nsddata_betas/ppdata/subj${xSN}/func1mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/ .
done

