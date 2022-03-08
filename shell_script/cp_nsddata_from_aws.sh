
!bin/bash

SNs=(01)
run=(01 02 03 04 05 06 07 08)


# download nsdsynthetic data from aws
for xSN in "${SNs[@]}"
do
SF_dir=/Users/auna/Dropbox/NYU/Projects/SF/NSD/nsddata_timeseries/ppdata/subj${xSN}/func1mm/timeseries


for mm in "${run[@]}"
do
aws s3 cp s3://natural-scenes-dataset/nsddata_timeseries/ppdata/subj01/func1mm/timeseries/timeseries_nsdsynthetic_run${mm}.nii.gz ${SF_dir} --profile nsd
done

done


# download nsdsynthetic experimental design 
for xSN in "${SNs[@]}"
do
mkdir -p /Users/auna/Dropbox/NYU/Projects/SF/NSD/nsddata_timeseries/ppdata/subj${xSN}/func1mm/design

SF_design_dir=/Users/auna/Dropbox/NYU/Projects/SF/NSD/nsddata_timeseries/ppdata/subj${xSN}/func1mm/design


for mm in "${run[@]}"
do
aws s3 cp s3://natural-scenes-dataset/nsddata_timeseries/ppdata/subj01/func1mm/design/design_nsdsynthetic_run${mm}.tsv ${SF_design_dir} --profile nsd
done

done

# download nsdsynthetic stimuli info from aws
for xSN in "${SNs[@]}"
do

mkdir -p /Users/auna/Dropbox/NYU/Projects/SF/NSD/nsddata/stimuli/nsdsynthetic/nsdsynthetic_subj${xSN}


SF_stim_dir=/Users/auna/Dropbox/NYU/Projects/SF/NSD/nsddata/stimuli/nsdsynthetic/nsdsynthetic_subj${xSN}

aws s3 cp --recursive s3://natural-scenes-dataset/nsddata/stimuli/nsdsynthetic/nsdsynthetic_subj${xSN}/ ${SF_stim_dir}/ --profile nsd

done


