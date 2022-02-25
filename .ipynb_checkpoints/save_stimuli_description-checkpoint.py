import pandas as pd
import numpy as np
import os

n_all_stims = 112 # number of sf stims presented in nsdsynthetic experiment
regular_sf_stims = ['pinwheel', 'forward spiral', 'annulus', 'reverse spiral']
base_sf_stims = ['intermediate 1', 'intermediate 2', 'intermediate 3', 'intermediate 4']
freq_levels = [6.0, 11.0, 20.0, 37.0, 69.0, 128.0] # from K. Kay's nsdsynthetic experiment design.m
phase_levels = [0.0, 1.570796327, 3.141592654, 4.71238898] # from K. Kay's nsdsynthetic experiment design.m
n_sf_stims = len(sf_stims)
n_sf_int_stims = 4
n_phase = len(phase_levels)
n_freq = len(freq_levels)

save_file_name = 'nsdsynthetic_sf_stim_description.csv'
save_path = os.path.join('/Users/auna/Dropbox/NYU/Projects/SF/natural-scenes-dataset/nsddata/stimuli/nsdsynthetic',
                         save_file_name)


# The intermediate spirals exist only at 37.0.
# load Broderick et al's spatialFrequencyStim.mat
original_stim_descriptions_path = '/Users/auna/Dropbox/NYU/Projects/SF/prac/task-sfp_stim_description.csv'
stim_df = pd.read_csv(original_stim_descriptions_path)


# column 1: image_idx
# each image's index in nsdsynthetic data (784 trials in total)
# images no. 105 to 216 are sf stimuli (written in MATLAB, starts from 1, ends at 784)
nsdsynthetic_image_idx = np.arange(0, n_all_stims) + 105

# column 2: stim class names
nsdsynthetic_names = [None]*n_all_stims
regular_names = np.repeat(regular_sf_stims, n_phase*n_freq)
intermediate_names = np.repeat(base_sf_stims, n_phase*1)
nsdsynthetic_names[0::] = np.append(regular_names, intermediate_names)

# column 3: name_idx
# 0: pinwheel
# 1: forward spiral
# 2: annulus
# 3: reverse spiral
# 4: intermediate1
# 5: intermediate2
# 6: intermediate3
# 7: intermediate4
nsdsynthetic_name_idx = np.empty((n_all_stims,))
regular_name_idx = np.arange(0, len(regular_sf_stims))
intermediate_name_idx = np.arange(0, len(base_sf_stims))+len(regular_sf_stims)
regular_name_idx = np.repeat(regular_name_idx, n_phase*n_freq)
intermediate_name_idx = np.repeat(intermediate_name_idx, n_phase*1)
nsdsynthetic_name_idx = np.append(regular_name_idx, intermediate_name_idx)

# column 4 & 5: stim_df.w_r & w_a
# repeat w_r and w_a for 4 different phases
# 0:4*6 = regular stims (4 classes, 6 frequencies)
# 4*6+1:end = intermediate stims (1 frequency)
stim_df_no_dups = stim_df.drop_duplicates('class_idx')
nsdsynthetic_w_r = np.repeat(stim_df_no_dups.w_r, n_phase)
nsdsynthetic_w_a = np.repeat(stim_df_no_dups.w_a, n_phase)

# column 6: phase info. (4 levels)
used_phase = np.unique(stim_df.phi)
used_phase = used_phase[used_phase_idx]
regular_phase = np.tile(used_phase, n_sf_stims*n_freq)
intermediate_phase = np.tile(used_phase, n_sf_int_stims*1)
nsdsynthetic_phase = np.append(regular_phase, intermediate_phase)

# column 7: phase index (originally 0-7, but in nsdsynthetic 0:2:7)
phase_idx = np.unique(stim_df.phase_idx)
used_phase_idx = phase_idx[0::2]
regular_phase_idx = np.tile(used_phase_idx, n_sf_stims*n_freq)
intermediate_phase_idx = np.tile(used_phase_idx, n_sf_int_stims*1)
nsdsynthetic_phase_idx = np.append(regular_phase_idx, intermediate_phase_idx)



# put all arrays into a dataframe

# Make a data frame
df = pd.DataFrame([nsdsynthetic_image_idx,
                   nsdsynthetic_names,
                   nsdsynthetic_name_idx,
                   nsdsynthetic_w_r,
                   nsdsynthetic_w_a,
                   nsdsynthetic_phase,
                   nsdsynthetic_phase_idx]).T

df = df.rename(columns={0:"image_idx",
                        1:"names",
                        2:"names_idx",
                        3:"w_r",
                        4:"w_a",
                        5:"phase",
                        6:"phase_idx"
                        })

df.to_csv(save_path, index=False)


