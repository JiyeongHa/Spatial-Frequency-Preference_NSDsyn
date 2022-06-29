import os
import sys
import numpy as np
import pandas as pd
import simulation as sim

configfile:
    "config.json"

rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['SUBJ_DF_DIR'],
        output_dir = config['OUTPUT_DIR']
    output:
        "syn_data_2d.csv"
    run:
        params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                               'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                               'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

        syn_data = sim.SynthesizeData(n_voxels=100,df=None,replace=True,p_dist="data",
            stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params,full_ver=False)
        syn_df_2d.to_csv(os.path.join(input.output_dir, output[0]))

rule generate_synthetic_BOLD:
    input:
        rules.generate_synthetic_data.output
    output:
        "/none"
    run:
        syn_df_2d = {input}.synthesize_BOLD_2d(params, full_ver=False)
        syn_df_2d.to_csv({output})





