import os
import sys
import numpy as np
import pandas as pd
import simulation as sim

configfile:
    "config.json"

LR_RATE = np.linspace(1e-4, 1e-3, 5)
NOISE_SD = [0]
MAX_EPOCH = [2]

rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['SUBJ_DF_DIR'],
    output:
        os.path.join(config['OUTPUT_DIR'], "syn_data_2d.csv")
    run:
        params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                               'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                               'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

        syn_data = sim.SynthesizeData(n_voxels=100,df=None,replace=True,p_dist="data",
            stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=False)
        syn_df_2d.to_csv(output[0])

rule run_simulation:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'], "syn_data_2d.csv"),
        save_dir = config['OUTPUT_DIR'],
    run:
        import itertools
        import sfp_nsd_utils as utils
        import two_dimensional_model as model
        # add noise
        for i, lr_rate, max_epoch in itertools.product(LR_RATE, NOISE_SD, MAX_EPOCH):
            syn_df = pd.read_csv(input.input_path)
            syn_df['betas'] = sim.add_noise(syn_df['betas'], noise_mean=0, noise_sd=i)
            syn_df['sigma_v'] = np.ones(syn_df.shape[0],dtype=np.float64)
            syn_dataset = model.SpatialFrequencyDataset(syn_df,beta_col='betas')
            syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=False)
            syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
                learning_rate=lr_rate, max_epoch=max_epoch, print_every=5000, anomaly_detection=False)
            str_noise_sd = str(i).replace('0.','p')
            str_lr_rate = str(lr_rate).replace('0.','p')
            model_f_name = f'model_history_w_noise_{str_noise_sd}_lr_{str_lr_rate}_eph_{max_epoch}.csv'
            loss_f_name = f'loss_history_w_noise_{str_noise_sd}_lr_{str_lr_rate}_eph_{max_epoch}.csv'
            ##
            utils.save_df_to_csv(syn_model_history, input.save_dir, model_f_name, indexing=False)
            utils.save_df_to_csv(syn_loss_history, input.save_dir, loss_f_name, indexing=False)








