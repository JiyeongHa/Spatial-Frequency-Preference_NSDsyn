import os
import sys
import numpy as np
import pandas as pd
import simulation as sim

configfile:
    "config.json"

LR_RATE = np.round(np.linspace(1e-4, 1e-3, 4), 7)
NOISE_SD = [0]
MAX_EPOCH = [35000]

rule run_all_simulations:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], 'simulation_results_2D', 'loss_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxel}.csv'), noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxel=N_VOXEL)

rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['SUBJ_DF_DIR'],
    output:
        os.path.join(config['OUTPUT_DIR'], "syn_data_2d_{n_voxel}.csv")
    run:
        params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                               'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                               'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

        syn_data = sim.SynthesizeData(n_voxels=int(wildcards.n_voxel),df=None,replace=True,p_dist="data",
            stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=False)
        syn_df_2d.to_csv(output[0])

rule run_simulation:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'], "syn_data_2d_{n_voxel}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], 'simulation_results_2D', 'model_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxel}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], 'simulation_results_2D', 'loss_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxel}.csv')
    run:
        import sfp_nsd_utils as utils
        import two_dimensional_model as model
        # add noise
        syn_df = pd.read_csv(input.input_path)
        syn_df['betas'] = sim.add_noise(syn_df['betas'], noise_mean=0, noise_sd=float(wildcards.noise_sd))
        syn_df['sigma_v'] = np.ones(syn_df.shape[0],dtype=np.float64)
        syn_dataset = model.SpatialFrequencyDataset(syn_df,beta_col='betas')
        syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=False)
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch), print_every=5000, anomaly_detection=False, amsgrad=False)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)










