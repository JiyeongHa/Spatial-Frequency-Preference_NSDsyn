import os
import sys
import numpy as np
import pandas as pd
import simulation as sim
import matplotlib
matplotlib.use("TkAgg")
import sfp_nsd_utils as utils
import two_dimensional_model as model

configfile:
    "config.json"
measured_noise_sd = 1.502063
LR_RATE = np.linspace(5,9,5)*1e-4
NOISE_SD =[measured_noise_sd*x for x in [1, 1.5, 2, 2.5, 3]]
MAX_EPOCH = [40000]
N_VOXEL = [100]

params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                       'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                       'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})


rule run_all_simulations:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], 'simulation_results_2D', 'loss_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxels}.csv'), noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)

rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['SUBJ_DF_DIR']
    output:
        os.path.join(config['OUTPUT_DIR'], "syn_data_2d_{n_voxels}.csv")
    run:
        params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                               'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                               'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

        syn_data = sim.SynthesizeData(n_voxels=int(wildcards.n_voxels),df=None,replace=True,p_dist="data",
            stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=False)
        syn_df_2d.to_csv(output[0])

rule run_simulation:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'], "syn_data_2d_{n_voxels}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], 'simulation_results_2D', 'model_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxels}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], 'simulation_results_2D', 'loss_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxels}.csv')
    run:
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

rule plot_loss_history:
    input:
        loss_history = os.path.join(config['OUTPUT_DIR'],'simulation_results_2D','loss_history_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxels}.csv')
    output:
        loss_fig = os.path.join(config['FIG_DIR'], 'Epoch_vs_Loss', 'loss_plot_noise-{noise_sd}_lr-{lr}_eph-{max_epoch}_n_vox-{n_voxels}.png')
    run:
        loss_history = pd.read_csv(input.loss_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(loss_history.columns) is False:
            loss_history['lr_rate'] = float(wildcards.lr)
            loss_history['noise_sd'] = float(wildcards.noise_sd)
            loss_history['max_epoch'] = int(wildcards.max_epoch)
        model.plot_loss_history(loss_history, to_x="epoch",to_y="loss", to_label=None,
            title=f'learning rate={wildcards.lr}, noise sd={wildcards.noise_sd}',
            save_fig=True, save_path=output.loss_fig, ci="sd", n_boot=100, log_y=True)

rule plot_param_history:
    input:
        param_history =





