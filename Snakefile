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
measured_noise_sd =0.03995  # unnormalized 1.502063
LR_RATE = [0.0007]#np.linspace(5,9,5)*1e-4
NOISE_SD = [np.round(measured_noise_sd*x, 2) for x in [0, 1]]#, 1.5, 2, 2.5, 3
MAX_EPOCH = [30000]
N_VOXEL = [100]
FULL_VER = ["True"]
params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                       'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                       'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})


rule plot_all_loss_and_param_history:
    input:
        loss_fig = expand(os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL),
        param_fig = expand(os.path.join(config['FIG_DIR'],"simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)

rule plot_all_loss_history:
    input:
        expand(os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)


rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['SUBJ_DF_DIR'],
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_sd-0_n_vox-{n_voxels}.csv")
    run:
        params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                               'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                               'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})
        syn_data = sim.SynthesizeData(n_voxels=int(wildcards.n_voxels),df=None,replace=True,p_dist="data",
            stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params,full_ver=(wildcards.full_ver=="True"))
        syn_df_2d['normed_betas'] = model.normalize(syn_df_2d, to_norm="betas", phase_info=False)
        syn_df_2d.to_csv(output[0])

rule generate_noisy_synthetic_data:
    input:
        syn_df_2d = os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D",  "original_syn_data_2d_full_ver-{full_ver}_sd-0_n_vox-{n_voxels}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}.csv")
    run:
        syn_df = pd.read_csv(input.syn_df_2d)
        noisy_df_2d = sim.copy_df_and_add_noise(syn_df, beta_col="normed_betas", noise_mean=0, noise_sd=float(wildcards.noise_sd))
        noisy_df_2d.to_csv(output[0])

rule plot_synthetic_data:
    input:
        all_files = expand(os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", "syn_data_2d_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}.csv"), full_ver=FULL_VER, n_voxels=N_VOXEL, noise_sd=NOISE_SD)
    output:
        os.path.join(config['FIG_DIR'], 'lineplot_syn_data_2d_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}.png')
    run:
        all_df = pd.DataFrame({})
        for file in input.all_files:
            print(file)
            tmp =  pd.read_csv(file)
            all_df = pd.concat((all_df, tmp), ignore_index=True)


rule run_simulation:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'],  "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    run:
        # add noise
        syn_df = pd.read_csv(input.input_path)
        syn_df['sigma_v'] = np.ones(syn_df.shape[0],dtype=np.float64)
        syn_dataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
        syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=(wildcards.full_ver=="True"))
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch), print_every=5000, anomaly_detection=False, amsgrad=False, eps=1e-8)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)

rule plot_loss_history:
    input:
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    output:
        loss_fig = os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'figures', 'Epoch_vs_Loss','loss_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}-%j.log')
    run:
        loss_history = pd.read_csv(input.loss_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(loss_history.columns) is False:
            loss_history['lr_rate'] = float(wildcards.lr)
            loss_history['noise_sd'] = float(wildcards.noise_sd)
            loss_history['max_epoch'] = int(wildcards.max_epoch)
        model.plot_loss_history(loss_history, to_x="epoch",to_y="loss", to_label=None,
            title=f'{input.loss_history.split(os.sep)[-1]}',
            save_fig=True, save_path=output.loss_fig, ci="sd", n_boot=100, log_y=True)

rule plot_model_param_history:
    input:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    output:
        param_fig = os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'figures', 'Epoch_vs_Param_values','param_history_plot_full_ver-{full_ver}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}-%j.log')
    run:

        params = pd.DataFrame({'sigma': [2.2], 'slope': [0.12], 'intercept': [0.35],
                               'p_1': [0.06], 'p_2': [-0.03], 'p_3': [0.07], 'p_4': [0.005],
                               'A_1': [0.04], 'A_2': [-0.01], 'A_3': [0], 'A_4': [0]})

        model_history = pd.read_csv(input.model_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(model_history.columns) is False:
            model_history['lr_rate'] = float(wildcards.lr)
            model_history['noise_sd'] = float(wildcards.noise_sd)
            model_history['max_epoch'] = int(wildcards.max_epoch)
        model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')
        params_col, params_group = sim.get_params_name_and_group(params, (wildcards.full_ver=="True"))
        model.plot_param_history(model_history,params=params_col, group=params_group,
            to_label=None,label_order=None, ground_truth=True, to_col=None,
            lgd_title=None, title=f'noise_sd:{wildcards.noise_sd}, lr_rate: {wildcards.lr}',
            save_fig=True, save_path=output.param_fig, ci=68, n_boot=100, log_y=True, sharey=False, adjust="tight")

