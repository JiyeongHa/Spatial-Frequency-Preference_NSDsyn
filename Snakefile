import os
import sys
import numpy as np
import pandas as pd
import simulation as sim
import matplotlib
import sfp_nsd_utils as utils
import two_dimensional_model as model

configfile:
    "config.json"
measured_noise_sd =0.03995  # unnormalized 1.502063
LR_RATE = [0.0005] #[0.0007]#np.linspace(5,9,5)*1e-4
NOISE_SD = [np.round(measured_noise_sd*x, 2) for x in [1]] #, 1.5, 2, 2.5, 3
MAX_EPOCH = [1000]
N_VOXEL = [100]
FULL_VER = ["True"]
PW = ["True", "False"]

rule plot_all_loss_and_param_history:
    input:
        loss_fig = expand(os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL),
        param_fig = expand(os.path.join(config['FIG_DIR'],"simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, noise_sd=NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)

rule plot_all_loss_history:
    input:
        expand(os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)


rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['DF_DIR'],
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-0_n_vox-{n_voxels}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-0_n_vox-{n_voxels}.log")
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        np.random.seed(1)
        syn_data = sim.SynthesizeData(n_voxels=int(wildcards.n_voxels), pw=(wildcards.pw == "True"), p_dist="data", stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=(wildcards.full_ver=="True"))
        syn_df_2d.to_csv(output[0])

rule generate_noisy_synthetic_data:
    input:
        syn_df_2d = os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D",  "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-0_n_vox-{n_voxels}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","synthetic_data_2D","syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}.csv")
    run:
        np.random.seed(1)
        syn_df = pd.read_csv(input.syn_df_2d)
        noisy_df_2d = sim.copy_df_and_add_noise(syn_df, beta_col="normed_betas", noise_mean=0, noise_sd=float(wildcards.noise_sd))
        noisy_df_2d.to_csv(output[0])


rule generate_noisy_synthetic_data_with_mtpl:
    input:
        syn_df_2d = os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D",  "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-0_n_vox-{n_voxels}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","synthetic_data_2D","syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.csv")
    run:
        np.random.seed(1)
        syn_df = pd.read_csv(input.syn_df_2d)
        noisy_df_2d = sim.copy_df_and_add_noise(syn_df, beta_col="normed_betas", noise_mean=0, noise_sd=syn_df['noise_SD']*float(wildcards.n_sd_mtpl))
        noisy_df_2d.to_csv(output[0])

rule plot_synthetic_data:
    input:
        all_files = expand(os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}.csv"), full_ver=FULL_VER, pw=PW, n_voxels=N_VOXEL, noise_sd=MULTIPLES_OF_NOISE_SD)
    output:
        os.path.join(config['FIG_DIR'], 'lineplot_syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}.png')
    log:
        os.path.join(config['FIG_DIR'],"logs", 'lineplot_syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}.png')
    run:
        all_df = pd.DataFrame({})
        for file in input.all_files:
            print(file)
            tmp =  pd.read_csv(file)
            all_df = pd.concat((all_df,tmp),ignore_index=True)


rule run_simulation:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'],  "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","results_2D",'loss_history_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        # add noise
        syn_df = pd.read_csv(input.input_path)
        syn_dataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
        syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=(wildcards.full_ver=="True"))
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch), print_every=2000, anomaly_detection=False, amsgrad=False, eps=1e-8)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)


rule run_simulation_mptl:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'],  "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","results_2D",'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        # add noise
        syn_df = pd.read_csv(input.input_path)
        syn_dataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
        syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=(wildcards.full_ver=="True"))
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch), print_every=2000, anomaly_detection=False, amsgrad=False, eps=1e-8)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)

rule plot_loss_history_mptl:
    input:
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    output:
        loss_fig = os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'figures', 'Epoch_vs_Loss','loss_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        loss_history = pd.read_csv(input.loss_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(loss_history.columns) is False:
            loss_history['lr_rate'] = float(wildcards.lr)
            loss_history['noise_sd'] = float(wildcards.n_sd_mtpl)
            loss_history['max_epoch'] = int(wildcards.max_epoch)
            loss_history['full_ver'] = wildcards.full_ver
            loss_history['pw'] = wildcards.pw
        model.plot_loss_history(loss_history, to_x="epoch",to_y="loss", to_label=None,
            save_fig=True, save_path=output.loss_fig, ci="sd", n_boot=100, log_y=True)

rule plot_loss_history:
    input:
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    output:
        loss_fig = os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'figures', 'Epoch_vs_Loss','loss_plot_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        loss_history = pd.read_csv(input.loss_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(loss_history.columns) is False:
            loss_history['lr_rate'] = float(wildcards.lr)
            loss_history['noise_sd'] = float(wildcards.noise_sd)
            loss_history['max_epoch'] = int(wildcards.max_epoch)
            loss_history['full_ver'] = wildcards.full_ver
            loss_history['pw'] = wildcards.pw
        model.plot_loss_history(loss_history, to_x="epoch",to_y="loss", to_label=None,
            save_fig=True, save_path=output.loss_fig, ci="sd", n_boot=100, log_y=True)



rule plot_model_param_history:
    input:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    output:
        param_fig = os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'figures', 'Epoch_vs_Param_values','param_history_plot_full_ver-{full_ver}_pw-{pw}_sd-{noise_sd}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        model_history = pd.read_csv(input.model_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(model_history.columns) is False:
            model_history['lr_rate'] = float(wildcards.lr)
            model_history['noise_sd'] = float(wildcards.noise_sd)
            model_history['max_epoch'] = int(wildcards.max_epoch)
            model_history['full_ver'] = wildcards.full_ver
        model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')
        params_col, params_group = sim.get_params_name_and_group(params, (wildcards.full_ver=="True"))
        model.plot_param_history(model_history,params=params_col, group=params_group,
            to_label=None,label_order=None, ground_truth=True,
            lgd_title=None, save_fig=True, save_path=output.param_fig, ci=68, n_boot=100, log_y=True, sharey=False)

