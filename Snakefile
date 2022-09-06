import os
import sys
import numpy as np
import pandas as pd
import simulation as sim
import matplotlib as mpl
#mpl.use('svg', warn=False)
import sfp_nsd_utils as utils
import two_dimensional_model as model
import pickle
pickle.HIGHEST_PROTOCOL = 4

configfile:
    "config.json"
measured_noise_sd =0.03995  # unnormalized 1.502063
LR_RATE = [0.0005] #[0.0007]#np.linspace(5,9,5)*1e-4
MULTIPLES_OF_NOISE_SD = [1]
NOISE_SD = [np.round(measured_noise_sd*x, 2) for x in [1]]
MAX_EPOCH = [20000]
N_VOXEL = [100]
FULL_VER = ["True"]
PW = ["True"]
SN_LIST = ["{:02d}".format(sn) for sn in np.arange(1,9)]
broderick_sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
DATASET=['nsdsyn']
SUBJ_OLD = [utils.sub_number_to_string(sn, dataset="broderick") for sn in broderick_sn_list]

def make_subj_list(wildcards):
    if wildcards.dset == "broderick":
        return [utils.sub_number_to_string(sn, dataset="broderick") for sn in [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]]
    elif wildcards.dset == "nsdsyn":
        return [utils.sub_number_to_string(sn, dataset="nsd") for sn in np.arange(1,9)]


def make_subj_list_(dset):
    if dset == "broderick":
        return [utils.sub_number_to_string(sn, dataset="broderick") for sn in [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]]
    elif dset == "nsdsyn":
        return [utils.sub_number_to_string(sn, dataset="nsd") for sn in np.arange(1,9)]

rule plot_all:
    input:
        expand(os.path.join(config['BD_DIR'],"figures", "sfp_model","results_2D",'{df_type}_history_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_V1.png'), df_type=["loss","model"], full_ver=FULL_VER, lr=LR_RATE, max_epoch=MAX_EPOCH),
        expand(os.path.join(config['BD_DIR'],"figures","sfp_model","results_2D",'scatterplot_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_V1.png'),full_ver=FULL_VER, lr=LR_RATE, max_epoch=MAX_EPOCH)

rule run_Broderick_all_subj:
    input:
        expand(os.path.join(config['BD_DIR'], "sfp_model", "results_2D", 'loss_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_V1.h5'), full_ver="True", subj=SUBJ, lr=LR_RATE, max_epoch=MAX_EPOCH)

rule run_simulation_all_subj:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}_lr-{lr}_eph-{max_epoch}.csv'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, sn=SN_LIST,  lr=LR_RATE, max_epoch=MAX_EPOCH)


rule plot_all_loss_and_param_history:
    input:
        loss_fig = expand(os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL),
        param_fig = expand(os.path.join(config['FIG_DIR'],"simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)

rule plot_all_loss_history:
    input:
        expand(os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)


rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['STIM_INFO_DIR'],'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = config['DF_DIR']
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-0_n_vox-{n_voxels}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-0_n_vox-{n_voxels}.log")
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        np.random.seed(1)
        syn_data = sim.SynthesizeData(n_voxels=int(wildcards.n_voxels), pw=(wildcards.pw == "True"), p_dist="data", stim_info_path=input.stim_info_path, subj_df_dir=input.subj_df_dir)
        syn_df_2d = syn_data.synthesize_BOLD_2d(params, full_ver=(wildcards.full_ver=="True"))
        syn_df_2d.to_csv(output[0])

rule generate_noisy_synthetic_data:
    input:
        syn_df_2d = os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D",  "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-0_n_vox-{n_voxels}.csv")
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
        all_files = expand(os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.csv"), full_ver=FULL_VER, pw=PW, n_voxels=N_VOXEL, n_sd_mtpl=MULTIPLES_OF_NOISE_SD)
    output:
        os.path.join(config['FIG_DIR'], 'lineplot_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.png')
    log:
        os.path.join(config['FIG_DIR'],"logs", 'lineplot_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.png')
    run:
        all_df = pd.DataFrame({})
        for file in input.all_files:
            print(file)
            tmp =  pd.read_csv(file)
            all_df = pd.concat((all_df,tmp),ignore_index=True)


rule run_simulation:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'],  "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv'),
        losses_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'losses_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","results_2D",'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        # add noise
        syn_df = pd.read_csv(input.input_path)
        syn_dataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
        syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=(wildcards.full_ver=="True"))
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch), print_every=2000, anomaly_detection=False, amsgrad=False, eps=1e-8)
        losses_history = model.shape_losses_history(losses, syn_df)
        utils.save_df_to_csv(losses_history, output.losses_history, indexing=False)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)


rule plot_loss_history:
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



rule plot_model_param_history:
    input:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.csv')
    output:
        param_fig = os.path.join(config['FIG_DIR'], "simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'figures', 'Epoch_vs_Param_values','param_history_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.log')
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        model_history = pd.read_csv(input.model_history)
        if {'lr_rate', 'noise_sd', 'max_epoch'}.issubset(model_history.columns) is False:
            model_history['lr_rate'] = float(wildcards.lr)
            model_history['noise_sd'] = float(wildcards.n_sd_mtpl)
            model_history['max_epoch'] = int(wildcards.max_epoch)
            model_history['full_ver'] = wildcards.full_ver
        model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')
        params_col, params_group = sim.get_params_name_and_group(params, (wildcards.full_ver=="True"))
        model.plot_param_history(model_history,params=params_col, group=params_group,
            to_label=None,label_order=None, ground_truth=True,
            lgd_title=None, save_fig=True, save_path=output.param_fig, ci=68, n_boot=100, log_y=True, sharey=False)



rule generate_synthetic_data_subj:
    input:
        subj_df_dir = config['DF_DIR']
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-0_subj-{sn}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', "simulation", "synthetic_data_2D", "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-0_subj-{sn}.log")
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        subj_data = sim.SynthesizeRealData(sn=int(wildcards.sn), pw=(wildcards.pw == "True"), subj_df_dir=input.subj_df_dir)
        subj_syn_df_2d = subj_data.synthesize_BOLD_2d(params, full_ver=(wildcards.full_ver=="True"))
        subj_syn_df_2d.to_csv(output[0])

rule generate_noisy_synthetic_data_subj:
    input:
        subj_syn_df_2d = os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D",  "original_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-0_subj-{sn}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","synthetic_data_2D","syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}.csv")
    run:
        np.random.seed(1)
        subj_syn_df = pd.read_csv(input.subj_syn_df_2d)
        noisy_df_2d = sim.copy_df_and_add_noise(subj_syn_df, beta_col="normed_betas", noise_mean=0, noise_sd=subj_syn_df['noise_SD']*float(wildcards.n_sd_mtpl))
        noisy_df_2d.to_csv(output[0])


rule run_simulation_subj:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'],  "simulation", "synthetic_data_2D", "syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}.csv"),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'model_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}_lr-{lr}_eph-{max_epoch}.csv'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}_lr-{lr}_eph-{max_epoch}.csv'),
        losses_history = os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'losses_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}_lr-{lr}_eph-{max_epoch}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "simulation","results_2D",'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}_lr-{lr}_eph-{max_epoch}.log')
    run:
        # add noise
        syn_df = pd.read_csv(input.input_path)
        syn_dataset = model.SpatialFrequencyDataset(syn_df, beta_col='normed_betas')
        syn_model = model.SpatialFrequencyModel(syn_dataset.my_tensor,full_ver=(wildcards.full_ver=="True"))
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model, syn_dataset,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch), print_every=2000, anomaly_detection=False, amsgrad=False, eps=1e-8)
        losses_history = model.shape_losses_history(losses, syn_df)
        utils.save_df_to_csv(losses_history, output.losses_history, indexing=False)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)

rule run_Broderick_subj:
    input:
        input_path = os.path.join(config['BD_DIR'],  "dataframes", "{subj}_stim_voxel_info_df_vs_md.csv")
    output:
        log_file = os.path.join(config['BD_DIR'],"logs","sfp_model","results_2D",'log_dset-Broderick_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.txt'),
        model_history = os.path.join(config['BD_DIR'],"sfp_model","results_2D",'model_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.h5'),
        loss_history = os.path.join(config['BD_DIR'],"sfp_model","results_2D",'loss_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.h5'),
#        losses_history = os.path.join(config['BD_DIR'],"sfp_model","results_2D",'losses_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}.h5')
    log:
        os.path.join(config['BD_DIR'],"logs","sfp_model","results_2D",'loss_history_dset-Broderick_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.log')
    benchmark:
        os.path.join(config['BD_DIR'],"benchmark","sfp_model","results_2D",'loss_history_dset-Broderick_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_benchmark_{roi}.txt')
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        subj_df = pd.read_csv(input.input_path)
        subj_df['angle'] = np.deg2rad(subj_df['angle'])
        subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
        subj_model = model.SpatialFrequencyModel(full_ver=(wildcards.full_ver=="True"))
        loss_history, model_history, elapsed_time, losses = model.fit_model(subj_model, subj_dataset, output.log_file,
            learning_rate=float(wildcards.lr), max_epoch=int(wildcards.max_epoch),
            print_every=100, loss_all_voxels=False,
            anomaly_detection=False, amsgrad=False, eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage',mode='w')
        loss_history.to_hdf(output.loss_history, key='stage',mode='w')
        #losses_history = model.shape_losses_history(losses,subj_df)
        #losses_history.to_hdf(output.losses_history,key='stage',mode='w')
        #
        # utils.save_df_to_csv(losses_history, output.losses_history, indexing=False)
        # utils.save_df_to_csv(model_history, output.model_history, indexing=False)
        # utils.save_df_to_csv(loss_history, output.loss_history, indexing=False)

rule plot_avg_subj_parameter_history:
    input:
        subj_files = expand(os.path.join(config['BD_DIR'],"sfp_model","results_2D",'model_history_dset-Broderick_bts-md_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=SUBJ),
        df_dir = os.path.join(config['BD_DIR'],"sfp_model","results_2D")
    output:
        history_fig = os.path.join(config['BD_DIR'],"figures", "sfp_model", "results_2D",'model_history_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png'),
    benchmark:
        os.path.join(config['BD_DIR'],"benchmark","sfp_model", "results_2D",'model_history_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_benchmark_{roi}txt'),
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
        model_history = model.load_history_df_Broderick_subj(input.df_dir, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "model",  [wildcards.roi])
        orig_subj = [utils.sub_number_to_string(x,"broderick") for x in sn_list]
        new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
        subj_replace_dict = dict(zip(orig_subj,new_subj))
        model_history = model_history.replace({'subj': subj_replace_dict})
        model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')
        params_col, params_group = sim.get_params_name_and_group(params,full_ver=(wildcards.full_ver=="True"))
        model.plot_param_history_horizontal(model_history, params=params_col, group=np.arange(0,9), to_label='subj',
            label_order=new_subj, ground_truth=True, lgd_title="Subjects", save_fig=True, save_path=output.history_fig,
            height=8, col_wrap=3, ci="sd", n_boot=100, log_y=False)


rule plot_avg_subj_loss_history:
    input:
        subj_files = expand(os.path.join(config['BD_DIR'],"sfp_model","results_2D", 'loss_history_dset-Broderick_bts-md_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=SUBJ),
        df_dir = os.path.join(config['BD_DIR'],"sfp_model","results_2D")
    output:
        history_fig = os.path.join(config['BD_DIR'],"figures", "sfp_model", "results_2D",'loss_history_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png'),
    benchmark:
        os.path.join(config['BD_DIR'],"benchmark","sfp_model", "results_2D",'loss_history_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_benchmark_{roi}.txt'),
    run:
        params = pd.read_csv(os.path.join(config['DF_DIR'], config['PARAMS']))
        sn_list=[1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
        loss_history = model.load_history_df_Broderick_subj(input.df_dir, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "loss", [wildcards.roi])
        model.plot_loss_history(loss_history,to_x="epoch",to_y="loss", to_label=None,to_col='lr_rate', height=5,
            lgd_title=None,to_row=None, save_fig=True, save_path=output.history_fig, ci=68, n_boot=100, log_y=True, sharey=True)


rule plot_scatterplot:
    input:
        bd_file = os.path.join(config['BD_DIR'], 'dataframes','Broderick_individual_subject_params_median_across_bootstraps.csv'),
        my_files = expand(os.path.join(config['BD_DIR'],"sfp_model","results_2D",'model_history_dset-Broderick_bts-md_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=SUBJ),
        df_dir= os.path.join(config['BD_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['BD_DIR'],"figures", "sfp_model", "results_2D",'scatterplot_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png')
    log:
        os.path.join(config['BD_DIR'],"logs","sfp_model","results_2D",'scatterplot_dset-Broderick_bts-md_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.log')
    run:
        bd_df = pd.read_csv(input.bd_file)
        sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
        model_history = model.load_history_df_Broderick_subj(input.df_dir, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)],"model", [wildcards.roi])
        max_epoch = model_history.epoch.max()
        fnl_df = model_history.query('epoch == @max_epoch')
        orig_subj = [utils.sub_number_to_string(x,"broderick") for x in sn_list]
        new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
        subj_replace_dict = dict(zip(orig_subj,new_subj))
        fnl_df = fnl_df.replace({'subj': subj_replace_dict})
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        fnl_df = pd.melt(fnl_df,id_vars=['subj'],value_vars=params_col,var_name='params',value_name='My_value')
        df = fnl_df.merge(bd_df, on=['subj','params'])
        f_name = 'scatter_comparison.png'
        grid = model.scatter_comparison(df.query('params in @params_col'),
            x="Broderick_value", y="My_value", col="params",
            col_order=params_col,label_order=new_subj,
            to_label='subj',lgd_title="Subjects",height=7,
            save_fig=True, save_path=output.scatter_fig)