import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('svg')
import pickle
pickle.HIGHEST_PROTOCOL = 4
from sfp_nsdsyn import *
from sfp_nsdsyn.preprocessing import convert_between_roi_num_and_vareas

configfile:
    "config.json"

STIM_LIST=['annulus', 'forward-spiral', 'pinwheel', 'reverse-spiral', 'avg']
ARGS_1D = ['sub', 'class', 'dset', 'lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
ARGS_2D = ['lr','eph','sub','roi','dset']
LR = [0.005]
MAX_EPOCH = [8000]
LR_2D = [0.0005]
MAX_EPOCH_2D = [30000]
measured_noise_sd =0.03995  # unnormalized 1.502063
LR_RATE = [0.0005] #[0.0007]#np.linspace(5,9,5)*1e-4
MULTIPLES_OF_NOISE_SD = [1]
NOISE_SD = [np.round(measured_noise_sd*x, 2) for x in [1]]
N_VOXEL = [100]
FULL_VER = ["True"]
PW = ["True"]
SN_LIST = ["{:02d}".format(sn) for sn in np.arange(1,9)]
broderick_sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
SUBJ_OLD = [utils.sub_number_to_string(sn, dataset="broderick") for sn in broderick_sn_list]
ROIS = ["V1","V2","V3"]
PARAMS_2D = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
PARAMS_GROUP_2D = [0,1,1,2,2,3,3,4,4]

# small tests to make sure snakemake is playing nicely with the job management
# system.
rule test_run:
     log: os.path.expanduser('~/test_run-%j.log')
     run:
         import numpy
         print("success!", numpy)

rule test_shell:
     log: os.path.expanduser('~/test_shell-%j.log')
     shell:
         "python -c 'import numpy; print(numpy)'; echo success!"

def get_sn_list(dset):
    if dset == "broderick":
        sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
    elif dset == "nsdsyn":
        sn_list = np.arange(1,9)
    return sn_list

def make_subj_list(dset):
    if dset == "broderick":
        return [utils.sub_number_to_string(sn, dataset="broderick") for sn in get_sn_list(dset)]
    elif dset == "nsdsyn":
        return [utils.sub_number_to_string(sn, dataset="nsdsyn") for sn in get_sn_list(dset)]

def interpret_bin_nums(wildcards):
    bin_list, bin_labels = get_ecc_bin_list(wildcards)
    if wildcards.ebin == "all":
        new_bin_labels = bin_labels
    else:
        new_bin_labels = [bin_labels[int(k)] for k in wildcards.ebin]
    return new_bin_labels

rule make_df_for_all_subj:
    input:
        #expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "nsdsyn", 'model-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_binned-ecc-{e1}-{e2}_nbin-{enum}_dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        #stim_class=STIM_LIST, lr=LR, max_epoch=MAX_EPOCH, e1=[0.5], e2=[4], enum=[7], subj=make_subj_list('nsdsyn'), roi=['V1'], vs=['pRFcenter','pRFsize'])
        expand(os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv'), dset='nsdsyn', subj=make_subj_list('nsdsyn'), roi=['V1','V2','V3'], vs=['pRFsize'])

def get_stim_size_in_degree(dset):
    if dset == 'nsdsyn':
        fixation_radius = vs.pix_to_deg(42.878)
        stim_radius = vs.pix_to_deg(714/2)
    else:
        fixation_radius = 1
        stim_radius = 12
    return fixation_radius, stim_radius

def _get_boolean_for_vs(vs):
    switcher = {'pRFcenter': False,
                'pRFsize': True}
    return switcher.get(vs, True)

rule prep_data:
    input:
        design_mat = os.path.join(config['NSD_DIR'], 'nsddata', 'experiments', 'nsdsynthetic', 'nsdsynthetic_expdesign.mat'),
        stim_info = os.path.join(config['NSD_DIR'], 'nsdsynthetic_sf_stim_description.csv'),
        lh_prfs = expand(os.path.join(config['NSD_DIR'], 'nsddata', 'freesurfer','{{subj}}', 'label', 'lh.prf{prf_param}.mgz'), prf_param= ["eccentricity", "angle", "size"]),
        lh_rois = expand(os.path.join(config['NSD_DIR'], 'nsddata', 'freesurfer','{{subj}}', 'label', 'lh.prf-{roi_file}.mgz'), roi_file= ["visualrois", "eccrois"]),
        lh_betas = os.path.join(config['NSD_DIR'], 'nsddata_betas', 'ppdata', '{subj}', 'nativesurface', 'nsdsyntheticbetas_fithrf_GLMdenoise_RR', 'lh.betas_nsdsynthetic.hdf5'),
        rh_prfs= expand(os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{{subj}}','label','rh.prf{prf_param}.mgz'), prf_param=["eccentricity", "angle", "size"]),
        rh_rois= expand(os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{{subj}}','label','rh.prf-{roi_file}.mgz'), roi_file=["visualrois", "eccrois"]),
        rh_betas= os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{subj}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','rh.betas_nsdsynthetic.hdf5')
    output:
        os.path.join(config['OUTPUT_DIR'], 'dataframes', 'nsdsyn','dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    params:
        rois_vals = lambda wildcards: [convert_between_roi_num_and_vareas(wildcards.roi), [1,2,3,4,5]],
        task_keys = ['fixation_task', 'memory_task'],
        stim_size= get_stim_size_in_degree('nsdsyn')
    run:
        from sfp_nsdsyn import prep
        from sfp_nsdsyn import vs
        lh_df = prep.make_sf_dataframe(stim_info=input.stim_info,
                                       design_mat=input.design_mat,
                                       rois=input.lh_rois, rois_vals=params.rois_vals,
                                       prfs=input.lh_prfs,
                                       betas=input.lh_betas,
                                       task_keys=params.task_keys, task_average=True,
                                       angle_to_radians=True)
        rh_df = prep.make_sf_dataframe(stim_info=input.stim_info,
                                       design_mat=input.design_mat,
                                       rois=input.rh_rois, rois_vals=params.rois_vals,
                                       prfs=input.rh_prfs,
                                       betas=input.rh_betas,
                                       task_keys=params.task_keys, task_average=True,
                                       angle_to_radians=True)
        sf_df = prep.concat_lh_rh_df(lh_df, rh_df)
        if wildcards.vs != 'None':
            sf_df = vs.select_voxels(sf_df, drop_by=wildcards.vs,
                                     inner_border=params.stim_size[0],
                                     outer_border=params.stim_size[1],
                                     to_group=['voxel'], return_voxel_list=False)
        sf_df['sub'] = wildcards.subj
        sf_df.to_csv(output[0],index=False)

def get_ecc_bin_list(wildcards):
    bin_list, bin_labels = tuning.get_bin_labels(wildcards.e1, wildcards.e2, wildcards.enum)
    return bin_list, bin_labels

rule binning:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}','dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'binned', 'binned_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'dataframes', '{dset}', 'binned', 'binned_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    params:
        bin_info = lambda wildcards: get_ecc_bin_list(wildcards)
    run:
        from sfp_nsdsyn import tuning
        df = pd.read_csv(input.subj_df)
        df = df.query('~names.str.contains("intermediate").values')
        df['ecc_bin'] = tuning.bin_ecc(df['eccentricity'], bin_list=params.bin_info[0], bin_labels=params.bin_info[1])
        c_df = tuning.summary_stat_for_ecc_bin(df,
                                               to_group= ['sub', 'ecc_bin', 'freq_lvl', 'names', 'vroinames'],
                                               to_bin=['betas', 'local_sf'],
                                               central_tendency='mean')
        c_df.to_csv(output[0], index=False)

def _get_bin_number(enum):
    only_num = int(enum.replace('log', ''))
    return [k for k in np.arange(1,only_num+1)]

def get_trained_model_for_all_bins(wildcards):
    only_num = int(wildcards.enum.replace('log', ''))
    BINS = [f'model-history_class-{wildcards.stim_class}_lr-{wildcards.lr}_eph-{wildcards.max_epoch}_e1-{wildcards.e1}_e2-{wildcards.e2}_nbin-{wildcards.enum}_dset-{wildcards.dset}_sub-{wildcards.subj}_roi-{wildcards.roi}_vs-{wildcards.vs}.pt' for bin in np.arange(1,only_num+1)]
    return [os.path.join(config['OUTPUT_DIR'], "sfp_model","results_1D", f'{wildcards.dset}', bin_path) for bin_path in BINS]


rule fit_tuning_curves:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'], 'dataframes', "{dset}", 'binned', 'binned_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'model-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'loss-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'model-params_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model", "results_1D", "{dset}", 'loss-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_set-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    resources:
        cpus_per_task = 1,
        mem_mb = 2000
    params:
        bin_info = lambda wildcards: get_ecc_bin_list(wildcards)
    run:
        subj_df = pd.read_csv(input.input_path)
        if wildcards.stim_class == "avg":
            subj_df = subj_df.groupby(['sub','ecc_bin','vroinames','freq_lvl'], group_keys=False).mean().reset_index()
        else:
            save_stim_type_name = wildcards.stim_class.replace('-',' ')
            subj_df = subj_df.query('names == @save_stim_type_name')

        save_ecc_bin_name = params.bin_info[1][int(wildcards.curbin)]
        subj_df = subj_df.query('ecc_bin == @save_ecc_bin_name')
        model_path_list = get_trained_model_for_all_bins(wildcards)
        my_model = tuning.LogGaussianTuningModel()
        my_dataset = tuning.LogGaussianTuningDataset(subj_df)
        loss_history, model_history = tuning.fit_tuning_curves(my_model, my_dataset,
                                                               learning_rate=float(wildcards.lr),
                                                               max_epoch=int(wildcards.max_epoch),
                                                               print_every=2000, save_path=output.model)
        model_history['ecc_bin'] = save_ecc_bin_name
        loss_history['ecc_bin'] = save_ecc_bin_name
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')

def _get_curbin(enum):
    return np.arange(0, int(enum.replace('log', '')))

rule plot_tuning_curves:
    input:
        model_df = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{{dset}}", 'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_dset-{{dset}}_sub-{{subj}}_roi-{roi}_vs-{{vs}}.pt'), curbin=_get_curbin(wildcards.enum), roi=ROIS),
        binned_df = expand(os.path.join(config['OUTPUT_DIR'], 'dataframes', "{{dset}}", 'binned', 'binned_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_dset-{{dset}}_sub-{{subj}}_roi-{roi}_vs-{{vs}}.csv'), roi=ROIS)
    output:
        os.path.join(config['OUTPUT_DIR'],"figures", "sfp_model","results_1D", "{dset}", 'fig-tuning_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-all_dset-{dset}_sub-{subj}_roi-all_vs-{vs}.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "figures", "sfp_model","results_1D", "{dset}", 'fig-tuning_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-all_dset-{dset}_sub-{subj}_roi-all_vs-{vs}.{fig_format}.log')
    run:
        final_params = tuning.load_all_models(input.model_df, *ARGS_1D)
        bin_df = tuning.load_history_files(input.binned_df, *['sub','dset','roi'])
        if wildcards.stim_class == "avg":
            bin_df = bin_df.groupby(['sub', 'ecc_bin', 'vroinames', 'freq_lvl'],group_keys=False).mean().reset_index()
        else:
            save_stim_type_name = wildcards.stim_class.replace('-',' ')
            bin_df = bin_df.query('names == @save_stim_type_name')
        vis1D.plot_sf_curves(df=bin_df,
                             params_df=final_params,
                             x='local_sf', y='betas', hue='ecc_bin',
                             col='vroinames', lgd_title='Eccentricity band',
                             save_path=output[0])

rule plot_curves_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"figures", "sfp_model","results_1D", "{dset}", 'fig-tuning_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-all_dset-{dset}_sub-{subj}_roi-all_vs-{vs}.{fig_format}'),
            stim_class=STIM_LIST, lr=LR, max_epoch=MAX_EPOCH, e1=0.5, e2=4, enum=['log3'], subj=make_subj_list('nsdsyn'), dset='nsdsyn', vs=['pRFsize','pRFcenter'], fig_format='svg')

rule make_precision_v_df:
    input:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}', 'precision', 'precision-v_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    params:
        p_dict = {1: 'noise_SD', 2: 'sigma_v_squared'}
    run:
        subj_vs_df = pd.read_csv(input[0])
        power_val = [1,2]
        power_var = [col for p, col in params.p_dict.items() if p in power_val]
        sigma_v_df = bts.get_multiple_sigma_vs(subj_vs_df,
                                               power=power_val,
                                               columns=power_var,
                                               to_sd='betas', to_group=['sub','voxel', 'vroinames'])
        sigma_v_df.to_csv(output[0], index=False)

rule make_precision_s_df:
    input:
        precision_v = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],'dataframes','{{dset}}','precision','precision-v_dset-{{dset}}_sub-{subj}_roi-{roi}_vs-{{vs}}.csv'), subj=make_subj_list(wildcards.dset), roi=ROIS)
    output:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-{vs}.csv')
    run:
        sigma_v_df = pd.DataFrame({})
        for subj_sigma_v_df in input.precision_v:
            tmp = pd.read_csv(subj_sigma_v_df)
            sigma_v_df = sigma_v_df.append(tmp)
        precision_s = sigma_v_df.groupby('sub')['sigma_v_squared'].mean().reset_index()
        precision_s['precision'] = 1 / precision_s['sigma_v_squared']
        precision_s.to_csv(output[0], index=False)


rule plot_preferred_period_1D:
    input:
        models = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{{dset}}", 'model-params_class-{stim_class}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.pt'), stim_class=[k.replace(' ', '-') for k in STIM_LIST], curbin=_get_curbin(wildcards.enum), subj=make_subj_list(wildcards.dset)),
        precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures', "sfp_model","results_1D", "{dset}",'fig-pperiod_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-avg_roi-{roi}_vs-{vs}.{fig_format}')
    run:
        args = ['sub', 'class','dset', 'lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
        final_params= tuning.load_all_models(pt_file_path_list=input.models, ecc_bin=True, args=args)
        precision_s = pd.read_csv(input.precision_s)
        params_precision_df = final_params.merge(precision_s[['sub', 'precision']], on='sub')
        vis1D.plot_preferred_period(params_precision_df, precision_col='precision',
                                    hue='names', hue_order=STIM_LIST, lgd_title='Stimulus class',
                                    height=8, save_path=output[0])

rule run_model:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}','dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
        precision = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}', 'precision', 'precision-v_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}", 'model-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}",'loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{dset}", 'model-params_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model","results_2D", "{dset}",'loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.log'),
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "sfp_model","results_2D", "{dset}",'loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.txt'),
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        subj_df = pd.read_csv(input.subj_df)
        precision_df = pd.read_csv(input.precision)
        df = subj_df.merge(precision_df, on=['sub', 'vroinames', 'voxel'])
        df = df.groupby(['sub','voxel','class_idx','vroinames']).mean().reset_index()
        subj_model = model.SpatialFrequencyModel(full_ver=True)
        subj_dataset = model.SpatialFrequencyDataset(df, beta_col='betas')
        loss_history, model_history, _ = model.fit_model(subj_model, subj_dataset,
                                                         learning_rate=float(wildcards.lr),
                                                         max_epoch=int(wildcards.max_epoch),
                                                         save_path=output.model,
                                                         print_every=10000,
                                                         loss_all_voxels=False,
                                                         anomaly_detection=False,
                                                         amsgrad=False,
                                                         eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')

rule calculate_Pv_based_on_model:
    input:
        stim = os.path.join(config['NSD_DIR'], 'nsdsyn_stim_description.csv'),
        model = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D","nsdsyn",'model-params_lr-{lr}_eph-{max_epoch}_dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    output:
        os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","nsdsyn",'prediction_frame-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}.h5')
    run:
        stim_info = vis2D.get_w_a_and_w_r_for_each_stim_class(input.stim)
        final_params = model.model_to_df(input.model, *ARGS_2D)
        syn_df = vis2D.calculate_preferred_period_for_synthetic_df(stim_info, final_params,
                                                                   ecc_range=(float(wildcards.ecc1), float(wildcards.ecc2)),
                                                                   n_ecc=int(wildcards.n_ecc),
                                                                   angle_range=(np.deg2rad(float(wildcards.ang1)), np.deg2rad(float(wildcards.ang2))),
                                                                   n_angle=int(wildcards.n_ang),
                                                                   ecc_col='eccentricity',
                                                                   angle_col='angle',
                                                                   angle_in_radians=True,
                                                                   reference_frame=wildcards.frame)
        syn_df.to_hdf(output[0], key='stage', mode='w')
#TODO: combine these two rules (Pv calculation rules) later

rule calculate_broderick_Pv_based_on_model:
    input:
        stim = os.path.join(config['NSD_DIR'], 'nsdsyn_stim_description.csv'),
        model = os.path.join(config['OUTPUT_OLD_DIR'], "sfp_model","results_2D",'model_history_dset-broderick_bts-median_full_ver-True_sub-{subj}_lr-0.0005_eph-30000_V1.h5'),
    output:
        os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","broderick",'prediction_frame-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-0.0005_eph-30000_dset-broderick_sub-{subj}_roi-V1_vs-pRFsize.h5')
    run:
        stim_info = vis2D.get_w_a_and_w_r_for_each_stim_class(input.stim)
        broderick_model_df = utils.load_history_files([input.model], *['dset','sub'])
        broderick_model_df['vroinames'] = 'V1'
        final_params = broderick_model_df.query('epoch == 29999')  #TODO: make it with new bd dataframes
        syn_df = vis2D.calculate_preferred_period_for_synthetic_df(stim_info, final_params,
                                                                   ecc_range=(float(wildcards.ecc1), float(wildcards.ecc2)),
                                                                   n_ecc=int(wildcards.n_ecc),
                                                                   angle_range=(np.deg2rad(float(wildcards.ang1)), np.deg2rad(float(wildcards.ang2))),
                                                                   n_angle=int(wildcards.n_ang),
                                                                   ecc_col='eccentricity',
                                                                   angle_col='angle',
                                                                   angle_in_radians=True,
                                                                   reference_frame=wildcards.frame)

        syn_df.to_hdf(output[0], key='stage', mode='w')

rule calculate_all:
    input:
        # expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","nsdsyn",'prediction_frame-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        #        subj=make_subj_list('nsdsyn'), frame=['relative', 'absolute'], xaxis='eccentricity', ecc1='0', ecc2='4', n_ecc='4', ang1='0', ang2='360', n_ang='360', dset='nsdsyn', vs='pRFsize', lr=LR_2D, max_epoch=MAX_EPOCH_2D, roi=ROIS),
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","broderick",'prediction_frame-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-0.0005_eph-30000_dset-broderick_sub-{subj}_roi-V1_vs-pRFsize.h5'),
               subj=make_subj_list('broderick'), frame=['relative', 'absolute'], ecc1='0', ecc2='10', n_ecc='3', ang1='0', ang2='360', n_ang='360', roi='V1', vs='pRFsize')

def Pv_projection(xaxis):
    if xaxis == "angle":
        return 'polar'
    else:
        return None

rule plot_preferred_period_2D:
    input:
        model_prediction = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","{{dset}}",'prediction_frame-{frame}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list(wildcards.dset)),
        precision = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],'dataframes','{{dset}}', 'precision', 'precision-v_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.csv'), subj=make_subj_list(wildcards.dset))
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-pperiod-prediction_frame-{frame}_xaxis-{xaxis}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-avg_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        projection = lambda wildcards: Pv_projection(wildcards.xaxis)
    run:
        precision_v = utils.load_history_files(input.precision)
        precision_s = precision_v.groupby(['sub','vroinames']).mean().reset_index()
        precision_s['precision'] = 1/precision_s['sigma_v_squared']
        df = utils.load_history_files(input.model_prediction, *ARGS_2D)
        df = df.merge(precision_s, on=['sub','vroinames'])
        if wildcards.xaxis == "angle":
            df = df.query('eccentricity == 5')
        df = df.groupby(['sub', 'names', wildcards.xaxis]).mean().reset_index()
        vis2D.plot_preferred_period(df, x=wildcards.xaxis, y='Pv', precision='precision',
                                    hue=None, hue_order=None,
                                    col=None, col_wrap=None,
                                    lgd_title=None, height=6,
                                    projection=params.projection, save_path=output[0])

rule plot_preferred_period_comparison:
    input:
        broderick_model_prediction=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","broderick",'prediction_frame-{{frame}}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-0.0005_eph-30000_dset-broderick_sub-{subj}_roi-V1_vs-pRFsize.h5'), subj=make_subj_list('broderick')),
        broderick_precision_v=expand(os.path.join(config['OUTPUT_DIR'],'dataframes','broderick','precision','precision-v_dset-broderick_sub-{subj}_roi-V1_vs-{{vs}}.csv'), subj=make_subj_list('broderick')),
        nsd_model_params=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","nsdsyn",'prediction_frame-{{frame}}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-{{lr}}_eph-{{max_epoch}}_dset-nsdsyn_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list('nsdsyn')),
        nsd_precision_v=expand(os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','precision-v_dset-nsdsyn_sub-{subj}_roi-V1_vs-{{vs}}.csv'), subj=make_subj_list('nsdsyn'))
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D","dset_comparison",'fig-pperiod-prediction_frame-{frame}_col-{col}_xaxis-{xaxis}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-all_sub-avg_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        col = lambda wildcards: None if wildcards.col == 'None' else wildcards.col
    run:
        broderick_model_df = utils.load_history_files(input.broderick_model_prediction, *PARAMS_2D)
        broderick_precision_v = utils.load_history_files(input.broderick_precision_v)
        broderick_precision_s = broderick_precision_v.groupby(['sub','vroinames'], group_keys=False).mean().reset_index()
        broderick_df = broderick_model_df.merge(broderick_precision_s[['sub','vroinames','sigma_v_squared']],
                                                on=['sub', 'vroinames'])

        nsd_model_df = utils.load_history_files(input.nsd_model_params, *ARGS_2D)
        nsd_precision_v = utils.load_history_files(input.nsd_precision_v)
        nsd_precision_s = nsd_precision_v.groupby(['sub','vroinames'], group_keys=False).mean().reset_index()
        nsd_df = nsd_model_df.merge(nsd_precision_s, on=['sub', 'vroinames'])

        all_df = broderick_df.append(nsd_df)
        all_df['precision'] = 1 / all_df['sigma_v_squared']
        vis2D.plot_preferred_period(all_df, x=wildcards.xaxis,
                                    hue='dset', hue_order=['broderick', 'nsdsyn'],
                                    col=params.col,
                                    lgd_title='Dataset', xlim=(0,10), save_path=output[0])

rule plot_pp:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D","dset_comparison",'fig-pperiod-prediction_frame-{frame}_col-{col}_xaxis-{xaxis}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-all_sub-avg_roi-{roi}_vs-{vs}.{fig_format}'),
        frame=['relative', 'absolute'], col=['None', 'names'], xaxis='eccentricity', ecc1='0', ecc2='10', n_ecc='3', ang1='0', ang2='360', n_ang='360', dset='nsdsyn', vs='pRFsize', lr=LR_2D, max_epoch=MAX_EPOCH_2D, roi='V1', fig_format='svg')

rule plot_avg_model_parameters:
    input:
        model_params = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{{dset}}", 'model-params_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{roi}_vs-{{vs}}.pt'), subj=make_subj_list(wildcards.dset), roi=ROIS),
        precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-params_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-avg_roi-V1V2V3_vs-{vs}.{fig_format}')
    params:
        params_list = PARAMS_2D,
        groups = PARAMS_GROUP_2D
    run:
        df = model.load_all_models(input.model_params, *ARGS_2D)
        precision_s = pd.read_csv(input.precision_s)
        final_params = df.merge(precision_s, on=['sub'])
        vis2D.plot_precision_weighted_avg_parameters(final_params,
                                                     params.params_list, params.groups,
                                                     weight='precision',
                                                     hue = 'vroinames', hue_order=ROIS,
                                                     lgd_title='Visual areas',
                                                     save_path=output[0])

rule plot_individual_model_parameters:
    input:
        model_params = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{{dset}}", 'model-params_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.pt'), subj=make_subj_list(wildcards.dset)),
        precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-params_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        params_list = PARAMS_2D,
        groups = PARAMS_GROUP_2D,
        subj_list = lambda wildcards: make_subj_list(wildcards.dset)
    run:
        df = model.load_all_models(input.model_params, *ARGS_2D)
        precision_s = pd.read_csv(input.precision_s)
        final_params = df.merge(precision_s, on=['sub'])
        vis2D.plot_precision_weighted_avg_parameters(final_params,
                                                     params.params_list, params.groups,
                                                     weight='precision',
                                                     hue = 'sub', hue_order= params.subj_list,
                                                     lgd_title='Subjects',
                                                     height=7, pal=utils.subject_color_palettes(wildcards.dset, params.subj_list),
                                                     save_path=output[0], suptitle=wildcards.roi)


rule plot_model_parameter_comparison:
    input:
        broderick_model_params = expand(os.path.join(config['OUTPUT_OLD_DIR'], "sfp_model","results_2D", 'model_history_dset-broderick_bts-median_full_ver-True_sub-{subj}_lr-0.0005_eph-30000_V1.h5'), subj=make_subj_list('broderick')),
        broderick_precision_v = expand(os.path.join(config['OUTPUT_DIR'],'dataframes','broderick', 'precision', 'precision-v_dset-broderick_sub-{subj}_roi-V1_vs-{{vs}}.csv'), subj=make_subj_list('broderick')),
        nsd_model_params = expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "nsdsyn", 'model-params_lr-{{lr}}_eph-{{max_epoch}}_dset-nsdsyn_sub-{subj}_roi-V1_vs-{{vs}}.pt'), subj=make_subj_list('nsdsyn')),
        nsd_precision_v = expand(os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn', 'precision', 'precision-v_dset-nsdsyn_sub-{subj}_roi-V1_vs-{{vs}}.csv'), subj=make_subj_list('nsdsyn'))
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "dset_comparison", 'fig-params_lr-{lr}_eph-{max_epoch}_dset-all_sub-all_roi-V1_vs-{vs}.{fig_format}')
    params:
        param_list = PARAMS_2D,
        param_group = PARAMS_GROUP_2D
    run:
        broderick_model_df = utils.load_history_files(input.broderick_model_params, *['sub','dset'])
        broderick_model_df['vroinames'] = 'V1'
        broderick_model_df = broderick_model_df.query('epoch == 29999') #TODO: make it with new bd dataframes
        broderick_precision_v = utils.load_history_files(input.broderick_precision_v)
        broderick_precision_s = broderick_precision_v.groupby(['sub','vroinames'], group_keys=False).mean().reset_index()
        broderick_df = broderick_model_df.merge(broderick_precision_s[['sub', 'vroinames', 'sigma_v_squared']], on=['sub', 'vroinames'])

        nsd_model_df = model.load_all_models(input.nsd_model_params, *ARGS_2D)
        nsd_precision_v = utils.load_history_files(input.nsd_precision_v)
        nsd_precision_s = nsd_precision_v.groupby(['sub','vroinames'], group_keys=False).mean().reset_index()
        nsd_df = nsd_model_df.merge(nsd_precision_s, on=['sub','vroinames'])

        all_df = broderick_df.append(nsd_df)
        all_df['precision'] = 1/all_df['sigma_v_squared']
        vis2D.plot_precision_weighted_avg_parameters(all_df,
                                                     params.param_list,
                                                     params.param_group,
                                                     hue='dset',
                                                     hue_order=['broderick','nsdsyn'],
                                                     lgd_title=['Dataset'],
                                                     height=7,
                                                     pal=utils.get_colors('dset', to_plot=['broderick','nsdsyn']),
                                                     save_path=output[0])





rule plot_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "{dset}", 'fig-{d_type}-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}'), d_type=['model','loss'], roi=['V1','V2','V3'], dset='nsdsyn', lr=LR_2D, max_epoch=MAX_EPOCH_2D, vs=['pRFsize'], fig_format=['svg']),
        expand(os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-params_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}'), roi=['V1','V2','V3'], dset='nsdsyn', lr=LR_2D, max_epoch=MAX_EPOCH_2D, vs=['pRFsize'], fig_format=['svg'])

rule plot_2D_model_loss_history:
    input:
        loss_history = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{{dset}}",'loss-history_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list(wildcards.dset))
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "{dset}",'fig-loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "figures", "sfp_model", "results_2D", "{dset}",'fig-loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}.log')
    params:
        args = ['dset', 'lr', 'eph', 'sub', 'roi']
    run:
        loss_history = utils.load_history_files(input.loss_history, *params.args)
        subj_list = make_subj_list(wildcards.dset)
        kwargs = {'palette': utils.subject_color_palettes(wildcards.dset, subj_list)}
        vis.plot_loss_history(loss_history,
                              hue='sub',
                              lgd_title='Subjects',
                              hue_order=subj_list,
                              col=None,
                              log_y=True,
                              suptitle=wildcards.roi,
                              save_path=output[0],
                              **kwargs)

rule plot_2D_model_param_history:
    input:
        model_history = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{{dset}}",'model-history_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list(wildcards.dset))
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "{dset}",'fig-model-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "figures", "sfp_model", "results_2D", "{dset}",'fig-model-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        args=['dset','lr','eph','sub','roi'],
        params_list = PARAMS_2D,
    run:
        model_history = utils.load_history_files(input.model_history, *params.args)
        subj_list = make_subj_list(wildcards.dset)
        kwargs = {'palette': utils.subject_color_palettes(wildcards.dset, subj_list)}
        vis.plot_param_history(model_history,
                               params.params_list,
                               hue='sub',
                               lgd_title='Subjects',
                               hue_order=subj_list,
                               col=None,
                               height=4,
                               suptitle=wildcards.roi,
                               save_path=output[0],
                               **kwargs)


rule plot_all_loss_and_param_history:
    input:
        loss_fig = expand(os.path.join(config['OUTPUT_DIR'], "figures", "simulation", "results_2D", 'Epoch_vs_Loss', 'loss_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL),
        param_fig = expand(os.path.join(config['OUTPUT_DIR'], "figures", "simulation", "results_2D", 'Epoch_vs_Param_values', 'param_history_plot_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}_lr-{lr}_eph-{max_epoch}.png'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, lr=LR_RATE, max_epoch=MAX_EPOCH, n_voxels=N_VOXEL)


rule plot_tuning_curves_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"figures", "sfp_model","results_1D", 'sftuning_plot_ebin-{ebin}_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.eps'), ebin='all', e1='0.5', e2='4', enum='log3', dset='nsdsyn', stat='mean', lr=LR_RATE, max_epoch=MAX_EPOCH, roi=ROIS, subj=make_subj_list("nsdsyn")),
        expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_1D",'sftuning_plot_ebin-{ebin}_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.eps'), ebin=['159','all'], e1='1',e2='12',enum='11',dset='broderick',stat='median', lr=LR_RATE, max_epoch=MAX_EPOCH, roi=ROIS, subj=make_subj_list("broderick"))

rule fit_tuning_curves_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_1D",'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.h5'), df_type=['loss','model'], e1='1', e2='12', enum='11', dset='broderick', stat='median', lr=LR_RATE, max_epoch=MAX_EPOCH, roi=ROIS, subj=make_subj_list("broderick")),
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_1D",'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.h5'), df_type=['loss','model'], e1='0.5', e2='4', enum='log3', dset='nsdsyn', stat='mean', lr=LR_RATE, max_epoch=MAX_EPOCH, roi=ROIS, subj=make_subj_list("nsdsyn"))

rule run_all_subj:
    input:
        #expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", 'loss_history_dset-{dset}_bts-{stat}_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.h5'), dset="broderick", stat="median", full_ver="True", subj=_make_subj_list("broderick"), lr=LR_RATE, max_epoch=MAX_EPOCH, roi=ROIS),
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'loss_history_dset-{dset}_bts-{stat}_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.h5'), dset="nsdsyn", stat="mean", full_ver="True", subj=make_subj_list("nsdsyn"), lr=LR_RATE, max_epoch=MAX_EPOCH, roi=['V1', 'V2', 'V3'])

rule run_simulation_all_subj:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "simulation", "results_2D", 'loss_history_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_subj-{sn}_lr-{lr}_eph-{max_epoch}.csv'), full_ver=FULL_VER, pw=PW, n_sd_mtpl=MULTIPLES_OF_NOISE_SD, sn=SN_LIST,  lr=LR_RATE, max_epoch=MAX_EPOCH)


rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn", 'nsdsynthetic_sf_stim_description.csv'),
        subj_df_dir = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn")
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
        os.path.join(config['OUTPUT_DIR'], "figures", 'lineplot_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.png')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", 'lineplot_syn_data_2d_full_ver-{full_ver}_pw-{pw}_noise_mtpl-{n_sd_mtpl}_n_vox-{n_voxels}.png')
    run:
        all_df = pd.DataFrame({})
        for file in input.all_files:
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
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model,syn_dataset,learning_rate=float(wildcards.lr),max_epoch=int(wildcards.max_epoch),print_every=2000,anomaly_detection=False,amsgrad=False,eps=1e-8)
        losses_history = model.shape_losses_history(losses, syn_df)
        utils.save_df_to_csv(losses_history, output.losses_history, indexing=False)
        utils.save_df_to_csv(syn_model_history, output.model_history, indexing=False)
        utils.save_df_to_csv(syn_loss_history, output.loss_history, indexing=False)

rule generate_synthetic_data_subj:
    input:
        subj_df_dir = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn")
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
        syn_loss_history, syn_model_history, syn_elapsed_time, losses = model.fit_model(syn_model,syn_dataset,learning_rate=float(wildcards.lr),max_epoch=int(wildcards.max_epoch),print_every=2000,anomaly_detection=False,amsgrad=False,eps=1e-8)
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
        subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
        subj_model = model.SpatialFrequencyModel(full_ver=(wildcards.full_ver=="True"))
        loss_history, model_history, elapsed_time, losses = model.fit_model(subj_model,subj_dataset,output.log_file,max_epoch=int(wildcards.max_epoch),print_every=100,loss_all_voxels=False,anomaly_detection=False,amsgrad=False,eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage',mode='w')
        loss_history.to_hdf(output.loss_history, key='stage',mode='w')




def get_df_type_and_subj_list(dset):
    if dset == "broderick":
        stat = "median"
        subj_list = [utils.sub_number_to_string(sn, dataset="broderick") for sn in [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]]
    elif dset == "nsdsyn":
        stat = "mean"
        subj_list = [utils.sub_number_to_string(sn, dataset="nsd") for sn in np.arange(1,9)]
    #file_name = [f'{wildcards.df_type}_dset-{wildcards.dset}_bts-{stat}_full_ver-{wildcards.full_ver}_{subj}_lr-{wildcards.lr}_eph-{wildcards.max_epoch}_{wildcards.roi}.h5' for subj in subj_list]
    return stat, subj_list


rule plot_avg_subj_parameter_history:
    input:
        subj_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D")
    output:
        history_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'model_history_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png')
    run:
        params = pd.read_csv(os.path.join(config['OUTPUT_DIR'], "dataframes", config['PARAMS']))
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir, wildcards.dset, wildcards.stat, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "model", [wildcards.roi])
        subj_list = [utils.sub_number_to_string(sn, dataset=wildcards.dset) for sn in sn_list]
        if wildcards.dset == "broderick":
            new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
            subj_replace_dict = dict(zip(subj_list, new_subj))
            model_history = model_history.replace({'subj': subj_replace_dict})
            subj_list = new_subj
        model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')
        params_col, params_group = sim.get_params_name_and_group(params,full_ver=(wildcards.full_ver=="True"))
        sfp_nsdsyn.visualization.plot_2D_model_results.plot_param_history_horizontal(model_history, params=params_col, group=np.arange(0,9), to_label='subj',
            label_order=subj_list, ground_truth=True, lgd_title="Subjects", save_fig=True, save_path=output.history_fig,
            height=8, col_wrap=3, ci="sd", n_boot=100, log_y=False)


rule plot_avg_subj_loss_history:
    input:
        subj_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D",'loss_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D")
    output:
        history_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'loss_history_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png')
    run:
        sn_list = get_sn_list(wildcards.dset)
        loss_history = model.load_history_df_subj(input.df_dir, wildcards.dset, wildcards.stat, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "loss", [wildcards.roi])
        sfp_nsdsyn.visualization.plot_2D_model_results.plot_loss_history(loss_history,hue=None,lgd_title=None,col='lr_rate',height=5,save_path=output.history_fig,log_y=True,sharey=True)


rule plot_scatterplot_subj:
    input:
        bd_file = os.path.join(config['OUTPUT_DIR'], 'dataframes','Broderick_individual_subject_params_median_across_bootstraps.csv'),
        my_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=make_subj_list(wildcards)),
        df_dir= os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_subj_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.eps')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs","figures", "sfp_model","results_2D",'scatterplot_subj_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.log')
    run:
        bd_df = pd.read_csv(input.bd_file)
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir, wildcards.dset, wildcards.stat, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "model", [wildcards.roi])
        max_epoch = model_history.epoch.max()
        fnl_df = model_history.query('epoch == @max_epoch')
        subj_list = [utils.sub_number_to_string(sn, dataset=wildcards.dset) for sn in sn_list]
        if wildcards.dset == "broderick":
            new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
            subj_replace_dict = dict(zip(subj_list, new_subj))
            fnl_df = fnl_df.replace({'subj': subj_replace_dict})
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        fnl_df = pd.melt(fnl_df,id_vars=['subj'],value_vars=params_col,var_name='params',value_name='My_value')
        df = fnl_df.merge(bd_df, on=['subj','params'])
        f_name = 'scatter_comparison.png'
        grid = sfp_nsdsyn.visualization.plot_2D_model_results.scatter_comparison(df.query('params in @params_col'),
            x="Broderick_value", y="My_value", col="params",
            col_order=params_col,label_order=new_subj,
            to_label='subj',lgd_title="Subjects",height=6,
            save_fig=True, save_path=output.scatter_fig)

rule plot_scatterplot_avgparams:
    input:
        bd_file = os.path.join(config['OUTPUT_DIR'],'dataframes','Broderick_individual_subject_params_allbootstraps.csv'),
        my_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_avgparams_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.eps')
    run:
        bd_df = pd.read_csv(input.bd_file)
        bd_fnl_df = bd_df.groupby(['subj', 'params']).median().reset_index()
        m_bd_fnl_df = model.get_mean_and_error_for_each_param(bd_fnl_df, err="sem")
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir,wildcards.dset,wildcards.stat,[
            wildcards.full_ver],sn_list,[float(wildcards.lr)],[int(wildcards.max_epoch)],"model",[wildcards.roi])
        m_epoch = model_history.epoch.max()
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        params_group = [0,1,1,2,2,2,2,3,3]
        fnl_df = model_history.query('epoch == @m_epoch')[params_col]
        m_fnl_df = model.get_mean_and_error_for_each_param(fnl_df, err="sem")
        sfp_nsdsyn.visualization.plot_2D_model_results.scatterplot_two_avg_params(m_bd_fnl_df, m_fnl_df, params_col, params_group, x_label='Broderick et al.(2022) values', y_label=f'My values: {wildcards.dset}', save_fig=True, save_path=output.scatter_fig)



rule plot_scatterplot_subj_betweenVareas:
    input:
        roi_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_subj_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_V1-vs-{roi}.eps')
    run:
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir,wildcards.dset,wildcards.stat,[wildcards.full_ver],sn_list,[float(wildcards.lr)],[int(wildcards.max_epoch)],"model",["V1", wildcards.roi])
        m_epoch = model_history.epoch.max()
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        params_group = [0,1,1,2,2,2,2,3,3]
        V1_df = model_history.query('epoch == @m_epoch & vroinames == "V1"')[params_col]
        roi_df = model_history.query('epoch == @m_epoch & vroinames == @wildcards.roi')[params_col]
        subj_list = [utils.sub_number_to_string(sn,dataset=wildcards.dset) for sn in sn_list]
        if wildcards.dset == "broderick":
            new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
            subj_replace_dict = dict(zip(subj_list,new_subj))
            V1_df = V1_df.replace({'subj': subj_replace_dict})
            roi_df = roi_df.replace({'subj': subj_replace_dict})
        else:
            new_subj = subj_list
        long_V1 = utils.melt_params(V1_df, value_name='V1_value')
        long_roi = utils.melt_params(roi_df, value_name=f'{wildcards.roi}_value')
        df = pd.concat((long_V1, long_roi),axis=0)
        sfp_nsdsyn.visualization.plot_2D_model_results.scatter_comparison(df.query('params in @params_col'),
            x="V1_value",y=f"{wildcards.roi}_value",col="params",
            col_order=params_col,label_order=new_subj,
            to_label='subj',lgd_title="Subjects",height=7,
            save_fig=True, save_path=output.scatter_fig)


rule plot_scatterplot_avgparams_betweenVareas:
    input:
        roi_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_avgparams_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_V1-vs-{roi}.eps')
    run:
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir,wildcards.dset,wildcards.stat,[
            wildcards.full_ver],sn_list,[float(wildcards.lr)],[int(wildcards.max_epoch)],"model",["V1", wildcards.roi])
        m_epoch = model_history.epoch.max()
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        params_group = [0,1,1,2,2,2,2,3,3]
        V1_df = model_history.query('epoch == @m_epoch & vroinames == "V1"')[params_col]
        roi_df = model_history.query('epoch == @m_epoch & vroinames == @wildcards.roi')[params_col]

        fnl_V1_df = model.get_mean_and_error_for_each_param(V1_df, err="sem")
        fnl_roi_df = model.get_mean_and_error_for_each_param(roi_df, err="sem")
        sfp_nsdsyn.visualization.plot_2D_model_results.scatterplot_two_avg_params(fnl_V1_df, fnl_roi_df, params_col, params_group, x_label=f'{wildcards.dset}: V1', y_label=f'{wildcards.dset}: {wildcards.roi}', save_fig=True, save_path=output.scatter_fig)

def make_file_name_list(wildcards, stim_list):
    f_names = [f'{wildcards.df_type}_history_dset-{wildcards.dset}_bts-{wildcards.stat}_{wildcards.subj}_lr-{wildcards.lr}_eph-{wildcards.max_epoch}_{wildcards.roi}_{stim}_vs-pRFcenter_e{wildcards.e1}-{wildcards.e2}_nbin-{wildcards.enum}.h5' for stim in stim_list]
    return f_names

def make_info_columns(wildcards, df):
    df['dset'] = wildcards.dset
    df['subj'] = wildcards.subj
    df['vroinames'] = wildcards.roi
    df['lr_rate'] = float(wildcards.lr)
    df['max_epoch'] = int(wildcards.max_epoch)
    return df

rule combine_all_stim:
    input:
        file_names = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{f_name}"), f_name=make_file_name_list(wildcards, stim_list))
    output:
        allstim = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_1D",'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.h5')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "sfp_model","results_1D",'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.log')
    run:
        all_df = pd.DataFrame({})
        for stim, f in zip(stim_list, input.file_names):
            df = pd.read_hdf(f)
            df['names'] = stim
            df = make_info_columns(wildcards, df)
            all_df = pd.concat((all_df, df), axis=0)
        all_df.to_hdf(output.allstim, key='stage', mode='w')
        # for f in input.file_names:
        #     os.remove(f)

rule save_precision_s:
    input:
        subj_df = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"dataframes","{{dset}}","{subj}_stim_voxel_info_df_vs-pRFsigma_{{roi}}.csv"),subj=make_subj_list(wildcards))
    output:
        os.path.join(config['OUTPUT_DIR'],"dataframes","{dset}","precision_s_{dset}_{roi}.csv")
    run:
        print(input.subj_df)
        from sfp_nsdsyn import bts
        precision_df = pd.DataFrame({})
        for df in input.subj_df:
            tmp = pd.read_csv(df)
            tmp = bts.get_precision_s(tmp, subset=['subj','vroinames'])
            precision_df = precision_df.append(tmp)
            precision_df.to_csv(output[0], index=False)

rule plot_precision_weighted_2D_parameters:
    input:
        model_history=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-True_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{roi}.h5'),subj=make_subj_list(wildcards), roi=['V1','V2','V3']),
        #subj_df=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"dataframes","{{dset}}","{subj}_stim_voxel_info_df_vs-pRFsigma_{roi}.csv"),subj=make_subj_list(wildcards), roi=['V1','V2','V3']),
        precision_df = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"dataframes","{{dset}}","precision_s_{{dset}}_{roi}.csv"), roi=['V1','V2','V3'])
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'pointplot-precision-weighted-params_avg-True_dset-{dset}_bts-{stat}_lr-{lr}_eph-{max_epoch}_vs-pRFsigma_roi-V1V2V3.{fig_format}'),
    params:
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D"),
        sn_list = lambda wildcards: get_sn_list(wildcards.dset),
        params_order = PARAMS_2D,
        params_group = PARAMS_GROUP_2D
    run:
        from sfp_nsdsyn import model
        from sfp_nsdsyn import bts
        from sfp_nsdsyn import vis
        model_history = model.load_history_files(input.model_history)
        #model_history = model.load_history_df_subj(output_dir=params.df_dir, dataset=wildcards.dset, stat=wildcards.stat, full_ver=[True], sn_list=params.sn_list, lr_rate=[float(wildcards.lr)], max_epoch=[int(wildcards.max_epoch)], df_type="model", roi=[wildcards.roi])
        final_params = model_history[model_history.epoch == int(wildcards.max_epoch) - 1]
        precision_df = pd.DataFrame({})
        for tmp in input.precision_df:
            tmp_df = pd.read_csv(tmp)
            tmp_df['vroinames'] = tmp.split('_')[-1][:2]
            precision_df = precision_df.append(tmp_df, ignore_index=True)
        final_params_with_precision = pd.merge(final_params, precision_df, on=['subj','vroinames'])
        grid = vis.plot_precision_weighted_avg_parameters(final_params_with_precision,
                                                          params.params_order,
                                                          params.params_group,
                                                          roi='all', hue='vroinames', hue_order=['V1','V2','V3'], lgd_title='Visual areas')
        utils.save_fig(save_fig=True,save_path=output[0])

rule plot_2D_parameters_individual:
    input:
        model_history=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-True_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        #subj_df=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"dataframes","{{dset}}","{subj}_stim_voxel_info_df_vs-pRFsigma_{{roi}}.csv"),subj=make_subj_list(wildcards)),
        precision_df=os.path.join(config['OUTPUT_DIR'],"dataframes","{dset}","precision_s_{dset}_{roi}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'pointplot-params_avg-False_dset-{dset}_bts-{stat}_lr-{lr}_eph-{max_epoch}_vs-pRFsigma_roi-{roi}.{fig_format}'),
    params:
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D"),
        params_order = PARAMS_2D,
        params_group = [1,2,3,4,4,4,4,4,4]
    run:
        from sfp_nsdsyn import model
        from sfp_nsdsyn import bts
        from sfp_nsdsyn import vis
        model_history = model.load_history_files(input.model_history)
        final_params = model_history[model_history.epoch == int(wildcards.max_epoch) - 1]
        precision_df = pd.read_csv(input.precision_df)
        final_params_with_precision = pd.merge(final_params, precision_df, on='subj')
        pal = vis.make_dset_palettes(wildcards.dset)
        grid = vis.plot_individual_parameters(final_params_with_precision, params.params_order, subplot_group=params.params_group, palette=pal,
            roi='all')
        utils.save_fig(save_fig=True, save_path=output[0])

def get_projection(x):
    if x == 'angle':
        projection = 'polar'
        despine = False
    else:
        projection = None
        despine = True
    return projection, despine

def get_stim_class(stim):
    stim_dict = {'a': 'annulus', 'f': 'forward spiral', 'p': 'pinwheel', 'r':'reverse spiral'}
    if stim == 'all':
        return stim_dict.values()
    else:
        return [v for k, v in stim_dict.items() if k in stim]

rule preferred_period:
    input:
        stim_info = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsynthetic_sf_stim_description.csv',
        model_history = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-True_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        precision_df = os.path.join(config['OUTPUT_DIR'],"dataframes","{dset}","precision_s_{dset}_{roi}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'y-preferred-period_x-{x}_stim-{stim}_avg-True_dset-{dset}_bts-{stat}_lr-{lr}_eph-{max_epoch}_vs-pRFsigma_roi-{roi}.{fig_format}')
    params:
        sn_list = lambda wildcards: get_sn_list(wildcards.dset),
        stim_class = lambda wildcards: get_stim_class(wildcards.stim),
        properties = lambda wildcards: get_projection(wildcards.x)
    run:
        from sfp_nsdsyn import prep
        from sfp_nsdsyn import vis
        from sfp_nsdsyn import model
        stim_info = model.stim_w_r_and_w_a(input.stim_info, params.stim_class)
        synthetic_voxels = model.create_synthetic_cols(stim_info,
                                                       var_list=['eccentricity', 'angle'],
                                                       val_range_list=[(0, 10), (0, np.pi * 2)],
                                                       n_val_list=[3, 360])
        synthetic_voxels['local_ori'] = prep.calculate_local_orientation(w_a=synthetic_voxels.w_a,w_r=synthetic_voxels.w_r,retinotopic_angle=synthetic_voxels.angle,angle_in_radians=True)

        model_history = model.load_history_files(input.model_history)
        final_params = model_history[model_history.epoch == int(wildcards.max_epoch) - 1]
        synthetic_voxels_all_subj = pd.DataFrame({})
        for sn in params.sn_list:
            tmp = synthetic_voxels.copy()
            subj = utils.sub_number_to_string(sn, wildcards.dset)
            subj_params = final_params.query('subj == @subj')
            tmp['subj'] = subj
            tmp['vroinames'] = wildcards.roi
            tmp['Pv'] = tmp.apply(model.get_Pv_row, params=subj_params, axis=1)
            synthetic_voxels_all_subj = synthetic_voxels_all_subj.append(tmp, ignore_index=True)

        precision_s = pd.read_csv(input.precision_df)
        synthetic_voxels_all_subj = pd.merge(synthetic_voxels_all_subj, precision_s, on=['subj','vroinames'])
        if wildcards.x == 'angle':
            synthetic_voxels_all_subj = synthetic_voxels_all_subj.query('eccentricity == 5')
        vis.plot_preferred_period(synthetic_voxels_all_subj,
                                  x=wildcards.x,
                                  col=None, col_wrap=None,
                                  hue='names', hue_order=params.stim_class,
                                  projection=params.properties[0],
                                  despine=params.properties[1])
        utils.save_fig(save_fig=True, save_path=output[0])


rule svg_all:
    input:
        a = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'pointplot-precision-weighted-params_avg-True_dset-nsdsyn_bts-mean_lr-0.0005_eph-20000_vs-pRFsigma_roi-{roi}.png'), roi=['V1V2V3']),
        b = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'y-preferred-period_x-{x}_stim-{stim}_avg-True_dset-nsdsyn_bts-mean_lr-0.0005_eph-20000_vs-pRFsigma_roi-{roi}.png'), x=['eccentricity','angle'], stim=['all'], roi=['V1', 'V2', 'V3'])

