#!/bin/bash
#SBATCH --export=NONE
# properties = {"type": "single", "rule": "plot_model_param_history", "local": false, "input": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/model_history_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.csv"], "output": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/figures/simulation/results_2D/Epoch_vs_Param_values/param_history_plot_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.png"], "wildcards": {"full_ver": "True", "noise_sd": "0", "n_voxels": "100", "lr": "0.0007", "max_epoch": "31"}, "params": {}, "log": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/figures/Epoch_vs_Param_values/param_history_plot_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.log"], "threads": 1, "resources": {"tmpdir": "/tmp"}, "jobid": 0, "cluster": {"nodes": 1, "tasks_per_node": 1, "mem": "80GB", "time": "12:00:00", "job_name": "plot_model_param_history.full_ver=True,lr=0.0007,max_epoch=31,n_voxels=100,noise_sd=0", "cpus_per_task": 10, "output": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/figures/Epoch_vs_Param_values/param_history_plot_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.log-%j", "error": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/figures/Epoch_vs_Param_values/param_history_plot_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.log-%j", "mail_type": "FAIL", "mail_user": "jiyeong.ha@nyu.edu"}}

env > /scratch/$USER/overlay/env.log

if [ "$SINGULARITY_CONTAINER" == "" ]; then
    export PATH=/scratch/$USER/overlay:$PATH
fi

 cd /home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn && \
/scratch/jh7685/sfp_nsd/sfp/bin/python3 \
-m snakemake /scratch/jh7685/sfp_nsd/derivatives_HPC/figures/simulation/results_2D/Epoch_vs_Param_values/param_history_plot_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.png --snakefile /home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/Snakefile \
--force --cores all --keep-target-files --keep-remote --max-inventory-time 0 \
--wait-for-files '/home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/.snakemake/tmp.qqya3zv4' '/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/model_history_full_ver-True_sd-0_n_vox-100_lr-0.0007_eph-31.csv' --latency-wait 5 \
 --attempt 1 --force-use-threads --scheduler greedy \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules plot_model_param_history --nocolor --notemp --no-hooks --nolock --scheduler-solver-path /scratch/jh7685/sfp_nsd/sfp/bin \
--mode 2  --default-resources "tmpdir=system_tmpdir"  && exit 0 || exit 1


