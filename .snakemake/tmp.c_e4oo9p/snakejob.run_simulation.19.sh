#!/bin/bash
#SBATCH --export=NONE
# properties = {"type": "single", "rule": "run_simulation", "local": false, "input": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.04_n_vox-100.csv"], "output": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/model_history_full_ver-True_pw-False_sd-0.04_n_vox-100_lr-0.0005_eph-40000.csv", "/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/loss_history_full_ver-True_pw-False_sd-0.04_n_vox-100_lr-0.0005_eph-40000.csv"], "wildcards": {"full_ver": "True", "noise_sd": "0.04", "n_voxels": "100", "lr": "0.0005", "max_epoch": "40000"}, "params": {}, "log": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/results_2D/loss_history_full_ver-True_pw-False_sd-0.04_n_vox-100_lr-0.0005_eph-40000.csv"], "threads": 1, "resources": {"tmpdir": "/tmp"}, "jobid": 19, "cluster": {"nodes": 1, "tasks_per_node": 1, "mem": "80GB", "time": "12:00:00", "job_name": "run_simulation.full_ver=True,lr=0.0005,max_epoch=40000,n_voxels=100,noise_sd=0.04", "cpus_per_task": 10, "output": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/results_2D/loss_history_full_ver-True_pw-False_sd-0.04_n_vox-100_lr-0.0005_eph-40000.csv-%j", "error": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/results_2D/loss_history_full_ver-True_pw-False_sd-0.04_n_vox-100_lr-0.0005_eph-40000.csv-%j", "mail_type": "FAIL", "mail_user": "jiyeong.ha@nyu.edu"}}

env > /scratch/$USER/overlay/env.log

if [ "$SINGULARITY_CONTAINER" == "" ]; then
    export PATH=/scratch/$USER/overlay:$PATH
fi

 cd /home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn && \
/scratch/jh7685/sfp_nsd/sfp/bin/python3 \
-m snakemake /scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/loss_history_full_ver-True_pw-False_sd-0.04_n_vox-100_lr-0.0005_eph-40000.csv --snakefile /home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/Snakefile \
--force --cores all --keep-target-files --keep-remote --max-inventory-time 0 \
--wait-for-files '/home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/.snakemake/tmp.c_e4oo9p' '/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.04_n_vox-100.csv' --latency-wait 5 \
 --attempt 2 --force-use-threads --scheduler greedy \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules run_simulation --nocolor --notemp --no-hooks --nolock --scheduler-solver-path /scratch/jh7685/sfp_nsd/sfp/bin \
--mode 2  --default-resources "tmpdir=system_tmpdir"  && exit 0 || exit 1


