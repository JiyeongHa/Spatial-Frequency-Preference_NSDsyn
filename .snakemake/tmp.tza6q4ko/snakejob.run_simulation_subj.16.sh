#!/bin/bash
#SBATCH --export=NONE
# properties = {"type": "single", "rule": "run_simulation_subj", "local": false, "input": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-True_noise_mtpl-1_subj-06.csv"], "output": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/model_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.csv", "/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/loss_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.csv", "/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/losses_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.csv"], "wildcards": {"full_ver": "True", "pw": "True", "n_sd_mtpl": "1", "sn": "06", "lr": "0.0005", "max_epoch": "20000"}, "params": {}, "log": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/results_2D/loss_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.log"], "threads": 1, "resources": {"mem_mb": 1000, "disk_mb": 1000, "tmpdir": "/tmp"}, "jobid": 16, "cluster": {"nodes": 1, "tasks_per_node": 1, "mem": "100GB", "time": "5-12:00:00", "job_name": "run_simulation_subj.full_ver=True,lr=0.0005,max_epoch=20000,n_sd_mtpl=1,pw=True,sn=06", "cpus_per_task": 20, "output": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/results_2D/loss_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.log-%j", "error": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/results_2D/loss_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.log-%j", "mail_type": "FAIL", "mail_user": "jiyeong.ha@nyu.edu"}}

env > /scratch/$USER/overlay/env.log

if [ "$SINGULARITY_CONTAINER" == "" ]; then
    export PATH=/scratch/$USER/overlay:$PATH
fi

cd '/home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn' && /scratch/jh7685/sfp_nsd/sfp/bin/python -m snakemake --snakefile '/home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/Snakefile' '/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/results_2D/loss_history_full_ver-True_pw-True_noise_mtpl-1_subj-06_lr-0.0005_eph-20000.csv' --allowed-rules 'run_simulation_subj' --cores 'all' --attempt 1 --force-use-threads  --wait-for-files '/home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/.snakemake/tmp.tza6q4ko' '/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-True_noise_mtpl-1_subj-06.csv' --force --keep-target-files --keep-remote --max-inventory-time 0 --nocolor --notemp --no-hooks --nolock --ignore-incomplete --rerun-triggers 'input' 'params' 'code' 'mtime' 'software-env' --skip-script-cleanup  --conda-frontend 'mamba' --wrapper-prefix 'https://github.com/snakemake/snakemake-wrappers/raw/' --latency-wait 5 --scheduler 'greedy' --scheduler-solver-path '/scratch/jh7685/sfp_nsd/sfp/bin' --default-resources 'mem_mb=max(2*input.size_mb, 1000)' 'disk_mb=max(2*input.size_mb, 1000)' 'tmpdir=system_tmpdir' --mode 2 && exit 0 || exit 1


