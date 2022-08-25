#!/bin/bash
#SBATCH --export=NONE
# properties = {"type": "single", "rule": "generate_noisy_synthetic_data", "local": false, "input": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/original_syn_data_2d_full_ver-True_pw-False_sd-0_n_vox-100.csv"], "output": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.0_n_vox-100.csv"], "wildcards": {"full_ver": "True", "noise_sd": "0.0", "n_voxels": "100"}, "params": {}, "log": ["/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.0_n_vox-100.csv"], "threads": 1, "resources": {"tmpdir": "/tmp"}, "jobid": 3, "cluster": {"nodes": 1, "tasks_per_node": 1, "mem": "80GB", "time": "12:00:00", "job_name": "generate_noisy_synthetic_data.full_ver=True,n_voxels=100,noise_sd=0.0", "cpus_per_task": 10, "output": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.0_n_vox-100.csv-%j", "error": "/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.0_n_vox-100.csv-%j", "mail_type": "FAIL", "mail_user": "jiyeong.ha@nyu.edu"}}

env > /scratch/$USER/overlay/env.log

if [ "$SINGULARITY_CONTAINER" == "" ]; then
    export PATH=/scratch/$USER/overlay:$PATH
fi

 cd /home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn && \
/scratch/jh7685/sfp_nsd/sfp/bin/python3 \
-m snakemake /scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/syn_data_2d_full_ver-True_pw-False_sd-0.0_n_vox-100.csv --snakefile /home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/Snakefile \
--force --cores all --keep-target-files --keep-remote --max-inventory-time 0 \
--wait-for-files '/home/jh7685/sfp_nsd/Spatial-Frequency-Preference_NSDsyn/.snakemake/tmp.c_e4oo9p' '/scratch/jh7685/sfp_nsd/derivatives_HPC/simulation/synthetic_data_2D/original_syn_data_2d_full_ver-True_pw-False_sd-0_n_vox-100.csv' --latency-wait 5 \
 --attempt 1 --force-use-threads --scheduler greedy \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules generate_noisy_synthetic_data --nocolor --notemp --no-hooks --nolock --scheduler-solver-path /scratch/jh7685/sfp_nsd/sfp/bin \
--mode 2  --default-resources "tmpdir=system_tmpdir"  && exit 0 || exit 1


