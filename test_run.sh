#!/bin/bash
#SBATCH --job-name=noisy_simulation
#SBATCH -a 1 # these numbers are read in to SLURM_ARRAY_TASK_ID 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=120g
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/out_%x-%a.txt
#SBATCH --error=/scratch/jh7685/sfp_nsd/derivatives_HPC/logs/simulation/error_%x-%a.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=jiyeong.ha@nyu.edu

conda activate sfp
snakemake -j 26 run_simulation_all_subj 


