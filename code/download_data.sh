#!/bin/bash -x

#SBATCH --job-name=rsc-era
##SBATCH -t 0-11:59
#SBATCH --partition=kraken_fast
##SBATCH --mail-type=ALL
##SBATCH --mail-user=roland.schlaefli@uzh.ch

echo "[$SLURM_ARRAY_TASK_ID] Activating conda environment"
source ~/anaconda3/bin/activate gru-minimal

~/anaconda3/envs/gru-minimal/bin/python 00_DATA/ERA/download_data.py
