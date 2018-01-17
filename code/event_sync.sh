#!/bin/bash -x

#SBATCH --job-name=trmm-sync
#SBATCH --array=0-2
##SBATCH --cpus-per-task=8
##SBATCH --partition=kraken_superfast
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem MaxMemPerNode
##SBATCH --partition=kraken_fast
#SBATCH --partition=kraken_slow
#SBATCH -o 00_LOGS/SYNC/%A_%a.log
#SBATCH -e 00_LOGS/SYNC/error.log

# params
AGGREGATION=0.75
LOG_DIR=00_LOGS/SYNC

# activate conda environment
echo "[$SLURM_ARRAY_TASK_ID] Activating conda environment"
source ~/anaconda3/bin/activate gru-minimal

# create necessary directories
mkdir -p $LOG_DIR/$SLURM_ARRAY_TASK_ID

# start logs
echo "---- START $SLURM_JOB_ID ----" >> $LOG_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out
echo "[$SLURM_ARRAY_TASK_ID] Calculating event synchronization #$SLURM_ARRAY_TASK_ID"

# calculate synchronization
~/anaconda3/envs/gru-minimal/bin/python ~/thesis/code/01_ANALYSIS/EventSync.py $SLURM_ARRAY_TASK_ID $AGGREGATION >> $LOG_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out

# end logs
echo "[$SLURM_ARRAY_TASK_ID] Finished event synchronization $SLURM_ARRAY_TASK_ID"
echo "---- END $SLURM_JOB_ID ----" >> $LOG_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out
