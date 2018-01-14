#!/bin/bash -x

#SBATCH --job-name=nn-v5
#SBATCH --array=9-11
##SBATCH --cpus-per-task=8
##SBATCH --partition=kraken_superfast
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem MaxMemPerNode
#SBATCH --partition=kraken_fast
##SBATCH --partition=kraken_slow
#SBATCH -o 00_LOGS/%A_%a.log
#SBATCH -e 00_LOGS/error.log

# params
EPOCHS=50
PATIENCE=0
LOG_DIR=~/thesis/code/00_LOGS/E1
# EVALUATE=false

# activate conda environment
echo "[$SLURM_ARRAY_TASK_ID] Activating conda environment"
source ~/anaconda3/bin/activate gru-minimal

# create necessary directories
mkdir -p $LOG_DIR/$SLURM_ARRAY_TASK_ID

# start logs
echo "---- START $SLURM_JOB_ID ----" >> $LOG_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out
echo "[$SLURM_ARRAY_TASK_ID] Training model $SLURM_ARRAY_TASK_ID"

# train model
~/anaconda3/envs/gru-minimal/bin/python ~/thesis/code/02_MODELLING/LSTM_E1.py $SLURM_ARRAY_TASK_ID $EPOCHS $PATIENCE >> $LOG_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out

# end logs
echo "[$SLURM_ARRAY_TASK_ID] Finished training model $SLURM_ARRAY_TASK_ID"
echo "---- END $SLURM_JOB_ID ----" >> $LOG_DIR/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out

# ~/anaconda3/envs/gru-minimal/bin/python LSTM_v5.py $SLURM_ARRAY_TASK_ID $EPOCHS $PATIENCE $EVALUATE >> 00_LOGS/v5/$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID.out
