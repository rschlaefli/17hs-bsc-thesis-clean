#!/bin/bash -x

# setup the log dir
MODEL="E4-final"
MODEL_VERSION="1"
PATIENCE="130"
LOG_DIR=~/17hs-bsc-thesis-clean/code/03_EVALUATION/logs/$MODEL

echo "> Activating conda environment..."

# activate the conda environment
source ~/anaconda3/bin/activate gru-minimal

echo "> Running configs..."

# go through all possible configs
for i in {0..4}
do
  # ensure the log dir for this version is existent
  mkdir -p $LOG_DIR

  # run the experiment, redirecting all output to the logfiles
  ~/anaconda3/envs/gru-minimal/bin/python ./02_MODELLING/LSTM_$1.py $i >> $LOG_DIR/$MODEL-$MODEL_VERSION_$PATIENCE_$i.txt
done

echo "> Finished running configs. Shutdown..."

# shutdown the machine after completion of all experiments
sudo shutdown
