import sys
import paths

from ERA import ERA
from ModelHelpers import ModelHelpers
from ModelERAv1 import ModelERAv1

# --- CONSTANTS ---
YEARS = range(1979, 2018)
YEARS_TRAIN = range(1979, 2012)
YEARS_DEV = range(2011, 2013)  # or none to use random validation data
YEARS_TEST = range(2013, 2018)
PRE_MONSOON = [3, 4, 5]
PREDICT_ON = '{}-05-11'

# --- PARAMETERS ---

# the tuning to actually train
INDEX = int(sys.argv[1])

# how many epochs for each tuning
EPOCHS = int(sys.argv[2])

# should early stopping be enabled?
# training will stop if val_loss increased for more than PATIENCE times in a row
# enabled if >0
PATIENCE = int(sys.argv[3])

# should evaluate early based on cache?
EVALUATE = sys.argv[4] if (len(sys.argv) >= 5) else None
"""
    List of tunings to try for each type of model (categorical and numerical outputs)
    Trained models and their training history will be cached in a file and loaded on next run

    Available params:
    :dropout: How much dropout to use after dense layers
    :dropout_recurrent: How much recurrent dropout to use in ConvLSTM2D
    :dropout_conv: How much dropout to use after convolutional layers
    :epochs: How many epochs to train for
    :optimizer: The optimizer to use
    :learning_rate: The learning rate to use with the optimizer
    :batch_size: The batch size of training
    :loss: The loss function to use
    :batch_norm: Whether to use batch normalization
    :conv_activation: The activation to use in the convolutional layers
    :conv_filters: The number of filters for all convolutional layers as a list
    :conv_kernels: The dimensions of the kernels for all convolutional layers as a list
    :conv_pooling: Dimensions of the max pooling layers => one after each conv layer, final one before flatten
    :recurrent_activation: The activation for the LSTM recurrent part of ConvLSTM2D
    :padding: Whether to apply padding or not
    :numerical: Whether to train classes or numerical output
    :dense_nodes: The number of dense nodes for all dense layers as a list
    :dense_activation: The activation to use in all dense nodes except the final one
    :verbose: The level of logging to be used
    :invalidate: Whether the cache should be invalidated
    :patience: How much patience to use for early stopping
    :tensorboard: Whether tensorboard should be used
    :validation_split: How much of the data to keep for validation
    :conv_kernel_regularizer: Regularizer function applied to the kernel weights matrix of ConvLSTM2D layers.
    :conv_recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix of ConvLSTM2D layers.
    :dense_kernel_regularizer: Regularizer function applied to the kernel weights matrix of Dense layers.
    :lr_plateau: Whether the learning rate should be dynamically decreased when learning stagnates
    :cache_id: The id of the respective experiment.
"""
TUNINGS = [
    {
        'epochs': 50,
        'patience': PATIENCE
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'dropout': 0.5,
        'dropout_recurrent': 0.3,
        'dropout_conv': 0.3,
        'conv_pooling': [0, 0, 3, 0]
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'dropout': 0.5,
        'dropout_recurrent': 0.3,
        'dropout_conv': 0.3,
        'conv_pooling': [0, 2, 2, 0]
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'dropout': 0.5,
        'dropout_recurrent': 0.3,
        'dropout_conv': 0.3,
        'dense_kernel_regularizer': ('L2', 0.01),
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01)
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'conv_filters': [32, 16, 8],
        'dense_nodes': [1024, 512, 512, 256]
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'learning_rate': 1
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'learning_rate': 0.1
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'dense_kernel_regularizer': ('L2', 0.05),
        'conv_kernel_regularizer': ('L2', 0.05),
        'conv_recurrent_regularizer': ('L2', 0.05)
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'dense_kernel_regularizer': ('L2', 0.03),
        'conv_kernel_regularizer': ('L2', 0.03),
        'conv_recurrent_regularizer': ('L2', 0.03)
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'dense_kernel_regularizer': ('L2', 0.1),
        'conv_kernel_regularizer': ('L2', 0.1),
        'conv_recurrent_regularizer': ('L2', 0.1),
        'lr_plateau': (0.1, 10, 0.0001)
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'lr_plateau': (0.1, 10, 0.0001)
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'lr_plateau': (0.1, 10, 0.0001),
        'dropout_conv': 0.3,
        'dropout_recurrent': 0.2
    },
]

# prepare onset dates and prediction timestamps
onset_dates, onset_ts = ModelHelpers.load_onset_dates()
prediction_ts = ModelHelpers.generate_prediction_ts(PREDICT_ON, YEARS)

# load the ERA dataset
print("> Loading Dataset")

data_temp, data_hum = ERA.load_dataset(YEARS)

# train test split
print("> Train-Test Split")
if YEARS_DEV:
    X_train, y_train, X_test, y_test, X_dev, y_dev, unstacked = ModelERAv1.train_test_split(
        [data_temp, data_hum],
        prediction_ts,
        onset_ts,
        years_train=YEARS_TRAIN,
        years_test=YEARS_TEST,
        years_dev=YEARS_DEV)
else:
    X_train, y_train, X_test, y_test, unstacked = ModelERAv1.train_test_split(
        [data_temp, data_hum],
        prediction_ts,
        onset_ts,
        years_train=YEARS_TRAIN,
        years_test=YEARS_TEST)

# build a model based on the above tunings
print("> Training Model")
models_num = ModelHelpers.run_configs(
    ModelERAv1, [TUNINGS[INDEX]],
    X_train,
    y_train,
    invalidate=True,
    evaluate=EVALUATE,
    validation_data=(X_dev, y_dev) if YEARS_DEV else None,
    cache_id=INDEX,
    version="E1")

# evaluation of the model above
print("> Evaluating model")
for model in models_num:
    print(model['config'])
    print(model['model'].model.evaluate(X_test, y_test))
