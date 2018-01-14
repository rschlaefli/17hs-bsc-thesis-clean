import sys
import paths

from ERA import ERA
from ModelHelpers import ModelHelpers
from ModelERAv2 import ModelERAv2

# --- CONSTANTS ---
YEARS = range(1979, 2018)
YEARS_TRAIN = range(1979, 2012)
YEARS_DEV = range(2011, 2013)  # or none to use random validation data
YEARS_TEST = range(2013, 2018)
PRE_MONSOON = [3, 4, 5]
PREDICT_ON = '{}-05-22'

# --- PARAMETERS ---

# the tuning to actually train
INDEX = int(sys.argv[1])

# how many epochs for each tuning
EPOCHS = int(sys.argv[2])

# should early stopping be enabled?
# training will stop if val_loss increased for more than PATIENCE times in a row
# enabled if >0
PATIENCE = int(sys.argv[3])

VERSION = str(sys.argv[4])

# should evaluate early based on cache?
EVALUATE = sys.argv[5] if (len(sys.argv) >= 6) else None

# list of hyperparameter tunings to try
TUNINGS = [
    {
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'tanh',
            'conv_dropout': 0.4,
            'conv_filters': [32, 16, 8],
            'conv_kernels': [7, 5, 3],
            'conv_pooling': [2, 2, 2, 0],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_activation': 'hard_sigmoid',
            'conv_recurrent_regularizer': ('L2', 0.02),
            'conv_recurrent_dropout': 0.3,
            'dense_dropout': 0.6,
            'dense_nodes': [1024, 512, 256],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'optimizer': 'rmsprop',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': EPOCHS,
            'lr_plateau': (0.5, 5, 0.0001),
            'patience': PATIENCE,
            'tensorboard': False,
            'validation_split': 0.1
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    {
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'tanh',
            'conv_dropout': 0.0,
            'conv_filters': [32, 16, 8],
            'conv_kernels': [7, 5, 3],
            'conv_pooling': [2, 2, 2, 0],
            'conv_kernel_regularizer': None,
            'conv_recurrent_activation': 'hard_sigmoid',
            'conv_recurrent_regularizer': None,
            'conv_recurrent_dropout': 0.0,
            'dense_dropout': 0.0,
            'dense_nodes': [1024, 512, 256],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': None,
            'learning_rate': 0.1,
            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': EPOCHS,
            'lr_plateau': (0.5, 5, 0.0001),
            'patience': PATIENCE,
            'tensorboard': False,
            'validation_split': 0.1
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    }
]

# get the tuning for the current index
TUNING = TUNINGS[INDEX]

# prepare onset dates and prediction timestamps
onset_dates, onset_ts = ModelHelpers.load_onset_dates(version='v2', objective=True if TUNING['objective_onsets'] else False)
prediction_ts = ModelHelpers.generate_prediction_ts(TUNING['predict_on'], TUNING['years'])

# load the ERA dataset
print("> Loading Dataset")


def filter_fun(df, year):
    # setup a filter function
    return ModelHelpers.filter_until(df, prediction_ts[year])


data_temp, data_hum = ERA.load_dataset(
    TUNING['years'],
    version='v3',
    filter_fun=filter_fun,
    aggregation_resolution=TUNING['aggregation_resolution'])

# train test split
print("> Train-Test Split")
X_train, y_train, X_test, y_test, X_dev, y_dev, unstacked = ModelERAv2.train_test_split(
    [data_temp, data_hum],
    prediction_ts,
    onset_ts,
    years_train=TUNING['years_train'],
    years_test=TUNING['years_test'],
    years_dev=TUNING['years_dev'])

# build a model based on the above tunings
print("> Training Model")
model, config, history = ModelHelpers.run_config(
    ModelERAv2,
    TUNING,
    X_train,
    y_train,
    invalidate=True,
    evaluate=EVALUATE,
    validation_data=(X_dev, y_dev) if TUNING['years_dev'] else None,
    cache_id=INDEX,
    version=VERSION)

# evaluation of the model above
print("> Evaluating model")
print(model.evaluate(X_test, y_test))
