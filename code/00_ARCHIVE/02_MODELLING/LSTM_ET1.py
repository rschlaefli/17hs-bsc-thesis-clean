import sys
import paths

from TRMM import TRMM
from ERA import ERA

from ModelHelpers import ModelHelpers
from ModelERAv2 import ModelERAv2

# --- CONSTANTS ---
YEARS = range(1998, 2018)
YEARS_TRAIN = range(1998, 2013)
YEARS_DEV = range(2013, 2015)
YEARS_TEST = range(2015, 2018)

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

TUNINGS = [
    {
        'aggregation_resolution': 1.0,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'tanh',
            'conv_dropout': 0.3,
            'conv_filters': [32, 32, 32],
            'conv_kernels': [7, 5, 3],
            'conv_pooling': [0, 0, 0, 2],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_activation': 'hard_sigmoid',
            'conv_recurrent_regularizer': ('L2', 0.02),
            'conv_recurrent_dropout': 0.2,
            'dense_dropout': 0.5,
            'dense_nodes': [1024, 512, 512, 256],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.1,
            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': EPOCHS,
            'lr_plateau': (0.5, 10, 0.0001),
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
        'aggregation_resolution': 1.0,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'tanh',
            'conv_dropout': 0.3,
            'conv_filters': [32, 32, 16],
            'conv_kernels': [15, 15, 3],
            'conv_strides': [1, 5, 1],
            'conv_pooling': [0, 0, 0, 0],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_activation': 'hard_sigmoid',
            'conv_recurrent_regularizer': ('L2', 0.02),
            'conv_recurrent_dropout': 0.2,
            'dense_dropout': 0.5,
            'dense_nodes': [1024, 512, 256],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.1,
            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': EPOCHS,
            'lr_plateau': (0.5, 10, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
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
        'aggregation_resolution': 1.0,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': 0.4,
            'conv_filters': [10, 10, 20],
            'conv_kernels': [7, 7, 15],
            'conv_strides': [1, 1, 1],
            'conv_pooling': [0, 0, 0, 0],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_activation': 'hard_sigmoid',
            'conv_recurrent_regularizer': ('L2', 0.02),
            'conv_recurrent_dropout': 0.2,
            'dense_dropout': 0.5,
            'dense_nodes': [1024, 512, 512, 256],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.001,
            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': EPOCHS,
            'lr_plateau': None,
            'patience': PATIENCE,
            'tensorboard': True,
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

# load data for the pre-monsoon period (MAM)
trmm_data = TRMM.load_dataset(
    TUNING['years'],
    range(3, 6),
    aggregation_resolution=TUNING['aggregation_resolution'],
    timestamp=True,
    invalidate=False,
    version='v2',
    filter_fun=filter_fun,
    bundled=False)

era_temp, era_hum = ERA.load_dataset(
    TUNING['years'],
    invalidate=False,
    timestamp=True,
    filter_fun=filter_fun,
    aggregation_resolution=TUNING['aggregation_resolution'])

# train test split
print("> Train-Test Split")
X_train, y_train, X_test, y_test, X_dev, y_dev, unstacked = ModelERAv2.train_test_split(
    [trmm_data, era_temp, era_hum],
    prediction_ts,
    onset_ts,
    years=TUNING['years'],
    years_train=TUNING['years_train'],
    years_test=TUNING['years_test'],
    years_dev=TUNING['years_dev'])

# build a model based on the above tunings
print("> Training Model")
model = ModelHelpers.run_config(
    ModelERAv2,
    TUNING,
    X_train,
    y_train,
    invalidate=True,
    evaluate=None,
    validation_data=(X_dev, y_dev) if TUNING['years_dev'] else None,
    cache_id=INDEX,
    version=VERSION)

# evaluation of the model above
print("> Evaluating model")
print(model['config'])
print(model['model'].evaluate(X_test, y_test))
print(model['history'])

