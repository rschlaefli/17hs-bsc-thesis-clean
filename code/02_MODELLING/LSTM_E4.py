import sys
import paths
from itertools import chain

from TRMM import TRMM
from ERA import ERA
from ModelHelpers import ModelHelpers
from ModelERAv3 import ModelERAv3

# --- CONSTANTS ---
YEARS = range(1979, 2018)
YEARS_TRAIN = range(1979, 2010)
YEARS_DEV = range(2010, 2013)  # or none to use random validation data
YEARS_TEST = range(2013, 2018)
PREDICT_ON = '{}-05-22'

# --- PARAMETERS ---

# the tuning to actually train
INDEX = int(sys.argv[1])

# should early stopping be enabled?
# training will stop if val_loss increased for more than PATIENCE times in a row
# enabled if >0
PATIENCE = int(sys.argv[2])

VERSION = f'E4-{str(sys.argv[3])}'

# should evaluate early based on cache?
EVALUATE = sys.argv[4] if (len(sys.argv) >= 5) else None

# list of hyperparameter tunings to try
TUNINGS = [
    # setup the hyperparameters for the model to be built
    { # 0
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3, 0.3, 0.3],
            'conv_filters': [30, 30, 30, 30],
            'conv_kernels': [3, 3, 3, 3],
            'conv_pooling': [0, 0, 0, 0, 2],
            'conv_strides': [1, 1, 1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.4, 0.4, 0.4],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [30, 30, 30],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 60,
            'lr_plateau': (0.1, 20, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 1
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0, 0, 0, 0.4],
            'conv_filters': [32, 32, 32, 32],
            'conv_kernels': [3, 3, 3, 3],
            'conv_pooling': [0, 0, 0, 0, 2],
            'conv_strides': [1, 1, 1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.3, 0.3, 0.3],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [32, 32, 32],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0, 0, 0.3],
            'lstm_recurrent_dropout': [0.1, 0.1, 0.1],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 60,
            'lr_plateau': (0.5, 15, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 2
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0, 0, 0],
            'conv_filters': [30, 30, 30],
            'conv_kernels': [3, 3, 3],
            'conv_pooling': [0, 0, 0, 2],
            'conv_strides': [1, 1, 1],
            'conv_kernel_regularizer': (None, 0.02),
            'conv_recurrent_regularizer': (None, 0.02),
            'dense_dropout': [0, 0, 0],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': (None, 0.02),
            'learning_rate': 0.1,
            'loss': 'mean_squared_error',
            'lstm_filters': [30, 30, 30],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0, 0, 0],
            'lstm_recurrent_dropout': [0, 0, 0],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 60,
            'lr_plateau': (0.5, 15, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 3
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [30, 30],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.3, 0.3, 0.3],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [30, 30],
            'lstm_kernels': [3, 3],
            'lstm_strides': [1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 500,
            'lr_plateau': (0.5, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 7,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 4
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3, 0.3],
            'conv_filters': [8, 8, 8],
            'conv_kernels': [3, 3, 3],
            'conv_pooling': [0, 0, 0, 2],
            'conv_strides': [1, 1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.3, 0.3, 0.3],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 500,
            'lr_plateau': (0.5, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 7,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 5
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3, 0.3, 0.3],
            'conv_filters': [12, 12, 12, 12],
            'conv_kernels': [3, 3, 3, 3],
            'conv_pooling': [0, 0, 0, 0, 2],
            'conv_strides': [1, 1, 1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16, 16, 16],
            'lstm_kernels': [3, 3, 3, 3, 3],
            'lstm_strides': [1, 1, 1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 500,
            'lr_plateau': (0.5, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 7,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 6
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3],
            'conv_filters': [16],
            'conv_kernels': [3],
            'conv_pooling': [0, 2],
            'conv_strides': [1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.5, 0.5],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 500,
            'lr_plateau': (0.5, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 20,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 7
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.5, 0.5],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.1,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'sgd',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 150,
            'lr_plateau': (0.1, 25, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 20,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 8
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.5, 0.5],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.1,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'rmsprop',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 150,
            'lr_plateau': (0.1, 25, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 20,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 9
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3, 0.3, 0.3],
            'conv_filters': [30, 30, 30, 30],
            'conv_kernels': [3, 3, 3, 3],
            'conv_pooling': [0, 0, 0, 0, 2],
            'conv_strides': [1, 1, 1, 1],
            'conv_kernel_regularizer': ('L2', 0.02),
            'conv_recurrent_regularizer': ('L2', 0.02),
            'dense_dropout': [0.4, 0.4, 0.4],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.02),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [30, 30, 30],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'relu',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same',
            'with_sequences': True
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': 60,
            'lr_plateau': (0.1, 20, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 10
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 22,
            'epochs': 150,
            'lr_plateau': (0.1, 25, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 11
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [],
            'conv_filters': [],
            'conv_kernels': [],
            'conv_pooling': [2],
            'conv_strides': [],
            'conv_kernel_regularizer': ('L2', 0.002),
            'conv_recurrent_regularizer': ('L2', 0.002),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.002),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [32, 32, 32, 32],
            'lstm_kernels': [3, 3, 3, 3],
            'lstm_strides': [1, 1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.4, 0.4, 0.4, 0.4],
            'lstm_recurrent_dropout': [0.3, 0.3, 0.3, 0.3],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 22,
            'epochs': 200,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 12
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [32, 32],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [32, 32, 32, 32],
            'lstm_kernels': [3, 3, 3, 3],
            'lstm_strides': [1, 1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.4, 0.4, 0.4, 0.4],
            'lstm_recurrent_dropout': [0.3, 0.3, 0.3, 0.3],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 22,
            'epochs': 200,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 13
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'tanh',
            'conv_dropout': [0.4, 0.4, 0.4],
            'conv_filters': [16, 16, 16],
            'conv_kernels': [3, 3, 3],
            'conv_pooling': [0, 0, 0, 2],
            'conv_strides': [1, 1, 1],
            'conv_kernel_regularizer': ('L2', 0.0005),
            'conv_recurrent_regularizer': ('L2', 0.0005),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.0005),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.4, 0.4, 0.4],
            'lstm_recurrent_dropout': [0.4, 0.4, 0.4],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 22,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 14
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.4, 0.4],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.0005),
            'conv_recurrent_regularizer': ('L2', 0.0005),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.0005),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16],
            'lstm_kernels': [3, 3],
            'lstm_strides': [1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.4, 0.4],
            'lstm_recurrent_dropout': [0.4, 0.4],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 22,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 21,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 15
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.5, 0.5],
            'conv_filters': [20, 20],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [20, 20, 20],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 15,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 16
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.5, 0.5],
            'conv_filters': [20, 20],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [20, 20, 20],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 10,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 90,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 17
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.5, 0.5],
            'conv_filters': [20, 20],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.5, 0.5, 0.5],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [20, 20, 20],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 30,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 18
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.6, 0.6],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.6, 0.6, 0.6],
            'dense_nodes': [2048, 1024, 512],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [32, 32, 32],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_dropout': [0.4, 0.4, 0.4],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v'],
            200: ['u']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 40,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 19
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.6, 0.6],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.6, 0.6, 0.6],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.6, 0.6, 0.6],
            'lstm_recurrent_dropout': [0.5, 0.5, 0.5],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v'],
            200: ['u']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 20
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.7, 0.7, 0.7],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 21
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.7, 0.7, 0.7],
            'dense_nodes': [1024, 1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
    { # 22
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': None,
            'conv_dropout': [0.3, 0.3],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.001),
            'conv_recurrent_regularizer': ('L2', 0.001),
            'dense_dropout': [0.7, 0.7],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.001),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [16, 16, 16],
            'lstm_kernels': [3, 3, 3],
            'lstm_strides': [1, 1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_dropout': [0.3, 0.3, 0.3],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 300,
            'lr_plateau': (0.1, 50, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 0,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': YEARS_TRAIN,
        'years_dev': YEARS_DEV,
        'years_test': YEARS_TEST,
    },
]

# get the tuning for the current index
TUNING = TUNINGS[INDEX]

# load onset dates
onset_dates, onset_ts = ModelHelpers.load_onset_dates(version='v2', objective=True if TUNING['objective_onsets'] else False)

# prepare prediction timestamps
# generate a sequence of timestamps for train and validation
# generate the timestamp of the 22nd of each test year
prediction_ts = ModelHelpers.generate_prediction_ts(TUNING['predict_on'], chain(TUNING['years_train'], TUNING['years_dev']), onset_dates=onset_dates, sequence_length=TUNING['prediction_sequence'], sequence_offset=TUNING['prediction_offset'], example_length=TUNING['prediction_example_length'])
prediction_ts_test = ModelHelpers.generate_prediction_ts(TUNING['predict_on'], TUNING['years_test'], fake_sequence=True, example_length=TUNING['prediction_example_length'])

# setup a filter function
# this later prevents any data after the prediction timestamp from being fed as input
# we do this externally to allow overriding or extending the filter function if needed
def filter_fun(df, year):
    return ModelHelpers.filter_until(df, onset_ts[year])

# load the ERA dataset
features = []
print("> Loading Dataset")
for era_level in ['invariant', 'surface', 1000, 700, 200]:
    if era_level in TUNING['features']:
        dataset = ERA.load_dataset_v2(TUNING['years'], invalidate=False, level=era_level, variables=TUNING['features'][era_level], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])
        features = features + [dataset[feature] for feature in TUNING['features'][era_level]]

if 'trmm' in TUNING['features']:
    features = features + TRMM.load_dataset(TUNING['years'], range(1, 6), invalidate=True, aggregation_resolution=0.75, version='v3', default_slice=True)

# train test split
print("> Train-Test Split")
X_train, y_train, X_test, y_test, X_dev, y_dev = ModelERAv3.train_test_split(
    features,
    prediction_ts,
    prediction_ts_test,
    onset_ts,
    years_train=TUNING['years_train'],
    years_test=TUNING['years_test'],
    years_dev=TUNING['years_dev'])

# ensure that the mean of all values is close to 0 and the normalization thus works correctly (channel-based)
assert round(X_train[:, :, :, :, 0].mean()) == 0
assert round(X_test[:, :, :, :, 0].mean()) == 0

if TUNING['years_dev']:
    assert round(X_dev[:, :, :, :, 0].mean()) == 0

# build a model based on the above tunings
print("> Training Model")
model, config, history = ModelHelpers.run_config(
    ModelERAv3,
    TUNING,
    X_train,
    y_train,
    invalidate=True,
    evaluate=EVALUATE,
    validation_data=(X_dev, y_dev) if TUNING['years_dev'] else None,
    cache_id=INDEX,
    version=f'{VERSION}_seq{TUNING["prediction_sequence"]}')

# evaluate the latest state of the model above
print('Train (latest):', model.evaluate(X_train, y_train), model.predict(X_train))

dev_preds = model.predict(X_dev)
print('Dev (latest):', model.evaluate(X_dev, y_dev), dev_preds, y_dev)

test_preds = model.predict(X_test)
print('Test (latest):', model.evaluate(X_test, y_test), test_preds, y_test)

# evaluate the best state of the model above
best_instance = ModelERAv3(version=f'{VERSION}_seq{TUNING["prediction_sequence"]}', cache_id=INDEX)
best_model = best_instance.load_model()

print('Train (best):', best_model.evaluate(X_train, y_train), best_model.predict(X_train))

dev_preds_best = best_model.predict(X_dev)
print('Dev (best):', best_model.evaluate(X_dev, y_dev), dev_preds_best, y_dev)

test_preds_best = best_model.predict(X_test)
print('Test (best):', best_model.evaluate(X_test, y_test), test_preds_best, y_test)
