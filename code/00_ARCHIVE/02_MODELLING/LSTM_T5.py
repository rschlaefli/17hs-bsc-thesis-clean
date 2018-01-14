import sys
import paths
from TRMM import TRMM
from ModelHelpers import ModelHelpers
from ModelTRMMv4 import ModelTRMMv4

# --- CONSTANTS ---
YEARS = range(1998, 2017)
YEARS_TRAIN = range(1998, 2013)
YEARS_DEV = range(2013, 2015)  # or none to use random validation data
YEARS_TEST = range(2015, 2017)
PRE_MONSOON = [3, 4, 5]
PREDICT_ON = '{}-05-22'

# --- PARAMETERS ---
# should the data be aggregated => specify degrees
# otherwise use None to get the full 140x140 grid
AGGREGATION_RESOLUTION = None

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
"""
TUNINGS = [
    {
        'epochs': EPOCHS,
        'patience': PATIENCE
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernels': [8, 4],
        'conv_filters': [32, 16],
        'conv_pooling': [3, 3, 0],
        'dense_nodes': [512, 256]
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernels': [8, 6, 4],
        'dense_nodes': [1024, 512, 256, 128]
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernels': [8, 6, 4],
        'dense_nodes': [1024, 512, 256, 128],
        'recurrent_activation': 'tanh'
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernels': [9, 5],
        'conv_filters': [16, 8],
        'conv_pooling': [3, 3, 0],
        'dropout_conv': 0.5,
        'dropout_recurrent': 0.4,
        'dropout': 0.7
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernels': [9, 5],
        'conv_filters': [16, 8],
        'conv_pooling': [3, 3, 0],
        'dropout_conv': 0.6,
        'dropout_recurrent': 0.5,
        'dropout': 0.8
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'batch_size': 2
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regular
        izer': ('L2', 0.01)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_recurrent_regularizer': ('L2', 0.01)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.01)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'batch_size': 2,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.01),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'batch_size': 2,
        'dropout_conv': 0.3,
        'dropout_recurrent': 0.2,
        'dropout': 0.5,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.01),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.01),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'batch_size': 2,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.005),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'batch_size': 2,
        'conv_kernel_regularizer': ('L2', 0.02),
        'conv_recurrent_regularizer': ('L2', 0.02),
        'dense_kernel_regularizer': ('L2', 0.02),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'batch_size': 2,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.02),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.005),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.02),
        'conv_recurrent_regularizer': ('L2', 0.02),
        'dense_kernel_regularizer': ('L2', 0.02),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.02),
        'lr_plateau': (0.1, 5, 0.0001)
    },
    {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.01),
        'lr_plateau': (0.1, 5, 0.0001),
        'learning_rate': 0.1
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.01),
        'lr_plateau': (0.1, 10, 0.0001),
        'learning_rate': 0.1,
        'dropout_conv': 0.3,
        'dropout_recurrent': 0.2
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.01),
        'conv_recurrent_regularizer': ('L2', 0.01),
        'dense_kernel_regularizer': ('L2', 0.1),
        'lr_plateau': (0.1, 10, 0.0001),
        'learning_rate': 0.1,
        'dropout_conv': 0.3,
        'dropout_recurrent': 0.2
    },
    {
        'epochs': 50,
        'patience': PATIENCE,
        'conv_kernel_regularizer': ('L2', 0.1),
        'conv_recurrent_regularizer': ('L2', 0.1),
        'dense_kernel_regularizer': ('L2', 0.2),
        'lr_plateau': (0.1, 10, 0.0001),
        'learning_rate': 0.1,
        'dropout_conv': 0.3,
        'dropout_recurrent': 0.2
    },
]

# prepare onset dates and prediction timestamps
onset_dates, onset_ts = ModelHelpers.load_onset_dates()
prediction_ts = ModelHelpers.generate_prediction_ts(PREDICT_ON, YEARS)


def filter_fun(df, year):
    # setup a filter function
    return ModelHelpers.filter_until(df, prediction_ts[year])


# load the TRMM dataset
print("> Loading Dataset")
data_trmm = TRMM.load_dataset(
    YEARS,
    PRE_MONSOON,
    invalidate=False,
    filter_fun=filter_fun,
    aggregation_resolution=AGGREGATION_RESOLUTION,
    bundled=False)

# train test split
print("> Train-Test Split")
if YEARS_DEV:
    X_train, y_train, X_test, y_test, X_dev, y_dev, unstacked = ModelTRMMv4.train_test_split(
        data_trmm,
        prediction_ts,
        onset_ts,
        years_train=YEARS_TRAIN,
        years_test=YEARS_TEST,
        years_dev=YEARS_DEV)
else:
    X_train, y_train, X_test, y_test, unstacked = ModelTRMMv4.train_test_split(
        data_trmm,
        prediction_ts,
        onset_ts,
        years_train=YEARS_TRAIN,
        years_test=YEARS_TEST)

# build a model based on the above tunings
print("> Training Model")
models_num = ModelHelpers.run_configs(
    ModelTRMMv4, [TUNINGS[INDEX]],
    X_train,
    y_train,
    invalidate=False,
    evaluate=EVALUATE,
    validation_data=(X_dev, y_dev) if YEARS_DEV else None,
    cache_id=INDEX,
    version="T4")

# evaluation of the model above
print("> Evaluating model")
for model in models_num:
    print(model['config'])
    print(model['model'].model.evaluate(X_test, y_test))
