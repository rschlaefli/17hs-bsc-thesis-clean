import sys
import paths
from TRMM import TRMM
from ModelHelpers import ModelHelpers
from ModelTRMMv3 import ModelTRMMv3

# --- CONSTANTS ---
YEARS = range(1998, 2017)
YEARS_TRAIN = range(1998, 2015)
YEARS_TEST = range(2015, 2017)
PRE_MONSOON = [3, 4, 5]
PREDICT_ON = '{}-05-11'

# --- PARAMETERS ---
# should the data be aggregated => specify degrees
# otherwise use None to get the full 140x140 grid
AGGREGATION_RESOLUTION = None

# how many epochs for each tuning
EPOCHS = int(sys.argv[2])

# should early stopping be enabled?
# training will stop if val_loss increased for more than PATIENCE times in a row
# enabled if >0
PATIENCE = int(sys.argv[3])
"""
    List of tunings to try for each type of model (categorical and numerical outputs)
    Trained models and their training history will be cached in a file and loaded on next run

    Available params:
    :dropout: How much dropout to use after dense layers (default: 0.6)
    :dropout_recurrent: How much recurrent dropout to use in ConvLSTM2D (default: None)
    :dropout_conv: How much dropout to use after convolutional layers (default: 0.4)
    :epochs: How many epochs to train for (default: 50)
    :batch_size: The batch size of training (default: 1)
    :numerical_loss: The loss function to use for numerical prediction (default: 'mean_squared_error')
    :categorical_loss: The loss function to use for categorical prediction (default: 'categorical_crossentropy')
    :optimizer: The optimizer to use (default: 'rmsprop')
    :learning_rate: The learning rate to use with the optimizer (default: None)
    :batch_norm: Whether to use batch normalization (default: True)
    :num_filters: The number of filters for the first, second and third conv layer as a tuple (default: (32, 16, 8))
    :kernel_dims: The dimensions of the kernels from first to third as a tuple (default: (7, 5, 3); squared, so 7 means 7x7)
    :padding: Whether to apply padding or not (default: 'same'; options: 'valid', 'same')
    :pool_dims: Dimension of the max pooling layers (default: (0, 0, 0, 4); squared, so 4 means 4x4) => one after each conv layer, final one before flatten
    :dense_nodes: The number of dense nodes at the start of dense layers (default: 1024)
    :dense_activation: The activation to use in dense nodes (default: 'relu')
    :patience: Patience for early stopping callback (default: 0)
    :validation_split: The size of the validation split (default: 0.2, percentage)
"""
TUNINGS = [{
    'epochs': EPOCHS,
    'patience': PATIENCE
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'batch_size': 4
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'dropout': 0.7,
    'dropout_conv': 0.5
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'dropout': 0.4,
    'dropout_conv': 0.3
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'optimizer': 'rmsprop',
    'learning_rate': 0.01
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'optimizer': 'rmsprop',
    'learning_rate': 0.003
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'optimizer': 'adam'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'optimizer': 'sgd'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'numerical_loss': 'mean_absolute_error'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'batch_norm': False
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'num_filters': (64, 32, 16)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'num_filters': (16, 16, 16),
    'kernel_dims': (6, 5, 4)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'num_filters': (8, 16, 32),
    'kernel_dims': (3, 5, 7)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'num_filters': (16, 16, 16),
    'kernel_dims': (4, 5, 6)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (0, 2, 0, 2)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0),
    'dense_activation': 'tanh'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0),
    'dense_activation': 'tanh'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0),
    'recurrent_activation': 'tanh'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (0, 0, 0, 3),
    'recurrent_activation': 'tanh'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (0, 0, 0, 3),
    'recurrent_activation': 'tanh'
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (0, 0, 0, 3),
    'recurrent_activation': None
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0),
    'dense_activation': 'tanh',
    'recurrent_activation': None,
    'num_filters': (16, 8, 4),
    'kernel_dims': (6, 4, 2)
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0),
    'dense_activation': 'tanh',
    'dropout_recurrent': 0.3
}, {
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'pool_dims': (2, 2, 2, 0),
    'dense_activation': 'tanh',
    'dense_nodes': 512
}]

# --- Loading the dataset ---
print("> Loading Dataset")
onset_dates, onset_ts = ModelHelpers.load_onset_dates()
prediction_ts = ModelHelpers.generate_prediction_ts(PREDICT_ON, YEARS)



def filter_fun(df, year):
    return ModelHelpers.filter_until(df, prediction_ts[year])


data_trmm = TRMM.load_dataset(
    YEARS,
    PRE_MONSOON,
    invalidate=False,
    filter_fun=filter_fun,
    aggregation_resolution=AGGREGATION_RESOLUTION,
    bundled=False)

# --- Building a model with numerical output based on the above tunings ---
print("> Train-Test Split")
X_train, y_train, X_test, y_test, unstacked = ModelTRMMv3.train_test_split(
    data_trmm,
    prediction_ts,
    onset_ts,
    years_train=YEARS_TRAIN,
    years_test=YEARS_TEST,
    numerical=True)

print("> Training Model")
models_num = ModelHelpers.run_configs(
    ModelTRMMv3, TUNINGS,
    X_train,
    y_train,
    numerical=True,
    invalidate=True,
    version='T3-200-0.25deg')

# evaluation of the models above
print("> Evaluating model")
for model in models_num:
    print(model['config'])
    print(model['model'].model.evaluate(X_train, y_train))
    print(model['model'].model.evaluate(X_test, y_test))
