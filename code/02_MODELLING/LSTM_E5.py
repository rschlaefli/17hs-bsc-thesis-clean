import sys
import paths

from ERA import ERA
from ModelHelpers import ModelHelpers
from ModelERAv3B import ModelERAv3B

# --- CONSTANTS ---
YEARS = range(1979, 2018)
YEARS_TRAIN = range(1979, 2010)
YEARS_DEV = range(2010, 2013)  # or none to use random validation data
YEARS_TEST = range(2013, 2018)
PRE_MONSOON = [3, 4, 5]
PREDICT_ON = '{}-05-22'

# --- PARAMETERS ---

# the tuning to actually train
INDEX = int(sys.argv[1])

# how many epochs for each tuning
EPOCHS = int(sys.argv[2])

# the length of the sequence of predictions for each year
SEQUENCE_LENGTH = int(sys.argv[3])

# should early stopping be enabled?
# training will stop if val_loss increased for more than PATIENCE times in a row
# enabled if >0
PATIENCE = int(sys.argv[4])

VERSION = f'E5-{EPOCHS}-{str(sys.argv[5])}'

# should evaluate early based on cache?
EVALUATE = sys.argv[6] if (len(sys.argv) >= 7) else None

# list of hyperparameter tunings to try
TUNINGS = [
    # setup the hyperparameters for the model to be built
    {
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
            'epochs': EPOCHS,
            'lr_plateau': (0.5, 10, 0.0001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.1
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': SEQUENCE_LENGTH,
        'prediction_offset': 0,
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

prediction_ts = ModelHelpers.generate_prediction_ts(TUNING['predict_on'], TUNING['years'], onset_dates=onset_dates, sequence_length=TUNING['prediction_sequence'], sequence_offset=TUNING['prediction_offset'])

# setup a filter function
# this later prevents any data after the prediction timestamp from being fed as input
def filter_fun(df, year):
    return ModelHelpers.filter_until(df, onset_ts[year])

# load the ERA dataset
print("> Loading Dataset")
# era_invariant = ERA.load_dataset_v2(TUNING['years'], level='invariant', variables=['z'], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])
era_surface = ERA.load_dataset_v2(TUNING['years'], level='surface', variables=['msl'], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])
era_1000 = ERA.load_dataset_v2(TUNING['years'], level=1000, variables=['r', 't'], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])
era_700 = ERA.load_dataset_v2(TUNING['years'], level=700, variables=['u', 'v'], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])
era_200 = ERA.load_dataset_v2(TUNING['years'], level=200, variables=['z', 'u'], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])

# train test split
print("> Train-Test Split")
X_train, y_train, X_test, y_test, X_dev, y_dev = ModelERAv3B.train_test_split(
    [era_surface['msl'], # mean sea-level pressure
     era_1000['r'], # relative humidity
     era_1000['t'], # temperature
     era_700['u'], # u-component of wind
     era_700['v'], # v-component of wind
     # era_200['u'], # u-component of wind at 200hPa
     era_200['z']], # geopotential at 200hPa
    prediction_ts,
    onset_ts,
    1,
    years_train=TUNING['years_train'],
    years_test=TUNING['years_test'],
    years_dev=TUNING['years_dev'])

# ensure that the mean of all values is very close to 0
assert round(X_train[:, :, :, :, 0].mean()) == 0
assert round(X_test[:, :, :, :, 0].mean()) == 0

if TUNING['years_dev']:
    assert round(X_dev[:, :, :, :, 0].mean()) == 0

# build a model based on the above tunings
print("> Training Model")
model, config, history = ModelHelpers.run_config(
    ModelERAv3B,
    TUNING,
    X_train,
    y_train,
    invalidate=True,
    evaluate=EVALUATE,
    validation_data=(X_dev, y_dev) if TUNING['years_dev'] else None,
    cache_id=INDEX,
    version=VERSION)

# evaluate the latest state of the model above
print('Train:', model.evaluate(X_train, y_train), model.predict(X_train))
print('Dev:', model.evaluate(X_dev, y_dev), model.predict(X_dev), y_dev)
print('Test:', model.evaluate(X_test, y_test), model.predict(X_test), y_test)
