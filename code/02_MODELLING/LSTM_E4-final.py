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
PATIENCE = 130

VERSION = 'E4-final'

# should evaluate early based on cache?
EVALUATE = None

# list of hyperparameter tunings to try
TUNINGS = [
    { # 0 (E32)
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.0, 0.0],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.0015),
            'conv_recurrent_regularizer': ('L2', 0.0015),
            'dense_dropout': [0.75, 0.75],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.00075),
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
            'batch_size': 30,
            'epochs': 500,
            'lr_plateau': (0.1, 50, 0.00001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.2
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 1,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': [1979, 1980, 1981, 1982, 1983, 1984, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2016],
        'years_dev': None,
        'years_test': [1985, 1995, 2003, 2004, 2005, 2014, 2015, 2017]
    },
    { # 1 (E33)
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.0, 0.0],
            'conv_filters': [16, 16],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.0015),
            'conv_recurrent_regularizer': ('L2', 0.0015),
            'dense_dropout': [0.75, 0.75],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.00075),
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
            'batch_size': 30,
            'epochs': 500,
            'lr_plateau': (0.1, 50, 0.00001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.2
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't'],
            700: ['u', 'v']
        },
        'objective_onsets': False,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 1,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': [1979, 1980, 1981, 1982, 1983, 1984, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2016],
        'years_dev': None,
        'years_test': [1985, 1995, 2003, 2004, 2005, 2014, 2015, 2017]
    },
    { # 2 (E41)
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [],
            'conv_filters': [],
            'conv_kernels': [],
            'conv_pooling': [2],
            'conv_strides': [],
            'conv_kernel_regularizer': ('L2', 0.002),
            'conv_recurrent_regularizer': ('L2', 0.002),
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
            'lstm_recurrent_dropout': [0.2, 0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 30,
            'epochs': 500,
            'lr_plateau': (0.1, 50, 0.00001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.2
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't']
        },
        'objective_onsets': False,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 1,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': [1979, 1980, 1981, 1982, 1983, 1984, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2016],
        'years_dev': None,
        'years_test': [1985, 1995, 2003, 2004, 2005, 2014, 2015, 2017]
    },
    { # 3 (E42)
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.0, 0.3],
            'conv_filters': [10, 10],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.002),
            'conv_recurrent_regularizer': ('L2', 0.002),
            'dense_dropout': [0.7, 0.7],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.0005),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [20, 20],
            'lstm_kernels': [3, 3],
            'lstm_strides': [1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 60,
            'epochs': 500,
            'lr_plateau': (0.1, 50, 0.00001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.2
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't']
        },
        'objective_onsets': False,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 1,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': [1979, 1980, 1981, 1982, 1983, 1984, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2016],
        'years_dev': None,
        'years_test': [1985, 1995, 2003, 2004, 2005, 2014, 2015, 2017]
    },
    { # 4 (E43)
        'aggregation_resolution': None,
        'config_build': {
            'batch_norm': True,
            'conv_activation': 'relu',
            'conv_dropout': [0.0, 0.3],
            'conv_filters': [10, 10],
            'conv_kernels': [3, 3],
            'conv_pooling': [0, 0, 2],
            'conv_strides': [1, 1],
            'conv_kernel_regularizer': ('L2', 0.002),
            'conv_recurrent_regularizer': ('L2', 0.002),
            'dense_dropout': [0.7, 0.7],
            'dense_nodes': [1024, 1024],
            'dense_activation': 'relu',
            'dense_kernel_regularizer': ('L2', 0.0005),
            'learning_rate': 0.01,
            'loss': 'mean_squared_error',
            'lstm_filters': [20, 20],
            'lstm_kernels': [3, 3],
            'lstm_strides': [1, 1],
            'lstm_activation': 'tanh',
            'lstm_dropout': [0.3, 0.3],
            'lstm_recurrent_dropout': [0.2, 0.2],
            'lstm_recurrent_activation': 'hard_sigmoid',
            'optimizer': 'adam',
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 60,
            'epochs': 500,
            'lr_plateau': (0.1, 50, 0.00001),
            'patience': PATIENCE,
            'tensorboard': True,
            'validation_split': 0.2
        },
        'features': {
            'surface': ['msl'],
            1000: ['r', 't']
        },
        'objective_onsets': True,
        'predict_on': PREDICT_ON,
        'prediction_sequence': 29,
        'prediction_offset': 1,
        'prediction_example_length': 60,
        'years': YEARS,
        'years_train': [1979, 1980, 1981, 1982, 1983, 1984, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2016],
        'years_dev': None,
        'years_test': [1985, 1995, 2003, 2004, 2005, 2014, 2015, 2017]
    }
]

# get the tuning for the current index
TUNING = TUNINGS[INDEX]

# load onset dates
onset_dates, onset_ts = ModelHelpers.load_onset_dates(version='v2', objective=True if TUNING['objective_onsets'] else False)

# prepare prediction timestamps
# generate a sequence of timestamps for train and validation (and, optionally, test)
prediction_ts = ModelHelpers.generate_prediction_ts(TUNING['predict_on'], TUNING['years'], onset_dates=onset_dates, sequence_length=TUNING['prediction_sequence'], sequence_offset=TUNING['prediction_offset'], example_length=TUNING['prediction_example_length'])
# prediction_ts_test = ModelHelpers.generate_prediction_ts(TUNING['predict_on'], TUNING['years_test'], onset_dates=onset_dates, sequence_length=TUNING['prediction_sequence'], sequence_offset=TUNING['prediction_offset'], example_length=TUNING['prediction_example_length'])

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
        dataset = ERA.load_dataset_v2(TUNING['years'], invalidate=True, level=era_level, variables=TUNING['features'][era_level], filter_fun=filter_fun, aggregation_resolution=TUNING['aggregation_resolution'])
        features = features + [dataset[feature] for feature in TUNING['features'][era_level]]

if 'trmm' in TUNING['features']:
    features = features + TRMM.load_dataset(range(1998, 2017), range(1, 6), invalidate=True, aggregation_resolution=0.75, version='v3', lon_slice=slice(61.125, 97.625), lat_slice=slice(4.125, 40.625))

# train test split
print("> Train-Test Split")
X_train, y_train, X_test, y_test, X_dev, y_dev = ModelERAv3.train_test_split(
    features,
    prediction_ts,
    # prediction_ts_test,
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
for iteration in range(3):
    version_id = f'{VERSION}-{iteration}_seq{TUNING["prediction_sequence"]}'
    cache_id = f'{INDEX}-{iteration}'

    print("> Training Model")
    model, config, history = ModelHelpers.run_config(
        ModelERAv3,
        TUNING,
        X_train,
        y_train,
        invalidate=True,
        evaluate=None,
        validation_data=None,
        cache_id=cache_id,
        version=version_id)

    # evaluate the latest state of the model above
    print('Train (latest):', model.evaluate(X_train, y_train), model.predict(X_train))

    # dev_preds = model.predict(X_dev)
    # print('Dev (latest):', model.evaluate(X_dev, y_dev), dev_preds, y_dev)

    test_preds = model.predict(X_test)
    print('Test (latest):', model.evaluate(X_test, y_test), test_preds, y_test)

    # evaluate the best state of the model above
    best_instance = ModelERAv3(version=version_id, cache_id=cache_id)
    best_model = best_instance.load_model()
    print('Train (best):', best_model.evaluate(X_train, y_train), best_model.predict(X_train))

    # dev_preds_best = best_model.predict(X_dev)
    # print('Dev (best):', best_model.evaluate(X_dev, y_dev), dev_preds_best, y_dev)

    test_preds_best = best_model.predict(X_test)
    print('Test (best):', best_model.evaluate(X_test, y_test), test_preds_best, y_test)
