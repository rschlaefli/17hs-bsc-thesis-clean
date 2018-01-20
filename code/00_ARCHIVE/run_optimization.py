import sys
import paths
import hashlib
import csv
import os
from keras.models import load_model
from keras import backend as K

from TRMM import TRMM
from ERA import ERA
from ModelHelpers import ModelHelpers
from ModelERAv2 import ModelERAv2

import GPy
import GPyOpt

# --- CONSTANTS ---
YEARS = range(1998, 2018)
YEARS_TRAIN = range(1998, 2013)
YEARS_DEV = range(2013, 2015)  # or none to use random validation data
YEARS_TEST = range(2015, 2018)
PRE_MONSOON = [3, 4, 5]
PREDICT_ON = '{}-05-22'
VERSION = 'B3'

EPOCHS = 100
PATIENCE = 30

# redirect all logging into a file
sys.stdout = open(f'03_EVALUATION/bayesian/{VERSION}.log', 'w')

mapping = {
    'conv_activation': [
        'tanh',
        'relu'
    ],
    'network_structure': [{
        # default for E3
        'conv_filters': [20, 20, 20],
        'conv_kernels': [7, 5, 3],
        'conv_strides': [1, 1, 1],
        'conv_pooling': [0, 0, 0, 3],
        'dense_nodes': [1024, 512, 256]
    }, {
        'conv_filters': [20, 10, 10, 10],
        'conv_kernels': [15, 9, 9, 5],
        'conv_strides': [1, 1, 1, 1],
        'conv_pooling': [0, 0, 0, 0, 3],
        'dense_nodes': [1024, 1024, 256]
    }, {
        'conv_filters': [10, 10, 10, 20],
        'conv_kernels': [5, 9, 9, 15],
        'conv_strides': [1, 1, 1, 1],
        'conv_pooling': [0, 0, 0, 0, 3],
        'dense_nodes': [1024, 1024, 256]
    }, {
        'conv_filters': [10, 10, 20, 10],
        'conv_kernels': [5, 9, 15, 9],
        'conv_strides': [1, 1, 1, 1],
        'conv_pooling': [0, 0, 0, 0, 3],
        'dense_nodes': [1024, 1024, 256]
    }],
    'dense_activation': [
        'tanh',
        'relu'
    ],
    'initial_lr': [
        None,
        1.0,
        0.5,
        0.1,
        0.05,
        0.01
    ],
    'optimizer': [
        'adam',
        'rmsprop'
    ],
    'regularizer': [
        None,
        'L1',
        'L2',
        'L1_L2'
    ]
}

# boundaries for hyperparameter tuning
# the bounds dict should be in order of continuous type and then discrete type
bounds = [{  # 0
    'name': 'dropout',
    'type': 'continuous',
    'domain': (0, 0.7)
}, {  # 1
    'name': 'dropout_conv',
    'type': 'continuous',
    'domain': (0, 0.7)
}, {  # 2
    'name': 'dropout_recurrent',
    'type': 'continuous',
    'domain': (0, 0.7)
}, { # 3
    'name': 'regularizer_weight',
    'type': 'continuous',
    'domain': (0.05, 0.1)
}, {  # 4
    'name': 'regularizer_type',
    'type': 'discrete',
    'domain': (0, 1, 2, 3)
}, {  # 5
    'name': 'optimizer',
    'type': 'discrete',
    'domain': (0, 1)
}, {  # 6
    'name': 'initial_learning_rate',
    'type': 'discrete',
    'domain': (0, 1, 2, 3, 4, 5)
}, {  # 7
    'name': 'network_structure',
    'type': 'discrete',
    'domain': (0, 1, 2, 3)
}, {  # 8
    'name': 'conv_activation',
    'type': 'discrete',
    'domain': (0, 1)
}, {  # 9
    'name': 'dense_activation',
    'type': 'discrete',
    'domain': (0, 1)
}]


# setup the objective function for optimization
def train_model(x):
    config = {
        'aggregation_resolution': 1.0,
        'config_build': {
            'batch_norm': True,
            'conv_activation': mapping['conv_activation'][int(x[:, 8])],
            'conv_dropout': float(x[:, 1]),
            'conv_filters': mapping['network_structure'][int(x[0, 7])]['conv_filters'],
            'conv_kernels': mapping['network_structure'][int(x[0, 7])]['conv_kernels'],
            'conv_strides': mapping['network_structure'][int(x[0, 7])]['conv_strides'],
            'conv_pooling': mapping['network_structure'][int(x[0, 7])]['conv_pooling'],
            'conv_kernel_regularizer': (mapping['regularizer'][int(x[:, 4])], float(x[:, 3])),
            'conv_recurrent_activation': 'hard_sigmoid',
            'conv_recurrent_regularizer': (mapping['regularizer'][int(x[:, 4])], float(x[:, 3])),
            'conv_recurrent_dropout': float(x[:, 2]),
            'dense_dropout': float(x[:, 0]),
            'dense_nodes': mapping['network_structure'][int(x[0, 7])]['dense_nodes'],
            'dense_activation': mapping['dense_activation'][int(x[:, 9])],
            'dense_kernel_regularizer': (mapping['regularizer'][int(x[:, 4])], float(x[:, 3])),
            'learning_rate': mapping['initial_lr'][int(x[:, 6])],
            'loss': 'mean_squared_error',
            'optimizer': mapping['optimizer'][int(x[:, 5])],
            'padding': 'same'
        },
        'config_fit': {
            'batch_size': 1,
            'epochs': EPOCHS,
            'lr_plateau': (0.5, 10, 0.00001),
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

    # hash the config that was passed down
    hashed_params = hashlib.md5(str(config).encode()).hexdigest()

    # build a model based on the above tunings
    print("> Training Model")
    print(f">> Parameters: {x}")
    print(f">> Config: {config}")

    # prepare onset dates and prediction timestamps
    onset_dates, onset_ts = ModelHelpers.load_onset_dates(version='v2', objective=True)
    prediction_ts = ModelHelpers.generate_prediction_ts(PREDICT_ON, YEARS)

    def filter_fun(df, year):
        # setup a filter function
        return ModelHelpers.filter_until(df, prediction_ts[year])

    # load the TRMM dataset
    print("> Loading Dataset")
    # load data for the pre-monsoon period (MAM)
    trmm_data = TRMM.load_dataset(
        YEARS,
        range(3, 6),
        aggregation_resolution=config['aggregation_resolution'],
        timestamp=True,
        invalidate=False,
        version='v2',
        filter_fun=filter_fun,
        bundled=False)

    era_temp, era_hum = ERA.load_dataset(
        YEARS,
        invalidate=False,
        timestamp=True,
        filter_fun=filter_fun,
        aggregation_resolution=config['aggregation_resolution'])

    # train test split
    print("> Train-Test Split")
    X_train, y_train, X_test, y_test, X_dev, y_dev, unstacked = ModelERAv2.train_test_split(
        [trmm_data, era_temp, era_hum],
        prediction_ts,
        onset_ts,
        years=config['years'],
        years_train=config['years_train'],
        years_test=config['years_test'],
        years_dev=config['years_dev'])

    # if the model was already trained, load it from cache
    print('>> Building a new model')

    # throw away the models in memory
    K.clear_session()

    # train a model based on the given config
    results = ModelHelpers.run_config(
        ModelERAv2,
        config,
        X_train,
        y_train,
        invalidate=True,
        validation_data=(X_dev, y_dev) if config['years_dev'] else None,
        cache_id=hashed_params,
        version=VERSION)

    model = results['model']

    # evaluate the latest model
    eval_latest_train = model.evaluate(X_train, y_train)
    print(model.predict(X_train, batch_size=1), y_train)
    print(f'Train (latest): {eval_latest_train}')

    eval_latest_dev = 'NO_DEV'
    if X_dev is not None:
        eval_latest_dev = model.evaluate(X_dev, y_dev)
        print(model.predict(X_dev, batch_size=1), y_dev)
        print(f'Dev (latest): {eval_latest_dev}')

    eval_latest_test = model.evaluate(X_test, y_test)
    print(model.predict(X_test, batch_size=1), y_test)
    print(f'Test (latest): {eval_latest_test}')

    # evaluate the best model
    best_path = f'00_CACHE/lstm_{VERSION}_{hashed_params}_best.h5'
    if os.path.isfile(best_path):
        best_model = load_model(best_path)

        eval_best_train = best_model.evaluate(X_train, y_train)
        print(best_model.predict(X_train, batch_size=1), y_train)
        print(f'Train (best): {eval_best_train}')

        eval_best_dev = 'NO_DEV'
        if X_dev is not None:
            eval_best_dev = best_model.evaluate(X_dev, y_dev)
            print(best_model.predict(X_dev, batch_size=1), y_dev)
            print(f'Dev (best): {eval_best_dev}')

        eval_best_test = best_model.evaluate(X_test, y_test)
        print(best_model.predict(X_test, batch_size=1), y_test)
        print(f'Test (best): {eval_best_test}')

        with open(f'03_EVALUATION/bayesian/{VERSION}_optimization_out.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(hashed_params), str(config), str(eval_latest_train), str(eval_best_train), str(eval_latest_dev), str(eval_best_dev), str(eval_latest_test), str(eval_best_test)])
    else:
        with open(f'03_EVALUATION/bayesian/{VERSION}_optimization_out.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(hashed_params), str(config), str(eval_latest_train), '-', str(eval_latest_dev), '-', str(eval_latest_test), '-'])

    # return MSE over train and dev set as a parameter to optimizer
    # TODO: should this be weighted in any way?
    # TODO: latest or best? or both?
    # TODO: we could overfit on the dev set by doing this?!
    # but this is the same as optimizing validation error manually...
    return eval_latest_train[1] + eval_latest_dev[1]


# create the optimizer
optimizer = GPyOpt.methods.BayesianOptimization(f=train_model, domain=bounds)

# specify the maximum number of iterations for the optimizer
optimizer.run_optimization(max_iter=100)
