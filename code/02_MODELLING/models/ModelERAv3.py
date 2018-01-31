import numpy as np
from itertools import chain

from ModelHelpers import ModelHelpers
from ModelBASEv2 import ModelBASEv2


class ModelERAv3(ModelBASEv2):
    """ Version 3 of ERA-based LSTM model """

    def __init__(self, version='E3', cache_id=None):
        super().__init__(version, cache_id=cache_id)

    @staticmethod
    def train_test_split(datasets,
                         prediction_ts,
                         # prediction_ts_test,
                         onset_ts,
                         years=range(1979, 2018),
                         years_train=range(1979, 2010),
                         years_dev=range(2010, 2013),
                         years_test=range(2013, 2018)):
        """
        Prepare data to be in a digestible format for the model

        :datasets: List of datasets to use as features (e.g., t and r dataframes)
        :prediction_ts: Sequences of prediction timestamps to use
        :onset_ts: Onset dates to use for outcome calculations
        :years: The overall years of all sets
        :years_train: The years to use for the training set
        :years_dev: The years to optionally use for the validation set
        :years_test: The years to use for the test set

        :return:
        """

        # generate outcomes
        outcomes = ModelHelpers.generate_outcomes(prediction_ts, onset_ts, years, numerical=True, sequence=True)
        # outcomes_test = ModelHelpers.generate_outcomes(prediction_ts_test, onset_ts, years_test, numerical=True, sequence=True)
        # print(outcomes_test)

        # generate training data
        X_train = ModelHelpers.prepare_datasets(years_train, datasets, prediction_ts)
        y_train = ModelHelpers.stack_outcomes(outcomes, years_train, augmented=True)
        print(X_train[0][0][0])
        print('> X_train', X_train.shape, 'y_train', y_train.shape)

        # standardize the training set, extracting mean and std
        X_train, X_mean, X_std = ModelHelpers.normalize_channels(X_train, seperate=True)

        # generate test data
        X_test = ModelHelpers.prepare_datasets(years_test, datasets, prediction_ts)
        y_test = ModelHelpers.stack_outcomes(outcomes, years_test, augmented=True)
        print(X_test)
        print('> X_test', X_test.shape, 'y_test', y_test.shape)

        # standardize the test set using mean and std from the training set
        X_test = ModelHelpers.normalize_channels(X_test, mean=X_mean, std=X_std)

        if years_dev:
            X_dev = ModelHelpers.prepare_datasets(years_dev, datasets, prediction_ts)
            y_dev = ModelHelpers.stack_outcomes(outcomes, years_dev, augmented=True)
            print(X_dev.shape)
            print('> X_dev', X_dev.shape, 'y_dev', y_dev.shape)

            # standardize the dev set using mean and std from the training set
            X_dev = ModelHelpers.normalize_channels(X_dev, mean=X_mean, std=X_std)

            return X_train, y_train, X_test, y_test, X_dev, y_dev

        return X_train, y_train, X_test, y_test, None, None
