import os
import pathlib
import numpy as np
import pandas as pd
import time
import pickle
import arrow as ar

from pymongo import MongoClient
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.callbacks import TensorBoard


class ModelHelpers:
    """ Basic helper methods for modeling """
    client = None
    db = None

    @staticmethod
    def normalize_channels(arr, standardize=True, dims=(0, 1), seperate=False, mean=None, std=None):
        """
        Normalize or standardize channels of a 4D tensor

        :arr: A numpy array with data to be standardized
        :standardize: Whether to use z-scores or [0, 1] normalization
        :dims: The dimensions to standardize over
        :separate: Whether the mean and std should be returned (bias prevention)
        :mean: The mean to be applied instead of calculated (bias prevention)
        :std: The std to be applied instead of calculated (bias prevention)
        """

        # see: https://stackoverflow.com/questions/42460217/how-to-normalize-a-4d-numpy-array
        # and: https://stackoverflow.com/questions/40956114/numpy-standardize-2d-subsets-of-a-4d-array

        # the channels should be standardized to zero mean and unit variance seperately
        if standardize:
            # if the mean and std over the training set are passed in, use these
            # to prevent any optimistical bias
            if mean is not None and std is not None:
                return (arr - mean) / std

            # calculate the mean and std over the specified dimensions
            mean = np.mean(arr, axis=dims, keepdims=True)
            std = np.std(arr, axis=dims, keepdims=True)

            # check whether the mean and std should additionally be returned
            # normally only done for the training set
            if seperate:
                return (arr - mean) / std, mean, std

            return (arr - mean) / std

        # the channels should be normalized to the range [0, 1] sepeartely
        arr_min = arr.min(axis=dims, keepdims=True)
        arr_max = arr.max(axis=dims, keepdims=True)

        return (arr - arr_min) / (arr_max - arr_min)

    @staticmethod
    def unstack_year(df):
        """ Unstack a single year and return an unstacked sequence of grids """

        return np.array([df.iloc[:, i].unstack().values for i in range(df.shape[1])])

    @staticmethod
    def unstack_all(dataframes, years):
        """ Unstack all years and return the resulting dict """

        result = {}

        for year in years:
            result[year] = np.stack([ModelHelpers.unstack_year(df[year]) for df in dataframes], axis=-1)

        return result

    @staticmethod
    def reshape_years(arr, num_channels):
        """ Reshape all passed down years appropriately and stack them into a single numpy array """

        return np.array(list(map(lambda year: year.reshape((year.shape[0], year.shape[1], year.shape[2], num_channels)), arr)))

    @staticmethod
    def stack_outcomes(outcomes, years, augmented=False):
        """ Stack the outcomes into a single array """

        if augmented:
            return np.concatenate([outcomes[year] for year in years])

        return [outcomes[year] for year in years]

    @staticmethod
    def load_onset_dates(data_path='../00_DATA/',
                         version='v1',
                         objective=False):
        """
        Load and preprocess the monsoon onset dates from csv

        :data_path: Path to the data directory
        :version: The version of onset dates to use (v1/v2)
        :objective: Whether to read objective or IMD dates for v2

        :return: A dataframe, a timestamps series
        """

        current_path = pathlib.Path(__file__).resolve().parent.parent.parent

        # read the onset dates from the csv
        df = pd.read_csv(current_path / f'00_DATA/onset_dates_{version}.csv', sep=';')

        # convert the date strings to datetime objects
        if version == 'v1':
            df['date'] = df['onset'].apply(lambda r: ar.get(r, 'D MMM YYYY'))
            df = df.drop('onset', axis=1)

        elif version == 'v2':
            df = df.set_index('year')
            df['date'] = df['objective' if objective else 'imd'].apply(lambda r: ar.get(r, 'D MMM YYYY'))
            df = df.drop(['imd', 'objective'], axis=1)

        # extract only dates after 01.01.1970 (first possible epoch)
        valid_dates = df['date'][df['date'] > ar.get('1970-01-01')]

        # calculate epoch timestamps from datetime
        df['timestamp'] = valid_dates.apply(lambda arr: int(arr.datetime.timestamp()))

        # create a new series containing timestamps indexed by year
        timestamps = pd.Series(df['timestamp'].dropna(), dtype='int64')
        timestamps.index = valid_dates.apply(lambda arr: arr.datetime.year)

        return df, timestamps

    @staticmethod
    def filter_until(df, epoch):
        """
        Filter a dataframe by epochs up to the given timestamp

        :df: Dataframe
        :epoch: The epoch timestamp of the last day to be included

        :return: A filtered dataframe
        """

        return df.loc[:, :epoch]

    @staticmethod
    def filter_between(df, after_ts, before_ts):
        """
        Filter a dataframe by epochs in between two timestamps

        :df: The dataframe to be filtered
        :after_ts: The timestamp after which data should be extracted
        :before_ts: The timestamp until which data should be extracted
        """

        return df.loc[:, after_ts:before_ts]

    @staticmethod
    def generate_prediction_ts(predict_on, years, onset_dates=None, sequence_length=None, sequence_offset=0, fake_sequence=False, example_length=60):
        """
        Generate epochs for the days to predict on as specified

        :param predict_on: Date template like '{}-05-11'
        :param years: The years to generate epochs for
        :example_length: The sequence length of training examples (i.e., how many days are in a training example)
        :sequence_length: The length of the offset sequence (i.e., how many different sequences are generated per year)
        :sequence_offset: The distance of the sequence to the onset
        :fake_sequence: If only one prediction should be performed on the test set, we return a "fake" sequence containing only the specified offset
        :example_length: The length of a single training example in days

        :return: A dictionary of epochs indexed by year
        """

        # if a sequence of prediction timestamps should be generated
        if sequence_length is not None:
            prediction_ts = {}

            # for each year calculate the border timestamps
            # only the data inside the borsers will be fed to the network
            for year in years:
                onset_date = onset_dates.loc[year]['date']
                prediction_ts[year] = [
                    (
                        onset_date.shift(days=-sequence_offset-i).datetime.timestamp(), onset_date.shift(days=-sequence_offset-example_length-i).datetime.timestamp())
                    for i in range(0, sequence_length + 1)]

            return prediction_ts

        # return a "fake" sequence for the test years in E4
        if fake_sequence:
            prediction_ts = {}
            for year in years:
                prediction_date = ar.get(predict_on.format(year), 'YYYY-MM-DD')
                prediction_ts[year] = [(prediction_date.datetime.timestamp(), prediction_date.shift(days=-example_length).datetime.timestamp())]
            return prediction_ts

        # if a single prediction timestamp should be generated for each year
        return {
            year: ar.get(predict_on.format(year), 'YYYY-MM-DD').datetime.timestamp() for year in years
        }

    @staticmethod
    def generate_outcomes(prediction_ts, onset_ts, years, numerical=False, sequence=False, true_offset=None):
        """
        Calculate the difference in days between prediction date and onset date

        :param prediction_ts: Epoch timestamp of the prediction
        :param onset_ts: Epoch timestamp of monsoon onset
        :param years: Years to generate outcomes for
        :numerical: Whether to apply one-hot encoding to the results
        :sequence: Whether to generate a sequence of outcomes (for E4)
        :true_offset: Whether to generate binary outcomes (1 only for the true offset)

        :return: Dictionary of outcomes indexed by year
        """

        #if we get a list of prediction timestamps, return a list of outcomes
        if sequence:
            outcomes = {}

            if true_offset is not None:
                for year in years:
                    # add 1 for the example at the true offset, 0 for all other examples
                    outcomes[year] = [
                        (1 if int((onset_ts[year] - prediction_ts[year][i][0]) / 86400) == true_offset else 0)
                        for i in range(0, len(prediction_ts[year]))]

                    # add the stratified example outcomes in the end (such that the number of 1 and 0 in the dataset is equal)
                    outcomes[year] = outcomes[year] + [1] * (len(prediction_ts[year]) - 2)
            else:
                for year in years:
                    # just calculate the difference between onset and prediction for each year
                    outcomes[year] = [
                        int((onset_ts[year] - prediction_ts[year][i][0]) / 86400)
                        for i in range(0, len(prediction_ts[year]))]

            return outcomes

        if numerical:
            return {
                year: int((onset_ts[year] - prediction_ts[year]) / 86400)
                for year in years
            }
        else:
            return {
                year: to_categorical(
                    int((onset_ts[year] - prediction_ts[year]) / 86400), 40)
                for year in years
            }

    @staticmethod
    def run_config(model_cls, config, X_train, y_train, version, invalidate=False, evaluate=None, validation_data=None, cache_id=None):
        # instantiate a new model
        print('>> Instantiating model')
        model_instance = model_cls(version=version, cache_id=cache_id)

        # build the model
        print('>> Building model')
        model_built = model_instance.build(X_train, **config['config_build'])

        print('>> Fitting model')
        # fit the model
        model_fitted, history = model_instance.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            invalidate=invalidate,
            tensorboard_id=f'{version}{cache_id}',
            **config['config_fit'])

        return model_fitted, config, history


    @staticmethod
    def run_config_bayesian(model_cls,
                   config,
                   X_train,
                   y_train,
                   version,
                   invalidate=False,
                   evaluate=None,
                   validation_data=None,
                   verbose=1,
                   numerical=False,
                   cache_id=None):

        print(f'> Training config {config!s}')

        if evaluate is not None:
            print(f'>> Eager evaluation: {evaluate}')

        model = model_cls()

        model.build(
            X_train,
            y_train,
            validation_data=validation_data,
            verbose=verbose,
            numerical=numerical,
            invalidate=invalidate,
            evaluate=evaluate,
            cache_id=cache_id,
            version=version,
            **config)

        return model.model

    @staticmethod
    def prepare_dataset_year(year, datasets, prediction_ts, true_offset=None):
        """
        Prepare a year of one or multiple datasets for processing by the E4 model architecture

        :year: The year to process from the datasets
        :datasets: An array of datasets to process
        :prediction_ts: A dictionary with sequences of prediction timestamps for each year
        :true_offset: Binary prediction (legacy)
        """

        channels = []

        # prepare each dataset seperately
        for dataset in datasets:
            # extract the dataframe for the current year
            df = dataset[year]

            # prepare an array for augmented data
            filtered = []
            stratified = None
            for index, (before_ts, after_ts) in enumerate(prediction_ts[year]):
                # for each sequence of the same year, filter the dataset
                filtered_df = ModelHelpers.filter_between(df, after_ts, before_ts)

                # unstack the filtered dataframe
                unstacked_df = np.array([filtered_df.iloc[:, i].unstack().values for i in range(filtered_df.shape[1])])

                # if we are currently processing the single true element of the sequence
                # we need to duplicate it to stratify the dataset
                # such that 0 and 1 are represented equally often

                filtered.append(unstacked_df)

                if index == true_offset:
                    stratified = unstacked_df

                # print('Unstacked', unstacked_df.shape)

            # stack all the different sequences of the same year
            # stacked = np.stack(filtered)

            if true_offset is not None:
                    filtered = filtered + [stratified for _ in range(0, len(prediction_ts[year]) - 2)]

            channels.append(filtered)

            # print('Filtered', stacked.shape)

        # stack the different datasets in the channel dimension
        with_channels = np.stack(channels, axis=-1)

        print('Processed', year, with_channels.shape)

        return with_channels

    @staticmethod
    def prepare_datasets(years, datasets, prediction_ts, true_offset=None):
        """
        Prepare one or multiple datasets for processing by the E4 architecture

        :years: The years available in the datasets
        :datasets: An array of datasets to process
        :prediction_ts: A dictionary with sequences of prediction timestamps for each year
        :true_offset: Binary prediction (legacy)
        """

        # pass through each year in the datasets
        result = None
        for year in years:
            year_result = ModelHelpers.prepare_dataset_year(year, datasets, prediction_ts, true_offset=true_offset)

            if result is not None:
                result = np.concatenate([result, year_result])
            else:
                result = year_result

        return result


    """
    LEGACY methods are not used in the latest models
    """

    @staticmethod
    def perform_split_v1(data, years, outcomes, scaling):
        """
        LEGACY

        Perform a split of datasets into train/dev/test sets
        => Split such that each grid cell represents a separate row
        => Prepend the grid location to the front of the time series

        :param data:
        :param years:
        :param outcomes:
        :param scaling:

        :return:
        """
        x = list()
        y = list()

        for year in years:
            values = data[year].reset_index().values
            # values = data[year].values

            # scale everything except the coordinates
            if scaling:
                scaler = preprocessing.StandardScaler()
                values[2:] = scaler.fit_transform(values[2:])

            x.append(np.asarray(values))
            y.append(np.full((values.shape[0], 40), outcomes[year]))

        x = np.concatenate(x)

        x = np.hstack(x).reshape(len(x), 1, 74)
        y = np.concatenate(y)

        return x, y

    @staticmethod
    def perform_split_v2(data, years, outcomes, scaling):
        """
        LEGACY

        Perform a split of datasets into train/dev/test sets
        => Split such that each row contains the entire grid encoded as features in the feature vector
        """

        x = list()
        y = list()

        for year in years:
            values = data[year].values

            if scaling:
                scaler = preprocessing.StandardScaler()
                values = scaler.fit_transform(values)

            # transpose such that coordinates are columns and timesteps are rows
            transposed = np.asarray(values).T

            x.append(transposed)
            y.append(outcomes[year])

        x = np.asarray(x)
        y = np.concatenate(y)

        return x, y

    @staticmethod
    def train_test_split(data,
                         years_train,
                         years_test,
                         prediction_ts,
                         onset_ts,
                         scaling=False,
                         split_fun=None):
        """
        LEGACY

        Split the dataframe into train/dev/test sets based on years

        :param data: Input dictionary with dataframes indexed by years
        :param years_train: The years to be contained in the training set
        :param years_test: The years to be contained in the test set
        :param prediction_ts: The directory of prediction timestamps
        :param onset_ts: The directory of onset timestamps
        :param scaling: Whether the input data should be normalized

        :return: Training, test and dev set
        """

        if split_fun is None:
            split_fun = ModelHelpers.perform_split_v1

        # calculate the differences in days between prediction date and onset date
        # convert the numerical difference into categories
        outcomes = ModelHelpers.generate_outcomes(
            prediction_ts, onset_ts,
            list(years_train) + list(years_test))

        x_train, y_train = split_fun(data, years_train[:-2], outcomes, scaling)
        x_dev, y_dev = split_fun(data, years_train[-2:], outcomes, scaling)
        x_test, y_test = split_fun(data, years_test, outcomes, scaling)

        return x_train, x_dev, x_test, y_train, y_dev, y_test

    @staticmethod
    def run_configs(model_cls,
                    configs,
                    X_train,
                    y_train,
                    version,
                    invalidate=False,
                    evaluate=None,
                    validation_data=None,
                    verbose=1,
                    numerical=False):
        """
        LEGACY

        Run multiple configs sequentially
        """

        models = []

        for index, config in enumerate(configs):
            print(f'> Training config {config!s}')

            if evaluate is not None:
                print(f'>> Eager evaluation: {evaluate}')

            model = model_cls()

            model.build(
                X_train,
                y_train,
                **config,
                validation_data=validation_data,
                verbose=verbose,
                numerical=numerical,
                invalidate=invalidate,
                evaluate=evaluate,
                cache_id=index,
                version=version)

            models.append(dict(config=config, model=model))

        return models

    @classmethod
    def connect_mongo(cls):
        """
        LEGACY

        Connect to the result database
        """

        if cls.client is not None:
            return cls.client, cls.db

        client = MongoClient('mongodb://rschlaefli:3wP0cS7dlSyd@ds131137.mlab.com:31137')
        db = client['nn-results']

        cls.client = client
        cls.db = db

        return client, db

    @staticmethod
    def normalize(column):
        """
        LEGACY

        Normalize a dataframe column (MinMax)
        """

        return (column - column.min()) / (column.max() - column.min())
        # standardized = (column - column.mean()) / column.std()
        # return standardized + abs(standardized.min())
