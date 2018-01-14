import os
import pathlib
import numpy as np
import pandas as pd
import time
import pickle

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping, TensorBoard, EarlyStopping, CSVLogger
from keras.utils import to_categorical

from ModelHelpers import ModelHelpers

# limit gpu resource usage of tensorflow
# see https://github.com/keras-team/keras/issues/1538
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class ModelTRMMv2:
    """ Version 2 of LSTM model """

    def __init__(self):
        self.cache_path = None
        self.model = None
        self.history = None
        self.results = None
        self.log_path = None
        self.log_name = None

    @staticmethod
    def to_classes(data):
        """ Perform argmax over columns """

        return np.argmax(data, axis=1)

    @staticmethod
    def calculate_diff(y, pred, thresh=7):
        """ calculate the difference in days between the prediction and the actual onset """

        diff = np.abs(y - pred)

        return diff, np.where(diff < thresh)[0], np.where(diff >= thresh)[0]

    @staticmethod
    def calculate_results(mapping, diff, diff_lt, normalize=True):
        """ Calculate results """

        lats = []
        lons = []
        text = []
        val_std = []

        for index in diff_lt:
            ix = mapping.index[index % mapping.shape[0]]
            lat = mapping.loc[ix, 'latitude']
            lon = mapping.loc[ix, 'longitude']
            lats.append(lat)
            lons.append(lon)
            text.append(f'({lat}, {lon})={diff[index]}')

            diff_ix = diff[index]
            val_std.append(1.5 / diff_ix if diff_ix > 0 else 2)

        result = pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'text': text,
            'val_std': val_std
        })

        if normalize:
            result['val_std'] = ModelHelpers.normalize(result['val_std'])

        return result

    def build(self,
              X_train,
              y_train,
              validation_data,
              dropout=None,
              dropout_recurrent=None,
              epochs=200,
              input_shape=(1, 74),
              num_classes=40,
              batch_size=169,
              nodes_lstm=128,
              nodes_dense=256,
              optimizer='rmsprop',
              verbose=1,
              invalidate=False,
              patience=0,
              tensorboard=False,
              version='T2',
              numerical=None,
              evaluate=False,
              cache_id=None):
        """
        Build a stateless LSTM model

        :param X_train: Training features
        :param y_train: Training outcomes
        :param validation_data: Validation data for X and y
        :param dropout:
        :param dropout_recurrent:
        :param epochs: How many epochs to train for
        :param seq_length: H
        ow long the sequences should be
        :param num_classes: How many classes (days) should be predicted
        :param batch_size: The batch size of training
        :param nodes_lstm:
        :param nodes_dense:
        :param optimizer:
        :param verbose:
        :param invalidate:
        :param patience:
        :param tensorboard:

        :return: The fitted model
        :return: The history of training the model
        """

        current_path = pathlib.Path(__file__).resolve().parent.parent.parent

        # define a path for the model to be cached to
        self.cache_path = current_path / '00_CACHE'
        self.cache_name = f'lstm_{version}_{cache_id}_epochs-{epochs}_batch-{batch_size}_lstm-{nodes_lstm}_dense-{nodes_dense}_optimizer-{optimizer!s}_dropout-{dropout}_dropoutR-{dropout_recurrent}_patience-{patience}'
        self.log_path = current_path.parent / '03_EVALUATION/histories'
        self.log_name = f'lstm_{version}_{cache_id}'

        # try to read the model and its history from cache
        if not invalidate and os.path.isfile(
                self.cache_path / f'{self.cache_name}.h5') and os.path.isfile(
                    self.cache_path / f'{self.cache_name}.history'):
            model = load_model(self.cache_path / f'{self.cache_name}.h5')

            history = None
            with open(self.cache_path / f'{self.cache_name}.history',
                      'rb') as file:
                history = pickle.load(file)

            self.model = model
            self.history = history
            return

        # if there is no cache available, initialize a new sequential model
        model = Sequential()

        # add an LSTM layer for initial input transformation
        model.add(
            LSTM(
                nodes_lstm,
                input_shape=input_shape,
                dropout=dropout_recurrent,
                recurrent_dropout=dropout_recurrent,
                return_sequences=False))

        # add several dense layers
        # optionally add dropout after each layer
        model.add(Dense(nodes_dense, activation='relu'))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(nodes_dense * 2, activation='relu'))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(nodes_dense, activation='relu'))
        if dropout:
            model.add(Dropout(dropout))

        # add a softmax layer for classification
        model.add(Dense(num_classes, activation='softmax'))

        # compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=[top_k_categorical_accuracy])

        # print an overview about the model
        print(model.summary())
        print('\n')

        # if early stopping patience is set, configure the callback
        callbacks = []
        if patience > 0:
            callbacks.append(
                EarlyStopping(monitor='val_loss', patience=patience))
        if tensorboard:
            callbacks.append(
                TensorBoard(log_dir=current_path / (
                    f'00_LOGS/{time.time()!s}')))

        log_path = str(self.log_path / f'{self.log_name}.csv')
        callbacks.append(CSVLogger(log_path))

        # fit the model to the training data
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=validation_data,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks)

        # save the model and its history to cache
        model.save(self.cache_path / f'{self.cache_name}.h5')
        with open(self.cache_path / f'{self.cache_name}.history',
                  'wb') as file:
            pickle.dump(history.history, file)

        self.model = model
        self.history = history.history

    def predict_outcomes(self, X, y):
        """ Predict outcomes """

        pred = self.model.predict(X)
        pred_cls = ModelTRMMv2.to_classes(pred)
        y_cls = ModelTRMMv2.to_classes(y)

        return pred, pred_cls, y_cls

    def evaluate(self,
                 mapping,
                 X_train,
                 X_dev,
                 X_test,
                 y_train,
                 y_dev,
                 y_test,
                 invalidate=False):
        """ Evaluate the performance of the model """

        # try to read the results from cache
        if not invalidate and os.path.isfile(
                self.cache_path / f'{self.cache_name}.results'):
            with open(self.cache_path / f'{self.cache_name}.results',
                      'rb') as file:
                self.results = pickle.load(file)

            print('> Evaluation finished.')

            return self.results

        if self.results is not None and not invalidate:
            print('> Evaluation finished.')

            return self.results

        print('> Evaluating model...', end='')

        # predict the training set (this should be good!)
        pred_train, pred_train_cls, y_train_cls = self.predict_outcomes(
            X_train, y_train)

        # predict the validation set
        pred_dev, pred_dev_cls, y_dev_cls = self.predict_outcomes(X_dev, y_dev)
        diff_dev, diff_dev_lt, diff_dev_gte = ModelTRMMv2.calculate_diff(
            y_dev_cls, pred_dev_cls)

        # predict the test set
        pred_test, pred_test_cls, y_test_cls = self.predict_outcomes(
            X_test, y_test)
        diff_test, diff_test_lt, diff_test_gte = ModelTRMMv2.calculate_diff(
            y_test_cls, pred_test_cls)

        # strict commons calculation (in all 4 years)
        modulized = [x % 169 for x in list(diff_dev_lt) + list(diff_test_lt)]
        diff_commons = set([x for x in modulized if modulized.count(x) == 4])

        results_dev = ModelTRMMv2.calculate_results(mapping, diff_dev,
                                                    diff_dev_lt)
        results_test = ModelTRMMv2.calculate_results(mapping, diff_test,
                                                     diff_test_lt)
        results_commons = ModelTRMMv2.calculate_results(
            mapping, diff_dev, diff_commons, normalize=False)

        results = dict(
            loss=pd.Series(self.history['loss']),
            val_loss=pd.Series(self.history['val_loss']),
            pred_dev=pred_dev,
            pred_test=pred_test,
            diff_dev=diff_dev,
            diff_test=diff_test,
            diff_commons=diff_commons,
            diff_dev_lt=diff_dev_lt,
            # calc1=np.mean(np.abs(np.argmax(pred_train, axis=1) - np.argmax(y_train, axis=1))),
            # calc2=np.mean(np.argmax(pred_dev[:169], axis=1)),
            # calc3=np.mean(np.argmax(pred_dev[169:], axis=1)),
            results_dev=results_dev,
            results_test=results_test,
            results_commons=results_commons)

        self.results = results

        with open(self.cache_path / f'{self.cache_name}.results',
                  'wb') as file:
            pickle.dump(results, file)

        print(' Evaluation finished.')

        return results
