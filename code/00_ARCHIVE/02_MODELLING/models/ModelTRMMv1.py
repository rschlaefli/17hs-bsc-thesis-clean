import numpy as np

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

# limit gpu resource usage of tensorflow
# see https://github.com/keras-team/keras/issues/1538
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class ModelTRMMv1:
    @staticmethod
    def build_stateful(X_train,
                       y_train,
                       validation_data,
                       epochs=200,
                       seq_length=74,
                       num_classes=40,
                       batch_size=169):
        model = Sequential()

        model.add(
            LSTM(
                64,
                batch_input_shape=(batch_size, 1, seq_length),
                dropout=0.3,
                recurrent_dropout=0.3,
                return_sequences=False,
                stateful=True))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=[top_k_categorical_accuracy])
        print(model.summary())

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=validation_data,
            batch_size=batch_size,
            shuffle=False)

        return model

    @staticmethod
    def build_dense(X_train, y_train, epochs=500):
        """
        Train a dense neural net

        :param X_train:
        :param y_train:
        :param epochs:

        :return: A fitted model
        """

        model = Sequential()

        model.add(Dense(200, input_shape=(130, )))
        model.add(Dense(130, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='relu'))

        # compile the model
        model.compile(
            loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        # fit the model
        model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)

        return model

    @staticmethod
    def build(X_train, y_train, epochs=500):
        """
        Train an LSTM

        :param X_train:
        :param y_train:
        :param epochs:

        :return: A fitted model
        """

        model = Sequential()

        model.add(LSTM(50, input_shape=(1, 130)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='relu'))

        # compile the model
        model.compile(
            loss='mean_absolute_error',
            optimizer='rmsprop',
            metrics=['accuracy'])
        print(model.summary())

        # modify the shape of the inputs to work with a keras lstm
        X_train_reshaped = np.hstack(X_train).reshape(len(X_train), 1, 130)
        y_train_reshaped = np.hstack(np.asarray(y_train)).reshape(
            len(y_train), 1)

        # fit the model
        model.fit(
            X_train_reshaped,
            y_train_reshaped,
            epochs=epochs,
            validation_split=0.1,
            batch_size=150)

        return model
