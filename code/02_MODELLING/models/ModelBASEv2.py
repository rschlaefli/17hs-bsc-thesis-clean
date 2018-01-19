import os
import pathlib
import time
import pickle
import re
import hashlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from keras.models import Model, Sequential, load_model
from keras.layers import BatchNormalization, Dropout, LSTM, Dense, ConvLSTM2D, Flatten, MaxPooling2D, MaxPooling3D, TimeDistributed, Conv2D, LSTM, Input, Conv3D
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, EarlyStopping, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.regularizers import l1, l2, l1_l2

from ModelHelpers import ModelHelpers

# limit gpu resource usage of tensorflow
# see https://github.com/keras-team/keras/issues/1538
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class ModelBASEv2:
    """ Base model, extend with specific models for each dataset """

    def __init__(self, version, cache_id=None):
        self.cache_id = cache_id
        self.cache_path = None
        self.cache_name = None
        self.log_path = None
        self.log_name = None
        self.model = None
        self.history = None
        self.version = version

        # find the root folder of the repository
        self.current_path = pathlib.Path(__file__).resolve().parent.parent.parent

        if cache_id is not None:
            # define a path for the model to be cached to
            # => use md5 or similar for cache path..
            # define a path for the model to be cached to
            # use the given cache id as a reference (array index)
            self.cache_path = self.current_path / '00_CACHE'
            self.cache_name = f'lstm_{version}_{cache_id}'
            self.log_path = self.current_path / '03_EVALUATION/histories'
            self.log_name = f'lstm_{version}_{cache_id}'


    def load_model(self, which='best', overwrite=False):
        """
        Try to load a model from cache
        """

        if self.cache_id is not None:

            file_path = self.cache_path / (self.cache_name + f'_{which}.h5')
            if os.path.isfile(file_path):
                model = load_model(file_path)

                if overwrite:
                    self.model = model

                return model

        return None


    def build(self,
              X_train,
              optimizer='rmsprop',
              learning_rate=0.001,
              loss='mean_squared_error',
              batch_norm=True,
              padding='same',
              conv_activation='tanh',
              conv_dropout=[0, 0, 0],
              conv_filters=[30, 30, 30],
              conv_kernels=[3, 3, 3],
              conv_strides=[1, 1, 1],
              conv_pooling=[0, 0, 0, 3],
              conv_kernel_regularizer=('L2', 0.02),
              conv_recurrent_regularizer=('L2', 0.02),
              dense_dropout=[0.0, 0.0, 0.0],
              dense_nodes=[1024, 512, 256],
              dense_activation='relu',
              dense_activation_final=None,
              dense_kernel_regularizer=('L2', 0.02),
              lstm_dropout=[0.0, 0.0, 0.0],
              lstm_filters=[30, 30, 30],
              lstm_kernels=[3, 3, 3],
              lstm_strides=[1, 1, 1],
              lstm_activation='tanh',
              lstm_recurrent_activation='hard_sigmoid',
              lstm_recurrent_dropout=[0.0, 0.0, 0.0],
              with_sequences=False):

        """
        Build a stateless LSTM model

        :X_train: Training features for input_shape calculation
        :dropout: How much dropout to use after dense layers
        :dropout_recurrent: How much recurrent dropout to use in ConvLSTM2D
        :dropout_conv: How much dropout to use after convolutional layers
        :batch_norm: Whether to use batch normalization
        :conv_activation: The activation to use in the convolutional layers
        :conv_filters: The number of filters for all convolutional layers as a list
        :conv_kernels: The dimensions of the kernels for all convolutional layers as a list
        :conv_pooling: Dimensions of the max pooling layers => one after each conv layer, final one before flatten
        :recurrent_activation: The activation for the LSTM recurrent part of ConvLSTM2D
        :padding: Whether to apply padding or not
        :dense_nodes: The number of dense nodes for all dense layers as a list
        :dense_activation: The activation to use in all dense nodes except the final one
        :conv_kernel_regularizer: Regularizer function applied to the kernel weights matrix of ConvLSTM2D layers.
        :conv_recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix of ConvLSTM2D layers.
        :dense_kernel_regularizer: Regularizer function applied to the kernel weights matrix of Dense layers.

        :return: The fitted model
        :return: The history of training the model
        """

        # prepare regularizers for later use in the model
        regularizers = {'L2': l2, 'L1': l1, 'L1_L2': l1_l2}
        conv_regularize_params = dict()
        dense_regularize_params = dict()
        if (conv_kernel_regularizer is not None and conv_kernel_regularizer[0] is not None):
            reg = regularizers[conv_kernel_regularizer[0]]
            conv_regularize_params['kernel_regularizer'] = reg(conv_kernel_regularizer[1])
        if (conv_recurrent_regularizer is not None and conv_recurrent_regularizer[0] is not None):
            reg = regularizers[conv_recurrent_regularizer[0]]
            conv_regularize_params['recurrent_regularizer'] = reg(conv_recurrent_regularizer[1])
        if (dense_kernel_regularizer is not None and dense_kernel_regularizer[0] is not None):
            reg = regularizers[dense_kernel_regularizer[0]]
            dense_regularize_params['kernel_regularizer'] = reg(dense_kernel_regularizer[1])

        # --- INPUT LAYER ---
        # calculate the wanted input shape
        input_shape = (
            X_train.shape[1],
            X_train.shape[2],
            X_train.shape[3],
            X_train.shape[4])

        print(f'> input_shape: {input_shape!s}')

        # setup an input layer based on the above
        main_input = Input(shape=input_shape, name='main_input')
        x = None

        # --- CONVOLUTIONAL LAYERS ---
        # go through all convolutional layers
        for (index, filters) in enumerate(conv_filters):
            last_layer = index == len(conv_filters) - 1

            # prepare parameters that are common for all convolutional layers
            conv_params = {
                'filters': filters,
                'kernel_size': (conv_kernels[index], conv_kernels[index]),
                'strides': (conv_strides[index], conv_strides[index]),
                'activation': conv_activation,
                'padding': padding
            }

            # add a ConvLSTM2D layer
            x = TimeDistributed(
                Conv2D(**conv_params),
                name=f'pre-conv-{index}'
            )(main_input if index == 0 else x)

            if conv_dropout[index] > 0:
                x = TimeDistributed(
                    Dropout(conv_dropout[index]),
                    name=f'pre-conv-{index}-dropout'
                )(x)

            # add a pooling layer if configured
            if conv_pooling[index] > 0:
                # pool in 2D for the last layer as we don't have return_sequences enabled anymore
                x = TimeDistributed(
                    MaxPooling2D(
                        pool_size=(conv_pooling[index], conv_pooling[index])),
                    name=f'pre-conv-pool-{index}'
                )(x)

            # add batch normalization
            if batch_norm:
                x = BatchNormalization(
                    name=f'pre-conv-norm-{index}'
                )(x)

        # add max pooling before flattening to reduce the dimensionality
        if conv_pooling[len(conv_filters)] > 0:
            if x is None:
                x = main_input

            x = TimeDistributed(
                MaxPooling2D(
                    pool_size=(
                        conv_pooling[len(conv_filters)],
                        conv_pooling[len(conv_filters)]),
                    padding=padding),
                name=f'pre-lstm-pool'
            )(x)

        # --- LSTM LAYERS ---
        # go through all convolutional lstm layers
        for (index, filters) in enumerate(lstm_filters):
            if x is None:
                x = main_input

            last_layer = index == len(lstm_filters) - 1

            lstm_params = {
                'filters': filters,
                'kernel_size': (lstm_kernels[index], lstm_kernels[index]),
                'strides': (lstm_strides[index], lstm_strides[index]),
                'activation': lstm_activation,
                'recurrent_activation': lstm_recurrent_activation,
                'padding': padding,
                'dropout': lstm_dropout[index],
                'recurrent_dropout': lstm_recurrent_dropout[index]
            }

            x = ConvLSTM2D(
                **lstm_params,
                **conv_regularize_params,
                return_sequences=with_sequences or not last_layer,
                name=f'lstm-{index}'
            )(x)

            if batch_norm:
                x = BatchNormalization(
                    name=f'lstm-{index}-norm'
                )(x)

        # optional 3D convolutions before flattening
        # replaces return_sequences=false in last ConvLSTM2D
        if with_sequences:
            x = MaxPooling3D(pool_size=(3, 1, 1))(x)
            x = Conv3D(30, kernel_size=(3, 3, 3))(x)
            x = Conv3D(30, kernel_size=(3, 3, 3))(x)
            x = Conv3D(30, kernel_size=(3, 3, 3))(x)

        # --- DENSE LAYERS ---
        # flatten to make data digestible for dense layers
        x = Flatten(
            name='lstm-flatten'
        )(x)

        # go through all passed dense layers
        for index, dense in enumerate(dense_nodes):
            # add a new dense layer
            x = Dense(
                dense,
                **dense_regularize_params,
                activation=dense_activation,
                name=f'dense-{index}'
            )(x)

            # add batch normalization
            if batch_norm:
                x = BatchNormalization(
                    name=f'dense-norm-{index}'
                )(x)

            # add dropout
            if dense_dropout[index] > 0:
                x = Dropout(
                    dense_dropout[index],
                    name=f'dense-dropout-{index}'
                )(x)

        # final dense layer for numerical prediction
        main_output = Dense(1, name='main_output', activation=dense_activation_final)(x)

        # --- MODEL CREATION ---
        model = Model(inputs=main_input, outputs=main_output)

        # prepare optimizer
        if optimizer == 'rmsprop':
            optimizer = RMSprop(lr=learning_rate if learning_rate else 0.001)
        elif optimizer == 'adam':
            optimizer = Adam(lr=learning_rate if learning_rate else 0.001)
        elif optimizer == 'sgd':
            optimizer = SGD(lr=learning_rate if learning_rate else 0.01)

        # compile the model
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['mean_squared_error', 'mean_absolute_error'])

        self.model = model

        return model


    def fit(self, X_train, y_train, epochs=50, batch_size=1, validation_data=None, validation_split=0.1, invalidate=False, patience=0, tensorboard=False, lr_plateau=(0.5, 5, 0.0001), tensorboard_id=None):
        """
        Fit a stateless LSTM model

        :epochs: The number of epochs to train for
        """

        # load the model from cache if it already exists
        if not invalidate and os.path.isfile(self.cache_path / self.cache_name):
            self.model = load_model(self.cache_path / (self.cache_name + '.h5'))

        # print an overview about the model
        print(self.model.summary())
        print('\n')

        # if early stopping patience is set, configure the callback
        callbacks = []
        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=patience))
        if tensorboard:
            callbacks.append(TensorBoard(log_dir=str(self.current_path / (f'00_LOGS/{time.time()!s}_{tensorboard_id}')), batch_size=batch_size, histogram_freq=0, write_graph=True, write_images=True))
        if lr_plateau:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_plateau[0],
                    patience=lr_plateau[1],
                    min_lr=lr_plateau[2]))

        # always save the models in a checkpoint file
        if self.cache_path is not None:
            # save the best overall model to _best.h5
            best_path = str(self.cache_path / f'{self.cache_name}_best.h5')
            callbacks.append(ModelCheckpoint(filepath=best_path, verbose=1, save_best_only=True))

            # save the latest model to _latest.h5
            latest_path = str(self.cache_path / f'{self.cache_name}_latest.h5')
            callbacks.append(ModelCheckpoint(filepath=latest_path, verbose=1))

        # log results to CSV
        if self.log_path is not None:
            log_path = str(self.log_path / f'{self.log_name}.csv')
            callbacks.append(CSVLogger(log_path))

        # fit the model to the training data
        if validation_data is not None:
            # always use the data given as validation_data for validation purposes
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=validation_data,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks)
        else:
            # use a percentage of the years as validation data (and shuffle)
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_split=validation_split,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks)

        self.history = history

        return self.model, history

    def evaluate(self, which='best'):
        print(f'>> Eager evaluation: {which}')

        # load the existing model
        load_model(self.cache_path / (self.cache_name + f'_{which}.h5'))
