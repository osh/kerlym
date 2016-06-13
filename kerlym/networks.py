import keras.backend as K
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Lambda, TimeDistributed
from keras.optimizers import RMSprop, Adadelta, Adam


# Some example networks which can be used as the Q function approximation ...

def simple_dnn(agent, env, dropout=0.5, **args):
    S = Input(shape=[agent.input_dim])
    h = Dense(256, activation='relu', init='he_normal')(S)
    h = Dropout(dropout)(h)
    h = Dense(256, activation='relu', init='he_normal')(h)
    h = Dropout(dropout)(h)
    V = Dense(env.action_space.n, activation='linear',init='zero')(h)
    model = Model(S,V)
    model.compile(loss='mse', optimizer=RMSprop(lr=args["learning_rate"]) )
    return model

def simple_rnn(agent, env, dropout=0, h0_width=8, h1_width=8, **args):
    S = Input(shape=[agent.input_dim])
    h = Reshape([agent.nframes, agent.input_dim/agent.nframes])(S)
    h = TimeDistributed(Dense(h0_width, activation='relu', init='he_normal'))(h)
    h = Dropout(dropout)(h)
    h = LSTM(h1_width, return_sequences=True)(h)
    h = Dropout(dropout)(h)
    h = LSTM(h1_width)(h)
    h = Dropout(dropout)(h)
    V = Dense(env.action_space.n, activation='linear',init='zero')(h)
    model = Model(S,V)
    model.compile(loss='mse', optimizer=RMSprop(lr=args["learning_rate"]) )
    return model

def simple_cnn(agent, env, dropout=0, learning_rate=1e-3, **args):
    S = Input(shape=[agent.input_dim])
    h = Reshape( agent.input_dim_orig )(S)
    h = TimeDistributed( Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu'))(h)
    h = TimeDistributed( Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', activation='relu'))(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    V = Dense(env.action_space.n, activation='linear',init='zero')(h)
    model = Model(S, V)
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate) )
    return model


def karpathy_simple_pgnet(agent, env, dropout=0.5, learning_rate=1e-4, **args):
    S = Input(shape=[agent.input_dim])
    h = Dense(200, activation='relu', init='he_normal')(S)
    h = Dropout(dropout)(h)
    V = Dense(env.action_space.n, activation='sigmoid',init='zero')(h)
    model = Model(S,V)
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate) )
    return model

def pgconvnet(agent, env, dropout=0.5, learning_rate=1e-4, **args):
    S = Input(shape=[agent.input_dim])
    h = Reshape( agent.input_dim_orig )(S)
    h = TimeDistributed( Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu', dim_ordering='tf'))(h)
    h = Flatten()(h)
    h = Dropout(dropout)(h)
    h = Dense(64, activation='relu', init='he_normal')(h)
    h = Dropout(dropout)(h)
    h = Dense(16, activation='relu', init='he_normal')(h)
    h = Dropout(dropout)(h)
    V = Dense(env.action_space.n, activation='sigmoid',init='zero')(h)
    model = Model(S,V)
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate) )
    return model
