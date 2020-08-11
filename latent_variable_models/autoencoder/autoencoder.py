from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Layer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np
import math


class SparsityRegularizer(Layer):
    '''Layer that adds an L1-loss'''

    def __init__(self, rate=2.0):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(
            self.rate/2 * tf.reduce_sum(tf.exp(-self.rate * tf.abs(inputs))))
        return inputs


class SparseAE(Model):
    '''
    autoencoder with one hidden layer which puts an 
    L1-regularizer on the output of the encoder. 
    '''

    def __init__(self, input_dim=2000, hidden_dim=10, rate=10.0):
        super().__init__()
        self.encoder = Dense(hidden_dim, activation='relu')
        self.sparsity_reg = SparsityRegularizer(rate)
        self.decoder = Dense(input_dim, activation='relu')

    def __call__(self, inputs, training=True):
        x = self.encoder(inputs)
        if training:
            x = self.sparsity_reg(x)
            return self.decoder(x)
        return x


class Noisy(Sequence):
    '''
    adds a gaussian noise to each batch
    '''

    def __init__(self, train, sg=0.05, batch_size=64):
        self.x = train
        self.sg = sg
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_y = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_x = batch_y + np.random.normal(scale=self.sg, size=batch_y.shape)
        return batch_x, batch_y
