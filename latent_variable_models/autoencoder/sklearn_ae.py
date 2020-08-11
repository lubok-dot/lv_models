import autoencoder
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SparseAE(BaseEstimator, TransformerMixin):
    '''
    sklearn wrapper for the Sparse AE
    '''

    def __init__(self, hidden_dim=10, rate=10.0, batch_size=64, epochs=25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.rate = rate

    def fit(self, X):
        # define model
        self.sparse_ae = autoencoder.SparseAE(
            X.shape[1], self.hidden_dim, self.rate)

        # compile model
        self.sparse_ae.compile(
            optimizer='adam',
            loss="mean_squared_error",
        )

        self.history = self.sparse_ae.fit(X, X, batch_size=self.batch_size,
                                          epochs=self.epochs, verbose=1)
        return self

    def loss(self):
        return self.history.history['loss'][-1]

    def transform(self, X):
        return self.sparse_ae.predict(X)


class DNAE(SparseAE):

    def __init__(self, hidden_dim=10, noise=0.05, batch_size=64, epochs=25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise = noise
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X):
        # define model
        self.de_ae = autoencoder.SparseAE(X.shape[1], self.hidden_dim, 0)

        # compile model
        self.de_ae.compile(
            optimizer='adam',
            loss="mean_squared_error",
        )

        data = autoencoder.Noisy(X, self.noise, self.batch_size)
        self.history = self.de_ae.fit(data, epochs=self.epochs, verbose=1)
