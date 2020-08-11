from sklearn.base import BaseEstimator, TransformerMixin
import ppca
import numpy as np


class PPCA(BaseEstimator, TransformerMixin):
    '''
    sklearn wrapper for the probabilistic PCAs
    '''
    sg = 1  # initial value for the standard deviation

    def __init__(self, q, prior=True, dual=False):
        self.q = q  # dimension of the latent space
        self.prior = prior  # true if we adopt a Bayesian approach
        self.dual = dual  # true if we consider a dual model
        self.missing = None  # true if we assume missing values
        self.model = None  # placeholder for the model PCA model

    def fit(self, t, epsilon=1e-4):
        '''
        t is feature matrix where each column is a data point
        '''
        # determine whether we deal with missing values
        if self.missing is None:
            self.missing = np.isnan(t).any()
        # initialize model if necessary
        if self.model is None:
            # dual models
            if self.dual:
                W = np.random.normal(size=(self.q, t.shape[1]))
                # models which support missing values
                if self.missing:
                    # Bayesian DPPCA with missing values
                    if self.prior:
                        self.model = ppca.Bayes_DPPCA_Missing(W, PPCA.sg)
                    # DPPCA with missing values
                    else:
                        self.model = ppca.DPPCA_Missing(W, PPCA.sg)
                # models without any missing values
                else:
                    # Bayesian DPPCA
                    if self.prior:
                        self.model = ppca.Bayes_DPPCA(W, PPCA.sg)
                    # DPPCA
                    else:
                        self.model = ppca.DPPCA(W, PPCA.sg)
            # non dual models
            else:
                W = np.random.normal(size=(t.shape[0], self.q))
                # models which support missing values
                if self.missing:
                    # Bayesian PPCA with missing values
                    if self.prior:
                        self.model = ppca.Bayes_PPCA_Missing(W, PPCA.sg)
                    # PPCA with missing values
                    else:
                        self.model = ppca.PPCA_Missing(W, PPCA.sg)
                # models without any missing values
                else:
                    # Bayesian PPCA
                    if self.prior:
                        self.model = ppca.Bayes_PPCA(W, PPCA.sg)
                    # PPCA
                    else:
                        self.model = ppca.PPCA(W, PPCA.sg)
        # fit the model
        self.model.fit(t, epsilon)

    def _z_n(self, t_n):
        '''
        auxiliary function which returns t_n or its expected value z_n in
        the case t_n contains nan values
        '''
        o = ~np.isnan(t_n)
        if o.all():
            return t_n
        return self.model.z_n(t_n, o, self.model.A(o))

    def transform(self, t):
        '''
        computes the projection of t onto the latent space. Columns are
        data points
        '''
        if self.dual:
            return self.model.W.T
        if self.missing:
            return np.array([np.dot(self.model._T, self._z_n(t_n)) for t_n in t.T]).T
        return np.array([np.dot(self.model._T, t_n) for t_n in t.T]).T
