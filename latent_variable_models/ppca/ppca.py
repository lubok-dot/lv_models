import numpy as np


class PPCA(object):
    '''
    implements the probabilistic PCA according to Bishop & Tipping 1998.

    https://doi.org/10.1111/1467-9868.00196

    all equation numbers refer to this paper.

    every COLUMN of t is a centered data point or will be centered.
    '''

    def __init__(self, W=None, sg=None):
        # attributes of the PPCA
        self.W = W  # latent variable
        self.sg = sg  # noise standard deviation
        self.mu = None  # empirical mean
        # auxiliary attributes
        self._tr_S = None  # trace of the empirical covariance matrix
        self._SW = None  # covariance between latent and variables and data
        self._SL = None  # covariance matrix of the latent variables
        self._M_inv = None  # M = W^TW + sg**2
        self._T = None  # T = M^{-1}W^T

    def q(self):
        '''
        dimension of the latent space
        '''
        return self.W.shape[1]

    def d(self):
        '''
        dimension of the feature space t
        '''
        return self.W.shape[0]

    def M_inv(self):
        '''
        (W^TW + sg^2)^{-1}
        '''
        return np.linalg.inv(np.dot(self.W.T, self.W) + self.sg**2 * np.diag(np.ones(self.q())))

    def T(self):
        '''
        M^{-1}W^T
        '''
        return np.dot(self._M_inv, self.W.T)

    def SW_n(self, t_n):
        '''
        computes t_n<x_n>^T where <x_n> is defined in eq. (25). 
        t_n is assumed to be centered
        '''
        return np.outer(t_n, np.dot(self._T, t_n).T)

    def tr_n(self, t_n):
        '''
        computes the trace of the covariance matrix of a single data point. 
        Mean of t_n is assumed to be 0
        '''
        return sum(t_n**2)

    def SL(self):
        '''
        covariance matrix of the latent variables
        '''
        return self.sg**2 * self._M_inv + np.dot(self._T, self._SW)

    def _update(self, t):
        # update the auxiliary variables
        self._M_inv = self.M_inv()
        self._T = self.T()
        # compute SW and SL
        self._SW = 0
        for t_n in t.T:
            self._SW += self.SW_n(t_n)
        self._SW = self._SW / t.shape[1]
        self._SL = self.SL()

    def W_tilde(self):
        '''
        computes W_tilde in equation (27)
        '''
        return np.dot(self._SW, np.linalg.inv(self._SL))

    def sg_tilde(self):
        '''
        computes sg_tilde in equation (28)
        '''
        sg_tilde = self._tr_S - 2 * np.trace(np.dot(self.W.T, self._SW))
        sg_tilde += np.trace(np.dot(self._SL, np.dot(self.W.T, self.W)))
        return np.sqrt(sg_tilde / self.d())

    def em_step(self, t):
        '''
        single step of the EM algorithm
        '''
        # update M, T, and SW
        self._update(t)
        # update latent matrix W
        W = self.W_tilde()
        delta = np.linalg.norm(W - self.W)
        self.W = W
        # update standard deviation
        sg = self.sg_tilde()
        delta += abs(self.sg - sg)
        self.sg = sg
        # return error
        return delta

    def center(self, t):
        '''
        centers the data for the EM algorithm
        '''
        # initialize mean
        self.mu = np.nanmean(t, axis=1)
        # center the data
        return np.array([x - self.mu for x in t.T]).T

    def em_algo(self, t, epsilon):
        '''
        EM algorithm on the data t
        '''
        # slight consistency check
        assert t.shape[0] == self.d()
        # get empirical variance
        self._tr_S = sum([self.tr_n(t_n)
                          for t_n in t.T]) / t.shape[1]
        # EM algorithm
        delta = 2 * epsilon
        while delta > epsilon:
            delta = self.em_step(t)
            print(delta)
        # make final update of the parameters
        self._update(t)

    def fit(self, t, epsilon):
        '''
        Fit method which ensures that the data is centered
        '''
        self.em_algo(self.center(t), epsilon)


class Bayes_PPCA(PPCA):
    '''
    Implements the Bayesian probabilistic PCA in Bishop 

    https://papers.nips.cc/paper/1549-bayesian-pca.pdf

    all equation numbers refer to this paper.
    '''

    def __init__(self, W=None, sg=None):
        super().__init__(W, sg)
        self.alpha = self.alpha_tilde()  # prior weights

    def alpha_tilde(self):
        '''
        empirical Bayes estimation of the hyperparameters alpha_i in eq (9)
        '''
        return np.array([self.d() / np.dot(w, w) for w in self.W.T])

    def W_tilde(self):
        '''
        computes W_tilde in eq. (12)
        '''
        # compute denominator matrix
        A = self._SL + self.sg**2 * np.diag(self.alpha)
        return np.dot(self._SW, np.linalg.inv(A))

    def em_step(self, t):
        '''
        modified em-step for Bayesian PCA
        '''
        # find still active dimensions
        flag = self.alpha != np.inf
        # get rid of dead components
        self.W = self.W[:, flag]
        # update hyperparamters
        alpha = self.alpha_tilde() / t.shape[1]
        delta = np.linalg.norm(alpha - self.alpha[flag])
        self.alpha = alpha
        # perform EM step
        return super().em_step(t) + delta


class DPPCA(PPCA):
    '''
    implements the dual probabilistic PCA proposed by N. Lawrence in

    http://jmlr.csail.mit.edu/papers/volume6/lawrence05a/lawrence05a.pdf
    '''

    def __init__(self, X=None, sg=None):
        super().__init__(X.T, sg)

    def fit(self, t, epsilon):
        self.em_algo(self.center(t).T, epsilon)


class Bayes_DPPCA(Bayes_PPCA, DPPCA):
    '''
    implements a Bayesian dual probabilistic PCA
    '''
    pass


class PPCA_Missing(PPCA):
    '''
    implements PPCA with missing values according to T. Chen, et al.
    every COLUMN of t is a data

    https://doi.org/10.1016/j.csda.2009.03.014

    All equation numbers refer to this paper.

    As for the PPCA, every column of t is a data point with mean 0.
    Note: we skip the computation of the expected mean in the EM
    algorithm. Furthermore, the meaning of t anc x in the paper is
    swapped in our implementation.
    '''

    def __init__(self, W, sg):
        super().__init__(W, sg)
        self._C = self.C()

    def C(self):
        '''
        WW^T + sg^2
        '''
        return np.dot(self.W, self.W.T) + self.sg**2 * np.diag(np.ones(self.d()))

    def A(self, o):
        '''
        matrix C_uoC_oo^{-1}
        '''
        if o.all():
            return None
        C_uo = self._C[~o][:, o]
        C_oo = self._C[o][:, o]
        return np.dot(C_uo, np.linalg.inv(C_oo))

    def Q(self, o):
        '''
        lower right block of matrix Q in eq. (23)
        '''
        if o.all():
            return None
        C_uu = self._C[~o][:, ~o]
        C_ou = self._C[o][:, ~o]
        return C_uu - np.dot(self.A(o), C_ou)

    def z_n(self, t_n, o, A=None):
        '''
        the variable z_n in eq. (23) where A = C_uoC^{-1}_oo
        '''
        if A is None:
            return t_n
        z_n = np.empty(self.d())
        z_n[o] = t_n[o]
        z_n[~o] = np.dot(A, t_n[o])
        return z_n

    def SW_n(self, z_n, o, Q=None):
        '''
        computes S_nT^T where S_n is the empirical covariance matrix
        eq. (31) for a single data point
        '''
        SW_n = super().SW_n(z_n)
        if Q is None:
            return SW_n
        SW_n[~o] += np.dot(Q, self._T.T[~o])
        return SW_n

    def tr_n(self, z_n, Q=None):
        '''
        computes the trace of the empirical covariance matrix eq. (28)
        for a single data point
        '''
        tr_n = super().tr_n(z_n)
        if Q is None:
            return tr_n
        return tr_n + np.trace(Q)

    def _update(self, t):
        '''
        computes tr_S, SW, and SL
        '''
        # initialize the auxiliary variables
        self._M_inv = self.M_inv()
        self._T = self.T()
        self._C = self.C()
        prev_o = np.ones(self.d(), dtype=np.bool)
        A, Q = None, None
        # compute tr_S, SW, and SL
        self._tr_S, self._SW = 0, 0
        for t_n in t.T:
            o = ~np.isnan(t_n)
            if not (~(o ^ prev_o)).all():
                A = self.A(o)
                Q = self.Q(o)
                prev_o = o
            z_n = self.z_n(t_n, o, A)
            self._tr_S += self.tr_n(z_n, Q)
            self._SW += self.SW_n(z_n, o, Q)
        self._SW, self._tr_S = self._SW / t.shape[1], self._tr_S / t.shape[1]
        self._SL = self.SL()


class Bayes_PPCA_Missing(Bayes_PPCA, PPCA_Missing):
    '''
    Implements the Bayesian PCA with missing values
    '''
    pass


class DPPCA_Missing(DPPCA, PPCA_Missing):
    '''
    implements the dual probabilistic PCA with missing values
    '''
    pass


class Bayes_DPPCA_Missing(Bayes_DPPCA, PPCA_Missing):
    '''
    implements a Bayesian dual probabilistic PCA with missing values
    '''
    pass
