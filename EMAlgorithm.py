import sys

import numpy as np
from scipy.stats import multivariate_normal as mvn

class GMM:

    def __init__(self, n_components, features=[i for i in range(3)], verbose=False):
        self.n_components = n_components
        self.features = features
        n_features = len(features)
        self._verbose = verbose

        self._initialize_weights(n_components)
        self._initialize_means(n_components, n_features)
        self._initialize_sigmas(n_components, n_features)

    def _initialize_weights(self, n_components):
        self._weights = np.random.rand(n_components)
        self._weights = np.divide(self._weights, np.sum(self._weights))

    def _initialize_means(self, n_components, n_features):
        self._means = np.random.rand(n_components, n_features)

    def _initialize_sigmas(self, n_components, n_features, scale=1.):
        self._sigmas = np.zeros((n_components, n_features, n_features))
        for i in range(n_components):
            self._sigmas[i, :, :] = np.diag(scale * np.ones(n_features))

    def fit(self, train_data, max_iter=50, delta=1e-3, verbose=False, allow_singular=True):
        weights = self._weights
        means = self._means
        sigmas = self._sigmas

        n_components = len(means)
        n_samples = train_data.shape[0]

        ll_old = 0.
        for i in range(max_iter):
            self._vprint(f'Iteration: {i}')

            # E-stop
            gammas = np.zeros((n_components, n_samples))
            for c in range(n_components):
                self._vprint(f'Computing gamma for component {c}')
                gammas[c] = np.multiply(weights[c], mvn.pdf(train_data, means[c], sigmas[c],
                                                            allow_singular=allow_singular))
            gammas = np.divide(gammas, gammas.sum(0))

            # M-step
            weights = np.zeros(weights.shape)
            means = np.zeros(means.shape)
            sigmas = np.zeros(sigmas.shape)

            for c in range(n_components):
                gamma_sum = gammas[c].sum()

                self._vprint(f'Computing weight for component {c}')
                weights[c] = np.divide(sum(gammas[c]), n_samples)

                self._vprint(f'Computing mean for component {c}')
                means[c] = np.dot(gammas[c], train_data)
                means[c] = np.divide(means[c], gamma_sum)

                self._vprint(f'Computing sigma for component {c}')
                d = train_data - means[c]
                dsq = np.square(d)
                sigmas[c] = gammas[c].dot(dsq)
                sigmas[c] = np.diag(sigmas[c].diagonal())
                sigmas[c] = np.divide(sigmas[c], gamma_sum)

                if (np.linalg.cond(sigmas[c]) > 1. / sys.float_info.epsilon):
                    self._vprint("[!] Singular matrix detected")

            self._vprint("Computing log-likelihood")
            likelihoods = np.zeros((n_components, n_samples))
            for c in range(n_components):
                likelihoods[c] = np.multiply(weights[c], mvn.pdf(train_data, means[c], sigmas[c],
                                                                 allow_singular=allow_singular))
                ll_new = np.log(likelihoods.sum(0)).sum()

            self._vprint(f'log_likelihood: {ll_new:3.4f}\n')
            if np.abs(ll_new - ll_old) < delta * ll_old:
                break
            ll_old = ll_new

        self._weights = weights
        self._means = means
        self._sigmas = sigmas

    def _vprint(self, msg):
        if self._verbose:
            print(msg)

    def predict(self, test_data, features):
        n_components = len(self._means)
        n_samples = test_data.shape[0]
        means = self._means[:, features]
        sigmas = self._sigmas[:, features, features]

        likelihoods = np.zeros((n_components, n_samples))
        log_likelihoods = np.zeros(n_samples)
        for c in range(n_components):
            likelihoods[c] = np.multiply(self._weights[c],
                                         mvn.pdf(test_data,
                                                 means[c],
                                                 sigmas[c],
                                                 allow_singular=True))
            log_likelihoods = likelihoods.sum(0)
        return log_likelihoods