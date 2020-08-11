data{
    int<lower=0>N; // number of data points in entire dataset
    int<lower=0>K; // number of mixture components
    int<lower=0>D; // dimension
    vector[D]y[N]; // observations
    real<lower=0>alpha0; // dirichlet prior
    real<lower=0>mu_sigma0; // scale of prior means
    real<lower=0>sigma_shape; // shape of the variance prior
    real<lower=0>sigma_scale; // scale of the variance prior
    real<lower=0>nu; // degrees of freedom
}

transformed data{
    vector<lower=0>[K]alpha0_vec;
    for(k in 1:K){
        alpha0_vec[k]=alpha0;
    }
}

parameters{
    simplex[K]theta; // mixing proportions
    vector[D]mu[K]; // locations of mixturecomponents
    vector<lower=0>[D]sigma[K]; // standard deviations of mixture components
}

model{
// priors
theta ~ dirichlet(alpha0_vec);
    for(k in 1:K){
        mu[k] ~ normal(0.0,mu_sigma0);
        sigma[k] ~ inv_gamma(sigma_shape, sigma_scale);
    }
// log-likelihood
    for(n in 1:N){
        real ps[K];
        for(k in 1:K){
            ps[k] = log(theta[k]) + student_t_lpdf(y[n] | nu, mu[k], sigma[k]);
        }
        target += log_sum_exp(ps);
    }
}

generated quantities {
    row_vector[K] z[N];
    for (n in 1:N) {
        for (k in 1:K) {
            z[n, k] = log(theta[k]) + student_t_lpdf(y[n] | nu, mu[k], sigma[k]);
        }
    }
}