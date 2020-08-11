data{
    int<lower=0>N; // number of data points in entire dataset
    int<lower=0>K; // number of mixture components
    int<lower=0>D; // dimension
    vector[D]y[N]; // observations
    real<lower=0>alpha0; // dirichlet prior
    real<lower=0>mu_sigma0; // scale of prior mean
    real<lower=0>sigma_sigma0; // scale of prior variance
}

transformed data{
    vector<lower=0>[K]alpha0_vec;
    for(k in 1:K){
        alpha0_vec[k]=alpha0;
    }
}

parameters{
    simplex[K]theta; // mixingproportions
    vector[D]mu[K]; // locationsofmixturecomponents
    vector<lower=0>[D]sigma[K]; // standarddeviationsofmixturecomponents
}

model{
// priors
theta ~ dirichlet(alpha0_vec);
    for(k in 1:K){
        mu[k]~normal(0.0,mu_sigma0);
        sigma[k]~lognormal(0.0,sigma_sigma0);
    }
// log-likelihood
    for(n in 1:N){
        real ps[K];
        for (k in 1:K){
            ps[k] = log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]);
        }
        target += log_sum_exp(ps);
    }
}