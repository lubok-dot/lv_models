// data {
//     int<lower=1> G; // number of genes
//     int<lower=1> M; // number of batches
//     int<lower=1> N; // number of observations
//     int<lower=1, upper=G> gg[N]; // gene for observation n
//     int<lower=1, upper=M> bb[N]; // batch for observation n
//     vector[N] y; // expression values for observation n
// }

// transformed data {
//     vector[M] y_i = rep_vector(0.0, M); // empirical means of the batch clusters
//     vector[M] bb_count = rep_vector(0.0, M); // counts how often batch m occures
//     for (n in 1:N) {
//         y_i[bb[n]] += y[n];
//         bb_count[bb[n]] += 1;
//     }
//     for (m in 1:M) y_i[m] = y_i[m]/bb_count[m];
// }

// parameters {
//     row_vector[G] y_mean[M]; //  mean for batch m and gene g
//     row_vector<lower=0>[G] dt[M]; // std-deviation for batch m and gene g
// }

// model {
//     vector[M] sg_i = rep_vector(5, M); // standard deviation of the batches
//     vector[M] alpha_i = rep_vector(2.5, M); // shape of the inverse gamma prior
//     vector[M] beta_i = rep_vector(3, M); // scale of the inverse gamma prior
//     for (m in 1:M) y_mean[m] ~ normal(y_i[m], sg_i[m]);
//     for (m in 1:M) dt[m] ~ cauchy(alpha_i[m], beta_i[m]);
//     for (n in 1:N) y[n] ~ normal(y_mean[bb[n], gg[n]], dt[bb[n], gg[n]]);
// }


data {
    int<lower=1> G; // number of genes
    int<lower=1> M; // number of batches
    int<lower=1> N; // number of observations
    int<lower=1, upper=G> gg[N]; // gene for observation n
    vector[N] y; // expression values for observation n

    real<lower=0> alpha_0; // dirichlet prior
    real<lower=0> mu_sigma0; // std-deviation of the mean prior
    real<lower=0> delta_alpha0; // shape of the std-deviation prior
    real<lower=0> delta_beta0; // scale of the std-deviation prior
}

transformed data {
    vector<lower=0>[M] alpha0_vec;
    for (m in 1:M) alpha0_vec[m] = alpha_0;
}

parameters {
    simplex[M] theta; // mixing proportions of the batches
    row_vector[G] y_mean[M]; //  mean for batch m and gene g
    row_vector<lower=0>[G] dt[M]; // std-deviation for batch m and gene g
}

model {
    // priors
    theta ~ dirichlet(alpha0_vec);
    for (m in 1:M) {
        y_mean[m] ~ normal(0, mu_sigma0);
        dt[m] ~ inv_gamma(delta_alpha0, delta_beta0);
    }
    
    // likelihood
    for (n in 1:N) {
        vector[M] ps;
        for (m in 1:M) {
            ps[m] = log(theta[m]) + normal_lpdf(y[n] | y_mean[m, gg[n]], dt[m, gg[n]]);
        }
        target += log_sum_exp(ps)
    }
}