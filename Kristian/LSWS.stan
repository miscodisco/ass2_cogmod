
// The input data 
data {
 int<lower=1> n;
 array[n] int c;
 vector[n] outcome_hand; 
 
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real follow_bias;
}

// The model
model {
  // priors
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(follow_bias | 0, 1);
  
  // model
  target += bernoulli_logit_lpmf(c | bias + follow_bias * outcome_hand);
}

// Saving Priors
generated quantities{
  real <lower=0, upper=1> bias_prior;
  real <lower=0, upper=1> bias_posterior;
   
  real <lower=0, upper=1> follow_bias_prior;
  real <lower=0, upper=1> follow_bias_posterior;
  
  int <lower=0, upper=n> bias_prior_preds;
  int <lower=0, upper=n> bias_posterior_preds;
  
  int <lower=0, upper=n> follow_bias_prior_preds;
  int <lower=0, upper=n> follow_bias_posterior_preds;
  
  bias_prior = inv_logit(normal_rng(0,1));
  bias_posterior = inv_logit(bias);
  
  follow_bias_prior = inv_logit(normal_rng(0,1));
  follow_bias_posterior = inv_logit(follow_bias);
  
  bias_prior_preds = binomial_rng(n, bias_prior);
  bias_posterior_preds = binomial_rng(n, inv_logit(bias));
  
  follow_bias_prior_preds = binomial_rng(n, follow_bias_prior);
  follow_bias_posterior_preds = binomial_rng(n, inv_logit(follow_bias));
}
