
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
