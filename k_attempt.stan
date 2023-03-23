
// The input data 
data {
 int<lower=1> n; #number of trials
 array[n] int c; #An array of the sequence of the agent's choices (1 is right, 0 is left)
 vector[n] outcome_hand; #A vector of 1s and -1s   
 #the vector is computed as follows: #Right and win = -1, lose and left = -1, right and lose = 1, left and win = 1
 #the intuition is that -1 will drag the choice towards 0 (left) and 1 will drag the choice towards 1 (right)
}

// The parameters accepted by the model. 
parameters {
  #The bias parameter determines whether and to which rate the agent is prone to choose one hand over the other.
  real bias;
  #The follow_bias determines to which extent the agent follows the implemented strategy. 
  real follow_bias;
}

// The model
model {
  // priors
  # priors for bias and follow_bias are set to a mean of 0 and a standard deviation of 1
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(follow_bias | 0, 1);
  
  // model
  #The model syntax: The choice is predicted by the agent's bias and the rate to which the agent follows the strategy.
  #The bias and follow_bias parameters are in logodds, which is why it is specified using a logit link
  target += bernoulli_logit_lpmf(c | bias + follow_bias * outcome_hand);
}

// Saving Priors & posteriors
generated quantities{
  #generating prior and posterior distribution for the bias parameter
  real <lower=0, upper=1> bias_prior;
  real <lower=0, upper=1> bias_posterior;
  
  #generating prior and posterior distributions for the followbias parameter
  real <lower=0, upper=1> follow_bias_prior;
  real <lower=0, upper=1> follow_bias_posterior;
  
  #Generating a distribution of right-choices according to the prior and the posterior 
  int <lower=0, upper=n> prior_preds;
  int <lower=0, upper=n> posterior_preds;
  
  bias_prior = normal_rng(0,1);
  bias_posterior = bias;
  
  follow_bias_prior = normal_rng(0,1);
  follow_bias_posterior = follow_bias;
  
  prior_preds = binomial_rng(n, inv_logit(bias_prior + follow_bias_prior * outcome_hand));
  posterior_preds = binomial_rng(n, inv_logit(bias_posterior + follow_bias_posterior * outcome_hand));
  
}