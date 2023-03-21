// The input (data) for the model. n of trials and h for (right and left) hand
data {
 int<lower=1> n;
 array[n] int h;
 array[n] int other;
}

// The parameters accepted by the model. 
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
  real confidencerate;
}

transformed parameters{
  vector[n] memory;

  for (trial in 1:n){
  if (trial == 1) {
    memory[trial] = 0.5;
  } 
  if (trial < n){
      memory[trial + 1] = memory[trial] + ((other[trial] - memory[trial]) / trial);
      if (memory[trial + 1] == 0){memory[trial + 1] = 0.01;}
      if (memory[trial + 1] == 1){memory[trial + 1] = 0.99;}
    }
  }
  
  vector[n] confidence;
  for (trial in 1:n){
    if (trial == 1) {
        confidence[trial] = 0;} 
    if (trial < n){
      if (other[trial] == h[trial]){
        confidence[trial+1] = confidence[trial] + confidencerate;
       }
        else {
        confidence[trial+1] = confidence[trial] - confidencerate;
      }
    }
  }
}

// The model to be estimated. 
model {
  // Priors
  target += normal_lpdf(bias | 0, .7);
  target += normal_lpdf(beta | 0, 1);
  
  // Model, looping to keep track of memory
  for (trial in 1:n) {
    target += bernoulli_logit_lpmf(h[trial] | bias + (beta + confidence[trial]) * logit(memory[trial]));
  }
}
