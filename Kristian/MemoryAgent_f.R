# Now we create the memory agent whose beta changes based on winning or losing
MemoryAgent_f <- function(bias, beta, memory, confidence){
  choice = rbinom(1, 1, inv_logit_scaled(bias + (beta + confidence) * logit_scaled(memory)))
  return(choice)
}

