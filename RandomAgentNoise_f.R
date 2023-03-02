# random agent with noise
RandomAgentNoise_f <- function(rate, noise) {
  choice <- rbinom(1, 1, rate)
  if (rbinom(1, 1, noise) == 1) 
  {choice = rbinom(1, 1, 0.5)}
  return(choice)
}