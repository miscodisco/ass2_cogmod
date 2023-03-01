# load libraries
pacman::p_load(cmdstanr, future, purrr, furrr)

# load data

trials <- 120
d <- NULL 

for (noise in seq(0, 0.5, 0.1)) {
  for (rate in seq(0, 1, 0.1)) {
    randomChoice <- rep(NA, trials)
    for (t in seq(trials)) {randomChoice[t] <- RandomAgentNoise_f(rate, noise)}
    temp <- tibble(trial = seq(trials), choice = randomChoice, rate, noise)
    if (exists("d")) { d <- rbind(d, temp)} else{d <- temp} }}

#data <- read.csv("data.csv")

# load stan-file
file <- ("/Users/kristian/Documents/Skole/8. Semester/Advanced Computational Modelling/ass2_cogmod/rate_recov.stan")

# model 
mod <- cmdstan_model(file,
  cpp_options = list(stan_threads = TRUE),
  stanc_options = list("O1"))


# simulate data and fit function
sim_func <- function(seed, trials, rateLvl, noiseLvl){
  
  for(t in seq(trials)){
    randomChoice[t] <- RandomAgentNoise_f(rateLvl, noiseLvl)
  }
  temp <- tibble(trial = seq(trials), choice = randomChoice, rate, noise)
  
  data <- list(
    n = 120,
    h = temp$choice
  )
  
  samples <- mod$sample(
    data = data,
    seed = 43,
    chains = 1,
    parallel_chains = 1,
    threads_per_chain = 1,
    iter_warmup = 1000,
    iter_sampling = 2000,
    refresh = 0,
    max_treedepth = 20,
    adapt_delta = 0.99
  )
  
  draws_df <- as_draws_df(samples$draws())
  temp <- tibble(biasEst = draws_df$theta_posterior, biasTrue = rateLvl, noise = noiseLvl)
  
  return(temp)
  
}

plan(multisession, workers = 4)


temp <- tibble(unique(d[,c("rate", "noise")])) %>% 
  mutate(seed = 43, trials = 120) %>% 
  rename(rateLvl = rate, noiseLvl = noise)



recovery_df <- future_pmap_dfr(temp, sim_func, .options = furrr_options(seed = TRUE))













