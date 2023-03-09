# agent loose-stay-win-shift-withBias function
LSWSAgentBias_f <- function(prevChoice, Feedback, bias, follow_bias){ 
  # if the agent won on previous round - shift 
  if (Feedback == 1 & prevChoice == 0){
    choice = rbinom(1, 1, inv_logit_scaled(bias + follow_bias * 1))
  }
  else if (Feedback == 1 & prevChoice == 1){
    choice = rbinom(1, 1, inv_logit_scaled(bias + follow_bias * -1))
  }
  # if the agent lost - stay
  else if (Feedback == 0 & prevChoice == 0){
    choice = rbinom(1, 1, inv_logit_scaled(bias + follow_bias * -1))
  }
  else if (Feedback == 0 & prevChoice == 1){
    choice = rbinom(1, 1, inv_logit_scaled(bias + follow_bias * 1))
  }
  return(choice) 
}