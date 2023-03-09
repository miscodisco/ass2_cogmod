# agent loose-stay-win-shift-withNoise function
LSWSAgentNoise_f <- function(prevChoice, Feedback, noise){ 
  if (Feedback == 1){
    choice = 1-prevChoice} 
  else if (Feedback == 0) {
    choice = prevChoice 
  }
  if (rbinom(1,1,noise)==1){
    choice <- rbinom(1,1,.5)
  }
  return(choice) 
}