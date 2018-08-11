#--------------------------------Auxiliar functions---------------------------
#We need to create aux functions to return in format "forecast"
lambda_fets <- function(x, h) {
  forecast(ets(x,lambda = "auto"), h = h)
}

fets <- function(x, h) {
  forecast(ets(x), h = h)
}

farima <- function(x, h) {
  forecast(auto.arima(x,stepwise = FALSE), h = h)
}

lambda_farima <- function(x, h) {
  forecast(auto.arima(x,stepwise = FALSE,lambda = "auto"), h = h)
}

regression_farima <- function(x, h,regcoef) {
  forecast(auto.arima(x[1:h,], xreg = regcoef[1:h,], stepwise = FALSE,approximation = FALSE,lambda = 0, d=1),xreg=regcoef[h+1,]) 
}


