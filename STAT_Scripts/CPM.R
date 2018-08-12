libraries = c("tseries","fpp2","forecast","astsa","fUnitRoots","quantmod", "gtrendsR", "reshape2", "curl", "devtools","forecast","xts")

lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

source("auxiliarfunctions.R")


#-------------------------------------------------------------------------------------
#---------------------Recolection and preparation of the Data-set---------------------
#-------------------------------------------------------------------------------------

#----------------------------------Recolection----------------------------------------

#BTC
FinDataYH = new.env()
getSymbols(c("BTC-USD"), from = "2016-01-01", src="yahoo", env = FinDataYH)
FinData = as.list(FinDataYH)
btc<-FinData$`BTC-USD`$`BTC-USD.Close`
attr(btc, 'frequency') <- 1 #If we dont change frequency to 1 many methods do not work
#Gold
getSymbols(c("GC=F"), from = "2016-01-01", src="yahoo", env = FinDataYH)
FinData = as.list(FinDataYH)
gold<-FinData$`GC=F`$`GC=F.Close`
attr(gold, 'frequency') <- 1 
#EUR/USD 
getSymbols(c("EURUSD=X"), from = "2016-01-01", src="yahoo", env = FinDataYH)
FinData = as.list(FinDataYH)
eur_usd<-FinData$`EURUSD=X`$`EURUSD=X.Close`
attr(eur_usd, 'frequency') <- 1
#Dow-Jones
getSymbols(c("^DJI"), from = "2016-01-01", src="yahoo", env = FinDataYH)
FinData = as.list(FinDataYH)
DJ<-FinData$`DJI`$`DJI.Close`
attr(DJ, 'frequency') <- 1

#-------------------------------Matrix Preparation--------------------------------

#Creation of a matrix with all the variables to be used.
btc_new <- merge(btc,lag.xts(gold,-1),lag.xts(eur_usd,-1),lag.xts(DJ,-1))
#As we will use the value of the day h to predict h+1 we take lags int the regressors variable.
btc_new$GC.F.Close <- na.approx(btc_new$GC.F.Close,na.rm = FALSE) #Gold
btc_new$BTC.USD.Close <- na.approx(btc_new$BTC.USD.Close,na.rm = FALSE) #BTC
btc_new$EURUSD.X.Close<-na.approx(btc_new$EURUSD.X.Close,na.rm=FALSE) #EUR/USD
btc_new$DJI.Close<-na.approx(btc_new$DJI.Close,na.rm=FALSE) #Dow Jones

#---------------------------Auxiliar variables definition-------------------------

timeframe<-"20160201/20180430" #Time frame selected for our experiment.
btc_selected<-btc_new[, "BTC.USD.Close"][timeframe] #Vector with BTC values.
btc_new<-btc_new[timeframe] #Matrix with all values.
regresors<-btc_new[, c("GC.F.Close","EURUSD.X.Close","DJI.Close")]
#We save the auxiliar auxiliar separately (they will be used as regressors)

#-----------------------------------------------------------------------------
#------------------------------Structure Study--------------------------------
#-----------------------------------------------------------------------------

#--------------------------------Unitary Root---------------------------------

adf.test(btc,k=1) 
adf.test(btc,k=2) ##It seems clear that only exists one unitary root.

#---------------------------------Variance------------------------------------

lambda<-BoxCox.lambda(btc)   #Best Box Cox transformation (0 is equivalent to the logarithm)

#Visualization of the variance issue
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIgraphics to be improved
par(mfrow = c(3,2))
plot(btc[timeframe])
plot(diff(btc[timeframe]))

plot(log(btc[timeframe]))
plot(diff(log(btc[timeframe])))

plot(BoxCox(btc[timeframe], lambda = lambda))
plot(diff(BoxCox(btc[timeframe], lambda = lambda)))


#------------------------------Autocorrelation-------------------------------- 
#BTC
acf2(btc)
acf2(diff(btc))  

#log(BTC)
acf2(log(btc))
acf2(diff(log(btc)))  
#As we can see, is pretty difficult to determine the structure in terms of ARIMA(p,d,q) graphically.
#Taking logarithms makes it a little bit more visible, but still the structure is not clear.


#-------------------------------------------------------------------------------
#-------------------------------------Modeling----------------------------------
#-------------------------------------------------------------------------------
  
#--------------------------------------ARIMA------------------------------------

#Optimal Arima calculation.
fit_btc <- auto.arima(btc,stepwise = FALSE)
fit_logbtc <- auto.arima(log(btc),stepwise = FALSE) 
fit_BCbtc<-auto.arima(btc,stepwise = FALSE,lambda = "auto") 
fit_full_btc <- auto.arima(btc_selected,    xreg = log(regresors),       stepwise = FALSE,approximation = FALSE,lambda = 0,d=1)

#Visualization of the residuals.
par(mfrow = c(1,1))
checkresiduals(fit_btc)
checkresiduals(fit_logbtc)
checkresiduals(fit_BCbtc) 
checkresiduals(fit_full_btc)


#-----------------------------------Smoothing---------------------------------

#Optimal Smoothing calculation
sfit_btc <- ets(btc)
sfit_logbtc <- ets(log(btc))
sfit_BCbtc <- ets(btc,lambda = "auto")

#Visualization of the residuals.
par(mfrow = c(1,1))
checkresiduals(sfit_btc)
checkresiduals(sfit_logbtc) 
checkresiduals(sfit_BCbtc) 


#-----------------------------------Forecasting-------------------------------
#-----------------------------------------------------------------------------

#Creation of the train and  test data set to visualize graphically how our models are working.
trainframe<-"/20180105"
testframe<-"20180206/"

train_btc <- btc_selected[trainframe] #Training set till 5th Feb 2018 
test_btc <- btc_selected[testframe] #Testing set from 6th Feb 2018 
regresors_train<- regresors[trainframe] #Regressors matrix till 5th Feb 2018 
regresors_test<- regresors[testframe] #Regressors matrix from 6th Feb 2018 

#------------------------------------ARIMA------------------------------------

#Training of the model with the train Data-set.
tfit_btc <- auto.arima(train_btc,stepwise = FALSE)
tfit_logbtc <- auto.arima(log(train_btc),stepwise = FALSE) 
tfit_BCbtc<-auto.arima(train_btc,stepwise = FALSE,lambda = "auto")
tfit_regfull<-auto.arima(train_btc, xreg = log(regresors_train), stepwise = FALSE,approximation = FALSE,lambda = 0,d=1)

for_length <- nrow(btc_selected) - nrow(train_btc)#Number of variables to be predicted

#Visualizaion
par(mfrow = c(1,1))
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIgraphics to be improved

tfit_btc %>% forecast(h=for_length) %>% plot()
lines(as.ts(btc_selected[-length(btc_selected),])) 
#As an extra prediction is done, we delete last element to make it match with the length of our time series.

tfit_logbtc %>% forecast(h=for_length) %>% plot()
lines(as.ts(log(btc_selected[-length(btc_selected),])))

tfit_BCbtc %>% forecast(h=for_length) %>% plot()
lines(as.ts(btc_selected[-length(btc_selected),]))

#Arima-Regression
tfit_regfull %>% forecast(xreg=log(regresors_test)) %>% plot()
lines(as.ts(btc_selected[-length(btc_selected),])) 



#----------------------------------Smoothing--------------------------------

#Training of the model with the train Data-set.
stfit_btc <- ets(train_btc)
stfit_logbtc <- ets(log(train_btc)) 
stfit_BCbtc <- ets(train_btc,lambda = "auto") 


par(mfrow = c(1,1))
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIgraphics to be improved
stfit_btc %>% forecast(h=for_length) %>% plot()
lines(as.ts(btc_selected[-length(btc_selected),])) 

stfit_logbtc %>% forecast(h=for_length) %>% plot()
lines(as.ts(log(btc_selected[-length(btc_selected),])))

stfit_BCbtc %>% forecast(h=for_length) %>% plot()
lines(as.ts(btc_selected[-length(btc_selected),]))


#-----------------------------------------------------------------------------
#---------------------------------Performance---------------------------------
#-----------------------------------------------------------------------------

#For the 3 first models we just use the functions tsCV, but as as we can not use it for the ARIMA with regressors we need 
#to create a loop to get every prediction (we compute every prediction h+1 using the rest h elements, defined as train data set)

for (h in 10:(nrow(btc_new)-1))
{
  fullforecast[h+1]<-regression_farima(btc_selected,h ,log(regresors))$mean
}


#----------------------------Errors Calculation-------------------------------

errorsplot<-cbind(as.ts(fullforecast),btc_selected) #Matrix with predictions and real values for ARIMA with regressors.
#To be directly comparable with the errors calculated with tsCV we need to take one lag.
e_afullbtc <- (errorsplot[,1]-errorsplot[,2]) #Vector with the errors.
e_afullbtc<-lag.xts(e_afullbtc,-1) # We take a lag in the vector

#Calculation of the rest of errors using the function tsCV.
e_abtc <- tsCV(btc_selected, farima, h = 1)
e_a_logbtc <- tsCV(log(btc_selected), farima, h = 1)
e_a_BCbtc <- tsCV(btc_selected, lambda_farima, h = 1)
e_sbtc <- tsCV(btc_selected, fets, h = 1)
e_s_logbtc <- tsCV(log(btc_selected), fets, h = 1)
e_s_BCbtc <- tsCV(btc_selected, lambda_fets, h = 1)

#we chargue every error in a list and create a matrix with the mean squared error and the Root-mean-square deviation.
lerrors <- list(e_afullbtc,e_abtc,e_a_BCbtc,e_sbtc,e_s_BCbtc)
mean_errors <- sapply(lerrors, function (x) mean(x[10:length(x)]^2, na.rm = TRUE))
sq_mean_errors <- sapply(lerrors, function (x) sqrt(mean(x[10:length(x)]^2, na.rm = TRUE)))
#We don´t take in cosideration the first 10 erros, as we need certain ammount of data to compute our models.


#Configuration of the matrix with the mean/root-mean squared errors.
sq_mean_errors<-matrix(sq_mean_errors,nrow=1)
dimnames(sq_mean_errors)<-list("error",c("e_afullbtc","e_abtc","e_a_BCbtc","e_sbtc","e_s_BCbtc"))
sq_mean_errors<-as.data.frame(sq_mean_errors)

#With the function accurancy we can compute different error measures with the train data defined before 
#This function is not available for the ARIMA with regressions.
accuracy(stfit_btc %>% forecast(h=for_length),as.ts(btc_selected)) 
accuracy(tfit_btc %>% forecast(h=for_length),as.ts(btc_selected))
accuracy(stfit_BCbtc %>% forecast(h=for_length),as.ts(btc_selected))
accuracy(tfit_BCbtc %>% forecast(h=for_length),as.ts(btc_selected))
accuracy(tfit_regfull %>% forecast(xreg=regresors_test),as.ts(btc_selected)) 


#----------------------Errors behavior visualization--------------------------

#Root-mean-square deviation comparation.
par(mfrow = c(1,1))
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIGraphic to be improved.
barplot(as.matrix(sq_mean_errors),col="darkblue", beside = TRUE,
        angle=45)

#Comparation of square-error between 2 models.
sqerrors<-sapply(lerrors, function (x) x[10:length(x)]^2)
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIGraphic to be improved.
plot.zoo(sqerrors[500:nrow(sqerrors),c(1,5)],plot.type = "single", col = c("red", "blue")) #Red: e_afullbtc, Blue: e_s_BCbtc

#Comparation between predictions and real values.
par(mfrow = c(1,1))
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIGraphics to be improved
plot.zoo(errorsplot[650:823,],plot.type = "single", col = c("red", "blue")) #Red: prediction, Blue: Real value

#Accurancy with respect to the train data-set length.
mean_errors_aux<-c()
mean_errors_full<-matrix(ncol=5,nrow = length(lerrors[[1]]))
for (i in 10:(length(lerrors[[1]])))
{
mean_errors_aux<-sapply(lerrors, function (x) mean(x[i:length(x)]^2, na.rm = TRUE))
mean_errors_full[i,] <- matrix(mean_errors_aux,nrow=1)[1,]
}
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIGraphics to be improved
plot.zoo(mean_errors_full[300:822,c(1,2)],plot.type = "single", col = c("red", "blue")) #Red: e_afullbtc, Blue: e_s_BCbtc 


