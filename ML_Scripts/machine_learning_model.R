options(max.print = 70)

libraries = c("tseries","fpp2","forecast","astsa","fUnitRoots","quantmod", 
              "gtrendsR", "reshape2", "curl", "devtools","forecast","xts",
              "parallelMap","tidyr", "keras")

lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

if(!is_keras_available()){
  install_keras()
}

#-------------------------------------------------------------------------------------
#---------------------Data loading and preparation---------------------
#-------------------------------------------------------------------------------------

#----------------------------------Data loading----------------------------------------
downloadTimeSeries <- function(symbol){
  if(regexpr("^", symbol)) # the circumflex is part of the yahoo ticker symbol but is not used within the attribute hierarchy
    symbol2 <- sub("\\^", "", symbol)
  else
    symbol2 <- symbol
  file_name <- paste(symbol2,'.csv', sep='')
  if(file.exists(file_name)){
    current_time_series <- read.zoo(file_name, sep=",", index.column = 1, header = TRUE, format="%Y-%m-%d")
    current_time_series <- as.xts(current_time_series)
  }
  else {
    getSymbols(c(symbol), from = "2016-01-01", src="yahoo", env = FinDataYH)
    FinData = as.list(FinDataYH)
    
    current_time_series <- FinData[[symbol2]][,paste(symbol2,'.Close', sep='')]
    current_time_series <- make.index.unique(current_time_series,drop=TRUE)
    current_time_series <- na.omit(current_time_series)
    write.zoo(current_time_series, sep=",", file=file_name)
  }
  attr(current_time_series, 'frequency') <- 1 #If we dont change frequency to 1=daily many methods do not work
  return(current_time_series)
}

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

get_predicted_output_price <- function(model, ts_validation_X, batch_size, ts_values_dl_max,ts_values_dl_min, ts_values_lo_1){
  predicted_output <- model %>% predict(ts_validation_X, batch_size = batch_size)
  
  # transformation back to absolute numbers:
  predicted_output_dif <- (predicted_output +1)*(ts_values_dl_max - ts_values_dl_min)/2 + ts_values_dl_min
  predicted_output_log <- c(ts_values_lo_1, ts_values_lo_1 + cumsum(predicted_output_dif)) # undo diff()
  return(exp(predicted_output_log))
}

# evaluate the model on a dataset, returns RMSE in transformed units
evaluate_epoch <- function(model, epoch_id, ts_validation_X, ts_validation_y, batch_size, ts_values_dl_max,ts_values_dl_min, ts_values_lo_1, n_train_elem){
  predicted_output_price <- get_predicted_output_price(model, ts_validation_X, batch_size, ts_values_dl_max,ts_values_dl_min, ts_values_lo_1)
  
  # Error term: root mean square error (RMSE)
  rmse_ <- c(epoch_id, RMSE(predicted_output_price, ts_validation_y),
             RMSE(predicted_output_price[1:n_train_elem], ts_validation_y[1:n_train_elem]),
             RMSE(predicted_output_price[(n_train_elem+1):length(ts_index)], ts_validation_y[(n_train_elem+1):length(ts_index)])
  )
  names(rmse_) <- c("Epoch_ID", "Overall", "Train", "Test")
  return(rmse_)
}


plotTrainProgress <- function(model){
  predicted_output_price <- get_predicted_output_price(model, ts_validation_X, batch_size, ts_values_dl_max,ts_values_dl_min, ts_values_lo[[1,2]])
  ts_validation_y_xts <- merge(as.xts(ts_validation_y, order.by = index(ts_values)),
                               as.xts(predicted_output_price, order.by = index(ts_values)))
  ts.plot(ts_validation_y_xts)
}

plotRMSE_collection <- function(rmse_collection){
  names(rmse_collection) <- c("Epoch_ID", "Overall", "Train", "Test")
  matplot(rmse_collection[,1], rmse_collection[-1], type = 'l', xlab = "Epoch_ID", ylab = "RMSE", col = 2:4, pch = 1, lwd = 2)
  legend("topright", legend = names(rmse_collection)[-1], pch = 1, col=2:4)
}


FinDataYH = new.env()
options("getSymbols.warning4.0"=FALSE)
options("getSymbols.yahoo.warning"=FALSE)

#BTC
btc <- downloadTimeSeries("BTC-USD")

#Gold
gold <- downloadTimeSeries("GC=F")

#EUR/USD 
eur_usd <- downloadTimeSeries("EURUSD=X")

#Dow-Jones
DJ <- downloadTimeSeries("^DJI")

#-------------------------------Matrix Preparation--------------------------------

#Creation of a matrix with all the variables to be used.
ts_values_all <- merge(lag.xts(btc,1),lag.xts(gold,1),lag.xts(eur_usd,1),lag.xts(DJ,1), btc)
#ts_values_all <- merge(lag.xts(btc,1), btc)
# rename the columns to explicitly contain the lag "(t-1)" or "(t)"
names(ts_values_all) <- c("BTC.USD(t-1)", "GC.F(t-1)", "EURUSD(t-1)", "DJI(t-1)", "BTC.USD(t)")
#names(ts_values_all) <- c("BTC.USD(t-1)", "BTC.USD(t)")

#As we will use the value of the day h to predict h+1 we take lags int the regressors variable.
for(i in names(ts_values_all)){
  ts_values_all[,i] <- na.approx(ts_values_all[,i], na.rm = FALSE) #Replace NA values by Interpolation
}


#---------------------------Auxiliar variables definition-------------------------

timeframe<-"20160208/20180430" #Time frame selected for our experiment.
ts_values <- ts_values_all[timeframe] #time series with all values
write.zoo(ts_values, sep=",", file='full_set.csv')

#---------------------------Transformation and Rescale-------------------------
ts_values_lo <- log(coredata(ts_values))
ts_values_dl <- diff(ts_values_lo, lag= 1) #calculate log returns

# rescale to (-1..1)
ts_values_dl_min <- min(ts_values_dl)
ts_values_dl_max <- max(ts_values_dl)
ts_values_re <- 2*(ts_values_dl - ts_values_dl_min)/(ts_values_dl_max - ts_values_dl_min) - 1 


#---------------------------Prepare train and test sets-------------------------
ts_index <- index(ts_values_re)
n_train_elem <- length(ts_index)-10 # last 10 rows remain for testing
train <- ts_values_re[1:n_train_elem,] 
test <- ts_values_re[(n_train_elem+1):length(ts_index),] 


# train/test split
X_col_num <- dim(ts_values_re)[[2]]

train_X <- train[, -X_col_num]
dim_ <- dim(train_X)
train_X <- array_reshape(train_X, c(dim_[[1]], 1, dim_[[2]])) 
train_y <- train[, X_col_num]

test_X <- test[, -X_col_num]
dim_ <- dim(test_X)
test_X <- array_reshape(test_X, c(dim_[[1]], 1, dim_[[2]])) 
test_y <- test[, X_col_num]

ts_validation_X <- ts_values_re[,-X_col_num]
dim_ <- dim(ts_validation_X)
ts_validation_X <- array_reshape(ts_validation_X, c(dim_[[1]], 1, dim_[[2]])) 
ts_validation_y <- coredata(ts_values)[,X_col_num]

model_file_name <- 'best_lstm_model.hdf5'

# model metaparameters
tsteps <- 1
batch_size <- 1
epochs <- 25
lahead <- 1 # number of elements ahead that are used to make the prediction
lstm_units <- 3
while(TRUE){
  model <- keras_model_sequential()
  model %>%
    layer_lstm(units = lstm_units, input_shape = c(tsteps, X_col_num-1), batch_size = batch_size,
               return_sequences = FALSE, stateful = TRUE) %>% 
    layer_dense(units = 1)
  model %>% compile(loss = 'mse', optimizer = 'rmsprop', metrics = c( metric_mean_squared_error, 'accuracy'))
  
  ptm <- proc.time()
  cat('Training\n')
  rmse_collection <- data.frame()
  j <- 1
  if(!exists("best_model_test_RMSE")){ 
    if(file.exists(model_file_name)){
      best_model <- load_model_hdf5(model_file_name, custom_objects = NULL, compile = TRUE)
      epoch_result <- evaluate_epoch(model, j, ts_validation_X, ts_validation_y, batch_size, ts_values_dl_max,ts_values_dl_min, ts_values_lo[[1,2]], n_train_elem)
      best_model_test_RMSE <- epoch_result[["Test"]]
    } else {
      best_model_test_RMSE <- 4000  
    }
  }
  
  do_restart <- FALSE
  for (i in 1:epochs) {
    # this command produces high CPU load!
    history_ <-model %>% fit(train_X, train_y, batch_size = batch_size,
                             epochs = 1, verbose = 0, shuffle = FALSE, 
                             validation_data=list(test_X, test_y))
    
    epoch_result <- evaluate_epoch(model, j, ts_validation_X, ts_validation_y, batch_size, ts_values_dl_max,ts_values_dl_min, ts_values_lo[[1,2]], n_train_elem)
    if(epoch_result[["Test"]] < best_model_test_RMSE){
      best_model_test_RMSE <- epoch_result[["Test"]]
      save_model_hdf5(model, model_file_name, overwrite = TRUE, include_optimizer = TRUE)
      print(paste("Best RMSE so far:", best_model_test_RMSE))
    }
    rmse_collection <- rbind(rmse_collection, epoch_result)
    j <- j + 1
    if(i >= 5 && epoch_result[["Test"]] > 1000)
    {
      do_restart <- TRUE
      print(paste("<RESTART>", "Best RMSE so far:", best_model_test_RMSE))
      plotRMSE_collection(rmse_collection)
      break
    }
    model %>% reset_states()
  }
  if(do_restart)
    next
  time_elapsed <- (proc.time() - ptm)[[3]]
  
  cat('Predicting\n')
  
  plotRMSE_collection(rmse_collection)
  break
}
#-------------Visual Comparison between predictions and real values.

best_model <- load_model_hdf5(model_file_name, custom_objects = NULL, compile = TRUE)
plotTrainProgress(best_model)
