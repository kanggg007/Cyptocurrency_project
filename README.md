# Cyptocurrency_project

Auto Regression Time series analysis:
Concept:
BTC price is not randomly generated values instead it can be treated as a discrete time series model which is based on a set of well-defined numerical data items collected at consecutive
 points at regular intervals of time 
Step:
•	Load data and keep only open, close and date columns 
•	Plot data to see if there is a time series correlation 
•	Test if a series is stationary by using ADF test and separate seasonality if necessary 
•	Fit into auto_arima model 


Results:
•	Based on the report performance, around 2.5% MAPE(Mean Absolute Percentage Error) implies the model is about 97.5% accurate in predicting the test set observations. 



LSTM model:

There is another model we believe that it will be good way to predict the trend of value of a cryptocurrency in the future is LSTM (Long short-term memory) layers.

Concept:

Long short-term memory (LSTM) is a type of recurrent neural network (RNN) and powerful to model sequence data because it maintains an internal state to keep track of the data it has already seen.

Steps:
•	Data load
•	Data Normalization
•	Data split 
•	Build an LSTM model 
•	Plot prediction and actual price 


Results:
•	Based on the results, it seems like prediction did not really stick with actual price and we believe that it could be more accurate if we input more data 
