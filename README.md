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
=======
# Cyptocurrency Price Prediction

## Selected topic
Cryptocurrency Pricing

## Reason they selected the topic
Cryptocurrencies are getting traction as an investment opportunity. It is through understanding their behaviour that more people will apporach them as a new way to diverisfy their portfolio. 

## Description of the source of data
The data is sourced from LunarCRUSH, a platform that aims to bring transparency to cryptocurrency investing by providing clarity around community activity. This platform employs social data and machine learning to rank coins and prive traders broader visibility and context.

#### SQL
In SQL, we imported BTC.csv which includes crypto currency daily price change and the data for social media feeds volume and score for discussion of specifying cryptocurrency from Python. In the dataset, timestamp info is transferred to DateTime data for further comparison with other cryptocurrencies. After the import, we have performed the completeness check over the imported data from python with no variance noted.


#### ERD
We have 1 dataset that includes cryptocurrency daily price change and the data for social media feeds volume and score. For the purpose of the mockup database, we use Bitcoin to analyze the trend between social media feeds and the time series model to analyze the potential fluctuations of the currency. In the next module, we will be using other cryptocurrency and we will use the dates as a key to merge different currencies and to analyze there is an impact when other currencies has high discussion threads.

![](/img/BTC_test_QuickDBD.png)

## Questions they hope to answer with the data
The main objective is to predict the price of the different curriencies
Establish differences among top cryptocurrencies available
Determine the social media impact on its price

## Description of the communication protocols
Given the limitations due to the pandemic, our communication has been reduced to Slack and video conferences.

## Machine model selection
Cryptocurrency price is not randomly generated values instead it can be treated as a discrete time series model which is based on a set of well-defined numerical data items collected at consecutive points at regular intervals of time. Therefore, autoregression and LTSM are two models that we have decided to apply for this project. Based on the sample model of Autoregression, we have included pretty good performance with not big dataset regarding of accuracy, around 2.5% MAPE(Mean Absolute Percentage Error) implies the model is about 97.5% accurate in predicting the test set observations.

