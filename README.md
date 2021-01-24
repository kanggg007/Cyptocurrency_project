# Cyptocurrency Price Prediction

## Selected topic
Cryptocurrency Pricing

## Reason they selected the topic
Cryptocurrencies are getting traction as an investment opportunity. It is through understanding their behaviour that more people will apporach them as a new way to diverisfy their portfolio. 

## Description of the source of data
The data is sourced from LunarCRUSH, a platform that aims to bring transparency to cryptocurrency investing by providing clarity around community activity. This platform employs social data and machine learning to rank coins and prive traders broader visibility and context.

### Database
We present the mockup database using a SQL-based database, including an ERD of the database. 
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

