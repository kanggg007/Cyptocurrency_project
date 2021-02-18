import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from datetime import datetime as dt, timedelta
import plotly.graph_objects as go 
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sqlalchemy import MetaData
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.tools as tls
from sqlalchemy import create_engine
from sqlalchemy import Table
from sqlalchemy.orm import create_session
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sqlalchemy.ext.declarative import declarative_base
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import joblib
from dash.dependencies import Input, Output
import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from plotly.subplots import make_subplots

# setup app
app = dash.Dash()

from config2 import db_password, user_name, aws_password

url='cryptodb.crgu064gyupd.us-east-2.rds.amazonaws.com'
aws_string=f"postgresql://{user_name}:{aws_password}@{url}:5432/postgres"
engine = create_engine(aws_string)
data = pd.read_sql_query('SELECT * FROM all_coins_data', con=engine)

data1=data.dropna(subset=['close', 'open'])
data_lean=data1.fillna(0)
data_all = data_lean.copy()

##top 10 currency based on map 
data_top_10 = data_all.loc[data_all['time'] ==  '2021-01-31']
df_top_10 = data_top_10.sort_values('market_cap', ascending=False)[:10]
df_top_10_social = df_top_10[['symbol',
                             'url_shares',
                             'reddit_posts',
                             'tweets',
                             'news',
                             'youtube']]

df_top_10_social['social impact'] = df_top_10_social.sum(axis=1)


genral_market_cap = px.pie(data_frame = df_top_10, values = 'market_cap',names= 'symbol', hole=.3, title = 'Cryptocurrencies by Market Capital')
genral_social_media = px.pie(data_frame = df_top_10_social, values = 'social impact',names = 'symbol', hole=.3, title = 'Social Media Engagement by Cryptocurrency')
genral_price = px.line(data_frame= data_all, x ='time', y = 'close', color = 'symbol', title = 'Cryptocurrencies Price')

# setup layout
app.layout = html.Div([
                    html.H1(
                        children = 'MMM Cryptocurrency Dashboard',
                        style = {
                            'textAlign': 'center'
                        }
                    ),
            html.Div([dcc.Graph(id='data-plot-overview-mc', figure=genral_market_cap)], className='row3'),
            html.Div([dcc.Graph(id='data-plot-overview-social', figure=genral_social_media)], className='row3'),
            html.Div([dcc.Graph(id='data-plot-overview-price', figure=genral_price)], className='row3'),

                    dcc.Dropdown(
                        id = 'my-dropdown',
                        options = [{'label': i, 'value': i}for i in data_all['symbol'].unique()],
                        placeholder = 'please enter ticker'
                    ),

                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=dt(2015, 1, 1),
                        max_date_allowed=dt.today().date() - timedelta(days=1),
                        initial_visible_month=dt.today().date() - timedelta(days=1),
                        start_date = dt(2019, 3, 1),
                        end_date=dt.today().date() - timedelta(days=1)
                        ),
                        html.Div(id='output-container-date-picker-range'),

                    html.Div([
                    dcc.Tabs(id='tabs-styled-with-props', value='tab-1', children = [
                        dcc.Tab(label='Price', value='tab-1',children = [dcc.Graph(id = 'A')]),
                        dcc.Tab(label='Social Engagement', value='tab-2',children = [dcc.Graph(id = 'S')]),
                        dcc.Tab(label = 'LSTM Time Series Model ', value = 'tab-3', children = [dcc.Graph(id = 'L')]),
                        dcc.Tab(label = 'Arima Time Series Model ', value = 'tab-4', children = [dcc.Graph(id = 'Arima')]),
                        dcc.Tab(label='Random Forest Model', value='tab-5',children = [dcc.Graph(id = 'random-forest')]),

                    ]),
                    html.Div(id='tabs-example-content')
                ])
    ])
       
        
@app.callback(
    Output(component_id='A', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_data(start_date, end_date,selected_ticket):
    new_data =data_all.loc[data_all['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['symbol'] == selected_ticket]
    new_data_1 = new_data_1[['url_shares','reddit_posts','tweets','news','youtube','symbol','time','close']]
    new_data_1['total_socail'] = new_data_1.iloc[:,0:5].sum(axis=1)

    genral_fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    genral_fig4.add_trace(
    go.Scatter(x=new_data_1['time'], y=new_data_1['total_socail'], name="Social Engagement"),
    secondary_y=False,)
    genral_fig4.add_trace(
    go.Scatter(x=new_data_1['time'], y=new_data_1['close'], name="Price [USD]"),
    secondary_y=True,)
    genral_fig4.update_layout(title_text="Overall Social Engagement vs. Price")

    genral_fig4.update_xaxes(title_text="Date")
    genral_fig4.update_yaxes(title_text="Social Engagement", secondary_y=False)
    genral_fig4.update_yaxes(title_text="Price", secondary_y=True)

    return genral_fig4




# social media callback    
@app.callback(
    Output(component_id='S', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_social(start_date, end_date,selected_ticket):

    new_data =data_all.loc[data_all['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['symbol'] == selected_ticket]

    new_data_1 = new_data_1[['url_shares','reddit_posts','tweets','news','youtube']].apply(pd.to_numeric)
    
    new_max = 100
    new_min = 0
    new_range = new_max - new_min

    factors = ['url_shares','reddit_posts','tweets','news','youtube']
    for factor in factors:
        max_val = new_data_1[factor].max()
        min_val = new_data_1[factor].min()
        val_range = max_val - min_val
        new_data_1[factor + '_Adj'] = new_data_1[factor].apply(
            lambda x: (((x - min_val) * new_range) / val_range) + new_min)


    df3 = new_data_1[['url_shares_Adj','reddit_posts_Adj','tweets_Adj','news_Adj','youtube_Adj']]

    df3.rename(columns={
        'url_shares_Adj': 'url shares',
        'reddit_posts_Adj': 'reddit posts',
        'tweets_Adj': 'tweets',
        'news_Adj': 'news',
        'youtube_Adj': 'youtube'
    }, inplace=True)
    #categories=list(df3)[1:]
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    sum_column = df3.sum(axis=0)
    fig_radar = px.line_polar(sum_column, r = list(sum_column), theta= list(df3)[0:], line_close=True)
    fig_radar.update_traces(fill='toself')
    return fig_radar


# model 
@app.callback(
    Output(component_id='L', component_property='figure'),
    [
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_LSTM(selected_ticket):
    coin_df = data_all.loc[data_all['symbol']== selected_ticket]
    coin_df = coin_df[['time', 'close']].copy()
        
        
    cl = coin_df.close.astype('float32')
    train = cl[0:int(len(cl)*0.80)]
    scl = MinMaxScaler()
    #Scale the data
    scl.fit(train.values.reshape(-1,1))
    cl =scl.transform(cl.values.reshape(-1,1))
        #Create a function to process the data into lb observations look back slices
        # and create the train test dataset (90-10)
    def processData(coin_df,lb):
        X,Y = [],[]
        for i in range(len(coin_df)-lb-1):
            X.append(coin_df[i:(i+lb),0])
            Y.append(coin_df[(i+lb),0])
        return np.array(X),np.array(Y)
    lb=10
    X,y = processData(cl,lb)
    X_train,X_test = X[:int(X.shape[0]*0.90)],X[int(X.shape[0]*0.90):]
    y_train,y_test = y[:int(y.shape[0]*0.90)],y[int(y.shape[0]*0.90):]
        
    path_lstm = 'LSTM_models/'+selected_ticket+'_LSTM.h5'
    model = load_model(path_lstm)
        
    # model = Sequential()
    # model.add(LSTM(256,input_shape=(lb,1)))
    # model.add(Dense(1))
    # model.compile(optimizer='adam',loss='mse')
        # #Reshape data for (Sample,Timestep,Features) 
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
        #Fit model with history to check for overfitting
    model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),shuffle=False)
        # model.summary() 

        # # plt.figure(figsize=(12,8))
    Xt = model.predict(X_train)
    list1 = scl.inverse_transform(y_test.reshape(-1,1)).tolist()
    list2 = scl.inverse_transform(Xt).tolist()
    test=pd.DataFrame(list1)
    test2=pd.DataFrame(list2)   
    test_merge = test.merge(test2, left_index=True, right_index=True)
    test_merge.columns=['Predicted', 'Actual']
    genral_fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    genral_fig6.add_trace(go.Scatter(y=test_merge['Predicted'], x=test_merge.index, name="Predicted"),secondary_y=False,)
    genral_fig6.add_trace(go.Scatter(y=test_merge['Actual'], x=test_merge.index, name="Actual"),secondary_y=True,)

    return genral_fig6     
    
    #scaler = MinMaxScaler()
    #last 60 days closing price values and convert the dataframe to an array
    #last_60_days = new_df[-60:].values
    # Scale he data to be values between 0 and 1
    #last_60_days_scaled = scaler.fit_transform(last_60_days)
    #X_test = []
    #X_test.append(last_60_days_scaled)
    #X_test = np.array(X_test)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #path_lstm = 'LSTM_models/'+selected_ticket+'_LSTM.h5'
    #modle_lstm = load_model(path_lstm)
    

    #showing next business day price
    #pred_price_lstm = modle_lstm.predict(X_test)
    #pred_price_lstm = scaler.inverse_transform(pred_price_lstm)
    #pred_price_lstm = pred_price_lstm[0][0]
    
    # showing plot with actual price and predict price 
    

    ## ARIMA model price prediction 

    #path_arima = 'ARIMA_models/'+selected_ticket+'_ARIMA.h5'    
    #Arima_model = joblib.load(path_arima)
    #xt = Arima_model.forecast()


# arima model
@app.callback(
    Output(component_id='Arima', component_property='figure'),
    [
    Input(component_id='my-dropdown', component_property='value')]
    )


def update_arima(selected_ticket):
    coin_df = data_all.loc[(data_all['symbol']=='selected_ticket')]
    coin_df = coin_df[['time', 'close']].copy()
    coin_df.index=coin_df['time']
    df_close = coin_df['close']
    df_log = np.log(df_close)
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

    #path_a = 'ARIMA_models/BTC_ARIMA.h5'
    #model_a = joblib.load(path_a)

    # model_a.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),shuffle=False)
    # model_a = ARIMA(train_data, order=(1, 1, 1))  
    # fitted = model_a.fit(disp=-1)  

    model_a = ARIMA(train_data, order=(1, 1, 1))  
    fitted = model_a.fit(disp=-1)  

    fc = fitted.forecast(72, alpha=0.05)  # 95% confidence
    fc_series = pd.Series(fc, index=test_data.index)
    test_data=pd.DataFrame(test_data)
    train_data=pd.DataFrame(train_data)
    fc_series=pd.DataFrame(fc_series)

    genral_fig7 = make_subplots(specs=[[{"secondary_y": False}]])
    genral_fig7.add_trace(
        go.Scatter(y=test_data['close'], x=test_data.index, name="Actual Price"),
        secondary_y=False,
        )

    genral_fig7.add_trace(
        go.Scatter(y=train_data['close'], x=train_data.index, name="Training"),
        secondary_y=False,
        )
    genral_fig7.add_trace(
        go.Scatter(y=fc_series[0], x=fc_series.index, name="Predicted Price"),
        secondary_y=False,
        )

    return genral_fig7




# random forest 

@app.callback(
    Output(component_id='random-forest', component_property='figure'),
    [
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_DF(selected_ticket):
    new_data = data_all.loc[data_all['symbol'] == selected_ticket]
    data1=new_data.dropna(subset=['close', 'open'])
    data_lean=data1.fillna(0)
    new_data = data_lean.copy()
    #new_data.drop('Unnamed: 0')

    target = new_data['close']
    inputs = new_data.drop(columns=["close", "index", "asset_id", "time", "symbol"])
    from sklearn.preprocessing import StandardScaler
    # Create a StandardScaler instance
    scaler = StandardScaler()
    # Fit the StandardScaler
    input_scaler = scaler.fit(inputs)
    input_scaled = input_scaler.transform(input_scaler)
    #X_train, y_train = train_test_split(inputs, target, random_state=1)
    #X_scaler = scaler.fit(X_train)
    #X_train_scaled = X_scaler.transform(X_train)
    #X_scaler = scaler.fit(X_train)
    #X_train_scaled = X_scaler.transform(X_train)
    #X_test_scaled = X_scaler.transform(X_test)
    #rf_model = RandomForestRegressor(n_estimators=128, random_state=78)
    #rf_model = rf_model.fit(X_train_scaled, y_train)

    #input_scaler = scaler.fit_transform(inputs)
    #predicted value
    import joblib
    load_model = joblib.load('DF_models/'+selected_ticket+'_RF.HDF5')
    y_pred = load_model.predict(input_scaled)
    prices_df =pd.DataFrame(list(zip(y_pred,target)), columns=['Predicted', 'Actual'])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=prices_df['Actual'],
            title = 'Actual'
        ))

    fig.add_trace(
        go.Scatter(
            y=prices_df['Predicted'],
            title = 'Predicted'
        ))
    fig.show()
    return fig

    #plot compare predicted vs real
   
@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-styled-with-props', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
        ])
    elif tab == 'tab-2':
        return html.Div([
        ])
    elif tab == 'tab-3':
        return html.Div([
        ])
    elif tab == 'tab-4':
        return html.Div([
        ])
    elif tab == 'tab-5':
        return html.Div([
    ])
   
   


if __name__ == '__main__':
    app.run_server(debug=True)