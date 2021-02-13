import matplotlib.pyplot as plt
from math import pi

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
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from sqlalchemy import *
import psycopg2
from sqlalchemy.orm import create_session


from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


from sklearn.preprocessing import MinMaxScaler
# from config import db_password




# setup app
app = dash.Dash()

#db_password='Kl@v5SQL'
#user_name='Crypto_Team'
#aws_password='Cryptocurrency_Project'

#url='cryptodb.crgu064gyupd.us-east-2.rds.amazonaws.com'
#aws_string=f"postgresql://{user_name}:{aws_password}@{url}:5432/postgres"
#engine = create_engine(aws_string)
#Create and engine and get the metadata
#Base = declarative_base()
#metadata = MetaData(bind=engine)

#reflect table
#coin = Table('all_coins_data', metadata, autoload=True, autoload_with=engine)
##Create a session to use the tables    
#session = create_session(bind=engine)

#Query database
#coin_list = session.query(btc).all()
#coin_df=pd.DataFrame(coin_list)
#coin_df = coin_df.drop('Unnamed: 0', axis= True)
#data = coin_df.copy()

data = pd.read_csv('coin_all.csv')

#top 10 currency based on map 
data_top_20 = data.loc[data['time'] ==  '2021-01-31']
df_top_20 = data_top_20.sort_values('market_cap', ascending=False)[:10]
df_top_20_social = df_top_20[['symbol',
                             'url_shares',
                             'reddit_posts',
                             'tweets',
                             'news',
                             'youtube']]

df_top_20_social['social impact'] = df_top_20_social.sum(axis=1)




#genral_sunburst = px.pie.gapminder()


genral_market_cap = px.pie(data_frame = df_top_20, values = 'market_cap',names= 'symbol', hole=.3)
genral_social_media = px.pie(data_frame = df_top_20_social, values = 'social impact',names = 'symbol', hole=.3)
genral_price = px.line(data_frame= data, x ='time', y = 'close', color = 'symbol')

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
                        options = [{'label': i, 'value': i}for i in data['symbol'].unique()],
                        placeholder = 'please enter ticker'
                    ),

                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=dt(2015, 1, 1),
                        max_date_allowed=dt.today().date() - timedelta(days=1),
                        initial_visible_month=dt.today().date() - timedelta(days=1),
                        end_date=dt.today().date() - timedelta(days=1)
                        ),
                        html.Div(id='output-container-date-picker-range'),

                    html.Div([
                    dcc.Tabs(id='tabs-styled-with-props', value='tab-1', children = [
                        dcc.Tab(label='Acutal Price', value='tab-1',children = [dcc.Graph(id = 'A')]),
                        dcc.Tab(label='Social Impact', value='tab-2',children = [dcc.Graph(id = 'S')]),
                        dcc.Tab(label = 'LSTM Predicted Price ', value = 'tab-3', children = [dcc.Graph(id = 'L')]),
                     ]),  
                    html.Div(id='tabs-example-content')
                ])
        
            
                    #dcc.Graph(id = 'm'),
                    #dcc.Graph(id= 'n')
    ])
        

class StartDateError(Exception):
    pass

class NoneValueError(Exception):
    pass

class TicketSelectError(Exception):
    pass




@app.callback(
    Output(component_id='A', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )
def update_data(start_date, end_date, selected_ticket):



    new_data =data.loc[data['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['symbol'] == selected_ticket]
    line_fig = px.line(new_data_1,
                    x='time', y='close',
                    title=f'{selected_ticket} Prices')

    return line_fig




# socail media callback    
@app.callback(
    Output(component_id='S', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_social(start_date, end_date,selected_ticket):

    new_data =data.loc[data['time'].between(start_date, end_date)]
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
        'url_shares_Adj': 'url_shares',
        'reddit_posts_Adj': 'reddit_posts',
        'tweets_Adj': 'tweets',
        'news_Adj': 'news',
        'youtube_Adj': 'youtube'
    }, inplace=True)


    categories=list(df3)[1:]
    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:


    sum_column = df3.sum(axis=0)


    fig = px.line_polar(sum_column, r = list(sum_column), theta= list(df3)[0:], line_close=True)
    fig.update_traces(fill='toself')
    return fig


# model 
@app.callback(
    Output(component_id='l', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_LSTM(start_date, end_date,selected_ticket):
    new_data = data[['time','symbol','close']]
    new_data =new_data.loc[data['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['symbol'] == selected_ticket]

    new_data_2 = new_data_1.copy()
    new_data_2.index = new_data_2.time
    new_data_2.drop('time', axis=1, inplace=True)
    new_data_2.drop('symbol', axis=1, inplace= True)
    final_dataset = new_data_2['close'].values.reshape(-1,1)
    train_data=final_dataset[0:int(len(final_dataset)*0.80)]
    valid_data=final_dataset[int(len(final_dataset)*0.80):]
    scaler = MinMaxScaler()
    #Scale the data
    scaler.fit(train_data)
    scaled_data =scaler.transform(final_dataset.values.reshape(-1,1))
    

    
    x_train_data,y_train_data=[],[]
    for i in range(10,len(train_data)):
        x_train_data.append(scaled_data[i-10:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
    model=load_model("saved_model.h5")

    inputs_data=new_data_2[len(new_data_2)-len(valid_data)-10:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)

    X_test=[]
    for i in range(10,inputs_data.shape[0]):
        X_test.append(inputs_data[i-10:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)

    train=new_data_2[0:int(len(final_dataset)*0.80)]
    valid=new_data_2[int(len(final_dataset)*0.80):]
    valid['Predictions']=closing_price

    line_fig_LSTM = px.line(new_data_2,
                            x= train.index, y=valid["Predictions"],mode = 'markers',
                            title ='LSTM')
    return line_fig_LSTM





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
   


if __name__ == '__main__':
    app.run_server(debug=True)