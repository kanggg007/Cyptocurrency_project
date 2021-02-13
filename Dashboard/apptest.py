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
from importlib import metadata
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.tools as tls
from sqlalchemy import create_engine
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sqlalchemy.ext.declarative import declarative_base
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from config2 import db_password, user_name, aws_password

# data =  pd.read_csv('test_data')

# from config import db_password
from config2 import password
#cloud
#cloud
url='cryptodb.crgu064gyupd.us-east-2.rds.amazonaws.com'
aws_string=f"postgresql://{user_name}:{aws_password}@{url}:5432/postgres"
engine = create_engine(aws_string)


# db_string = f"postgres://postgres:{password}@localhost/cryptocurrency_db"
# engine = create_engine(db_string)
data = pd.read_sql_query('SELECT * FROM all_coins_data', con=engine)
data1 = pd.read_sql_query('SELECT * FROM crypto_id', con=engine)
data = pd.merge(data, data1, how="left", on=["asset_id", "asset_id"])
genral_fig = px.line(data, x = 'time', y='close', color ='name', title = 'Top 20 currencies')


# setup app
app = dash.Dash()

#Plot top 20 currencies
df_top_20 = data.loc[data['time'] == '2021-01-25']
df_top_20 = df_top_20.sort_values('market_cap', ascending=False)
df_top_20 = df_top_20.reset_index()
df_top_20mc = df_top_20[['name','market_cap']]
df_top_20mc
# Set the x-axis to a list of strings for each month.
x_axis = df_top_20mc['name']
y_axis = df_top_20mc['market_cap']

genral_fig2 = px.bar(df_top_20mc, x=x_axis, y=y_axis, title="Top Market Cap cryptocurrencies")
# Plot social medial
df_top_20_social = df_top_20[['name',
                             'url_shares',
                             'reddit_posts',
                             'tweets',
                             'news',
                             'youtube']]
df_top_20_social['sum'] = df_top_20_social.sum(axis=1)
genral_fig3 = px.pie(df_top_20_social, values=df_top_20_social['sum'], names=df_top_20_social['name'], title='Social media posts of selected currency vs total posts')

# selected coin
# coin_df = data.loc[(data['asset_id']=='2')]
# genral_fig4 = px.line(coin_df, x = coin_df['time'], y=[coin_df['reddit_posts'], coin_df['close']], title = 'Reddit Posts vs. Price')

df2=df_top_20_social.drop(['sum'], axis=1)
df2[['url_shares','reddit_posts','tweets','news','youtube']]=df2[['url_shares','reddit_posts','tweets','news','youtube']].apply(pd.to_numeric)
new_max = 100
new_min = 0
new_range = new_max - new_min
factors = ['url_shares','reddit_posts','tweets','news','youtube']
for factor in factors:
  max_val = df2[factor].max()
  min_val = df2[factor].min()
  val_range = max_val - min_val
  df2[factor + '_Adj'] = df2[factor].apply(
      lambda x: (((x - min_val) * new_range) / val_range) + new_min)


# setup layout
app.layout = html.Div([
                    html.H1(
                        children = 'MMM Cryptocurrency Dashboard',
                        style = {
                            'textAlign': 'center'
                        }
                    ),
            html.Div([dcc.Graph(id='data-plot-overview', figure=genral_fig)], className='row3'),
            html.Div([dcc.Graph(id='data-plot-overview2', figure=genral_fig2)], className='row4'),
            html.Div([dcc.Graph(id='data-plot-overview3', figure=genral_fig3)], className='row5'),
                    
                    html.H1(
                        children = 'Select Cryptocurrency',
                        style = {
                            'textAlign': 'left'
                        }
                    ),
                    dcc.Dropdown(
                        id = 'my-dropdown',
                        options = [{'label': i, 'value': i}for i in data['name'].unique()],
                        placeholder = 'please enter ticker'
                    ),
                    

                    # dcc.DatePickerRange(
                    #     id='date-picker-range',
                    #     min_date_allowed=dt(2015, 1, 1),
                    #     max_date_allowed=dt.today().date() - timedelta(days=1),
                    #     initial_visible_month=dt.today().date() - timedelta(days=1),
                    #     end_date=dt.today().date() - timedelta(days=1)
                    #     ),
                    #     html.Div(id='output-container-date-picker-range'),

                    
                    

                    html.Button(
                        id='update-button', 
                        children = 'Submit',
                        n_clicks = 0,  
                    ),
                    # html.Div([dcc.Graph(id='data-plot-overview4', figure=genral_fig4)], className='row8'),
                    # html.Div([dcc.Graph(id='data-plot-overview5', figure=genral_fig5)], className='row9'),

                    html.Div([
                    dcc.Tabs(id='tabs-styled-with-props', value='tab-1', children = [
                        dcc.Tab(label='Acutal Price', value='tab-1',children = [dcc.Graph(id = 'm')]),
                        dcc.Tab(label='Social Radar', value='tab-2',children = [dcc.Graph(id = 'i')]),
                        dcc.Tab(label='Social Impact', value='tab-3',children = [dcc.Graph(id = 'n')]),
                        dcc.Tab(label = 'LSTM Predicted Price ', value = 'tab-4', children = [dcc.Graph(id = 'l')]),
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
    Output(component_id='m', component_property='figure'),
    [
    # Input(component_id='date-picker-range', component_property='start_date'),
    # Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )
def update_data(selected_ticket):
    # new_data =data.loc[data['time'].between(start_date, end_date)]
    # new_data_1 = new_data.loc[new_data['name'] == selected_ticket]
    coin_df = data.loc[(data['name']==selected_ticket)]
    genral_fig4 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
    genral_fig4.add_trace(
    go.Scatter(x=coin_df['time'], y=coin_df['tweets'], name="Tweets"),
    secondary_y=False,
    )

    genral_fig4.add_trace(
    go.Scatter(x=coin_df['time'], y=coin_df['close'], name="Price"),
    secondary_y=True,
    )

# Add figure title
    genral_fig4.update_layout(
    title_text="Tweets vs. Price"
    )

# Set x-axis title
    genral_fig4.update_xaxes(title_text="Date")

# Set y-axes titles
    genral_fig4.update_yaxes(title_text="Tweets", secondary_y=False)
    genral_fig4.update_yaxes(title_text="Price", secondary_y=True)
    
    # genral_fig4.add_trace(
    # go.Scatter(x=coin_df['time'], y=[coin_df['reddit_posts']]),
    # secondary_y=True)
    # genral_fig4.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)
    # line_fig = px.line(new_data_1,
    #                 x='time', y='close',
    #                 title=f'{selected_ticket} Prices')

    return genral_fig4
    
@app.callback(
    Output(component_id='n', component_property='figure'),
    [
    # Input(component_id='date-picker-range', component_property='start_date'),
    # Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_social(selected_ticket):
    # new_data =data.loc[data['time'].between(start_date, end_date)]
    new_data_1 = data.loc[(data['name']==selected_ticket)]
    

    line_fig_social = px.bar(new_data_1,
                            x='time', y= ['news','reddit_posts','tweets','youtube'],
                            title='soical impact')


    return line_fig_social

@app.callback(
    Output(component_id='i', component_property='figure'),
    [
    # Input(component_id='date-picker-range', component_property='start_date'),
    # Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_social(selected_ticket):
    df3 = df2.loc[:, ['name', 'url_shares_Adj','reddit_posts_Adj','tweets_Adj','news_Adj','youtube_Adj']]

    df3.rename(columns={
    'url_shares_Adj': 'url_shares',
    'reddit_posts_Adj': 'reddit_posts',
    'tweets_Adj': 'tweets',
    'news_Adj': 'news',
    'youtube_Adj': 'youtube'
    }, inplace=True)
    from math import pi
    categories=list(df3)[1:]
    N = len(categories)
    
    r=df3.loc[df3['name']==selected_ticket].drop(columns=['name']).values.flatten().tolist()
    df4 = pd.DataFrame(dict(r=r,theta=categories))

    genral_fig5 = px.line_polar(df4, r='r', theta='theta', line_close=True)    

    return genral_fig5

@app.callback(
    Output(component_id='l', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_LSTM(start_date, end_date,selected_ticket):
    new_data = pd.read_csv('test_data')[['time','name','close']]
    new_data =new_data.loc[data['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['name'] == selected_ticket]

    new_data_2 = new_data_1.copy()
    new_data_2.index = new_data_2.time
    new_data_2.drop('time', axis=1, inplace=True)
    new_data_2.drop('name', axis=1, inplace= True)
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

    line_fig_LSTM = go.line(new_data_2,
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