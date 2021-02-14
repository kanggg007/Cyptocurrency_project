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
import joblib
from config2 import db_password, user_name, aws_password
from dash.dependencies import Input, Output

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
                    
                    html.H3(
                        children = 'Select Cryptocurrency',
                        style = {
                            'textAlign': 'left'
                        }
                    ),
                    dcc.Dropdown(
                        id = 'my-dropdown',
                        options = [{'label': i, 'value': i}for i in data['name'].unique()],
                        placeholder = 'please enter ticker',
                        style={'width': '50%'}
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
                    html.H3(
                        children = 'Enter social media volume',
                        style = {
                            'textAlign': 'left'
                        }
                    ),
                    html.Div([
                    # dcc.Input(id="dfalse", type="number", placeholder="Debounce False"),
                    # dcc.Input(
                    #     id="dtrue", type="number",
                    #     debounce=True, placeholder="Debounce True",
                    # ),
                    html.Br(),
                    dcc.Input(
                    id="reddit", type="number", placeholder="reddit posts",
                    min=100, max=100000, step=1,
                    style={'width': '10%'}
                    ),
                    dcc.Input(
                    id="tweets", type="number", placeholder="tweets",
                    min=100, max=100000, step=1,
                    style={'width': '10%'}
                    ),
                    dcc.Input(
                    id="news", type="number", placeholder="news",
                    min=100, max=100000, step=1,
                    style={'width': '10%'}
                    ),
                    dcc.Input(
                    id="youtube", type="number", placeholder="youtube",
                    min=100, max=100000, step=1,
                    style={'width': '10%'}
                    ),
                    dcc.Input(
                    id="url_shares", type="number", placeholder="url_shares",
                    min=100, max=100000, step=1,
                    style={'width': '10%'}
                    ),
                    html.Hr(),
                    html.Div(id='output')
                    # html.Div([dcc.Graph(id='data-plot-overview4', figure=genral_fig4)], className='row8'),
                    # html.Div([dcc.Graph(id='data-plot-overview5', figure=genral_fig5)], className='row9'),
                    ]),
                    html.Div([
                    dcc.Tabs(id='tabs-styled-with-props', value='tab-1', children = [
                        dcc.Tab(label='Acutal Price', value='tab-1',children = [dcc.Graph(id = 'm')]),
                        dcc.Tab(label='Social Radar', value='tab-2',children = [dcc.Graph(id = 'i')]),
                        dcc.Tab(label='Social Impact', value='tab-3',children = [dcc.Graph(id = 'n')]),
                        dcc.Tab(label = 'LSTM Predicted Price ', value = 'tab-4', children = [dcc.Graph(id = 'l')]),
                        dcc.Tab(label = 'Arima Predicted Price ', value = 'tab-5', children = [dcc.Graph(id = 'k')]),
                        dcc.Tab(label = 'Random Forest Predicted Price ', value = 'tab-6'),
                        
                        ]),  
                    html.Div(id='tabs-example-content'),



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
    Output('output', "children"),
    Input("reddit", "value"),
    Input("tweets", "value"),
    Input("news", "value"),
    Input("youtube", "value"),
    Input("url_shares", "value"),
)   
def number_render(reddit, tweets, news, youtube, url_shares):
    return u'reddit {}, tweets {}, news {}, youtube {}, url_shares {}'.format(reddit, tweets, news, youtube, url_shares)



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

# def update_LSTM(selected_ticket):
    # model_imported = tf.keras.models.load_model("../../LSTM_models/BTC_LSTM.H5")

    # loaded_model = joblib.load('../../LSTM_models/BTC_LSTM)
    # result = as
    # new_data = pd.read_csv('test_data')[['time','name','close']]
    # new_data =new_data.loc[data['time'].between(start_date, end_date)]
    # data_1 = data.loc[data['name'] == selected_ticket]
    # data_1 =data_1.dropna()
    # data_1=data_1[['time', 'close']].copy()
    
    # from sklearn.preprocessing import MinMaxScaler
    # cl = data_1_clean.close.astype('float32')
    # train = cl[0:int(len(cl)*0.80)]
    # scl = MinMaxScaler()
    #     #Scale the data
    # scl.fit(train.values.reshape(-1,1))
    # cl =scl.transform(cl.values.reshape(-1,1))
    # #Create a function to process the data into lb observations look back slices
    # # and create the train test dataset (90-10)
    # def processData(coin_df,lb):
    #     X,Y = [],[]
    #     for i in range(len(coin_df)-lb-1):
    #         X.append(coin_df[i:(i+lb),0])
    #         Y.append(coin_df[(i+lb),0])
    #     return np.array(X),np.array(Y)
    # lb=10
    # X,y = processData(cl,lb)
    # X_train,X_test = X[:int(X.shape[0]*0.90)],X[int(X.shape[0]*0.90):]
    # y_train,y_test = y[:int(y.shape[0]*0.90)],y[int(y.shape[0]*0.90):]

    # import keras
    # from keras import Sequential
    # from keras.layers import LSTM
    # from keras.layers import Dense

    # model = Sequential()
    # model.add(LSTM(256,input_shape=(lb,1)))
    # model.add(Dense(1))
    # model.compile(optimizer='adam',loss='mse')
    # #Reshape data for (Sample,Timestep,Features) 
    # X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    # X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    # #Fit model with history to check for overfitting
    # history = model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),shuffle=False)
    # plt.figure(figsize=(12,8))
    # Xt = model.predict(X_train)
    # plt.plot(scl.inverse_transform(y_train.reshape(-1,1)), label="Actual")
    # plt.plot(scl.inverse_transform(Xt), label="Predicted")
    # plt.legend()
    # plt.title("Train Dataset")  
    # line_fig_LSTM = go.line(new_data_2,
    #                         x= train.index, y=valid["Predictions"],mode = 'markers',
    #                         title ='LSTM')
    # return line_fig_LSTM

#  app.layout = html.Div([
#     dcc.Textarea(
#         id='textarea-example',
#         value='Textarea content initialized\nwith multiple lines of text',
#         style={'width': '100%', 'height': 300},
#     ),
#     html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})
#     ])


# @app.callback(
#     Output('textarea-state-example-output', 'children'),
#     Input('textarea-state-example-button', 'n_clicks'),
#     State('input1', 'value'),State('input2', 'value'),State('input3', 'value'),State('input4', 'value'),State('input5', 'value')
# )
# def update_output(n_clicks, value):
#     if n_clicks > 0:
#         return 'You have entered: \n{}'.format(value)

def update_social(selected_ticket):
    # load, no need to initialize the loaded_rf
    rf = joblib.load("../DF_models/BTC_RF.HDF5")
    #separate inputs and output
    coin_df_clean = data.loc[data['name'] == 'Bitcoin']
    target = coin_df_clean['close']
    inputs = coin_df_clean.drop(columns=["close", "index", "asset_id", "time", "symbol"])
    #separate inputs and output
    target = coin_df_clean['close']
    inputs = coin_df_clean.drop(columns=["close", "index", "asset_id", "time", "symbol"])
    #predicted value
    y_pred = rf.predict(input_scaled)
    
    return y_pred

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