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


from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


from sklearn.preprocessing import MinMaxScaler


data =  pd.read_csv('test_data')
genral_fig = px.line(data, x = 'time', y='close', color ='name', title = 'Top 5 currencies')


# setup app
app = dash.Dash()









# setup layout
app.layout = html.Div([
                    html.H1(
                        children = 'MMM Cryptocurrency Dashboard',
                        style = {
                            'textAlign': 'center'
                        }
                    ),
            html.Div([dcc.Graph(id='data-plot-overview', figure=genral_fig)], className='row3'),

                    dcc.Dropdown(
                        id = 'my-dropdown',
                        options = [{'label': i, 'value': i}for i in data['name'].unique()],
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


                    html.Button(
                        id='update-button', 
                        children = 'Submit',
                        n_clicks = 0,  
                    ),

                    html.Div([
                    dcc.Tabs(id='tabs-styled-with-props', value='tab-1', children = [
                        dcc.Tab(label='Acutal Price', value='tab-1',children = [dcc.Graph(id = 'm')]),
                        dcc.Tab(label='Social Impact', value='tab-2',children = [dcc.Graph(id = 'n')]),
                        dcc.Tab(label = 'LSTM Predicted Price ', value = 'tab-3', children = [dcc.Graph(id = 'l')]),
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
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )
def update_data(start_date, end_date, selected_ticket):
    new_data =data.loc[data['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['name'] == selected_ticket]
    line_fig = px.line(new_data_1,
                    x='time', y='close',
                    title=f'{selected_ticket} Prices')

    return line_fig
    
@app.callback(
    Output(component_id='n', component_property='figure'),
    [
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='my-dropdown', component_property='value')]
    )

def update_social(start_date, end_date,selected_ticket):
    new_data =data.loc[data['time'].between(start_date, end_date)]
    new_data_1 = new_data.loc[new_data['name'] == selected_ticket]
    

    line_fig_social = px.bar(new_data_1,
                            x='time', y= ['news','reddit_posts','tweets','youtube'],
                            title='soical impact')

    return line_fig_social


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