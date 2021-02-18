import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
import plotly.express as px
from datetime import date


app = dash.Dash()

data =  pd.read_csv('test_data')
data['time'] = pd.to_datetime(data['time'])

genral_fig_1 = px.line(data, x = 'time', y='close', color ='name', title = 'Top 5 currencies')
genral_fig_2 = px.line(data, x = 'time', y='close', color ='name', title = 'Top 5 currencies')
genral_fig_3 = px.line(data, x = 'time', y='close', color ='name', title = 'Top 5 currencies')




app.layout = html.Div([
        html.Div([
           html.H1(children='Cryptocurrency Dashboard',
                      className='twelve columns',
                      style={'text-align': 'center',
                             'margin': '2% 0% 3% 0%',
                             'letter-spacing': 2}),],classname= 'title'),
        
        html.Div([
            dcc.Graph(id='data-plot-overview_1', figure=genral_fig_1),
            dcc.Graph(id='data-plot-overview_1', figure=genral_fig_1),
            dcc.Graph(id='data-plot-overview_1', figure=genral_fig_1)
        ], classname = 'overview-general'),


        html.Div([
            dcc.Dropdown(
                            id = 'my-dropdown',
                            options = [{'label': i, 'value': i}for i in data['name'].unique()],
                            placeholder = 'please enter ticker',
                            style={'height': '40px',
                                    'fontSize': 20,
                                    'margin': '2% 0% 7% 0%',
                                    'textAlign': 'left'}
                        )
        ], classname='dropdown'),


        html.Div([
            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=dt(2015, 1, 1),
                                max_date_allowed=dt.today().date() - timedelta(days=1),
                                initial_visible_month=dt.today().date() - timedelta(days=1),
                                end_date=dt.today().date() - timedelta(days=1)
                                )
        ], classname = 'data-picker'),

        html.Div([
            html.Button(id='update-button',
                            children='Analyze',
                            n_clicks=0,
                            style={'fontSize': 14,
                                'fontWeight': 'normal',
                                'height': '40px',
                                'width': '150px'},)
        ], classname = 'button')

])



if __name__ == '__main__':
    app.run_server(debug=True)