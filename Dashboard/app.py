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
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True



data =  pd.read_csv('test_data')
data['time'] = pd.to_datetime(data['time'])

genral_fig = px.line(data, x = 'time', y='close', color ='name', title = 'Top 5 currencies')

# Main Div (1st level)
app.layout = html.Div([

    # Sub-Div (2nd level)
    # Dashboard Title
    html.Div([html.H1(children='Cryptocurrency Dashboard',
                      className='twelve columns',
                      style={'text-align': 'center',
                             'margin': '2% 0% 3% 0%',
                             'letter-spacing': 2})
              ], className='title'),
    
             html.Div([dcc.Graph(id='data-plot-overview', figure=genral_fig)], className='row3'),

    # Sub-Div (2nd level)
    # DropDown
             dcc.Dropdown(
                        id = 'my-dropdown',
                        options = [{'label': i, 'value': i}for i in data['name'].unique()],
                        placeholder = 'please enter ticker',
                        style={'height': '40px',
                                  'fontSize': 20,
                                  'margin': '2% 0% 7% 0%',
                                  'textAlign': 'center'}
                    ),
    # Sub-Div (2nd level)
    # Date picker and Button
   
        # Sub-Div (3rd level)
        # Date Picker
              dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=dt(2015, 1, 1),
                        max_date_allowed=dt.today().date() - timedelta(days=1),
                        initial_visible_month=dt.today().date() - timedelta(days=1),
                        end_date=dt.today().date() - timedelta(days=1)
                        ),
                        html.Div(id='output-container-date-picker-range'),

        # Update Button
        html.Button(id='update-button',
                    children='Analyze',
                    n_clicks=0,
                    style={'fontSize': 14,
                           'fontWeight': 'normal',
                           'height': '40px',
                           'width': '150px'},
                    className='two columns button-primary'),

    # Sub-Div (2nd level)
    # Stocks Graph
    html.Div([dcc.Graph(id='data-plot2')], className='row2'),
    html.Div([dcc.Graph(id='data-plot3')], className='row3')

], className='ten columns offset-by-one')

class StartDateError(Exception):
    pass

class NoneValueError(Exception):
    pass

class TicketSelectError(Exception):
    pass






@app.callback(
            Output(component_id='data-plot2', component_property='figure'),
            [
             Input(component_id= 'my-dropdown', component_property= 'value'),
             Input(component_id='date-picker-range',component_property ='start-date'),
             Input(component_id='date-picker-range',component_property = 'end-date'),
             Input(component_id='update-button', component_property='n_clicks'),])
def update_graph(n_clicks,selected_ticket, start_date, end_date):

    empty_layout = dict(data=[], layout=go.Layout(title=f' machine model',
                                                  xaxis={'title': 'Date'},
                                                  yaxis={'title': 'social influence'},
                                                  font={'family': 'verdana', 'size': 15, 'color': '#606060'}))
    
    if n_clicks >0:
        try:
            if start_date is None or end_date is None or selected_ticket is None:
                raise NoneValueError("ERROR : Start/End date or selected symbols is None!")
            if start_date > end_date:
                raise StartDateError("ERROR : Start date is greater than End date!")
            if len(selected_ticket) == 0:
                raise TicketSelectError("ERROR : No stocks selected!")
            
            new_data =data.loc[data['time'].between(start_date, end_date)]
            new_data_1 = new_data.loc[new_data['name'] == selected_ticket]
    
    
            line_fig = px.line(new_data_1,
                            x='time', y='close',
                            title=f'{selected_ticket} Prices')
            return line_fig

        except StartDateError as e:
            print(e)
            return empty_layout
        except NoneValueError as e:
            print(e)
            return empty_layout
        except TicketSelectError as e:
            print(e)
            return empty_layout
        except Exception as e:
            print(e)
    else:
        return empty_layout


if __name__ == '__main__':
    app.run_server(debug=True)