import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np

# Your general exploration objective is to understand whether and how , 
# life expectancy correlates with: 
#   population, (DONE)
#   GDPperCapita, and (DONE) 
#   continent (DONE)
# over the time period covered by the dataset. (DONE) 
# 
# Interesting exploration objectives to consider are:

# 1. The evolution of the gap between high, middle and low income countries over time. The gap can be based on life expectancy or GDP per capita. (DONE)

# 2. The slope of the linear dependency between life expectancy and GDP per capita over time. (DONE)

# 3. How the GDP distribution per capita changes over time. (DONE)

# 4. What is the world GDP and world life expectancy over time as a reference level. The world GDP per capita / life expectancy is a weighted average of GDP per capita / life expectancy over the 
# world countries, where the weight is the population of each country. (DONE)

# Instead of a focus on continents, you may focus on the income level defined based on GDP per capita in US$ as: 
# - Low-income (0-1000), lower middle income (1000-4000), 
# - Upper middle income (4000-13500), and 
# - High income countries (13500+). 

# In the visualizations that do not explicitly represent GDP per capita you can use different colour for the different income levels. Countries near income level boundaries may switch income levels over time, which can form the basis of an exploration objective. 

# Load dataset
data = px.data.gapminder()

data['IncomeGroup'] = pd.cut(data['world_gdp'],
                        bins=[1000, 4000, 7000, float('inf')], 
                        labels=['Lower middle (1000 - 4000)', 'Upper middle (4001 - 7000)', 'High (7000+)'])
data['LifeExpectancyGroup'] = pd.cut(data['world_life_exp'],
                                    bins=[40, 50, 60, 101], 
                                    labels=['Lower middle (40 - 50)', 'Upper middle (51 - 60)', 'High (60+)'])

app = dash.Dash(__name__)

@app.callback(
    [Output('g1-continent-dropdown', 'style'),
     Output('g1-dropdown-header', 'style'),
     Output('g1-dropdown-header', 'children'),
     Output('g3-continent-dropdown', 'style'),
     Output('g3-dropdown-header', 'style'),
     Output('g3-dropdown-header', 'children')],
    [Input('g1-radio-value', 'value'),
     Input('g3-radio-value', 'value')]
)
def toggle_dropdown(g1_radio_value, g3_radio_value):
    def handle_visibility(radio_value):
        if radio_value == "Countries":
            return {'display': 'block', 'margin-right':'4%'}, {'display': 'block'}, "Select a continent to see all of its countries' growth over time"
        else:
            return {'display': 'none'}, {'display': 'none'}, ""

    return handle_visibility(g1_radio_value) + handle_visibility(g3_radio_value)


@app.callback(
    Output('life-exp-vs-gdp', 'figure'),
    [Input('g1-continent-dropdown', 'value'), Input('g1-radio-value', 'value')]
)
def update_graph(continent_value, radio_value):
    if radio_value == 'Countries':
        dff = data[data['continent'] == continent_value]
        color_word = "country"
        hover_word = "continent"
    else:
        dff = data
        hover_word = "country"
        color_word = "continent"

    fig = px.scatter(dff, x="gdpPercap", y="lifeExp",
                     size="pop", color=color_word, hover_name=hover_word,
                     log_x=True, size_max=60, animation_frame='year',
                     animation_group="country", range_x=[100, 10000000], 
                     range_y=[0,100], labels=dict(pop="Population", 
                     gdpPercap="GDP Per Capita", lifeExp="Life Expectency"))

    return fig

def update_graph_continent():
    fig = px.scatter(data, x="continent", y="lifeExp",
                     size="pop", color="continent", hover_name="country", size_max=60, animation_frame='year',
                     animation_group="country", range_y=[0,100], labels=dict(lifeExp="Life Expectency"))

    return fig

@app.callback(
    Output('life-exp-vs-population', 'figure'),
    [Input('g3-continent-dropdown', 'value'), Input('g3-radio-value', 'value')]
)
def update_graph_population(continent_value, radio_value):
    if radio_value == 'Countries':
        dff = data[data['continent'] == continent_value]
        color_word = "country"
        hover_word = "continent"
    else:
        dff = data
        hover_word = "country"
        color_word = "continent"

    fig = px.scatter(dff, x="pop", y="lifeExp",
                     size="pop", color=color_word, hover_name=hover_word,
                     log_x=True, size_max=60, animation_frame='year',
                     animation_group="country", range_y=[0,100],range_x=[100000,2000000000], labels=dict(pop="Population", 
                     lifeExp="Life Expectency"))

    return fig

# The evolution of the gap between high, middle and low income countries over time. The gap can be based on life expectancy or GDP per capita.
@app.callback(
        Output('countries-evolution', 'figure'),
        [Input('g4-continent-dropdown', 'value'),
         Input('g4-radio-value', 'value')]
)
def country_evolution(continent_value, radio_value):
    data['IncomeGroup'] = pd.cut(data['gdpPercap'],
                            bins=[0, 1000, 4000, 13500, float('inf')], 
                            labels=['Low (0 - 1000)', 'Lower middle (1001 - 4000)', 'Upper middle (4001 - 13500)', 'High (13500+)'])
    data['LifeExpectancyGroup'] = pd.cut(data['lifeExp'],
                                    bins=[0, 40, 55, 70, 101], 
                                    labels=['Low (0 - 40)', 'Lower middle (41 - 55)', 'Upper middle (56 - 70)', 'High (71 - 100)'])

    fig = px.bar(data[data['continent']==continent_value], x='country', y='lifeExp' if radio_value == "Life Expectency" else 'gdpPercap', 
                 color='LifeExpectancyGroup' if radio_value == "Life Expectency" else 'IncomeGroup', animation_frame="year", hover_name='country')

    max_value = data['lifeExp' if radio_value == "Life Expectency" else 'gdpPercap'].max()
    fig.update_layout(yaxis=dict(range=[0, max_value]))
    return fig

# The slope of the linear dependency between life expectancy and GDP per capita over time.
def gdp_linear_regression():
    fig = px.scatter(data, x='gdpPercap', y='lifeExp', animation_frame='year', trendline='ols', trendline_options=dict(log_x=True))
    return fig


# How GDP per capita changes over time for continents
@app.callback(
    Output('gdp-over-time-1', 'figure'),
    [Input('g6-1-radio-value', 'value')]
)
def update_gdp_overtime_1(y_type):
    dff = data.groupby(['continent', 'year']).agg({"gdpPercap":"mean"}).reset_index()
    fig = px.scatter(dff, x="year", y="gdpPercap", color="continent", hover_name='continent', labels=dict(gdpPercap="GDP per capita"))
    fig.update_traces(mode="lines+markers")
    fig.update_yaxes(type=y_type)
    fig.update_layout(hovermode='closest')
    return fig

# How GDP per capita changes over time for countries
@app.callback(
    Output('gdp-over-time-2', 'figure'),
    [Input('gdp-over-time-1', 'hoverData'),
     Input('g6-2-continent-dropdown', 'value'),
     Input('g6-2-radio-value', 'value'),
    ]
)
def update_gdp_overtime_2(hoverData, continent_value, y_type):
    dff= data
    if hoverData:
        dff = data[data['continent']==hoverData['points'][0]['hovertext']]
    elif continent_value:
        dff = data[data['continent'] == continent_value]
    fig = px.scatter(dff, x="year", y="gdpPercap", color="country", hover_name='country', labels=dict(gdpPercap="GDP per capita"))
    fig.update_traces(mode="lines+markers")
    fig.update_yaxes(type=y_type)
    fig.update_layout(hovermode='closest')
    return fig

# World GDP/ Life expectency as a reference level
@app.callback(
        Output('world-gdp-life-exp', 'figure'),
        [Input('g7-radio-value', 'value'),
         Input('g7-button', 'n_clicks')
        ]
)
def world_gdp_life(radio_value, n_clicks):

    dff = data.groupby('year').apply(lambda x: pd.Series({
    'world_gdp': np.average(x['gdpPercap'], weights=x['pop']),
    'world_life_exp': np.average(x['lifeExp'], weights=x['pop'])
    }))

    dff.reset_index(inplace=True)

    if n_clicks is None or n_clicks % 2 == 0:
        fig = px.scatter(dff, x='world_life_exp', y='world_gdp', color='IncomeGroup', 
                     labels={"IncomeGroup":"Income Group", "world_gdp":"World GDP Per Capita (Weighted Average)", 
                             "world_life_exp":"World Life Expectancy (Weighted Average)", "LifeExpectancyGroup":"Life Expectancy Group"})
        line = px.line(dff, x='world_life_exp', y='world_gdp').data[0]
    else:
        fig = px.scatter(dff, x='world_gdp', y='world_life_exp', color='LifeExpectancyGroup', 
                     labels={"IncomeGroup":"Income Group", "world_gdp":"World GDP Per Capita (Weighted Average)", 
                             "world_life_exp":"World Life Expectancy (Weighted Average)", "LifeExpectancyGroup":"Life Expectancy Group"})
        line = px.line(dff, y='world_life_exp', x='world_gdp').data[0]

    fig.update_xaxes(type='linear' if radio_value == 'linear' else 'log')
    fig.update_yaxes(type='linear' if radio_value == 'linear' else 'log')
    
    fig.add_trace(line)

    return fig

# Define app layout
app.layout = html.Div([
    html.H1(children="ASSIGNMENT - 3", style={"text-align":"center"}),
    
    # Graph-1
    html.Div([
        html.H3(children="Life Expectency vs GDP per capita graph", style={"margin-left":"4%"}),
        html.Div([
            html.H4(children="View this graph for "),
            dcc.RadioItems(id="g1-radio-value", options=[{'label': i, 'value': i} for i in ['Continents', 'Countries']], value='Continents', style={"display":"flex"}),
            html.H4(id='g1-dropdown-header', children="", style={'display': 'none'}),
            dcc.Dropdown(
                id='g1-continent-dropdown',
                options=[{'label': i, 'value': i} for i in data['continent'].unique()],
                style={'display': 'none', 'margin-right':'55%'}
            )],
            style={"margin-left":"4%"}
        ),
        dcc.Graph(id='life-exp-vs-gdp'),
    ]),

    # Graph-2
    html.Div([
        html.H3(children="Life Expectency vs Continents", style={"margin-left":"4%"}),
        dcc.Graph(id='life-exp-vs-continent', figure=update_graph_continent()),
    ]),
    
    # Graph-3
    html.Div([        
        html.H3(children="Life Expectency vs Population", style={"margin-left":"4%"}),
        html.Div([    
            html.H4(children="View this graph for "),    
            dcc.RadioItems(id="g3-radio-value", options=[{'label': i, 'value': i} for i in ['Continents', 'Countries']], value='Continents', style={"display":"flex"}),
            html.H4(id='g3-dropdown-header', children="", style={'display': 'none'}),
            dcc.Dropdown(
                id='g3-continent-dropdown',
                options=[{'label': i, 'value': i} for i in data['continent'].unique()],
                style={'display': 'none'}
            ),
        ],
        style={"margin-left":"4%"}
        ),
        dcc.Graph(id='life-exp-vs-population'),
    ]),

    # Graph-4
    html.Div([
        html.H3(children="Countries' Evolution", style={"margin-left":"4%"}),
        html.Div([    
                html.H4(id='g4-dropdown-header', children="View this graph for ", style={'margin-right':'4%'}),
                dcc.Dropdown(
                    id='g4-continent-dropdown',
                    options=[{'label': i, 'value': i} for i in data['continent'].unique()], value="Asia",
                    style={'margin-right':'4%'}
                ),
                dcc.RadioItems(id="g4-radio-value", options=[{'label': i, 'value': i} for i in ['Life Expectency', 'GDP Per Capita']], value='Life Expectency', style={"display":"flex", 'margin-right':'4%'}),
            ],
        style={"margin-left":"4%"}
        ),
        dcc.Graph(id='countries-evolution'),
    ]),

    # Graph-5 -- With and without outliers?
    html.Div([
        html.H3(children="The slope of the linear dependency between life expectancy and GDP per capita over time", style={"margin-left":"4%"}),
        dcc.Graph(id="gdp-linear-regression", figure=gdp_linear_regression())
    
    ]),

    # Graph-6 -- Dropdown needed?
    html.Div([
        html.Div([
            html.H3(children="How GDP per capita changes over time", style={"margin-left":"4%"}),
            html.H3(children="GDP growth for continents", style={"margin-left":"4%"}),
            html.Div([    
                dcc.RadioItems(id="g6-1-radio-value", options=[{'label': i, 'value': i} for i in ['linear', 'log']], value='linear', style={"display":"flex", 'margin-right':'4%'}),
            ],
            style={"margin-left":"4%"}
            ),
            dcc.Graph(id='gdp-over-time-1', hoverData={'points': [{'hovertext': 'Asia'}]}),
        ], style={"width": "49%"}),
        html.Div([
            html.Div([    
                html.H4(id='g6-2-dropdown-header', children="Hover over a point in the continent scatter plot or select a continent to see all of its countries' growth over time", style={'margin-right':'4%'}),
                dcc.Dropdown(
                    id='g6-2-continent-dropdown',
                    options=[{'label': i, 'value': i} for i in data['continent'].unique()], value="Asia",
                    style={'margin-right':'4%'}
                ),
                dcc.RadioItems(id="g6-2-radio-value", options=[{'label': i, 'value': i} for i in ['linear', 'log']], value='linear', style={"display":"flex", 'margin-right':'4%'}),
            ],
            style={"margin-left":"4%"}
            ),
            dcc.Graph(id='gdp-over-time-2'),
        ], style={"width": "49%"}),
    ], style={"display":"flex"}),

    # Graph-7
    html.Div([
        html.H3(children="World GDP and world life expectancy over time as a reference level.", style={"margin-left":"4%"}),
        html.Button(id="g7-button", children="Flip axes", style={"margin-left":"4%"}),
        dcc.RadioItems(id="g7-radio-value", options=[{'label': i, 'value': i} for i in ['linear', 'log']], value='linear', style={"display":"flex", 'margin-left':'4%'}),
        dcc.Graph(id='world-gdp-life-exp'),
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True)