# ================================================================
# ======= Importing Libraries ====================================
# ================================================================
import datetime
import os
import yaml

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import plotly.graph_objs as go

# ================================================================
# ======= Reading data ===========================================
# Lecture du fichier d'environnement
ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

# Lecture du fichier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df: _df['Last Update'].dt.date).drop('Last Update', axis=1)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020, 3, 10)])

countries = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]

regions = {
    'world': {'lat': 0, 'lon': 0, 'zoom': 1},
    'europe': {'lat': 50, 'lon': 0, 'zoom': 3},
    'north_america': {'lat': 40, 'lon': -100, 'zoom': 2},
    'south_america': {'lat': -15, 'lon': -60, 'zoom': 2},
    'africa': {'lat': 0, 'lon': 20, 'zoom': 2},
    'asia': {'lat': 30, 'lon': 100, 'zoom': 2},
    'oceania': {'lat': -10, 'lon': 130, 'zoom': 2},
}

# ================================================================
# ======= App implementation =====================================
# ================================================================
app = dash.Dash('Covid19: Data Explorations')
app.layout = html.Div([
    html.H1(['Corona Virus Data Explorations'], style={'textAlign': 'center'}),
    dcc.Tabs([
# ======= Time Series ============================================
            dcc.Tab(label='Time Series', children=[
                """
                Select your first country
                """,
                html.Div([
                    dcc.Dropdown(
                        id='country',
                        options=countries,
                        style={"display": "inline-block", 
                               "margin-left": "auto", 
                               "margin-right": "auto",
                               "width": "30%"} 
                    )
                ]),
                """
                Select your second country
                """,
                html.Div([
                    dcc.Dropdown(
                        id='country2',
                        options=countries,
                        style={"display": "inline-block",
                               "margin-left": "auto", 
                               "margin-right": "auto",
                               "width": "30%"}
                    )
                ]),
                """
                Change your variable
                """,
                html.Div([
                    dcc.RadioItems(
                        id='variable',
                        options=[
                            {'label': 'Confirmed', 'value': 'Confirmed'},
                            {'label': 'Deaths', 'value': 'Deaths'},
                            {'label': 'Recovered', 'value': 'Recovered'}
                        ],
                        value='Confirmed'
                    )
                ]),
                html.Div([
                    dcc.Graph(id='graph1')
                ]), 
            ]),
# ======= Map ====================================================       
        dcc.Tab(label='Map', children=[
            """
            Variable selection
            """,
            html.Div([
                dcc.Dropdown(
                    id='var',
                    options=[
                        {'label': 'Confirmed', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    style={"display": "inline-block", 
                           "margin-left": "auto", 
                           "margin-right": "auto",
                           "width": "30%"}                    
                )
            ]),            
            """
            Area selection
            """,
            html.Div([
                dcc.Dropdown(
                    id='region',
                    className='six columns',
                    options=[
                        {'label': 'World', 'value': 'world'},
                        {'label': 'Europe', 'value': 'europe'},
                        {'label': 'North America', 'value': 'north_america'},
                        {'label': 'South America', 'value': 'south_america'},
                        {'label': 'Africa', 'value': 'africa'},
                        {'label': 'Asia', 'value': 'asia'},
                        {'label': 'Oceania', 'value': 'oceania'},
                    ],
                    value='europe',
                    style={"display": "inline-block", 
                           "margin-left": "auto", 
                           "margin-right": "auto",
                           "width": "30%"}
                ),
                
            ]),
            """
            Map Style selection
            """,            
            html.Div([
                dcc.Dropdown(
                    id='style',
                    options=[
                        {'label': 'Street', 'value': 'open-street-map'},
                        {'label': 'Light', 'value': 'light'},
                        {'label': 'Dark', 'value': 'dark'},
                        {'label': 'Satellite', 'value': 'satellite'},
                        {'label': 'Custom','value': 'mapbox://styles/jackdbd/cj6nva4oi14542rqr3djx1liz'}
                    ],
                    value='dark',
                    style={"display": "inline-block", 
                           "margin-left": "auto", 
                           "margin-right": "auto",
                           "width": "30%"}                    
                )
            ]),
            html.Div([
                dcc.Graph(id='map')
            ]),
            """
            Time
            """,
            html.Div([
                dcc.Slider(
                    id='map_day',
                    min=0,
                    max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                    value=0,
                    marks={i:'t'+str(i) for i, date in enumerate(epidemie_df['day'].unique())})
            ]),
        ]),
# ======= SIR Model ===============================================       
        dcc.Tab(label='SIR Model', children=[
            """
            Select your country here
            """,
            html.Div([
                dcc.Dropdown(
                    id='pays',
                    options=countries,
                    value='Italy',
                    style={"display": "inline-block", 
                           "margin-left": "auto", 
                           "margin-right": "auto",
                           "width": "40%"}
                )
            ]),
            """
            Choose your beta, gamma and population here
            """,
            html.Div([
                dcc.Input(
                    id='beta',
                    placeholder='Beta parameter',
                    min  =-20.0, 
                    max  =99.0,
                    step =0.05,
                    value=0.03,
                    type ='number',
                    style={'float': 'left', 'width': '5%'}) 
               ]),

            html.Div([
                dcc.Input(
                    id='gamma',
                    placeholder='Gamma parameter',
                    min  =-20.0, 
                    max  =90.0,
                    step =5.00,
                    value=20.0,
                    type ='number',
                    style={'float':'left', 'width':'5%'})
            ]),
            html.Div([
                dcc.Input(
                    id='population',
                    placeholder='Population',
                    min  =1000, 
                    max  =2e9,
                    step =1000,
                    value=12000,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
            html.Div([
                dcc.Graph(id='SIRmodel')
            ]),
            html.Div([
                dcc.Checklist(id = 'optimal',
                    options=[{'label': 'Optimize your parameters here', 'value': 'optmal'}])
            ]),
            
        ]),
# ======= SEIR Model ==============================================       
        dcc.Tab(label='SEIR Model', children=[
            """
            Select your country here
            """,
            html.Div([
                dcc.Dropdown(
                    id='provence',
                    options=countries,
                    value='France',
                    style={"display": "inline-block", 
                           "margin-left": "auto", 
                           "margin-right": "auto",
                           "width": "40%"}
                )
            ]),
            """
            Define all variable of SEIR model
            """,
            html.Div([
                dcc.Input(
                    id='T_inc',
                    placeholder='Incubation period',
                    min  =0.0, 
                    max  =999.0,
                    step =7,
                    value=7,
                    type ='number',
                    style={'float': 'left', 'width': '5%'}) 
               ]),

            html.Div([
                dcc.Input(
                    id='T_inf',
                    placeholder='Infection period',
                    min  =0.0, 
                    max  =999.0,
                    step =7,
                    value=3,
                    type ='number',
                    style={'float':'left', 'width':'5%'})
            ]),
            html.Div([
                dcc.Input(
                    id='R_0',
                    placeholder='Reproduction number',
                    min  =10, 
                    max  =1_200_000,
                    step =2,
                    value=5,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
            """
            Total Population
            """,
            html.Div([
                dcc.Input(
                    id='N',
                    placeholder='Population',
                    min  =1000, 
                    max  =2e9,
                    step =1000,
                    value=1_200_000,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
            
            html.Div([
                dcc.Graph(id='SEIRmodel')
            ]),
            
            #Initial stat for SEIR model
            """
            Your S parameter
            """,
            html.Div([
                dcc.Input(
                    id='s',
                    placeholder='S parameter',
                    value=1,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
            """
            Your I parameter
            """,
            html.Div([
                dcc.Input(
                    id='i',
                    placeholder='I parameter',
                    value=3,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
            """
            Your E parameter
            """,
            html.Div([
                dcc.Input(
                    id='e',
                    placeholder='E parameter',
                    min  =0, 
                    max  =1,
                    step =.003,
                    value=0,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
            """
            Your R parameter
            """,
            html.Div([
                dcc.Input(
                    id='r',
                    placeholder='R parameter',
                    min  =0, 
                    max  =1,
                    step =.003,
                    value=0,
                    type ='number',
                    style={'int':'left','width':'5%'})
            ]),
             
            html.Div([
                dcc.Checklist(id = 'SEIROptimal',
                    options=[{'label': 'Optimize your parameters here', 'value': 'optmal'}])
            ]),
            
        ]),
    ]),
])

# ================================================================
# ======= Callback: Time Series Plot =============================
@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),
        Input('country2', 'value'),
        Input('variable', 'value'),        
    ]
)

def update_graph(country, country2, variable):
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index())
    if country2 is not None:
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                     .groupby(['Country/Region', 'day'])
                     .agg({variable: 'sum'})
                     .reset_index())
        
    return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total')
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2)            
        ] if country2 is not None else [])
    }
theme = {
    'font-family': 'Raleway',
    'background-color': '#787878',
}

colorscale = [
    [0, 'red'],
    [0.25, 'blue'],
    [0.5, '#fd8d3c'],
    [0.75, '#f03b20'],
    [1, '#bd0026'],
]
colorscale_Deaths = [
    [0, '#f0f0f0'],
    [0.5, '#bdbdbd'],
    [0.1, '#636363'],
]
regions = {
    'world': {'lat': 0, 'lon': 0, 'zoom': 1},
    'europe': {'lat': 50, 'lon': 0, 'zoom': 3},
    'north_america': {'lat': 40, 'lon': -100, 'zoom': 2},
    'south_america': {'lat': -15, 'lon': -60, 'zoom': 2},
    'africa': {'lat': 0, 'lon': 20, 'zoom': 2},
    'asia': {'lat': 30, 'lon': 100, 'zoom': 2},
    'oceania': {'lat': -10, 'lon': 130, 'zoom': 2},
}

# ==========================================================
# ======= Callback: Map ====================================
@app.callback(
    Output('map', 'figure'),
    [
        Input('map_day', 'value'),
        Input('region', 'value'),
        Input('var', 'value'),
        Input('style', 'value'),
    ]
)
def update_map(map_day, region, var, style):
    day = epidemie_df['day'].unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({var: 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index())
    
    #radius_multiplier = {'inner': 1.5, 'outer': 3}
    return {
        'data':[
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region']+' ('+str(r[var])+')',
                                  axis=1),
                mode='markers',
                marker=dict(
                    #size=map_df[var]*radius_multiplier['outer']/1000,
                    colorscale=colorscale,
                    color=map_df[var],
                    opacity=1,
                    size=np.maximum(map_df[var] / 1_000, 7)
                ),
                
            )
        ],
        'layout': dict(
            title=var+' on '+str(day),
            #autosize=True,
            hovermode='closest',
            height=750,
            geo=dict(showland=True),
            font=dict(family=theme['font-family']),
            style=style,
            zoom=regions[region]['zoom'],
            #margin={"r":0,"t":0,"l":0,"b":0},
            center=dict(
                    lat=regions[region]['lat'],
                    lon=regions[region]['lon'],
            ),
        ),
    }

# ================================================================
# ======= Callback: SIR Model ====================================
@app.callback(
    Output('SIRmodel', 'figure'),
    [
        Input('pays', 'value') ,
        Input('beta', 'value'),
        Input('gamma','value'),
        Input('population','value'),
        Input('optimal','value'),
    ]
)
def update_model(pays, beta, gamma, population, optimal):
    
    if pays is None:
        pays = 'World'
        pays_df = epidemie_df.groupby('day').agg({'Confirmed': 'sum'}).reset_index()
        pays_df['infected'] = pays_df['Confirmed'].diff()
        
    else:     
        pays_df = (epidemie_df[epidemie_df['Country/Region'] == pays]
                   .groupby(['Country/Region', 'day'])
                   .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
                   .reset_index())
        pays_df['infected'] = pays_df['Confirmed'].diff()
        
    nb_steps = len(pays_df['infected'])
    
    # ======= SIR Model ==============================================            
    def SIR(t, y):
        S = y[0]; I = y[1]; R = y[2]
        return([-beta*S*I, beta*S*I-gamma*I, gamma*I]) 
    solution = solve_ivp(SIR, [0, nb_steps], [population, 1, 0], t_eval=np.arange(0, nb_steps, 1))
    
    # ======= Parameter optimization =================================
    def sumsq_error(p):
        beta, gamma = p
        def SIR(t, y):
            S = y[0]; I = y[1]; R = y[2]
            return([-beta*S*I, beta*S*I-gamma*I, gamma*I])
        sol = solve_ivp(SIR, [0, nb_steps], [population, 1, 0], t_eval=np.arange(0, nb_steps, 1))
        return(sum((sol.y[1]-pays_df['infected'])**2))
        
    if optimal is not None:
        msol = minimize(sumsq_error, [0.001, 0.1], method='Nelder-Mead')
        print(msol.x)
        beta, gamma = float(msol.x[0]), float(msol.x[1])
        
        def SIR(t, y):
            S = y[0]; I = y[1]; R = y[2]
            return([-beta*S*I, beta*S*I-gamma*I, gamma*I]) 
        solution = solve_ivp(SIR, [0, nb_steps], [population, 1, 0], t_eval=np.arange(0, nb_steps, 1))

    # ======= Return results and plot Graph ==========================
    return {
        'data': [
            dict(
                x=solution.t,
                y=solution.y[0],
                type='line',
                name=pays+': Susceptible')
        ] + ([
            dict(
                x=solution.t,
                y=solution.y[1],
                type='line',
                name=pays+': Infected')
        ]) + ([
            dict(
                x=solution.t,
                y=solution.y[2],
                type='line',
                name=pays+': Removed')
        ]) + ([
            dict(
                x=solution.t,
                y=pays_df['infected'],
                type='line',
                name=pays+': Original Data(Infected)')
        ])
    }

# ================================================================
# ======= Callback: SEIR Model ====================================
    # équation des susceptibles
def dS_dt(S, I, R_t, T_inf):
    return -(R_t / T_inf) * I * S

# équation des exposés
def dE_dt(S, E, I, R_t, T_inf, T_inc):
    return (R_t / T_inf) * I * S - (T_inc**-1) * E

# équation des infectés
def dI_dt(I, E, T_inc, T_inf):
    return (T_inc**-1) * E - (T_inf**-1) * I

# equation des guéris
def dR_dt(I, T_inf):
    return (T_inf**-1) * I

def SEIR_model(t, y, R_t, T_inf, T_inc):
    if callable(R_t):
        reproduction = R_t(t)
    else:
        reproduction = R_t
    S, E, I, R = y
    
    S_out = dS_dt(S, I, reproduction, T_inf)
    E_out = dE_dt(S, E, I, reproduction, T_inf, T_inc)
    I_out = dI_dt(I, E, T_inc, T_inf)
    R_out = dR_dt(I, T_inf)
    
    return [S_out, E_out, I_out, R_out]

@app.callback(
    Output('SEIRmodel', 'figure'),
    [
        Input('provence', 'value') ,
        Input('T_inc', 'value'),
        Input('T_inf','value'),
        Input('R_0','value'),
        Input('N','value'),
        Input('s','value'),
        Input('e','value'),
        Input('i','value'),
        Input('r','value'),
        Input('SEIROptimal','value'),
    ]
)
def update_model(provence, T_inc, T_inf, R_0, N,s,e,i,r, SEIROptimal):
    
    if provence is None:
        provence = 'World'
        provence_df = epidemie_df.groupby('day').agg({'Confirmed': 'sum'}).reset_index()
        provence_df['infected'] = provence_df['Confirmed'].diff()
        
    else:     
        provence_df = (epidemie_df[epidemie_df['Country/Region'] == provence]
                   .groupby(['Country/Region', 'day'])
                   .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
                   .reset_index())
        provence_df['infected'] = provence_df['Confirmed'].diff()
            
    # ======= SEIR Model ==============================================            
    
    max_days = len(provence_df['infected'])
    s=(N-max_days)/N; 
    i=max_days/N;
    SEIRsolution = solve_ivp(SEIR_model, [0, max_days], [s, e, i, r],
                             args=(R_0, T_inf, T_inc), t_eval=np.arange(max_days))
    
    # ======= Return results and plot Graph ==========================
    return {
        'data': [
            dict(
                x=provence_df['day'],
                y=SEIRsolution.y[1],
                type='line',
                name=provence+': Exposed')
        ] + ([
            dict(
                x=provence_df['day'],
                y=SEIRsolution.y[2],
                type='line',
                name=provence+': Infected')
        ]) + ([
            dict(
                x=provence_df['day'],
                y=SEIRsolution.y[3],
                type='line',
                name=provence+': Recovered/deceased')
        ]) + ([
            dict(
                x=provence_df['day'],
                y=SEIRsolution.y[1],
                type='line',
                name=provence+': Predict Confirmed')
        ])
    }


if __name__ == '__main__':
    app.run_server(debug=True)