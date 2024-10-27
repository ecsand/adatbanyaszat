import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

allowed = (
    "number"
)
BGR_COLOR = '#129aa1'
BGR_COLOR2 = '#c9c6c5'
BGR_COLOR3 = '#80edc3'

def make_empty_fig():

    fig = go.Figure()
    fig.layout.paper_bgcolor = BGR_COLOR
    fig.layout.plot_bgcolor = BGR_COLOR
    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

solvers=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
current_dir = os.path.dirname(os.path.abspath(__file__))  # Aktuális script mappája
devdata= pd.read_csv(os.path.join(current_dir, '1_emberi_fejlettseg.csv'))
uniquevars=devdata.columns.str.replace(r'\s*\(\d{4}\)', '', regex=True).unique()
uniquevars=uniquevars[5:]
years = devdata.columns.str.extract(r'\((\d{4})\)')[0].dropna().unique()
years=[int(y) for y in years]
miny=min(years)
maxy=max(years)

app.layout = html.Div(style={'backgroundColor': BGR_COLOR2, 'padding': '20px'}, children=[

    html.H1('Emberi Fejlettség'),

    dbc.Tabs([
        dbc.Tab([
             dcc.Markdown(
                """
                    Név:

                    *Ecsédi András*

                    E-mail cím

                    *ecsedi.andris@gmail.com*

                    Neptun-kód

                    *DLZTAT*
                    """)

        ], label='Személyes adataim'),
        dbc.Tab([
             dcc.Markdown(
            """
                Projekt célja

                *A projekt célja az emberi fejlettség országokon átívelő elemzése az elmúlt évtizedek adatain keresztül.*

                Adatállományra vonatkozó információk:

                *1990-2021 közt adatokkal rendelkezik.*

                *Az életminőséget többfélén mérő metrikákkal dolgozik.*

                *Egyes országok esetén hiányosak az adatok.*

                Megvalósítás módja:

                *Többféle statisztikai kimutatás révén.*

                *Diagramok segítségével.*

                *Interaktívan.*
                """)
        ], label='A projekt adatai')
    ]),



    

    
    dcc.Dropdown(
        id='Human Development Groups',
        options=[
            {'label': str(hdg),'value': str(hdg)} for hdg in devdata['Human Development Groups'].unique() if hdg is not None
        ], placeholder="3. feladat"
        
    ),
    html.Div(id='hdgoutput'),
    html.Br(),
    dcc.Dropdown(
        id='Countries',
        options=[
            {'label': country, 'value': country} for country in devdata['Country'].unique()
        ], placeholder="4. feladat"
        
    ),
html.Div(id='coutput'),

    dcc.Dropdown(
        id='Countries2',
        options=[
            {'label': country, 'value': country} for country in devdata['Country'].unique()
        ], placeholder="5. feladat", multi=True
        
    ),

 dcc.Graph(id='otodik'),
html.Br(),
    dbc.Row([
        dbc.Col(dcc.Slider(miny, maxy, 1, marks={year: str(year) for year in years}, id='hategy'), width=8),
        dbc.Col(dcc.Dropdown(
            id='hatketto',
            options=[{'label': var, 'value': var} for var in uniquevars],
            placeholder="Choose a variable"
        ), width=2),
        dbc.Col(dcc.Input(
            id="classint",
            placeholder="input",
        ), width=2),
    ], align='center'),
        
    dcc.Graph(id='hatodik'),
    html.Br(),
    #hetedik

        dcc.Dropdown(
        id='hetketto',
        options=[
            {'label': var, 'value': var} for var in uniquevars
        ], placeholder="Choose a variable"
        
    ),
    dcc.Graph(id='hetes'),
    html.Br(),
    #nyolcadik
        dcc.Dropdown(
        id='nyketto',
        options=[
            {'label': country, 'value': country} for country in devdata['Country'].unique()
        ], placeholder="Choose a country"
        
    ),

        dcc.Dropdown(
        id='nyegy',
        options=[
            {'label': var, 'value': var} for var in uniquevars
        ], placeholder="Choose a variable"
        
    ),

        dcc.Dropdown(
        id='solver',
        options=[
            {'label': s, 'value': s} for s in solvers
        ], placeholder="Choose a solver"
        
    ),

        dcc.Input(
            id="polydeg",

            placeholder="poly deg",
        ),
        
        dcc.Input(
            id="ridgealpha",

            placeholder="ridge_alpha",
        ),

                dcc.Input(
            id="testsize",

            placeholder="testsize",
        ),

    dcc.Graph(id='nyolcas'),
    
    
])

#3. feladat
@app.callback(
    Output('hdgoutput', 'children'),
    Input('Human Development Groups', 'value')
)
def hdg_members(hdg):
    if hdg is None:
        return ''

    filtered_df = devdata[(devdata['Human Development Groups'] == hdg)].sort_values(by='Country')
    header = html.Thead(html.Tr([html.Th("Country"), html.Th("HDI Rank 2021")]))

    body = html.Tbody([
        html.Tr([html.Td(country), html.Td(rank)]) for country, rank in zip(filtered_df["Country"], filtered_df["HDI Rank (2021)"])
    ])
    
    hdgoutput=dbc.Table([header, body])
    return hdgoutput

#4. feladat
@app.callback(
    Output('coutput', 'children'),
    Input('Countries', 'value')
)
def countries(country):
    if country is None:
        return ''

    filtered_df = devdata[(devdata['Country'] == country)]
    header = html.Thead(html.Tr([html.Th(x) for x in filtered_df.columns]))
    
    body =html.Tbody([
             html.Tr([html.Td(filtered_df[x].values[0]) for x in filtered_df.columns])
        ])
    
    coutput=dbc.Table([header, body])
    return coutput

#5. feladat
@app.callback(
    Output('otodik', 'figure'),
    Input('Countries2', 'value')
)
def countries2(country):

    if not country:
        return make_empty_fig()

    filtered_df = devdata[devdata['Country'].isin(country)]
    filtered_df = filtered_df.loc[:, (filtered_df.columns.str.contains('Human Development Index \(') | filtered_df.columns.str.contains('Country'))& ~filtered_df.columns.str.contains('adjusted')]

    otod = filtered_df.transpose()
    otod.columns = filtered_df['Country'].values  
    otod.reset_index(inplace=True) 
    otod=otod[~otod['index'].str.contains("Country")]
    otod['index']=otod['index'].str.replace('Human Development Index (', "").str.replace(')', "")

    fig = px.line(otod, x='index', y=otod.columns[1:], title='Human Development Index by Year and by Country', labels={'index': 'Year'})
    fig.update_layout(
        paper_bgcolor=BGR_COLOR3,
        plot_bgcolor=BGR_COLOR3
    )

    return fig

#6. feladat
@app.callback(
    Output('hatodik', 'figure'),
    [Input('hategy', 'value'), Input('hatketto', 'value')], [Input('classint', 'value')]
)
def histi(year, var, val):

    
    if not year or not var:
        return make_empty_fig()
    
    if val:
        print(val)
    filtered_columns = devdata.columns[devdata.columns.str.startswith(var+" (") & devdata.columns.str.contains(str(year))]
    filtered_df = devdata[['Country'] + list(filtered_columns)]
    hat = filtered_df.transpose()
    hat.columns = filtered_df['Country'].values  
    hat.reset_index(inplace=True) 
    hat=hat[~hat['index'].str.contains("Country")]

    hat_melted = hat.melt(id_vars=['index'], var_name='Country', value_name='Value')


    fig = px.histogram(
        data_frame=hat_melted,
        x="Value",  
        color="Country",
        nbins=int(val) if val else 0,
        title=f'Frequency of {var}', 
        labels={'x':'Frequency', 'y':'count'})
    fig.update_layout(
        paper_bgcolor=BGR_COLOR3,
        plot_bgcolor=BGR_COLOR3
    )
    return fig

#7. feladat
@app.callback(
    Output('hetes', 'figure'),
    Input('hetketto', 'value')
)
def themmap(var):
    if not var:
        return make_empty_fig()

    filtered_columns= devdata.loc[:, (devdata.columns.str.startswith(f"{var} ("))]
    filtered_df = devdata[['Country'] + list(filtered_columns)]
    melted_df = pd.melt(
    filtered_df,
    id_vars=['Country'],  
    var_name='Year',      
    value_name='Variable' 
    )
    melted_df['Year'] = melted_df['Year'].str.extract(r'\((\d{4})\)')[0]
    melted_df['Year'] = pd.to_numeric(melted_df['Year'])
    fig = px.choropleth(
        melted_df,
        color_continuous_scale='cividis',
        locations='Country',
        locationmode='country names',
        color='Variable',
        animation_frame='Year', 
        title='Thematic Map'
    )
    fig.update_layout(
        paper_bgcolor=BGR_COLOR3,
        plot_bgcolor=BGR_COLOR3
    )
    return fig


#8. feladat
@app.callback(
    Output('nyolcas', 'figure'),
    [Input('nyegy', 'value'), Input('nyketto', 'value'), Input('solver', 'value'), Input('ridgealpha', 'value'), Input('polydeg', 'value'), Input('testsize', 'value')]
)
def regger(var, country, solver, ridgealpha, polydeg, testsize):
    if not var or not country or not solver or not ridgealpha or not polydeg or not testsize:
        return make_empty_fig()

    filtcol = devdata.loc[:, devdata.columns.str.startswith(f"{var} (") | devdata.columns.str.startswith("Gross National Income Per Capita (")] 

    filt = devdata[['Country'] + list(filtcol)]
    filt = filt[filt['Country'] == country]


    melted_var = pd.melt(
        filt,
        id_vars=['Country'],
        value_vars=filt.columns[~filt.columns.str.startswith("Gross National Income Per Capita (")],
        var_name='Year',
        value_name='Var'
    )  
    melted_gni = pd.melt(
        filt,
        id_vars=['Country'],
        value_vars=filt.columns[filt.columns.str.startswith("Gross National Income Per Capita (")],
        var_name='Year',
        value_name='GNI'
    )
    filt2 = pd.merge(melted_gni, melted_var, on=['Country', 'Year'], how='outer')  
    print("Merged DataFrame:", filt2)


    filt2['Year'] = filt2['Year'].str.extract(r'(\d{4})')[0]
    filt2['Year'] = pd.to_numeric(filt2['Year'], errors='coerce')  
    filt2['GNI'] = pd.to_numeric(filt2['GNI'], errors='coerce')
    filt2['Var'] = pd.to_numeric(filt2['Var'], errors='coerce')

    
    filt2.fillna(0, inplace=True)


    #regression
    solve=solver
    poly_deg = int(polydeg)
    ridge_alpha = float(ridgealpha)
    
    testsiz=float(testsize)

    rid = Ridge(alpha=ridge_alpha, solver=solve, random_state=42) 


    X=filt2['Var'].values.reshape(-1, 1)
    y=filt2['GNI'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsiz, random_state=17)



    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=poly_deg, include_bias=False)),  
        ("std_scaler", StandardScaler()),  
        ("regul_reg", rid)
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    train_results = pd.DataFrame({
        'Year': filt2.loc[X_train.flatten().argsort(), 'Year'].values,
        'Actual GNI': y_train[X_train.flatten().argsort()],
        'Predicted GNI': y_pred[X_train.flatten().argsort()]
    })


    fig = px.line(
        filt2,
        x='Year',
        y=['GNI', 'Var'],
        title=f'{var} and GNI by year',     
    )

    fig.update_layout(
        paper_bgcolor=BGR_COLOR3,
        plot_bgcolor=BGR_COLOR3
    )
    fig.add_scatter(
    x=train_results['Year'],
    y=train_results['Predicted GNI'],
    mode='lines',
    name='Predicted GNI (Train)',
    line=dict(dash='dash', color='red')



)
    
    return fig









if __name__ == '__main__':
    app.run_server(debug=True)
    app.layout.paper_bgcolor = BGR_COLOR2