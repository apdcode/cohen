# %%

# imports
import warnings
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, ctx
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import datasets
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from jupyter_dash import JupyterDash
import statsmodels.api as sm
import os
import plotly.figure_factory as ff
import polars as pl
import json

import sys
sys.path.insert(1, 'c:/repos/pyfx/')
from pdfx import RegressionList_StepWise  # nopep8
from pdfx import txt2image  # nopep8
from pdfx import powerpoint_com02  # nopep8
from pdfx import ply_bar_single  # nopep8
from pdfx import ply_tSerier2  # nopep8
from pdfx import ply_boxScatter01  # nopep8
from pdfx import ply_box  # nopep8
from pdfx import ply_tSerier  # nopep8
from pdfx import table_offset_V2  # nopep8
from pdfx import LinReg_tTest  # nopep8
from pdfx import LinReg_params2  # nopep8
from pdfx import LinReg_models  # nopep8
from pdfx import LinReg01_MultipleResults  # nopep8
from pdfx import Regresjonslister_stegvis  # nopep8
from pdfx import RollRegression_resultsOnly  # nopep8
from pdfx import plot_sns_RegressionV2  # nopep8
from pdfx import plot_sns_Regression1  # nopep8
from pdfx import RegressionList_1by1  # nopep8
from pdfx import pptKommentar  # nopep8
from pdfx import pptPrint_sns  # nopep8
from pdfx import plt_corr  # nopep8
from pdfx import pd_timediff  # nopep8
from datetime import timedelta  # nopep8


# system setup
os.listdir(os.getcwd())
# os.getcwd()
os.chdir(r'C:\repos\fun\cohen\data')
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(1, 'c:/repos/pyfx/')

# local imports
# data
df_alb = pd.read_csv(r"C:\repos\fun\cohen\data\CohenAlbums.txt", sep='|')
df = pd.read_csv('C:/repos/fun/cohen/data/CohenList.txt', sep='|')
df = df.dropna()
df

albums = list(df_alb['Album'])

albums1 = list(df['Album 1'].dropna())
albums2 = list(df['Album 2'].dropna())

albums1

album_not_1 = [a for a in albums if a not in albums1]
album_not_1

album_not_2 = [a for a in albums if a not in albums2]
album_not_2
#
df_albumrank1 = df.groupby('Album 1')['Title 1'].count(
).to_frame().reset_index().sort_values('Title 1', ascending=False)
#
df_albumrank2 = df.groupby('Album 2')['Title 2'].count(
).to_frame().reset_index().sort_values('Title 2', ascending=False)
#
# print(df_albumrank1)
# print(df_albumrank2)
#

df_titlerank1 = pd.merge(df[['Position', 'Title 1']], df[[
                         'Position', 'Title 2']], how='left', left_on=['Title 1'], right_on=['Title 2'])
df_titlerank1 = df_titlerank1[['Position_x', 'Title 1', 'Position_y']]
df_titlerank1.columns = ['AP', 'Title', 'Haakon']
df_titlerank1 = df_titlerank1.dropna()
# %%
fig_rank = go.Figure()
for index, row in df_titlerank1.iterrows():
    # print(row)
    fig_rank.add_trace(go.Scatter(
        x=[1, 2], y=[row['AP']*-1, row['Haakon']*-1], line_shape='spline'))

    fig_rank.add_annotation(dict(font=dict(  # color="green",
        size=14),
        # x=x_loc,
        x=1,
        y=row['AP']*-1,
        showarrow=False,
        # text="<i>"+k+"</i>",
        text=row['Title'],
        # textposition = "right",
        align="right",
        textangle=0,
        standoff=100,
        xshift=-10,
        xanchor="right",
        xref="x",
        yref="y"
    ))

    fig_rank.add_annotation(dict(font=dict(  # color="green",
        size=14),
        # x=x_loc,
        x=2,
        y=row['Haakon']*-1,
        showarrow=False,
        # text="<i>"+k+"</i>",
        text=row['Title'],
        # textposition = "right",
        align="right",
        textangle=0,
        standoff=100,
        xshift=10,
        xanchor="left",
        xref="x",
        yref="y"
    ))
# fig.show()
# %%
# fig=go.Figure()
# fig.add_trace(go.Scatter(x=df['Year 1'], y = df['Position'], mode = 'markers', trendline = 'ols'))
# fig.add_trace(go.Scatter(x=df['Year 2'], y = df['Position'], mode = 'markers'))
# fig.show()

# pd.wide_to_long()


df_long_year = pd.melt(df, id_vars='Position', value_vars=['Year 1', 'Year 2', ],
                       var_name='Man', value_name='Year'
                       )
df_long_year['Man'] = df_long_year['Man'].map(
    {'Year 1': 'AP', 'Year 2': 'Haakon'})
df_long_title = pd.melt(df, id_vars='Position', value_vars=['Title 1', 'Title 2', ],
                        var_name='Man', value_name='Title'
                        )
df_long_title['Man'] = df_long_title['Man'].map(
    {'Title 1': 'AP', 'Title 2': 'Haakon'})
df_long_title

#
df_long_album = pd.melt(df, id_vars='Position', value_vars=['Album 1', 'Album 2', ],
                        var_name='Man', value_name='Album'
                        )
df_long_album['Man'] = df_long_album['Man'].map(
    {'Album 1': 'AP', 'Album 2': 'Haakon'})
df_long_album

df_long1 = pd.merge(df_long_year, df_long_title, how='left', left_on=[
                    "Position", "Man"], right_on=["Position", "Man"])
df_long1

df_long = pd.merge(df_long1, df_long_album, how='left', left_on=[
                   "Position", "Man"], right_on=["Position", "Man"])

df_long['Position'] = df_long['Position']*-1

fig = px.scatter(df_long, x='Year', y='Position', color='Man', trendline='ols',
                 hover_data=['Title', 'Album']
                 )
# fig.show()


# marks = [{i: {'label': value, 'style': {'font-size': '10px'}}} for i, value in enumerate(list(df['Title 2'][::-1]))]
# markers = list(df['Title 2'][::-1])
# marks = [{i: {'label': markers[1], 'style': {'font-size': '10px'}}}
#          for i in range(-len(df), 0)]

# %%

# slider settings
sliderStart = - len(df)
sliderStops = 0
sliderSteps = 1

app = Dash(external_stylesheets=[dbc.themes.SLATE])

app.layout = dbc.Container(
    [html.H1("Cohen connoisseurs",
             style={"text-align": "center"}),
     dbc.Row([dbc.Col([dbc.Row([dbc.Col([html.H4("List AP",
             style={"text-align": "center"}), dash_table.DataTable(data=df[['Title 1', 'Album 1']].to_dict('records'),
                                                                   columns=[{"name": i, "id": i} for i in df[[
                                                                       'Title 1', 'Album 1']].columns],
                                                                   style_cell={
                 'fontSize': 10, 'font-family': 'sans-serif'}
     )], className="mt-3")]), dbc.Row([dbc.Col([html.H4("Cherished albums",
                                                        style={"text-align": "center"}), dash_table.DataTable(data=df_albumrank1[['Album 1', 'Title 1']].to_dict('records'),
                                                                                                              columns=[{"name": i, "id": i} for i in df_albumrank1[[
                                                                                                                  'Album 1', 'Title 1']].columns],
                                                                                                              style_cell={
                                                            'fontSize': 10, 'font-family': 'sans-serif'},
         # style={'margin-top': '5px'}


     )], className="mt-3")])], width=3, className="bg-primary"),


         dbc.Col([dcc.Graph(id="fig1", figure=fig),
                  dcc.Graph(id="fig2", figure=fig_rank)]),
         dbc.Col([dash_table.DataTable(data=df[['Title 2', 'Album 2']].to_dict('records'),
                                       columns=[{"name": i, "id": i} for i in df[[
                                           'Title 2', 'Album 2']].columns],
                                       style_cell={
             'fontSize': 10, 'font-family': 'sans-serif'}
         ),
             dash_table.DataTable(data=df_albumrank2[['Album 2', 'Title 2']].to_dict('records'),
                                  columns=[{"name": i, "id": i} for i in df_albumrank2[[
                                           'Album 2', 'Title 2']].columns],
                                  style_cell={
                 'fontSize': 10, 'font-family': 'sans-serif'}
         )], width=3)], className="mt-2"),
        dbc.Row([]),
        dbc.Row([])


     ])
# %%

if __name__ == "__main__":
    app.run_server(debug=True, threaded=True, port=8099)
