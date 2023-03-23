# %%

# imports
import warnings
import dash
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
dfi = pd.read_csv('C:/repos/fun/cohen/data/CohenList.txt', sep='|')
df = dfi.dropna()

# reappering figure adjustments

template = "plotly_dark"


def benchmarkmodels_layout(fig):

    fig.update_layout(
        margin={'t': 65, 'b': 20, 'r': 0, 'l': 0, 'pad': 0})
#     fig.update_layout(yaxis1=dict(range=[df[y].min()*0.5, df[y].max()*2]))
#     fig.update_layout(yaxis2=dict(range=[fitted.min()*0.5, fitted.max()*2]))
    fig.update_layout(hovermode='x')
    fig.update_layout(showlegend=True, legend=dict(x=0.4, y=1.1))
    fig.update_layout(uirevision='constant')
    fig.update_layout(template=template,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      modebar_bgcolor='rgba(0, 0, 0, 0.0)'

                      )
    return fig


# make a rankindex for subsetting with dcc.RangeSlider
df['rankIndex'] = np.arange(df['Index'].iloc[0], 26)*-1

albums = list(df_alb['Album'])

albums1 = list(df['Album 1'].dropna())
albums2 = list(df['Album 2'].dropna())

# albums1

album_not_1 = [a for a in albums if a not in albums1]
# album_not_1

album_not_2 = [a for a in albums if a not in albums2]
# album_not_2
#
df_albumrank1 = df.groupby('Album 1')['Title 1'].count(
).to_frame().reset_index().sort_values('Title 1', ascending=False)
#

df_album_forgotten1 = pd.DataFrame(
    [album for album in albums if album not in df_albumrank1['Album 1'].values])
df_album_forgotten1.columns = ['Forgotten1']

df_albumrank2 = df.groupby('Album 2')['Title 2'].count(
).to_frame().reset_index().sort_values('Title 2', ascending=False)


df_album_forgotten2 = pd.DataFrame(
    [album for album in albums if album not in df_albumrank2['Album 2'].values])
df_album_forgotten2.columns = ['Forgotten2']

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

fig_rank = benchmarkmodels_layout(fig_rank)
fig_rank.update_layout(showlegend=False)
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
# %%
# merged albumranks
dfa = df_albumrank1.copy()
dfb = df_albumrank2.copy()

dfa.columns, dfb.columns = ['Album', 'Count'], ['Album', 'Count']
dft = pd.concat([dfa, dfb]).groupby(['Album']).sum(['Count']).reset_index()
ignored = [a for a in albums if a not in dft['Album']]
ignored = [a for a in albums if a not in list(dft['Album'].values)]
df_ignored = pd.DataFrame({'Album': ignored, 'Count': [0]*len(ignored)})
df_album_total = pd.concat([dft, df_ignored])
df_album_total

fig_bar = px.bar(df_album_total, x='Album', y='Count')
fig_bar.update_xaxes(categoryarray=albums)
# fig_bar.show()
# %%


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
sliderStart = df['rankIndex'].iloc[len(df)-1]
sliderStops = df['rankIndex'].iloc[0]
sliderSteps = 1

app = Dash(external_stylesheets=[dbc.themes.SLATE])

cell_font_size = 18

app.layout = dbc.Container(
    dbc.Row([html.H1("Tower of Songs",
             style={"text-align": "center"}),
             dbc.Row([dbc.Col([dbc.Row([dbc.Col([html.H4("List AP",
                                                         style={"text-align": "center"}), dash_table.DataTable(id='tbl_1', data=df[['Title 1', 'Album 1']].to_dict('records'),
                                                                                                               columns=[{"name": i, "id": i} for i in df[[
                                                                                                                   'Title 1', 'Album 1']].columns],
                                                                                                               style_header={
                                                             'backgroundColor': '#1f1f1f',
                                                             'fontWeight': 'bold',
                                                             'color': 'white',
                                                             'border': '1px solid #2c2c2c',
                                                         },
                 style_cell={
                                                             'backgroundColor': '#2c2c2c',
                                                             'color': 'white',
                                                             'border': '1px solid #343434',
                                                         },
             )], className="mt-3")]), dbc.Row([dbc.Col([html.H4("Cherished albums",
                                                                style={"text-align": "center"}), dash_table.DataTable(id="tbl_2", data=df_albumrank1[['Album 1', 'Title 1']].to_dict('records'),
                                                                                                                      columns=[{"name": i, "id": i} for i in df_albumrank1[[
                                                                                                                          'Album 1', 'Title 1']].columns],
                                                                                                                      style_cell={
                                                                    'backgroundColor': '#2c2c2c',
                                                                    'color': 'white',
                                                                    'border': '1px solid #343434',
                                                                },
                 style_header={
                                                                    'backgroundColor': '#1f1f1f',
                                                                    'fontWeight': 'bold',
                                                                    'color': 'white',
                                                                    'border': '1px solid #2c2c2c',
                                                                },
                 # style={'margin-top': '5px'}


             )], className="mt-3")]),  dbc.Row([dbc.Col([html.H4("Forgotten albums",
                                                                 style={"text-align": "center"}), dash_table.DataTable(id="tbl_5", data=df_album_forgotten1[['Forgotten1']].to_dict('records'),
                                                                                                                       columns=[
                                                                     {"name": i, "id": i} for i in df_album_forgotten1.columns],
                 style_cell={
                                                                     'backgroundColor': '#2c2c2c',
                                                                     'color': 'white',
                                                                     'border': '1px solid #343434',
                                                                 },
                 style_header={
                                                                     'backgroundColor': '#1f1f1f',
                                                                     'fontWeight': 'bold',
                                                                     'color': 'white',
                                                                     'border': '1px solid #2c2c2c',
                                                                 },
                 # style={'margin-top': '5px'}


             )], className="mt-3")])], width=3, className="bg-primary"),


                 dbc.Col([dcc.Graph(id="fig1", figure=fig),
                          dcc.Graph(id="fig2", figure=fig_rank),
                          dcc.Graph(id="fig3", figure=fig_bar),
                          dcc.RangeSlider(sliderStart, sliderStops, sliderSteps,
                                          id='slider_rank',
                                          #         min=5,
                                          # max=len(df),
                                          # step=1,
                                          #  pushable=True,
                                          # value=[0, len(dfi)],
                                          value=[sliderStart, sliderStops],
                                          updatemode='mouseup',
                                          marks=None,
                                          allowCross=False,
                                          # style={'width':'50%'}
                                          tooltip={'always_visible': True,
                                                   'placement': 'bottom'},
                                          )], width=4),
                 dbc.Col([dbc.Row([dbc.Col([html.H4("List Haakon",
                                            style={"text-align": "center"}), dash_table.DataTable(id='tbl_3', data=df[['Title 2', 'Album 2']].to_dict('records'),
                                                                                                  columns=[{"name": i, "id": i} for i in df[[
                                                                                                      'Title 2', 'Album 2']].columns],
                                                                                                  style_cell={
                                                'backgroundColor': '#2c2c2c',
                                                'color': 'white',
                                                'border': '1px solid #343434',
                                            },
                     style_header={
                                                'backgroundColor': '#1f1f1f',
                                                'fontWeight': 'bold',
                                                'color': 'white',
                                                'border': '1px solid #2c2c2c',
                                            },
                 )], className="mt-3")]), dbc.Row([dbc.Col([html.H4("Cherished albums",
                                                                    style={"text-align": "center"}), dash_table.DataTable(id="tbl_4", data=df_albumrank2[['Album 2', 'Title 2']].to_dict('records'),
                                                                                                                          columns=[{"name": i, "id": i} for i in df_albumrank2[[
                                                                                                                              'Album 2', 'Title 2']].columns],
                                                                                                                          style_cell={
                                                                        'backgroundColor': '#2c2c2c',
                                                                        'color': 'white',
                                                                        'border': '1px solid #343434',
                                                                    },
                     style_header={
                                                                        'backgroundColor': '#1f1f1f',
                                                                        'fontWeight': 'bold',
                                                                        'color': 'white',
                                                                        'border': '1px solid #2c2c2c',
                                                                    },
                     # style={'margin-top': '5px'}


                 )], className="mt-3")]),  dbc.Row([dbc.Col([html.H4("Forgotten albums",
                                                                     style={"text-align": "center"}), dash_table.DataTable(id="tbl_6", data=df_album_forgotten2[['Forgotten2']].to_dict('records'),
                                                                                                                           columns=[
                                                                         {"name": i, "id": i} for i in df_album_forgotten2.columns],
                     style_cell={
                                                                         'backgroundColor': '#2c2c2c',
                                                                         'color': 'white',
                                                                         'border': '1px solid #343434',
                                                                     },
                     style_header={
                                                                         'backgroundColor': '#1f1f1f',
                                                                         'fontWeight': 'bold',
                                                                         'color': 'white',
                                                                         'border': '1px solid #2c2c2c',
                                                                     },
                     # style={'margin-top': '5px'}


                 )], className="mt-3")])], width=3, className="bg-primary")], justify='center', className="mt-3"),
             dbc.Row([]),
             dbc.Row([])


             ],
            justify='center'), fluid=True)
# %%


@app.callback(Output('fig1', 'figure'),
              Output('fig2', 'figure'),
              Output("fig3", 'figure'),
              Output("tbl_1", 'data'),
              Output("tbl_2", 'data'),
              Output("tbl_3", 'data'),
              Output("tbl_4", 'data'),
              Output("tbl_5", 'data'),
              Output("tbl_6", 'data'),
              Input('slider_rank', 'value'))
def slide_dfi(rankslider):
    # df = dfi
    # df = dfi.iloc[rankslider[0]*-1:rankslider[1]*-1,]
    df = dfi.iloc[(rankslider[1]+1)*-1: rankslider[0]*-1]

    print(rankslider)
    # print(df)

    # print(rankslider)

    albums = list(df_alb['Album'])

    albums1 = list(df['Album 1'].dropna())
    albums2 = list(df['Album 2'].dropna())

    # albums1

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

    fig_rank = benchmarkmodels_layout(fig_rank)
    fig_rank.update_layout(showlegend=False)
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

    df_albumrank1 = df.groupby('Album 1')['Title 1'].count(
    ).to_frame().reset_index().sort_values('Title 1', ascending=False)
    #

    df_album_forgotten1 = pd.DataFrame(
        [album for album in albums if album not in df_albumrank1['Album 1'].values])
    df_album_forgotten1.columns = ['Forgotten1']

    df_albumrank2 = df.groupby('Album 2')['Title 2'].count(
    ).to_frame().reset_index().sort_values('Title 2', ascending=False)

    df_album_forgotten2 = pd.DataFrame(
        [album for album in albums if album not in df_albumrank2['Album 2'].values])
    df_album_forgotten2.columns = ['Forgotten2']

    fig.update_layout(xaxis_range=[1965, 2020])
    fig.update_layout(yaxis_range=[-26, 0])
  # , yanchor="bottom", y=-1.1, xanchor="left", x=0))
    # return go.Figure(), go.Figure()
    fig = benchmarkmodels_layout(fig)
    fig.update_layout(legend=dict(orientation="h"))
    fig.update_layout(hovermode="closest")
    # fig.update_layout(showlegend=False)

    # merged albumranks
    dfa = df_albumrank1.copy()
    dfb = df_albumrank2.copy()

    dfa.columns, dfb.columns = ['Album', 'Count'], ['Album', 'Count']
    dft = pd.concat([dfa, dfb]).groupby(['Album']).sum(['Count']).reset_index()
    ignored = [a for a in albums if a not in dft['Album']]
    ignored = [a for a in albums if a not in list(dft['Album'].values)]
    df_ignored = pd.DataFrame({'Album': ignored, 'Count': [0]*len(ignored)})
    df_album_total = pd.concat([dft, df_ignored])
    df_album_total

    fig_bar = px.bar(df_album_total, x='Album', y='Count')
    fig_bar.update_xaxes(categoryarray=albums)
    fig_bar.update_layout(yaxis_range=[0, 16])
    fig_bar = benchmarkmodels_layout(fig_bar)
    # fig_bar.show()

    # return to fig1, fig2, tbl_1, tbl_2, tbl_3, tbl_4, tbl_5
    return(fig,
           fig_rank,
           fig_bar,
           df[['Title 1', 'Album 1']].to_dict('records'),
           df_albumrank1[['Album 1', 'Title 1']].to_dict('records'),
           df[['Title 2', 'Album 2']].to_dict('records'),
           df_albumrank2[['Album 2', 'Title 2']].to_dict('records'),
           df_album_forgotten1[['Forgotten1']].to_dict('records'),
           df_album_forgotten2[['Forgotten2']].to_dict('records'))


if __name__ == "__main__":
    app.run_server(debug=False, threaded=True, port=8099)
