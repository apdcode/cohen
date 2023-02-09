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
df_alb = pd.read_csv(r"C:\repos\fun\cohen\data\CohenAlbums.txt", sep=',')
df_alb
# dfi = pd.read_csv( sorted(
