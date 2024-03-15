import streamlit as st
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import scipy as sp
import seaborn as sns
import pickle
import sklearn
import time
import xgboost as xgb
from xgboost import XGBClassifier
import optuna

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py

pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_white,
        'ytick.color': pl_white,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     }
    )

marker_colors = {
    'FF':'#d22d49', 
    'SI':'#c57a02',
    'FS':'#00a1c5',  
    'FC':'#933f2c', 
    'SL':'#9300c7',  
    'ST':'#C95EBE',
    'CU':'#3c44cd',
    'CH':'#07b526', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
}

pitch_names = {
    'FF':'Four-Seamer', 
    'SI':'Sinker',
    'FS':'Splitter',  
    'FC':'Cutter', 
    'SL':'Slider', 
    'ST':'Sweeper',
    'CU':'Curveball',
    'CH':'Changeup', 
    'KN':'Knuckleball',
    'SC':'Screwball', 
    'UN':'Unknown', 
}

year_data = pd.DataFrame()

player = 'Tyler Glasnow'
hand = 'L' if year_data.loc[(year_data['pitchername']==player),'pitcherside_L'].values[0] == 1 else 'R'

chart_df = year_data.loc[(year_data['pitchername']==player)].copy()
chart_df['3d_stuff_plus'] = 100
for pitchtype in chart_df['pitchtype'].unique():
    knn=KNeighborsRegressor(n_neighbors=min(30,int(chart_df.loc[chart_df['pitchtype']==pitchtype].shape[0]/2)))
    model_knn=knn.fit(chart_df.loc[chart_df['pitchtype']==pitchtype,['IHB','IVB','velo']],chart_df.loc[chart_df['pitchtype']==pitchtype,'plv_stuff_plus'])
    chart_df.loc[chart_df['pitchtype']==pitchtype,'3d_stuff_plus'] = model_knn.predict(chart_df.loc[chart_df['pitchtype']==pitchtype,['IHB','IVB','velo']])

Scene = dict(camera=dict(eye=dict(x=1.35, y=-1.6, z=0.9),
                        center=dict(x=-0.05,y=0,z=-0.1)
                        ),
    aspectmode='cube',
             xaxis = dict(title='Glove <-- HB --> Arm' if hand=='R' else 'Arm <-- HB --> Glove',
                         backgroundcolor=pl_background,
                         range=(-25,25) if hand=='R' else (25,-25)),
             yaxis = dict(title='velo',
                         backgroundcolor=pl_background),
             zaxis = dict(title='IVB',
                         backgroundcolor=pl_background,
                         range=(-25,25)),
             )

# model.labels_ is nothing but the predicted clusters i.e y_clusters
labels = chart_df['3d_stuff_plus']
trace = go.Scatter3d(x=chart_df['IHB'].mul(-1 if hand=='R' else 1), y=chart_df['velo'], z=chart_df['IVB'], 
                     mode='markers', marker=dict(color = labels, size= 5, line=dict(width = 0), 
                                                 cmin=50,cmax=150,
                                                 colorscale=[[x/100,'rgb'+str(tuple([int(y*255) for y in sns.color_palette('vlag',n_colors=101)[x]]))] for x in range(101)], 
                                                 colorbar=dict(
                                                     title="plvStuff+\n",
                                                     titleside="top",
                                                     tickmode="array",
                                                     tickvals=[50, 75, 100, 125, 150],
                                                     ticks="outside"
                                                 )
                                                ),
                     text=chart_df['pitchtype'].map(pitch_names),
                     hovertemplate =
                     '<b>%{text}</b>'+
                     '<br><b>plvStuff+: %{marker.color:.1f}</b>'+
                     '<br>Arm-Side Break: %{x:.1f}"'+
                     '<br>IVB: %{z:.1f}"'+
                     '<br>Velo: %{y}mph<extra></extra>'
                     )
layout = go.Layout(margin=dict(l=30,r=0,t=45, b=30
                              ),
                   scene = Scene,
                   height = 800,width = 800,
                   
                  )
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(
    title={
        'text': f"{player}'s plvStuff+",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.show()
