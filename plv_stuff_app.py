import streamlit as st
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import seaborn as sns

import sklearn
from sklearn.neighbors import KNeighborsRegressor

import urllib
from PIL import Image

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

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

# Year
years = [2023,2022,2021,2020]
year = st.radio('Choose a year:', years)

pitch_threshold = st.number_input(f'Min # of Pitches:',
                                  min_value=0, 
                                  max_value=2000,
                                  step=50, 
                                  value=500 if year != 2024 else 0)

@st.cache_data
def load_data(year):
    file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_Stuff_App_Data.parquet?raw=true'
    return pd.read_parquet(file_name)

year_data = load_data(year)

players = list(year_data
               .groupby('pitchername')
               [['pitch_id','plv_stuff_plus']]
               .agg({'pitch_id':'count','plv_stuff_plus':'mean'})
               .query(f'pitch_id >={pitch_threshold}')
               .reset_index()
               .sort_values('plv_stuff_plus', ascending=False)
               ['pitchername']
              )
default_ix = players.index('Zack Wheeler')
player = st.selectbox('Choose a player:', players, index=default_ix)

hand = 'L' if year_data.loc[(year_data['pitchername']==player),'pitcherside_L'].values[0] == 1 else 'R'

def stuff_chart(df,player):
    chart_df = df.loc[(df['pitchername']==player)].copy()
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
                       height = 500,width = 500,
                       gridwidth=2                       
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
    st.plotly_chart(fig)
stuff_chart(year_data,player)
