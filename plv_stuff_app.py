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

st.title("PLV Stuff App")

# Year
years = [2023,2022,2021,2020]
year = st.radio('Choose a year:', years)

pitch_threshold = st.number_input(f'Min # of Pitches:',
                                  min_value=0, 
                                  max_value=2000,
                                  step=50, 
                                  value=500 if year != 2024 else 0)

@st.cache_data(ttl=2*3600,show_spinner=f"Loading {year} data")
def load_data(year):
    file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_Stuff_App_Data.parquet?raw=true'
    return pd.read_parquet(file_name)

year_data = load_data(year)

pitch_order = ['FF','SI','FC','SL','ST','CU','CH','FS']
st.dataframe(pd.pivot_table((year_data
                     .loc[(year_data['pitchtype'].isin(pitch_order)) & 
                          (year_data['pitch_id'].groupby([year_data['pitchername'],year_data['pitchtype']]).transform('count')>=10)]), 
                   values=['plv_stuff_plus','pitch_id'], index=['pitchername'],
                   columns='pitchtype', aggfunc={'plv_stuff_plus':'mean','pitch_id':'count'})
             .assign(Num_Pitches = lambda x: x[[('pitch_id',y) for y in pitch_order]].sum(axis=1),
                     plvStuff = lambda x: x[[('plv_stuff_plus',y) for y in pitch_order]].mul(x[[('pitch_id',y) for y in pitch_order]].droplevel(0, axis=1)).sum(axis=1) / x['Num_Pitches'])
             .drop(columns=[('pitch_id',y) for y in pitch_order+['KN','SC','UN'])
             .droplevel(0, axis=1)
             .reset_index()
             .set_axis(['Pitcher','CH','CU','FC','FF','FS','SI','SL','ST','Pitches','plvStuff+'], axis=1)
             .set_index('Pitcher')
             [['Pitches','plvStuff+']+pitch_order]
             .query(f'Pitches >= {pitch_threshold}')
             .sort_values('plvStuff+',ascending=False)
             .fillna(-100)
             .astype({
                 'CH':'float',
                 'CU':'float',
                 'FC':'float',
                 'FF':'float',
                 'FS':'float',
                 'SI':'float',
                 'SL':'float',
                 'ST':'float',
                 'Pitches':'int',
                 'plvStuff+':'float'
             })
             .reset_index()
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=50, vmax=150,
                                  cmap="vlag", subset=['plvStuff+']+pitch_order)
             .map(lambda x: 'color: transparent; background-color: transparent' if x==-100 else ''),
             hide_index=True
             )

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

st.write(f"{player}'s Repertoire")
st.dataframe(year_data
             .loc[(year_data['pitchername']==player)]
             .groupby('pitchtype')
             [['pitch_id','velo','IVB','IHB','plv_stuff_plus']]
             .agg({
                 'pitch_id':'count',
                 'velo':'mean',
                 'IVB':'mean',
                 'IHB':'mean',
                 'plv_stuff_plus':'mean'
                 })
             .astype({
                 'pitch_id':'int',
                 'velo':'float',
                 'IVB':'float',
                 'IHB':'float',
                 'plv_stuff_plus':'float'
                 })
             .reset_index()
             .assign(pitchtype = lambda x: x['pitchtype'].map(pitch_names),
                     IHB = lambda x: x['IHB'].mul(-1 if hand=='R' else 1)
                    )
             .rename(columns={
                 'pitchtype':'Pitch Type',
                 'pitch_id':'Pitches',
                 'velo':'Velo',
                 'IHB':'Arm-Side Break',
                 'plv_stuff_plus':'plvStuff+'
                 })
             .set_index('Pitch Type')
             .dropna()
             .sort_values('Pitches',ascending=False)
             .reset_index()
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=50, vmax=150,
                                  cmap="vlag", subset=['plvStuff+']),
             hide_index=True
            )

st.title("Interactive 3D Stuff Plot")
st.write('Controls:\n- Hover to see pitch details\n- Left click + drag to rotate the chart\n- Scroll to zoom\n- Right click + drag to move the chart')

palette = st.radio('Choose a color palette:', ['plvStuff+','Pitch Type'])

def stuff_chart(df,player,palette):
    chart_df = df.loc[(df['pitchername']==player)].copy()
    chart_df['3d_stuff_plus'] = 100

    ax_lim = max(25,chart_df[['IVB','IHB']].abs().max().max())
    for pitchtype in chart_df['pitchtype'].unique():
        if chart_df.loc[chart_df['pitchtype']==pitchtype].shape[0]==1:
            chart_df.loc[chart_df['pitchtype']==pitchtype,'3d_stuff_plus'] = chart_df.loc[chart_df['pitchtype']==pitchtype,'plv_stuff_plus']
        else:
            knn=KNeighborsRegressor(n_neighbors=min(30,int(chart_df.loc[chart_df['pitchtype']==pitchtype].shape[0]/2)))
            model_knn=knn.fit(chart_df.loc[chart_df['pitchtype']==pitchtype,['IHB','IVB','velo']],chart_df.loc[chart_df['pitchtype']==pitchtype,'plv_stuff_plus'])
            chart_df.loc[chart_df['pitchtype']==pitchtype,'3d_stuff_plus'] = model_knn.predict(chart_df.loc[chart_df['pitchtype']==pitchtype,['IHB','IVB','velo']])
    
    Scene = dict(camera=dict(eye=dict(x=1.35, y=-1.6, z=0.9),
                            center=dict(x=-0.05,y=0,z=-0.1)
                            ),
        aspectmode='cube',
                 xaxis = dict(title='Glove <-- HB --> Arm' if hand=='R' else 'Arm <-- HB --> Glove',
                             backgroundcolor=pl_background,
                             range=(-ax_lim,ax_lim) if hand=='R' else (ax_lim,-ax_lim)),
                 yaxis = dict(title='velo',
                             backgroundcolor=pl_background),
                 zaxis = dict(title='IVB',
                             backgroundcolor=pl_background,
                             range=(-ax_lim,ax_lim)),
                 )

    if palette=='plvStuff+':
        labels = chart_df['3d_stuff_plus']
        marker_dict = dict(color = labels, size= 5, line=dict(width = 0), 
                                                     cmin=50,cmax=150,
                                                     colorscale=[[x/100,'rgb'+str(tuple([int(y*255) for y in sns.color_palette('vlag',n_colors=101)[x]]))] for x in range(101)], 
                                                     colorbar=dict(
                                                         title="plvStuff+\n",
                                                         titleside="top",
                                                         tickmode="array",
                                                         tickvals=[50, 75, 100, 125, 150],
                                                         ticks="outside"
                                                     ))
        bonus_text = chart_df['pitchtype'].map(pitch_names)
        hover_text = '<b>%{text}</b><br><b>plvStuff+: %{marker.color:.1f}</b><br>Velo: %{y}mph<br>IVB: %{z:.1f}"<br>Arm-Side Break: %{x:.1f}"<extra></extra>'
    else:
        labels = chart_df['pitchtype'].map(marker_colors)
        marker_dict = dict(color=labels,size=5,line=dict(width=0.25,color=pl_line_color))
        bonus_text = chart_df['3d_stuff_plus']
        hover_text = '<b>%{customdata}</b><br><b>plvStuff+: %{text:.1f}</b><br>Velo: %{y}mph<br>IVB: %{z:.1f}"<br>Arm-Side Break: %{x:.1f}"<extra></extra>'
    trace = go.Scatter3d(x=chart_df['IHB'].mul(-1 if hand=='R' else 1), y=chart_df['velo'], z=chart_df['IVB'], 
                         mode='markers', marker=marker_dict,
                         text=bonus_text,
                         customdata=chart_df['pitchtype'].map(pitch_names),
                         hovertemplate=hover_text
                         )
    layout = go.Layout(margin=dict(l=30,r=0,t=45, b=30
                                  ),
                       scene = Scene,
                       height = 500,width = 500                  
                      )
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    overall_stuff = chart_df['plv_stuff_plus'].mean()
    fig.update_layout(
        title={
            'text': f"{player}'s<br>plvStuff+: {overall_stuff:.1f}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='white')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='white')
    st.plotly_chart(fig,use_container_width=True, theme=None)
stuff_chart(year_data,player,palette)
