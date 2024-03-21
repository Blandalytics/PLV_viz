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

st.title("PLV Location App")

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
    file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_Loc_App_Data.parquet?raw=true'
    return pd.read_parquet(file_name)

year_data = load_data(year)

pitch_order = ['FF','SI','FC','SL','ST','CU','CH','FS']
st.dataframe(pd.pivot_table((year_data
                     .loc[(year_data['pitchtype'].isin(pitch_order)) & 
                          (year_data['pitch_id'].groupby([year_data['pitchername'],year_data['pitchtype']]).transform('count')>=10)]), 
                   values=['PLV_loc_plus','pitch_id'], index=['pitchername'],
                   columns='pitchtype', aggfunc={'PLV_loc_plus':'mean','pitch_id':'count'})
             .assign(Num_Pitches = lambda x: x[[('pitch_id',y) for y in pitch_order]].sum(axis=1),
                     plvLocation = lambda x: x[[('PLV_loc_plus',y) for y in pitch_order]].mul(x[[('pitch_id',y) for y in pitch_order]].droplevel(0, axis=1)).sum(axis=1) / x['Num_Pitches'])
             .drop(columns=[('pitch_id',y) for y in pitch_order+['KN','SC','UN']])
             .droplevel(0, axis=1)
             .reset_index()
             .set_axis(['Pitcher','CH','CU','FC','FF','FS','SI','SL','ST','Pitches','plvLocation+'], axis=1)
             .set_index('Pitcher')
             [['Pitches','plvLocation+']+pitch_order]
             .query(f'Pitches >= {pitch_threshold}')
             .sort_values('plvLocation+',ascending=False)
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
                 'plvLocation+':'float'
             })
             .reset_index()
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=50, vmax=150,
                                  cmap="vlag", subset=['plvLocation+']+pitch_order)
             .map(lambda x: 'color: transparent; background-color: transparent' if x==-100 else ''),
             hide_index=True
             )

players = list(year_data
               .groupby('pitchername')
               [['pitch_id','PLV_loc_plus']]
               .agg({'pitch_id':'count','PLV_loc_plus':'mean'})
               .query(f'pitch_id >={pitch_threshold}')
               .reset_index()
               .sort_values('PLV_loc_plus', ascending=False)
               ['pitchername']
              )
default_ix = players.index('Zack Wheeler')
player = st.selectbox('Choose a player:', players, index=default_ix)

st.write(f"{player}'s Repertoire")
st.dataframe(year_data
             .loc[(year_data['pitchername']==player)]
             .groupby('pitchtype')
             [['pitch_id','csw_pred','wOBAcon_pred','PLV_loc_plus']]
             .agg({
                 'pitch_id':'count',
                 'csw_pred':'mean',
                 'wOBAcon_pred':'mean',
                 'PLV_loc_plus':'mean'
                 })
             .astype({
                 'pitch_id':'int',
                 'csw_pred':'float',
                 'wOBAcon_pred':'float',
                 'PLV_loc_plus':'float'
                 })
             .reset_index()
             .assign(pitchtype = lambda x: x['pitchtype'].map(pitch_names))
             .rename(columns={
                 'pitchtype':'Pitch Type',
                 'pitch_id':'Pitches',
                 'csw_pred':'locCSW',
                 'wOBAcon_pred':'loc wOBAcon',
                 'PLV_loc_plus':'plvLocation+'
                 })
             .set_index('Pitch Type')
             .dropna()
             .sort_values('Pitches',ascending=False)
             .reset_index()
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=50, vmax=150,
                                  cmap="vlag", subset=['plvLocation+']),
             hide_index=True
            )

st.title("Interactive 3D Location Plot")
st.write('Controls:\n- Hover to see pitch details\n- Left click + drag to rotate the chart\n- Scroll to zoom\n- Right click + drag to move the chart')

pitches = {'All':1}
pitches.update(year_data
    .loc[year_data['pitchername']==player,'pitchtype']
    .map(pitch_names)
    .value_counts(normalize=True)
    .where(lambda x : x>0)
    .dropna()
    .to_dict()
)

select_list = []
for pitch in pitches.keys():
    select_list += [f'{pitch} ({pitches[pitch]:.1%})']
pitch_type = st.selectbox('Choose a pitch (season usage):', select_list)
pitch_type = pitch_type.split('(')[0][:-1]

def location_chart(df,player,pitch_type):
    if pitch_type=='All':
        chart_df = df.loc[(df['pitchername']==player)].copy()
    else:
        chart_df = df.loc[(df['pitchername']==player) & (df['pitchtype']=={v: k for k, v in pitch_names.items()}[pitch_type])].copy()
    chart_df['smoothed_csw'] = 0.288
    chart_df['smoothed_wOBAcon'] = 0.3284

    plate_y = -.25

    # for pitchtype in chart_df['pitchtype'].unique():
    #     if chart_df.loc[chart_df['pitchtype']==pitchtype].shape[0]==1:
    #         chart_df.loc[chart_df['pitchtype']==pitchtype,'smoothed_csw'] = chart_df.loc[chart_df['pitchtype']==pitchtype,'csw_pred']
    #         chart_df.loc[chart_df['pitchtype']==pitchtype,'smoothed_wOBAcon'] = chart_df.loc[chart_df['pitchtype']==pitchtype,'wOBAcon_pred']
    #     else:
    #         knn=KNeighborsRegressor(n_neighbors=min(10,int(chart_df.loc[chart_df['pitchtype']==pitchtype].shape[0]/2)))
    #         model_knn=knn.fit(chart_df.loc[chart_df['pitchtype']==pitchtype,['p_x','p_z','balls','strikes']],chart_df.loc[chart_df['pitchtype']==pitchtype,'csw_pred'])
    #         chart_df.loc[chart_df['pitchtype']==pitchtype,'smoothed_csw'] = model_knn.predict(chart_df.loc[chart_df['pitchtype']==pitchtype,['p_x','p_z','balls','strikes']])
    #         model_knn=knn.fit(chart_df.loc[chart_df['pitchtype']==pitchtype,['p_x','p_z','balls','strikes']],chart_df.loc[chart_df['pitchtype']==pitchtype,'wOBAcon_pred'])
    #         chart_df.loc[chart_df['pitchtype']==pitchtype,'smoothed_wOBAcon'] = model_knn.predict(chart_df.loc[chart_df['pitchtype']==pitchtype,['p_x','p_z','balls','strikes']])
    
    layout = go.Layout(height = 600,width = 500,xaxis_range=[-2.5,2.5], yaxis_range=[-1,6])

    labels = chart_df['PLV_loc_plus']
    bonus_text = chart_df['pitchtype'].map(pitch_names)
    hover_text = '<b>%{text}</b><br><b>plvLoc+: %{marker.color:.1f}</b><br>Count: %{customdata[0]}-%{customdata[1]}<br>X Loc: %{x:.1f}ft<br>Y Loc: %{y:.1f}ft<br>locCSW: %{customdata[2]:.1%}<br>loc wOBAcon: %{customdata[3]:.3f}<extra></extra>'
    marker_dict = dict(color = labels, size= 5, line=dict(width=0), 
                               cmin=50,cmax=150,
                               colorscale=[[x/100,'rgb'+str(tuple([int(y*255) for y in sns.color_palette('vlag',n_colors=101)[x]]))] for x in range(101)], 
                               colorbar=dict(
                                   title="plvLocation+\n",
                                   titleside="top",
                                   tickmode="array",
                                   tickvals=[50, 75, 100, 125, 150],
                                   ticks="outside"
                                   ))
    trace = go.Scatter(x=chart_df['p_x'].mul(-1), y=chart_df['p_z'], mode='markers', 
                       marker=marker_dict,text=bonus_text,
                       customdata=chart_df[['balls','strikes','csw_pred','wOBAcon_pred']],
                       hovertemplate=hover_text,
                        showlegend=False)
    
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    fig.add_trace(go.Scatter(x=[10/12,10/12], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,-10/12], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[1.5,1.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[3.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[10/36,10/36], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/36,-10/36], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[1.5+2/3,1.5+2/3],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[3.5-2/3,3.5-2/3],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    
    # Plate
    fig.add_trace(go.Scatter(x=[-8.5/12,8.5/12], y=[plate_y,plate_y],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-8.5/12,-8.25/12], y=[plate_y,plate_y+0.15],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[8.5/12,8.25/12], y=[plate_y,plate_y+0.15],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[8.28/12,0], y=[plate_y+0.15,plate_y+0.25],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-8.28/12,0], y=[plate_y+0.15,plate_y+0.25],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    
    overall_loc = chart_df['PLV_loc_plus'].mean()
    type_text = '' if pitch_type=='All' else ' '+pitch_type+'s'
    fig.update_layout(
            template='simple_white',
            title={
                'text': f"{player}'s{type_text}<br>plvLocation+: {overall_loc:.1f}",
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            legend={
                "x": 0.8,
                "y": 0.67}
        )
    fig.show()
    st.plotly_chart(fig,
                    theme=None
                   )
location_chart(year_data,player,pitch_type)
