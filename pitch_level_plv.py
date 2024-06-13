import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import urllib
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image
from scipy import stats

# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_background,
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

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

st.title("Pitcher Characteristics")

agg_dict = {
    'pitch_id':'count',
    'velo':'mean',
    'pitch_extension':'mean',
    'raw_vaa':'mean',
    'adj_vaa':'mean',
    'pfx_x':'mean',
    'pfx_z':'mean',
    'IHB':'mean',
    'IVB':'mean',
    'p_x':'mean',
    'p_z':'mean',
    'plv':'mean'
}

stat_names = {
    'pitch_id':'# Pitches',
    'index':'Pitch ID',
    'pitchtype':'Type',
    'pitchername':'Name',
    'pitcher_mlb_id':'MLBAMID',
    'velo':'Velo',
    'pitch_extension':'Ext',
    'raw_vaa':'VAA',
    'adj_vaa':'HAVAA',
    'pfx_x':'pfx_x',
    'pfx_z':'pfx_z',
    'IHB':'Arm-Side Break',
    'IVB':'IVB',
    'p_x':'Plate X',
    'p_z':'Plate Z',
    'plv':'PLV'
}

type_dict = {
    'pitch_id':'int',
    'velo':'float',
    'pitch_extension':'float',
    'raw_vaa':'float',
    'adj_vaa':'float',
    'pfx_x':'float',
    'pfx_z':'float',
    'IHB':'float',
    'IVB':'float',
    'plv':'float'
}

round_dict = {
    'pitch_id':0,
    'index':0,
    'velo':1,
    'pitch_extension':2,
    'pfx_x':1,
    'pfx_z':1,
    'IHB':1,
    'IVB':1,
    'raw_vaa':2,
    'adj_vaa':2,
    'p_x':2,
    'p_z':2,
    'plv':2
}

marker_colors = {
    'FF':'#d22d49', 
    'SI':'#c57a02',
    'FS':'#00a1c5',  
    'FC':'#933f2c', 
    'SL':'#9300c7',  
    'SL/ST':'#9300c7',  
    'ST':'#C95EBE',
    'CU':'#3c44cd',
    'CH':'#07b526', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
}

years = [2024,2023,2022,2021,2020]
year = st.selectbox('Choose a year:', years)


@st.cache_data(ttl=60*15,show_spinner=f"Loading {year} data")
def load_data(year):
    df = pd.DataFrame()
    for month in range(3,11):
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_Stuff_App_Data-{month}.parquet?raw=true'
        df = pd.concat([df,pd.read_parquet(file_name)], ignore_index=True).query('pitchtype != "UN"')
    return df.reset_index(drop=True)

year_data = load_data(year)
year_data['game_played'] = pd.to_datetime(year_data['game_played']).dt.date

metrics = ['PLV','Velo', 'Ext', 'VAA', 'HAVAA','Arm-Side Break','IVB','pfx_x','pfx_z','Plate X','Plate Z']

st.header('League-Wide Metrics')
col1, col2, col3 = st.columns(3)

with col1:
    pitch_threshold = st.number_input(f'Min # of Pitches:',
                                      min_value=0, 
                                      max_value=int(year_data.groupby('pitcher_mlb_id')['pitch_id'].count().quantile(0.8).round(-2)),
                                      step=25, 
                                      value=int(year_data.groupby('pitcher_mlb_id')['pitch_id'].count().quantile(0.2).round(-2)))

with col2:
    usage_threshold = st.number_input(f'Min Usage %:',
                                      min_value=0.0, 
                                      max_value=50.0,
                                      step=1.0, 
                                      value=5.0,format='%f')
    usage_threshold = usage_threshold/100
    
with col3:
    szn_metric = st.selectbox('Choose a metric:', metrics)
    szn_round_val = round_dict[{v: k for k, v in stat_names.items()}[szn_metric]]
    if szn_metric=='PLV':
        szn_metric = 'type_plv'

st.dataframe(pd.pivot_table((year_data
                             .assign(IHB = lambda x: np.where(x['pitcherside_L']==0,x['IHB']*-1,x['IHB']))
                             .groupby(['pitcher_mlb_id','pitchername','pitchtype'])
                             [list(agg_dict.keys())]
                             .agg(agg_dict)
                             .dropna()
                             .astype(type_dict)
                             .round(round_dict)
                             .rename(columns=stat_names)
                             .rename(columns={'PLV':'type_plv'})
                             .reset_index()
                             .assign(num_pitches = lambda x: x['# Pitches'].groupby([x['pitcher_mlb_id'],x['pitchername']]).transform('sum'),
                                     usage = lambda x: x['# Pitches'] / x['num_pitches'],
                                     PLV = lambda x: x['# Pitches'].mul(x['type_plv']).groupby([x['pitcher_mlb_id'],x['pitchername']]).transform('sum') / x['num_pitches'])
                             ).query(f'usage >= {usage_threshold}'), 
                            values=szn_metric, index=['pitcher_mlb_id','pitchername','num_pitches','PLV'],
                            columns=['pitchtype'], aggfunc="mean")
             .query(f'num_pitches >={pitch_threshold}')
             .sort_values('PLV',ascending=False)
             .fillna(0)
             .reset_index()
             .rename(columns={'pitcher_mlb_id':'MLBAMID',
                              'pitchername':'Name',
                             'num_pitches':'# Pitches'})
             .set_index('MLBAMID')
             [['Name','# Pitches','PLV','FF','SI','FC','SL','ST','CU','CH','FS','KN']]
             .style
             .format(precision=szn_round_val,thousands=',', na_rep="")
             .format(precision=2,subset=['PLV'])
             .background_gradient(axis=0, vmin=4.25, vmax=5.75,
                                  cmap="vlag", subset=['PLV'] if szn_metric != 'type_plv' else ['PLV']+list(year_data['pitchtype'].unique()))
             .map(lambda x: 'color: transparent; background-color: transparent' if x==0 else ''),
            hide_index=True
            )

fig, ax = plt.subplots(figsize=(6,4))
sns.kdeplot((year_data
             .assign(IHB = lambda x: np.where(x['pitcherside_L']==0,x['IHB']*-1,x['IHB']))
             .query('pitchtype!="KN"')
             .replace('ST','SL' if szn_metric=='type_plv' else 'ST')
             .replace('SL','SL/ST' if szn_metric=='type_plv' else 'SL')
             .rename(columns=stat_names)
             .rename(columns={'PLV':'type_plv'})
             .groupby(['MLBAMID','Name','Type'])
             [['# Pitches',szn_metric]]
             .agg({'# Pitches':'count',
                   szn_metric:'mean'})
             .reset_index()
             .assign(num_pitches = lambda x: x['# Pitches'].groupby([x['MLBAMID'],x['Name']]).transform('sum'),
                     usage = lambda x: x['# Pitches'] / x['num_pitches'])
             .query(f'usage >= {usage_threshold}')).query(f'num_pitches >={pitch_threshold}'),
            x=szn_metric,
            hue='Type',
            palette=marker_colors,
            linewidth=3,
            # common_norm=True if szn_metric != 'type_plv' else False
           )
fig.suptitle(f'MLB {szn_metric} Distribution')
ax.get_yaxis().set_visible(False)
ax.get_legend().set_title('')

sns.despine(left=True)
st.pyplot(fig)

st.header('Per-Game Metrics')
col1, col2 = st.columns([0.5,0.5])

with col1:
    players = list(year_data.groupby('pitchername').filter(lambda x: len(x) >= pitch_threshold)['pitchername'].value_counts().index)
    default_player = players.index('Zack Wheeler')
    player = st.selectbox('Choose a pitcher:', players, index=default_player)

with col2:
    player_metric = st.selectbox('Choose a metric: ', metrics)
    round_val = round_dict[{v: k for k, v in stat_names.items()}[player_metric]]
    if player_metric=='PLV':
        player_metric = 'type_plv'

st.dataframe(pd.pivot_table((year_data
                             .assign(IHB = lambda x: np.where(x['pitcherside_L']==0,x['IHB']*-1,x['IHB']))
                             .loc[year_data['pitchername']==player]
                             .groupby(['game_played','pitchername','pitchtype'])
                             [list(agg_dict.keys())]
                             .agg(agg_dict)
                             .dropna()
                             .astype(type_dict)
                             .round(round_dict)
                             .rename(columns=stat_names)
                             .rename(columns={'PLV':'type_plv'})
                             .reset_index()
                             .assign(num_pitches = lambda x: x['# Pitches'].groupby([x['pitchername'],x['game_played']]).transform('sum'),
                                     PLV = lambda x: x['# Pitches'].mul(x['type_plv']).groupby([x['pitchername'],x['game_played']]).transform('sum') / x['num_pitches'])
                             ),
                            values=player_metric, index=['pitchername','game_played','num_pitches','PLV'],
                            columns=['pitchtype'], aggfunc="mean")
             .fillna(0)
             .reset_index()
             .rename(columns={'pitchername':'Name',
                             'num_pitches':'# Pitches',
                             'game_played':'Game Date'})
             .set_index(['Name','Game Date'])
             [['# Pitches','PLV']+[x for x in ['FF','SI','FC','SL','ST','CU','CH','FS','KN'] if x in year_data.loc[year_data['pitchername']==player,'pitchtype'].unique()]]
             .style
             .format(precision=round_val,thousands=',')
             .format(precision=2,subset=['PLV'])
             .background_gradient(axis=0, vmin=4.25, vmax=5.75,
                                  cmap="vlag", subset=['PLV']+list(year_data.loc[year_data['pitchername']==player,'pitchtype'].unique()) if player_metric=='type_plv' else ['PLV'])
             .map(lambda x: 'color: transparent; background-color: transparent' if x==0 else '')
            )

st.header('Pitch-Level Metrics')
games = list(year_data.loc[year_data['pitchername']==player,'game_played'].unique())
game_date = st.selectbox('Choose a game:', games, index=len(games)-1)
# date_format = '%Y-%m-%d'
# game_date = datetime.strptime(game_date_str, date_format).date()


st.dataframe(year_data
             .assign(IHB = lambda x: np.where(x['pitcherside_L']==0,x['IHB']*-1,x['IHB']))
             .loc[(year_data['pitchername']==player) & 
                  (year_data['game_played']==game_date),
             ['game_played','pitchername','pitchtype']+list(agg_dict.keys())]
             .sort_values('pitch_id')
             .drop(columns=['pitch_id'])
             .reset_index(drop=True)
             .reset_index()
             .assign(index = lambda x: x['index']+1)
             .astype({
                 'velo':'float',
                 'pitch_extension':'float',
                 'raw_vaa':'float',
                 'adj_vaa':'float',
                 'pfx_x':'float',
                 'pfx_z':'float',
                 'IHB':'float',
                 'IVB':'float',
                 'plv':'float'
                 }
                     )
             .round(round_dict)
             .rename(columns=stat_names)
             .rename(columns={'Pitch ID':'Pitch #',
                             'game_played':'Game Date'})
             [['Name','Game Date','Pitch #','Type','PLV','Velo','Ext','VAA','HAVAA','pfx_x','pfx_z','Arm-Side Break','Plate X','Plate Z','IVB']],
             hide_index=True
)
