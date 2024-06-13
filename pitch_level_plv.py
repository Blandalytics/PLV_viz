import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import urllib

from PIL import Image
from scipy import stats

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

years = [2024,2023,2022,2021,2020]
year = st.selectbox('Choose a year:', years)

@st.cache_data(ttl=60*15,show_spinner=f"Loading {year} data")
def load_data(year):
    df = pd.DataFrame()
    for month in range(3,11):
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_Stuff_App_Data-{month}.parquet?raw=true'
        df = pd.concat([df,pd.read_parquet(file_name)], ignore_index=True)
    return df.reset_index(drop=True)

year_data = load_data(year)
year_data['game_played'] = pd.to_datetime(year_data['game_played']).dt.date

st.dataframe(pd.pivot_table((year_data
                             .groupby(['pitcher_mlb_id','pitchername','pitchtype'])
                             [list(agg_dict.keys())]
                             .agg(agg_dict)
                             .dropna()
                             .astype(type_dict)
                             .round(round_dict)
                             .rename(columns=stat_names)
                             .rename(columns={'PLV':'type_plv'})
                             .reset_index()
                             .assign(PLV = lambda x: x['# Pitches'].mul(x['type_plv']).groupby([x['pitcher_mlb_id'],x['pitchername']]).transform('sum') / x['# Pitches'].groupby([x['pitcher_mlb_id'],x['pitchername']]).transform('sum'))
                             ), 
                            values='type_plv', index=['pitcher_mlb_id','pitchername','PLV'],
                            columns=['pitchtype'], aggfunc="mean")
             .fillna(-100)
             .reset_index()
             .set_index(['pitcher_mlb_id','pitchername'])
             .style
             .format(precision=2, thousands=',')
             .background_gradient(axis=0, vmin=4, vmax=6,
                                  cmap="vlag")
             .map(lambda x: 'color: transparent; background-color: transparent' if x==-100 else ''))

list(year_data['pitchername'].value_counts().index)
