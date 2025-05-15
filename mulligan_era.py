import streamlit as st
import numpy as np
import pandas as pd
st.title('Mulligan ERA')
st.write(
    "Using starts from the 2024 and 2025 (through 5/13) seasons, choose a timeframe of most recent starts to analyze, \
    and the number of each pitcher's worst ERA starts to drop, and this will highlight the difference in ERA between the two samples."
)
start_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1klECnPdLB1GmSZ5dND9-At0Z76DPdlS3x0zLhrdIoT4/export?gid=1317572214&format=csv')
start_data['Date'] = pd.to_datetime(start_data['Date'])
start_data['num_starts'] = start_data['Date'].groupby(start_data['playerId']).transform('count')
start_data['start_recency'] = start_data.groupby("playerId")["Date"].rank(method="first") 
start_data['start_recency'] = start_data['start_recency'].groupby(start_data['playerId']).transform('max') - start_data['start_recency']

col1, col2, col3, col4, col5 = st.columns([1/7,2/7,1/7,2/7,1/7])
with col2:
    last_starts = st.slider('Last X Starts (across 2024 & 2025)',min_value=10,max_value=30,value=20)
with col4:
    worst_drop = st.number_input('Number of worst ERA starts to drop:',min_value=1,max_value=5)

filter_df = start_data.loc[(start_data['num_starts']>=last_starts) & (start_data['start_recency']<=last_starts)].reset_index(drop=True)
filter_df['worst_era'] = filter_df.groupby("playerId")["ERA"].rank(ascending=False, method='min')
filter_df['worst_era_sub'] = filter_df.groupby(["playerId",'worst_era'])["Date"].rank(ascending=False)
filter_df['worst_era'] = filter_df['worst_era'].add(filter_df['worst_era_sub']).sub(1).astype('int')

st.dataframe(
    pd.merge(
        filter_df.groupby(['playerId','Name'])[['IP','ER']].sum().assign(ERA = lambda x: x['ER'].div(x['IP']).mul(9)),
        filter_df.loc[filter_df['worst_era']>worst_drop].groupby(['playerId','Name'])[['IP','ER']].sum().assign(ERA = lambda x: x['ER'].div(x['IP']).mul(9)),
        how='inner',
        left_index=True,
        right_index=True,
        suffixes=['','_mull']
    )
    .assign(diff = lambda x: x['ERA'].sub(x['ERA_mull']))
    .sort_values('diff',ascending=False)
    .rename(columns={
        'IP_mull':'mIP',
        'ER_mull':'mER', 
        'ERA_mull':'mERA'})
    .reset_index()
)
