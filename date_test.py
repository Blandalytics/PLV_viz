import streamlit as st
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import urllib

from datetime import time
from PIL import Image
from collections import Counter
from scipy import stats

test_df = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/data/date_pitch_map.parquet?raw=true')
test_df['year_played'] = pd.to_datetime(test_df['game_played']).dt.year
test_df['game_played'] = pd.to_datetime(test_df['game_played']).dt.date
st.write(test_df.dtypes)
years = [2023,2022,2021,2020]
year = st.radio('Choose a year:', years)
test_df = test_df.loc[test_df['year_played']==year].copy()

date_range = st.slider(
    "Date range:",
    value=(test_df['game_played'].min(), 
           test_df['game_played'].max()),
    min_value=test_df['game_played'].min(),
    max_value=test_df['game_played'].max(),
    format="MM/DD")
st.write(f'Date range: {date_range[0]:%m/%d/%Y} - {date_range[1]:%m/%d/%Y}')

season_start = test_df['game_played'].min()
season_end = test_df['game_played'].max()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Date Start", season_start,
                               min_value=season_start,
                               max_value=season_end,
                               format="MM/DD/YYYY")
with col2:
    end_date = st.date_input("Date End", season_end,
                             min_value=season_start,
                             max_value=season_end,
                             format="MM/DD/YYYY")

st.write(f'Date range: {start_date:%b %d} - {end_date:%b %d}')
