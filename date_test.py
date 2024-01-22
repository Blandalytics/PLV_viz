import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import urllib

from PIL import Image
from collections import Counter
from scipy import stats

test_df = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/data/date_pitch_map.parquet?raw=true')
date_range = st.slider(
    "Date range:",
    value=(test_df['game_played'].min(), 
           test_df['game_played'].max()),
    min_value=test_df['game_played'].min(),
    max_value=test_df['game_played'].max(),
    format="MM/DD")
min_date = date_range[0]
max_date = date_range[1]
pitches = test_df.loc[(test_df['game_played']>=min_date) & (test_df['game_played']<=max_date)].shape[0]
st.write(f'Pitch count: {pitches}')
