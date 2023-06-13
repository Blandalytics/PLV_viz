import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

colors = {
  'Top':'#a60027',
  'Solid':'#fa9053',
  'Average':'#ffffbf',
  'Weak':'#87cc67',
  'Poor':'#006837'
}

def highlight_cols(x):
    df = x.copy()
    #select all values
    for col in ['Team','wOBA','Tier']:
      df[col] = df['Tier'].apply(lambda x: 'color: black; border: 1.5px solid white; background-color: '+colors[x])
    #return color df
    return df
  
rank_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1-vizwKykEEPNhUl9mtSR_2VaTslTXVjOLsHqxo3Jpfs/export?format=csv&gid=1365643765')[['Team','wOBA','Tier']].query("Tier != ''")

color_thresh = (rank_df['wOBA'].max() - rank_df['wOBA'].min())/2

st.title('MLB Offense Ranks')
st.dataframe(rank_df
             .style
             .format(precision=4)
             .apply(highlight_cols, axis=None),
#              .background_gradient(axis=0, gmap=(rank_df['wOBA']-rank_df['wOBA'].min())/(rank_df['wOBA'].max() - rank_df['wOBA'].min()), 
#                                   vmax=0.95, vmin=0.05, 
#                                   cmap='vlag'),
             width=500,
             height=800
            )
