import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

colors = {
  'Top':['white','#bb5f5d'],
  'Solid':['black','#dbaba8'],
  'Average':['black','#faf5f5'],
  'Weak':['black','#aebcd1'],
  'Poor':['white','#5a84bd']
}

def highlight_cols(x):
    df = x.copy()
    #select all values
    for col in ['Team','wOBA','Tier']:
      df[col] = df['Tier'].apply(lambda x: f'color: {colors[x][0]}; border: 1.5px solid white; background-color: {colors[x][1]}')
    #return color df
    return df
  
rank_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1-vizwKykEEPNhUl9mtSR_2VaTslTXVjOLsHqxo3Jpfs/export?format=csv&gid=1365643765')[['Team','wOBA',
#                                                                                                                                                'Tier'
                                                                                                                                              ]].query("wOBA != ''")

st.title('MLB Offense Ranks')
st.dataframe(rank_df
             .style
             .format(precision=4)
#              .apply(highlight_cols, axis=None)
             .background_gradient(axis=0,gmap=(rank_df['wOBA']-0.318)/0.016, 
                                  vmin=-2,vmax=2,
                                  cmap='vlag'),
             width=400,
             height=800,
#              hide_index=True
            )
