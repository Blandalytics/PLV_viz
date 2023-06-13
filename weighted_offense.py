import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

colors = {
  'Top':'#f4cccc',
  'Solid':'#fce5cd',
  'Average':'#fff2cc',
  'Weak':'#d9ead3',
  'Poor':'#c9daf8'
}

def highlight_cols(x):
    df = x.copy()
    #select all values
    for col in ['Team','wOBA','Tier']:
      df[col] = df['Tier'].apply(lambda x: 'color: black; border: 1.5px solid white; background-color: '+colors[x])
    #return color df
    return df
  
rank_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1-vizwKykEEPNhUl9mtSR_2VaTslTXVjOLsHqxo3Jpfs/export?format=csv&gid=1365643765')[['Team','wOBA','Tier']].query("Tier != ''")
st.dataframe(rank_df
             .style
             .format(precision=3)
             .apply(highlight_cols, axis=None),
             width=500,
             height=800
            )

# st.dataframe(highlight_cols(rank_df))
