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
      df[col] = df['Tier'].map(colors)
    #return color df
    return df[['Team','wOBA','Tier']]
  
rank_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1-vizwKykEEPNhUl9mtSR_2VaTslTXVjOLsHqxo3Jpfs/export?format=csv&gid=1365643765')[['Team','wOBA','Tier']]
st.dataframe(rank_df
             .style
             .format(precision=3)
             .apply(highlight_cols, axis=None)
            )

# st.dataframe(highlight_cols(rank_df))
