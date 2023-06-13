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

# def background_color(col):
#     return [f'background-color: {colors[tier]}' for i,x in col.iteritems()]
#   f"color: {colors[tier]}"

def highlight_cols(s):
#   if s in colors.keys():
#     color = colors[s]
#     return 'background-color: % s; color: black' % color
  
    df1 = pd.DataFrame('', index=s.index, columns=s.columns)
    df1[['Team','wOBA','Tier']] = rank_df['Tier'].map(colors)
    return df1
  
rank_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1-vizwKykEEPNhUl9mtSR_2VaTslTXVjOLsHqxo3Jpfs/export?format=csv&gid=1365643765')[['Team','wOBA','Tier']]
st.dataframe(rank_df
             .style
             .format(precision=3)
             .applymap(highlight_cols, axis=None)
            )
