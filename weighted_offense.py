import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

rank_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1-vizwKykEEPNhUl9mtSR_2VaTslTXVjOLsHqxo3Jpfs/export?format=csv&gid=1365643765')[['Team','wOBA','Tier']].set_index('Team')
st.dataframe(rank_df)
