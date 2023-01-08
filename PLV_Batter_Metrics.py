import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_background,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_white,
        'ytick.color': pl_white,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     }
    )

line_color = sns.color_palette('vlag', n_colors=20)[0]

st.title("Batter Ability Metrics")
## Selectors
# Year
year = st.radio('Choose a year:', [2022,2021,2020])

def z_score_scaler(series):
    return (series - series.mean()) / series.std()

stat_names = {
    'swing_agg':'Swing Agg',
    'strike_zone_judgement':'SZ Judge',
    'contact_over_expected':'Contact',
    'decision_value':'Sw Dec',
    'adj_power':'Adj Power',
    'batter_wOBA':'Value Added'
}

# Load Data
def load_season_data():
    file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_App_Data.parquet?raw=true'
    df = pd.read_parquet(file_name)
    return df.rename(columns=stat_names)

plv_df = load_season_data()
season_df = (plv_df
             .groupby('hittername')
             [['pitch_id']+list(stat_names.values())]
             .agg({
                 'pitch_id':'count',
                 'Swing Agg':'mean',
                 'SZ Judge':'mean',
                 'Contact':'mean',
                 'Sw Dec':'mean',
                 'Adj Power':'mean',
                 'Value Added':'mean'
             })
             .query('pitch_id >= 400')
             .rename(columns={'hittername':'Name',
                              'pitch_id':'Pitches Seen'})
             .astype({'Pitches Seen':'string'})
             .sort_values('Value Added', ascending=False)
            )

for stat in list(stat_names.values()):
    season_df[stat] = round(z_score_scaler(season_df[stat])*2+10,0)*5
    season_df[stat] = np.clip(season_df[stat], a_min=20, a_max=80).astype('int')

st.write('Metrics on a 20-80 scale')
def make_pretty(styler):
    styler.background_gradient(axis=None, vmin=20, vmax=80, cmap="vlag")
    return styler
st.dataframe(season_df.style.pipe(make_pretty))

st.title("Rolling Ability Charts")
# Player
players = list(plv_df['hittername'].unique())
default_player = players.index('Juan Soto')
player = st.selectbox('Choose a player:', players, index=default_player)

# Metric
metrics = ['Swing Aggression','Strikezone Judgement','Decision Value',
           'Contact Ability','Adjusted Power']
default_stat = metrics.index('Decision Value')
metric = st.selectbox('Choose a metric:', metrics, index=default_stat)

rolling_denom = {
    'Strikezone Judgement':'Pitches',
    'Swing Aggression':'Pitches',
    'Decision Value':'Pitches',
    'Contact Ability':'Swings',
    'Adjusted Power': 'BBE'
}

rolling_threshold = {
    'Strikezone Judgement':400,
    'Swing Aggression':400,
    'Decision Value':400,
    'Contact Ability':200,
    'Adjusted Power': 75
}

window_max = plv_df.dropna(subset=metric).groupby('battername')['pitch_id'].count().max()

# Rolling Window
window = st.slider(f'Choose a {rolling_denom[metric]} threshold:', 50, window_max,
                   value=rolling_threshold[metric])

def rolling_chart()
  rolling_df = (plv_df
                .sort_values('pitch_id')
                .loc[(plv_df['batter_name']==player),
                     ['battername',metric]]
                .dropna()
                .reset_index(drop=True)
                .reset_index()
                .assign(Rolling_Stat=lambda x: x[metric].rolling(window).mean())
                )

  fig, ax = plt.subplots(figsize=(7,7))
  sns.lineplot(data=rolling_df,
               x='index',
               y='Rolling_Stat',
               color=line_color)

  ax.axhline(rolling_df[metric].mean(), color=line_color)
  ax.text(rolling_df.shape[0]*1.05,
          rolling_df[metric].mean(),
          'Szn Avg',
          va='center',
          color=sns.color_palette('vlag', n_colors=20)[3])

  ax.axhline(plv_df[metric].mean(),
             color='w',
             linestyle='--',
             alpha=0.5)
  ax.text(rolling_df.shape[0]*1.05,
          pitch_data[metric].mean(),
          'MLB Avg' if abs(rolling_df[metric].mean() - rolling_df[metric].mean()) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
          va='center',
          color='w')

  min_value = 0.25 if metric=='Strikezone Judgement' else 0

  ax.set(xlabel='Season '+rolling_denom[metric],
         ylabel=metric,
         ylim=(min(min_value,rolling_df['Rolling_Stat'].min(),plv_df[metric].mean()) + ax.get_ylim()[0]/20,
               max(0,rolling_df['Rolling_Stat'].max(),plv_df[metric].mean()) + ax.get_ylim()[1]/20),
         title="{}'s {} Rolling {} ({} {})".format(player,
                                                   year,
                                                   metric,
                                                   window,
                                                   rolling_denom[metric]))

  sns.despine()
  st.pyplot(fig)
rolling_chart()
