import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import seaborn as sns

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
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

# Marker Style
marker_colors = {
    'FF':'#d22d49', 
    'SI':'#c57a02',
    'FS':'#00a1c5',  
    'FC':'#933f2c', 
    'SL':'#9300c7', 
    'CU':'#3c44cd',
    'CH':'#07b526', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
}

marker_list = {
    'FF':'^', 
    'SI':'v', 
    'FC':'D', 
    'FS':'o', 
    'SL':'s',
    'CH':'X', 
    'CU':'P',
    'SC':'*', 
    'KN':'*', 
    'UN':'*'
}

# Pitch Names
pitch_names = {
    'FF':'Four-Seamer', 
    'SI':'Sinker',
    'FS':'Splitter',  
    'FC':'Cutter', 
    'SL':'Slider', 
    'CU':'Curveball',
    'CH':'Changeup', 
    'KN':'Knuckleball',
    'SC':'Screwball', 
    'UN':'Unknown', 
}

# General constants for formatting charts and sizing
y_lim = 5.5
y_bot = -1
sz_bot = 1.5
sz_top = 3.5

# Year
years = [2022,2021,2020]
year = st.radio('Choose a year:', years)

# Load Data
def load_data():
    file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_App_Data.parquet?raw=true'
    df = pd.read_parquet(file_name).sort_values('pitch_id')
    return df
plv_df = load_data()

# Player
players = list(plv_df['pitchername'].unique())
default_ix = players.index('Sandy Alcantara')
player = st.selectbox('Choose a player:', players, index=default_ix)

st.write(plv_df.loc[plv_df['pitchername']==player,'p_x'].mean())

pitch_list = list(plv_df
                .loc[(plv_df['pitchername']==player)# &
                     #plv_df['b_hand'].isin(hand_map[handedness])
                    ]
                .groupby('pitchtype',as_index=False)
                ['pitch_id']
                .count()
                .dropna()
                .sort_values('pitch_id', ascending=False)
                .query(f'pitch_id >= {50}')
                ['pitchtype']
                )

def plv_card(pitch_threshold=200,scale_val=1.5):
  # Create df for only the pitcher's pitches
  graph_data = plv_df.loc[plv_df['pitchername']==player].iloc[::-1].reset_index(drop=True)
  graph_data['p_x'] = graph_data['p_x'].mul(-1)
  chart_data['PLV_clip'] = np.clip(chart_data['PLV'], a_min=0, a_max=10)

  # Update the pitch count threshold if the pitcher has a low season pitch count
  pitch_threshold = min(pitch_threshold,graph_data.shape[0])

  # Strikeouts and Walks need their own, as they're conditional
  # Card size
  fig = plt.figure(figsize=(7.5*scale_val,10.5*scale_val))

  # Parameters to divide card
  grid_height = 10
  pitch_feats = 8

  # Divide card into tiles
  grid = plt.GridSpec(grid_height, 7, wspace=0.1*scale_val, hspace=0.5*scale_val, width_ratios=[2.5,0.6,4.4,1.5,0.5,0.25,0.25],
                      height_ratios=[1.5]+[7/pitch_feats]*(pitch_feats)+[1])

  # Title of card (name, etc)
  title_ax = plt.subplot(grid[0, :3])
  title_ax.text(0,0,"{}'s {}\nPLV Card\nAvg PLV: {}".format(player,year,round(graph_data['PLV'].mean(),2)), ha='center', va='center', fontsize=round(16*scale_val),
           bbox=dict(facecolor='#162B50', alpha=0.6, edgecolor='#162B50'))
  title_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
  title_ax.set_xticklabels([])
  title_ax.set_yticklabels([])
  title_ax.tick_params(left=False, bottom=False)

  # Arsenal Distributions
  pitch_dist_ax = plt.subplot(grid[1:, 0])

  # Per game/appearance chart
  game_ax = plt.subplot(grid[1:4, 2:6])
  game_ax.grid(visible=True, which='major', axis='y', color='#FEFEFE', alpha=0.1)

  graph_data['game_played'] = pd.to_datetime(graph_data['game_played'])
  graph_data['appearance'] = graph_data['game_played'].rank(method='dense')

  # Subtle line to connect the dots
  sns.lineplot(
      data=graph_data.groupby(['game_played','pitchername'],as_index=False)[['PLV','appearance']].agg({
          'PLV':'mean',
          'appearance': 'mean'
          }), 
      x='game_played', 
      y='PLV',
      style='pitchername',
      color='#FEFEFE',
      linewidth=round(scale_val),
      alpha=0.1,
      ax=game_ax,
      legend=False)

  # Dots
  sns.scatterplot(
      data=graph_data.groupby('game_played',as_index=False)[['PLV','appearance']].agg({
          'PLV':'mean',
          'appearance': 'mean'
          }), 
      x='game_played', 
      y='PLV', 
      s=round(100*scale_val), 
      edgecolor=None, 
      hue='PLV', 
      hue_norm=game_norm, 
      palette='vlag', 
      alpha=1,
      ax=game_ax,
      legend=False)

  # League Average line
  game_ax.axhline(5, color='#FEFEFE', linewidth=round(scale_val), linestyle='--', alpha=0.75)
  
  game_ax.set(xlabel=None, ylabel=None, ylim=(0,10))
  x_ticks_format(game_ax,graph_data['game_played'],scale_val)
  game_ax.set_title('Avg PLV, per Game', fontsize=round(12*scale_val))
  
  # Plot of individual pitches
  pitch_plot_ax = plt.subplot(grid[4:, 1:5])
  sns.scatterplot(data=graph_data.loc[(graph_data['p_z']<=y_lim-0.25)&
                                      (graph_data['p_x']>-2.8)], 
                  x='p_x', 
                  y='p_z', 
                  s=round(70*scale_val), 
                  style='pitchtype',
                  hue='PLV_clip',
                  palette='vlag',
                  hue_norm=norm,
                  markers=marker_list,
                  edgecolor='#293a6b',
                  ax=pitch_plot_ax,
                  legend=False
                  )

  # Strike zone outline
  pitch_plot_ax.axvline(10/12, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='black', linewidth=4*scale_val)
  pitch_plot_ax.axvline(-10/12, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='black', linewidth=4*scale_val)
  pitch_plot_ax.axhline(sz_top, xmin=26/72, xmax=46/72, color='black', linewidth=4*scale_val)
  pitch_plot_ax.axhline(sz_bot, xmin=26/72, xmax=46/72, color='black', linewidth=4*scale_val)

  pitch_plot_ax.set(xlabel=None, xlim=(-3,3), ylabel=None, ylim=(y_bot,y_lim))
  pitch_plot_ax.set_xticklabels([])
  pitch_plot_ax.set_yticklabels([])
  pitch_plot_ax.tick_params(left=False, bottom=False)
  pitch_plot_ax.text(0.75,y_lim-0.6,"PLV per Pitch", ha='center', va='bottom', fontsize=round(12*scale_val), 
           bbox=dict(facecolor='#162B50', alpha=0.75, edgecolor='#162B50'))
  pitch_plot_ax.text(0.75,y_lim-0.7,"(From Pitcher's Perspective)", ha='center', va='top', fontsize=round(10*scale_val), alpha=0.7,
           bbox=dict(facecolor='#162B50', alpha=0.75, edgecolor='#162B50'))

  # Add custom legend for markers
  legend_markers = [Line2D([],[],
                           color='#FEFEFE',
                           label='\n'.join(wrap(x, 10)),
                           marker=marker_list[x],
                           markeredgecolor=pl_line_color,
                           markeredgewidth=round(scale_val),
                           markersize=round(10*scale_val),
                           linestyle='None') 
                    for x in graph_data.sort_values('avg_velo',ascending=False)['pitch_name'].unique()]

  pitch_plot_ax.legend(loc=(0.01,0.01),
             handles=legend_markers,
             edgecolor='#162B50',
             framealpha=0.5, fontsize=round(12*scale_val)
             )

  # Colorbar for pitch plot
  cb_ax = plt.subplot(grid[5:9, 5])
  sm = plt.cm.ScalarMappable(cmap='vlag', norm=norm)
  sm.set_array([])
  fig.colorbar(sm,
               cax=cb_ax
               )
  cb_ax.tick_params(labelsize=round(10*scale_val))
  
  # Chart ownership (PitcherList)
  owner_ax = plt.subplot(grid[0, 3:])
  owner_ax.text(0.5,0,'Created by Kyle Bland\n@blandalytics', ha='center', va='center',fontweight='normal', fontsize=round(12*scale_val),
            bbox=dict(facecolor='#162B50', alpha=0.6, edgecolor='#162B50'))
  owner_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
  owner_ax.set_xticklabels([])
  owner_ax.set_yticklabels([])
  owner_ax.tick_params(left=False, bottom=False)

  # Credit for the inspiration
  credit_ax = plt.subplot(grid[9:, 5:])
  credit_ax.text(-0.1,0.5,'Viz by\n@Blandalytics', ha='center', va='center', fontsize=round(10*scale_val),
           bbox=dict(facecolor='#162B50', alpha=0.6, edgecolor='#162B50'))
  credit_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
  credit_ax.set_xticklabels([])
  credit_ax.set_yticklabels([])
  credit_ax.tick_params(left=False, bottom=False)

  # Box the Pitchtype Charts
  fig.add_artist(Line2D([0.125, 0.302], [0.8, 0.8], linewidth=round(2*scale_val)))
  fig.add_artist(Line2D([0.125, 0.125], [0.125, 0.8], linewidth=round(2*scale_val)))
  fig.add_artist(Line2D([0.302, 0.302], [0.125, 0.8], linewidth=round(2*scale_val)))
  fig.add_artist(Line2D([0.125, 0.302], [0.125, 0.125], linewidth=round(2*scale_val)))

  #Box the games
  fig.add_artist(Line2D([0.32, 0.94], [0.8, 0.8], linewidth=round(2*scale_val))) # Top
  fig.add_artist(Line2D([0.32, 0.32], [0.546, 0.8], linewidth=round(2*scale_val))) # Left
  fig.add_artist(Line2D([0.94, 0.94], [0.546, 0.8], linewidth=round(2*scale_val))) # Right
  fig.add_artist(Line2D([0.32, 0.94], [0.546, 0.546], linewidth=round(2*scale_val))) # Bottom

  #Box the pitch chart
  fig.add_artist(Line2D([0.32, 0.94], [0.536, 0.536], linewidth=round(2*scale_val)))
  fig.add_artist(Line2D([0.32, 0.32], [0.125, 0.536], linewidth=round(2*scale_val)))
  fig.add_artist(Line2D([0.94, 0.94], [0.125, 0.536], linewidth=round(2*scale_val)))
  fig.add_artist(Line2D([0.32, 0.94], [0.125, 0.125], linewidth=round(2*scale_val)))

  sns.despine(left=True, bottom=True)
  st.pyplot(fig)
plv_card()
