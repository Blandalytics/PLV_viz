import numpy as np
import seaborn as sns
import os
import pandas as pd
import pickle
import psycopg2
import xgboost as xgb #v1.6.0
import zipfile

from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from xgboost import XGBClassifier
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

def strikezone_z(dataframe,top_column,bottom_column):
    dataframe[['p_z',top_column,bottom_column]] = dataframe[['p_z',top_column,bottom_column]].astype('float')
    
    # Ratio of 'strikezones' above/below midpoint of strikezone
    dataframe['sz_mid'] = dataframe[[top_column,bottom_column]].mean(axis=1)
    dataframe['sz_height'] = dataframe[top_column].sub(dataframe[bottom_column])
    
    return dataframe['p_z'].sub(dataframe['sz_mid']).div(dataframe['sz_height'])

model_df = pd.DataFrame()
for year in [2020,2021,2022,2023]:
    for chunk in [1,2,3]:
        model_df = pd.concat([model_df,
                             pd.read_parquet(f'{year}_PLV_App_Data-{chunk}.parquet')])
        
model_df['sz_z'] = strikezone_z(model_df,'strike_zone_top','strike_zone_bottom')
model_df = model_df.reset_index(drop=True)

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

sz_bot = 1.5
sz_top = 3.5
x_ft = 3.5
x_adj = 2
adj_x = x_ft - x_adj
y_bot = -0.25
y_lim = 6
plate_y = 0

def generate_df(df,player,stat,year,game_date):
    df['pitchtype'] = df['pitchtype'].astype('str')
    test_df = df.loc[(df['year_played'] == year) &
                     (df['pitchername']== player)].reset_index(drop=True).copy()
    
    test_df['pitch_group'] = 0
    test_df.loc[test_df['game_played'].astype('str')==game_date,'pitch_group'] = 1
    
    for group in [0,1]:
        for pitch in test_df['pitchtype'].unique():
            test_df.loc[(test_df['pitchtype']==pitch) &
                   (test_df['pitch_group']==group),'z_value'] = test_df.loc[(test_df['pitchtype']==pitch) &
                                                                  (test_df['pitch_group']==group),stat].sub(test_df.loc[(test_df['pitchtype']==pitch) &
                                                                                                              (test_df['pitch_group']==group),stat].mean()).div(test_df.loc[(test_df['pitchtype']==pitch) &
                                                                                                              (test_df['pitch_group']==group),stat].std())
    return test_df.copy()
  
player = 'Ross Stripling'
game_date = model_df.loc[model_df['pitchername']==player,'game_played'].max().strftime('%Y-%m-%d')
game_text = (game_date[5:7] if game_date[5]!='0' else game_date[6])+'/'+(game_date[-2:] if game_date[-2]!='0' else game_date[-1])+'/'+game_date[2:4]
stat = 'velo'
year = 2023
chart_df = generate_df(model_df,
                       player,
                       stat,
                       year,
                       game_date)
chart_df = chart_df.loc[chart_df['pitchtype'].isin(pitch_names.keys())].copy()
pitch_list = [x[0] for x in Counter(chart_df['pitchtype']).most_common() if x[1] > int(chart_df.shape[0]*0.05)]

fig, ax = plt.subplots(figsize=(7,5))
sns.violinplot(data=chart_df.loc[(chart_df['pitchtype'].isin(pitch_list)) &
                                 (chart_df['z_value'].abs()<3)], 
               x=stat, 
               y='pitchtype', 
               hue='pitch_group', 
               inner=None,
               split=True)
ax.legend(labels=['Season',game_text],
          loc='upper left')
sns.despine()

for pitch in pitch_list:
    fig, axs = plt.subplots(1,2,figsize=(7,5))
    for hand in ['R','L']:
        ax_num = 0 if hand=='L' else 1
        x_lim = ((x_ft,-adj_x)) if hand =='L' else ((adj_x,-x_ft))
        cmap = mpl.colors.ListedColormap(list(sns.light_palette(marker_colors[pitch], n_colors=5)[1:]))
        if chart_df.loc[(chart_df['pitchtype']==pitch) &
                        (chart_df['b_hand']==hand)].shape[0]>5:
            sns.kdeplot(data=chart_df.loc[(chart_df['pitchtype']==pitch) &
                                          (chart_df['b_hand']==hand)],
                        x='p_x',
                        y='p_z',
                        levels=5,
                        thresh=0.5,
                        ax=axs[ax_num],
                        cmap=cmap,
                        alpha=0.5,
                        clip=[x_lim,[y_bot,y_lim]],
                        fill=True)
        sns.scatterplot(data=chart_df.loc[(chart_df['pitchtype']==pitch) &
                                          (chart_df['pitch_group']==1) &
                                          (chart_df['b_hand']==hand)],
                        x='p_x',
                        y='p_z',
                        ax=axs[ax_num],
                        color=marker_colors[pitch],
                       alpha=1)
        # Strike zone outline
        axs[ax_num].plot([-10/12,10/12], [sz_bot,sz_bot], color='w', linewidth=1)
        axs[ax_num].plot([-10/12,10/12], [sz_top,sz_top], color='w', linewidth=1)
        axs[ax_num].plot([-10/12,-10/12], [sz_bot,sz_top], color='w', linewidth=1)
        axs[ax_num].plot([10/12,10/12], [sz_bot,sz_top], color='w', linewidth=1)

        # Inner Strike zone
        axs[ax_num].plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color='w', linewidth=1)
        axs[ax_num].plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color='w', linewidth=1)
        axs[ax_num].axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='w', linewidth=1)
        axs[ax_num].axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='w', linewidth=1)

        # Plate
        axs[ax_num].plot([-8.5/12,8.5/12], [plate_y,plate_y], color='w', linewidth=1)
        axs[ax_num].axvline(8.5/12, ymin=(plate_y-y_bot)/(y_lim-y_bot), ymax=(plate_y+0.1-y_bot)/(y_lim-y_bot), color='w', linewidth=1)
        axs[ax_num].axvline(-8.5/12, ymin=(plate_y-y_bot)/(y_lim-y_bot), ymax=(plate_y+0.1-y_bot)/(y_lim-y_bot), color='w', linewidth=1)
        axs[ax_num].plot([8.28/12,0], [plate_y+0.1,plate_y+0.2], color='w', linewidth=1)
        axs[ax_num].plot([-8.28/12,0], [plate_y+0.1,plate_y+0.2], color='w', linewidth=1)
        
        # Batter
        hand_mul = 1 if hand=='L' else -1
        axs[ax_num].plot([20/12*hand_mul,38/12*hand_mul], [2.5+0.9*2,2.5+1.5*2], color='w', linewidth=2) # Bat
        axs[ax_num].plot([20/12*hand_mul,20/12*hand_mul], [2.5+0.9*2,2.5+0.6*2], color='w', linewidth=2) # Forearm
        axs[ax_num].plot([20/12*hand_mul,26/12*hand_mul], [2.5+0.6*2,2.5+1*2], color='w', linewidth=2) # Upper Arm
        axs[ax_num].plot([26/12*hand_mul,32/12*hand_mul], [2.5+1*2,2.5+0], color='w', linewidth=2) # Torso
        axs[ax_num].plot([32/12*hand_mul,26/12*hand_mul], [2.5+0,2.5-0.49*2], color='w', linewidth=2) # Thigh
        axs[ax_num].plot([26/12*hand_mul,30/12*hand_mul], [2.5-0.49*2,plate_y], color='w', linewidth=2) # Shin
        head = mpl.patches.Ellipse((25/12*hand_mul, 2.5+1.2*2),
                           width=0.6,
                           height=0.8, 
                           color='w') # Head
        axs[ax_num].add_patch(head)
        axs[ax_num].text(0,4.5,f'{hand}HH',ha='center')
        
#         plt.text(-2.25*hand_mul,plate_y,'Season is\nShaded',ha='center',size=8)

        axs[ax_num].set(xlim=x_lim,
                        ylim=(y_bot,y_lim))
        fig.suptitle(f"{player}'s\n{pitch_names[pitch]} Location",x=0.525,y=0.875)
        fig.text(0.51,0.72,f'Scatter = {game_text}',ha='center',size=8)
        fig.text(0.51,0.685,f'Shaded = {year}',ha='center',size=8)
        axs[ax_num].axis('off')
    sns.despine()
