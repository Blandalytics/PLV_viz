import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp

from collections import Counter
from scipy import stats

# ## Set Styling
# # Plot Style
# pl_white = '#FEFEFE'
# pl_background = '#162B50'
# pl_text = '#72a3f7'
# pl_line_color = '#293a6b'

# sns.set_theme(
#     style={
#         'axes.edgecolor': pl_background,
#         'axes.facecolor': pl_background,
#         'axes.labelcolor': pl_white,
#         'xtick.color': pl_white,
#         'ytick.color': pl_white,
#         'figure.facecolor':pl_background,
#         'grid.color': pl_background,
#         'grid.linestyle': '-',
#         'legend.facecolor':pl_background,
#         'text.color': pl_white
#      }
#     )

sz_bot = 1.5
sz_top = 3.5
x_ft = 2.5
y_bot = -0.5
y_lim = 6
plate_y = -.25

st.title("Pitchtype Cards")
st.write(
  '''File needs to contain the following columns, with these exact names:\n
  pitch_id, name, pitchtype, pitcher_hand, velo\n
  horizontal_location, vertical_location, horizontal_movement, vertical_movement\n
  spin_rate, spin_axis, extension, vaa\n
  '''
)

# Load Data
pitch_file = st.file_uploader("Load a pitch-level CSV file")
if pitch_file is not None:
    pitch_df =  pd.read_csv(pitch_file)

# Marker Style
marker_colors = dict(zip(list(pitch_data['pitch_name'].value_counts().index),list(sns.color_palette('tab20',n_colors=len(pitch_data['pitch_name'].unique())))))

# Movement values should be in inches
for axis in ['horizontal','vertical']:
  if pitch_df[axis+'_movement'].std() <5: # Standard deviation of movement in feet is generally < 5
    pitch_df[axis+'_movement'] = pitch_df[axis+'_movement'].mul(12)

# Has at least 1 pitch with at least 20 thrown
pitcher_list = list(pitch_df.groupby(['name','pitchtype'])['pitch_id'].count().reset_index().query('pitch_id >=20')['pitchername'].sort_values().unique())

col1, col2 = st.columns(2)

with col1:
    # Player
    card_player = st.selectbox('Choose a pitcher:', pitcher_list)

with col2:
    # Pitch
    pitches = list(pitch_df.loc[pitch_df['name']==card_player].groupby('pitchtype')['pitch_id'].count().reset_index().sort_values('pitch_id',ascending=False).query('pitch_id>=20')['pitchtype'])
    pitch_type = st.selectbox('Choose a pitch:', pitches)

pitch_type = {v: k for k, v in pitch_names.items()}[pitch_type]

def pitch_analysis_card(card_player,pitch_type):
    # Find number of this pitchtype thrown by this pitcher 
    pitches_thrown = int(pitch_df.loc[(pitch_df['name']==card_player) & (pitch_df['pitchtype']==pitch_type)].shape[0]/100)*100

    # Threshold at number of pitches thrown by pitcher, or 75th %ile of population, whichever is higher
    pitch_num_thresh = max(20,
                           min(pitches_thrown,
                               int(pitch_df.loc[(pitch_df['pitchtype']==pitch_type)].groupby('pitchername')['pitch_id'].count().nlargest(75)[-1]/50)*50
                              )
                          )

    # Generate df for card bottom stats
    pitch_stats_df = (
        pitch_df
        .loc[(pitch_df['pitchtype']==pitch_type)]
        .groupby(['name'])
        [['pitch_id','pitcher_hand','velo','extension','vertical_movement','horizontal_movement','vaa','spin_rate','spin_axis']]
        .agg({
            'pitch_id':'count',
            'pitcher_hand':pd.Series.mode,
            'velo':'mean',
            'extension':'mean',
            'vertical_movement':'mean',
            'horizontal_movement':'mean',
            'vaa':'mean',
            'spin_rate':'mean',
            'spin_axis':'mean'
        })
        .query(f'pitch_id>={pitch_num_thresh}')
        .reset_index()
        .sort_values('velo', ascending=False)
    )

    # Scale pitches for violin plots (will all be 0-100)
    def min_max_scaler(x):
        return ((x-x.min())/(x.max()-x.min()))

    for col in ['velo','velo','extension','vertical_movement','horizontal_movement','vaa','spin_rate','spin_axis']:
        pitch_stats_df[col+'_scale'] = min_max_scaler(pitch_stats_df[col])

    chart_stats = ['velo','extension','vertical_movement','horizontal_movement','vaa','spin_rate','spin_axis']
    fig = plt.figure(figsize=(10,10))

    # Dictionaries for names and top/bottom text of each chart
    stat_name_dict = {
        'velo':'Velocity',
        'extension':'Release\nExtension',
        'vertical_movement':'Vertical\nBreak',
        'horizontal_movement':'Arm-Side\nBreak',
        'vaa':'Vertical\nApproach\nAngle',
        'spin_rate':'Spin Rate',
        'spin_axis':'Spin Axis',
    }

    stat_tops = {
        'velo':'Faster',
        'extension':'Longer',
        'vertical_movement':'Rise',
        'horizontal_movement':'Arm',
        'vaa':'Flatter',
        'spin_rate':'Higher',
        'spin_axis':'Up/Down',
    }
    stat_bottoms = {
        'velo':'Slower',
        'extension':'Shorter',
        'vertical_movement':'Drop',
        'horizontal_movement':'Glove',
        'vaa':'Steeper',
        'spin_rate':'Lower',
        'spin_axis':'Left/Right',
    }

    # Divide card into tiles
    grid = plt.GridSpec(2, len(chart_stats),height_ratios=[5,5],hspace=0.2)
    ax = plt.subplot(grid[0, :3])
    sns.scatterplot(data=(pitch_df
                          .loc[(pitch_df['name']==card_player) &
                               (pitch_df['pitchtype']==pitch_type)]
                          .assign(horizontal_location = lambda x: x['horizontal_location']*-1)),
                    x='horizontal_location',
                    y='vertical_location',
                    color=marker_colors[pitch_type],
                    alpha=1)

    # Strike zone outline
    ax.plot([-10/12,10/12], [sz_bot,sz_bot], color='k', linewidth=2)
    ax.plot([-10/12,10/12], [sz_top,sz_top], color='k', linewidth=2)
    ax.plot([-10/12,-10/12], [sz_bot,sz_top], color='k', linewidth=2)
    ax.plot([10/12,10/12], [sz_bot,sz_top], color='k', linewidth=2)

    # Inner Strike zone
    ax.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color='k', linewidth=1)
    ax.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color='k', linewidth=1)
    ax.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='k', linewidth=1)
    ax.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='k', linewidth=1)

    # Plate
    ax.plot([-8.5/12,8.5/12], [plate_y,plate_y], color='k', linewidth=2)
    ax.axvline(8.5/12, ymin=(plate_y-y_bot)/(y_lim-y_bot), ymax=(plate_y+0.15-y_bot)/(y_lim-y_bot), color='k', linewidth=2)
    ax.axvline(-8.5/12, ymin=(plate_y-y_bot)/(y_lim-y_bot), ymax=(plate_y+0.15-y_bot)/(y_lim-y_bot), color='k', linewidth=2)
    ax.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color='k', linewidth=2)
    ax.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color='k', linewidth=2)

    ax.set(xlim=(-x_ft,x_ft),
           ylim=(y_bot,y_lim),
           aspect=1)
    fig.text(0.23,0.89,'Locations',fontsize=18,bbox=dict(facecolor='w', alpha=0.75, edgecolor='w'))
    ax.axis('off')
    sns.despine()

    hand = pitch_df.loc[(pitch_df['name']==card_player),'pitcher_hand'].values[0]
    ax = plt.subplot(grid[0, 3:])
    sns.scatterplot(data=pitch_df.loc[(pitch_df['name']==card_player) &
                                      (pitch_df['pitchtype']==pitch_type)],
                    x='horizontal_movement',
                    y='vertical_movement',
                    color=marker_colors[pitch_type],
                    s=25,
                    alpha=1)

    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set(aspect=1)

    sns.scatterplot(data=(pitch_df
                          .loc[(pitch_df['name']==card_player) &
                               (pitch_df['pitchtype']==pitch_type)]
                          .groupby('pitchtype')
                          [['vertical_movement','horizontal_movement']]
                          .mean()
                          .reset_index()
                         ),
                    x='horizontal_movement',
                    y='vertical_movement',
                    color=marker_colors[pitch_type],
                    s=200,
                    legend=False,
                    linewidth=2
                   )

    ax_lim = max(25,
                 pitch_df.loc[(pitch_df['name']==card_player) &
                              (pitch_df['pitchtype']==pitch_type),
                              ['horizontal_movement','vertical_movement']].abs().quantile(0.999).max()+1
                )
    ax.set(xlim=(ax_lim,-ax_lim),
           ylim=(-ax_lim,ax_lim))
    plt.xlabel('Arm-Side Break', fontsize=12)
    plt.ylabel('Vertical Break', fontsize=12,labelpad=-1)
    ax.set_xticks([x*10 for x in range(-int(ax_lim/10),int(ax_lim/10)+1)][::-1])
    if hand=='R':
        ax.set_xticklabels([x*-1 for x in ax.get_xticks()])
    fig.text(0.62,0.89,'Movement',fontsize=18)
    sns.despine(left=True,bottom=True)

    fig.text(0.5,0.45,'Pitch Characteristics',ha='center',fontsize=18)
    fig.text(0.5,0.43,f'(Compared to league {pitch_type}s - Min {pitch_num_thresh} Thrown)',ha='center',fontsize=12)
    for stat in chart_stats:
        val = pitch_stats_df.loc[(pitch_stats_df['name']==card_player),
                                 stat].item()
        up_thresh = max(pitch_stats_df[stat].quantile(0.99),
                        val)
        low_thresh = min(pitch_stats_df[stat].quantile(0.01),
                         val)
        ax = plt.subplot(grid[1, chart_stats.index(stat)])
        sns.violinplot(data=pitch_stats_df.loc[(pitch_stats_df[stat] <= up_thresh) &
                                               (pitch_stats_df[stat] >= low_thresh)],
                       y=stat+'_scale',
                       inner=None,
                       orient='v',
                       cut=0,
                       color=marker_colors[pitch_type],
                       linewidth=1
                     )
        ax.collections[0].set_edgecolor('k')

        top = ax.get_ylim()[1]
        bot = ax.get_ylim()[0]
        plot_height = top - bot

        format_dict = {
            'velo':f'{val:.1f}mph',
            'extension':f'{val:.1f}ft',
            'vertical_movement':f'{val:.1f}"',
            'horizontal_movement':f'{val:.1f}"',
            'vaa':f'{val:.1f}°',
            'spin_rate':f'{val:.0f}rpm',
            'spin_axis':f'{val:.1f}°',
        }
        ax.axhline(pitch_stats_df[stat+'_scale'].median(),
                   linestyle='--',
                   color='k')
        ax.axhline(top + (0.25 * plot_height),
                   xmin=0.1,
                   xmax=0.9,
                   color='k')
        ax.text(0,
                pitch_stats_df.loc[(pitch_stats_df['pitchername']==card_player),
                                   stat+'_scale'],
                format_dict[stat],
                va='center',
                ha='center',
                fontsize=12 if stat=='velo' else 14,
                bbox=dict(facecolor='w', alpha=0.75, edgecolor='w'))
        ax.text(0,
                top + (0.5 * plot_height),
                stat_name_dict[stat],
                va='center',
                ha='center',
                fontsize=14)
        ax.text(0,
                top + (0.2 * plot_height),
                stat_tops[stat],
                va='top',
                ha='center',
                fontsize=12)
        ax.text(0,
                bot - (0.2 * plot_height),
                stat_bottoms[stat],
                va='bottom',
                ha='center',
                fontsize=12)
        ax.tick_params(left=False, bottom=False)
        ax.set_yticklabels([])
        ax.set(xlabel=None,ylabel=None,ylim=(bot - (0.15 * plot_height),
                                             top + plot_height))
        ax.xaxis.set_label_position('top')

    # Add PL logo
    pl_ax = fig.add_axes([0.41,0.025,0.2,0.2], anchor='S', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')

    apostrophe_text = "'" if card_player[-1]=='s' else "'s"
    fig.suptitle(f"{card_player}{apostrophe_text} {year} {pitch_type}",y=0.97,fontsize=20,x=0.525)
    fig.text(0.525,0.925,"(From Pitcher's Perspective)",ha='center',fontsize=12)
    sns.despine(left=True,bottom=True)
    st.pyplot(fig)
pitch_analysis_card(card_player,pitch_type)
