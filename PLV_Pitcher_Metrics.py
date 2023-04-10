import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import urllib

from PIL import Image
from scipy import stats

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

cb_colors = {
    'FF':'#920000', 
    'SI':'#ffdf4d',
    'FS':'#006ddb',  
    'FC':'#ff6db6', 
    'SL':'#b66dff', 
    'CU':'#009999',
    'CH':'#22cf22', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
}

diverging_palette = 'vlag'

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

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

# Year
years = [2023,2022,2021,2020]
year = st.radio('Choose a year:', years)

seasonal_constants = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/plv_seasonal_constants.csv?raw=true').set_index('year')

# Load Data
@st.cache_data
def load_data(year):
    df = pd.DataFrame()
    for chunk in [1,2,3]:
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_App_Data-{chunk}.parquet?raw=true'
        df = pd.concat([df,
                        pd.read_parquet(file_name)[['pitchername','pitcher_mlb_id','pitch_id',
                                                    'p_hand','b_hand','pitchtype','PLV',
                                                    'IHB','IVB'
                                                   ]]
                       ])
    df = (df
          .sort_values('pitch_id')
          .astype({'pitch_id':'int',
                   'pitcher_mlb_id':'int'})
          .query(f'pitchtype not in {["KN","SC"]}')
          .reset_index(drop=True)
         )
    
    df['pitch_runs'] = df['PLV'].mul(seasonal_constants.loc[year]['run_plv_coef']).add(seasonal_constants.loc[year]['run_plv_constant'])
    
    df['pitch_quality'] = 'Average'
    df.loc[df['PLV']>=5.5,'pitch_quality'] = 'Quality'
    df.loc[df['PLV']<4.5,'pitch_quality'] = 'Bad'

    for qual in df['pitch_quality'].unique():
      df[qual+' Pitch'] = 0
      df.loc[df['pitch_quality']==qual,qual+' Pitch'] = 1

    df['QP-BP'] = df['Quality Pitch'].sub(df['Bad Pitch'])
    
    return df
plv_df = load_data(year)

def get_ids():
    id_df = pd.DataFrame()
    for chunk in list(range(0,10))+['a','b','c','d','e','f']:
        chunk_df = pd.read_csv(f'https://github.com/chadwickbureau/register/blob/master/data/people-{chunk}.csv?raw=true')
        id_df = pd.concat([id_df,chunk_df])
    return id_df[['key_mlbam','key_fangraphs']].dropna().astype('int') 

st.title("Season PLA")
st.write('- ***Pitch Level Average (PLA)***: Value of all pitches (ERA scale), using IP and the total predicted run value of pitches thrown.')
st.write('- ***Pitchtype PLA***: Value of a given pitch type (ERA scale), using total predicted run values and an IP proxy for that pitch type (pitch usage % * Total IP).')

# pitch_threshold = 400

# Num Pitches threshold
pitch_threshold = st.number_input(f'Min # of Pitches:',
                              min_value=100 if year==2023 else 200, 
                              max_value=2000,
                              step=50, 
                              value=100 if year==2023 else 500)

def get_pla(year,pitch_threshold=pitch_threshold,p_hand=['L','R'],b_hand=['L','R']):
    pla_data = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/pla_data.csv?raw=true', encoding='latin1')
    season_df = (pla_data
             .loc[(pla_data['year_played']==year) &
                  pla_data['p_hand'].isin(p_hand) &
                  pla_data['b_hand'].isin(b_hand)]
             .assign(total_plv = lambda x: x['num_pitches'] * x['plv'])
      .groupby(['pitchername','pitchtype','pitcher_mlb_id'])
      [['num_pitches','pitch_runs','total_plv','subset_ip']]
      .agg({
          'num_pitches':'sum',
          'subset_ip':'sum',
          'pitch_runs':'sum',
          'total_plv':'sum'
      })
      .sort_values('pitch_runs', ascending=False)
      .query(f'num_pitches >={int(pitch_threshold/20)}') # 5% of total pitches threshold
      .reset_index()
      )

    # Clean IP to actual fractions
    season_df['season_IP'] = season_df['subset_ip'].groupby(season_df['pitcher_mlb_id']).transform('sum')
    season_df['season_pitches'] = season_df['num_pitches'].groupby(season_df['pitcher_mlb_id']).transform('sum')

    # Calculate PLV, in general, and per-pitchtype
    season_df['PLV'] = season_df['total_plv'].groupby(season_df['pitcher_mlb_id']).transform('sum').div(season_df['season_pitches']).astype('float')
    season_df['pitchtype_plv'] = season_df['total_plv'].div(season_df['num_pitches'])

    # Calculate PLA, in general, and per-pitchtype
    season_df['PLA'] = season_df['pitch_runs'].groupby(season_df['pitcher_mlb_id']).transform('sum').mul(9).div(season_df['season_IP']).astype('float')
    season_df['pitchtype_pla'] = season_df['pitch_runs'].mul(9).div(season_df['subset_ip']) # ERA Scale

    season_df = season_df.sort_values('PLA')

    # Pivot a dataframe of per-pitchtype PLAs
    pitchtype_df = season_df.pivot_table(index=['pitcher_mlb_id'], 
                                          columns='pitchtype', 
                                          values='pitchtype_pla',
                                          aggfunc='sum'
                                        ).replace({0:None})

    # Merge season-long PLA with pitchtype PLAs
    df = (season_df
          .drop_duplicates('pitcher_mlb_id')
          [['pitcher_mlb_id','pitchername','season_pitches','PLA','PLV']]
          .merge(pitchtype_df, how='inner',left_on='pitcher_mlb_id',right_index=True)
          .query(f'season_pitches >= {pitch_threshold}')
          .rename(columns={'pitchername':'Pitcher',
                           'season_pitches':'Num_Pitches'})
          .drop(columns=['pitcher_mlb_id'])
          .fillna(np.nan)
          .set_index('Pitcher')
          [['Num_Pitches','PLV','PLA','FF','SI','SL','CH','CU','FC','FS']]
          .copy()
          )
    return df

# Season data
pla_df = get_pla(year,pitch_threshold)

def get_movement(year,player):
    move_data = pd.read_csv(f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_pitch_movement.csv?raw=true', encoding='latin1')
    return move_data.loc[(move_data['pitchername']==player) &
                         (move_data['pitchtype']!='UN')].copy()

mean_plv = pla_df['PLV'].mul(pla_df['Num_Pitches']).sum() / pla_df['Num_Pitches'].sum()

format_cols = ['PLA','FF','SI','SL','CH','CU','FC','FS']

fill_val = pla_df[format_cols].max().max()+0.01

def pitchtype_color(s):
    return f"background-color: {marker_colors[s]}" if s in list(marker_colors.keys()) else None

st.write('At least 20 pitches thrown, per pitch type. Table is sortable.')
if year == 2023:
    st.write('Note: PLV and PLA begin to stabilize at ~500 pitches.')    
st.dataframe(pla_df
             .astype({'Num_Pitches': 'int'})
             .fillna(fill_val)
             .style
             .format(precision=2, thousands=',')
             .background_gradient(axis=0, vmin=2, vmax=6,
                                  cmap=f"{diverging_palette}_r", subset=format_cols)
             .applymap(lambda x: 'color: transparent; background-color: transparent' if x==fill_val else '')
             #.applymap_index(pitchtype_color, axis='columns') # Apparently Streamlit doesn't style headers
            )


st.title("Pitcher Charts")

palettes = ['Pitcher List','Color Blind-Friendly']
palette = st.radio('Choose a palette:', 
                 palettes,
                 horizontal=True)

color_palette = cb_colors if palette=='Color Blind-Friendly' else marker_colors

## Selectors
# Player
players = list(pla_df
               .reset_index()
               .sort_values('PLV', ascending=False)
               ['Pitcher']
              )
default_ix = players.index('Sandy Alcantara')
player = st.selectbox('Choose a player:', players, index=default_ix)

# Chart Select
charts = ['Pitch Quality','Pitch Distribution',
          'Pitch Movement'
         ]
chart = st.radio('Choose a chart type:', 
                 charts,
                 horizontal=True)

if chart=='Pitch Distribution':
    if year==2023:
        st.write('2023 is under construction. Sorry!')
        exit()
    # Hitter Handedness
    handedness = st.select_slider(
        'Hitter Handedness',
        options=['Left', 'All', 'Right'],
        value='All')
    # Pitcher Handedness
    if handedness=='All':
        pitcher_hand = ['L','R']
    else:
        pitcher_hand = list(plv_df.loc[(plv_df['pitchername']==player),'p_hand'].unique())

    hand_map = {
        'Left':['L'],
        'All':['L','R'],
        'Right':['R']
    }

    pitches_thrown = plv_df.loc[(plv_df['pitchername']==player) &
                                plv_df['b_hand'].isin(hand_map[handedness])].shape[0]

    st.write('Distribution of PLV for all pitches thrown by {}{} in {}'.format(player,
                                                                               '' if handedness=='All' else f' to {handedness} Handed Hitters',
                                                                               year))
    st.write('Pitches Thrown: {:,}'.format(pitches_thrown))

    if pitches_thrown >= pitch_threshold:
        pitch_type_thresh = 20
        pitch_list = list(plv_df
                    .loc[(plv_df['pitchername']==player) &
                         plv_df['b_hand'].isin(hand_map[handedness])]
                    .groupby('pitchtype',as_index=False)
                    ['pitch_id']
                    .count()
                    .dropna()
                    .sort_values('pitch_id', ascending=False)
                    .query(f'pitch_id > {pitch_type_thresh}')
                    ['pitchtype']
                    )

    ## Chart function
        def arsenal_dist():
            # Subplots based off of # of pitchtypes
            fig, axs = plt.subplots(len(pitch_list),1,figsize=(8,8), sharex='row', sharey='row', constrained_layout=True)
            ax_num = 0
            max_count = 0
            for pitch in pitch_list:
                # Data just for that pitch type
                chart_data = plv_df.loc[(plv_df['pitchtype']==pitch) &
                                        plv_df['b_hand'].isin(hand_map[handedness])].copy()
                # Restrict to 0-10
                chart_data['PLV_clip'] = np.clip(chart_data['PLV'], a_min=0, a_max=10)
                num_pitches = chart_data.loc[chart_data['pitchername']==player].shape[0]

                # Plotting
                sns.histplot(data=chart_data.loc[chart_data['pitchername']==player],
                            x='PLV_clip',
                            hue='pitchtype',
                            palette=color_palette,
                            binwidth=0.5,
                            binrange=(0,10),
                            alpha=1,
                            ax=axs[ax_num],
                            legend=False
                            )
                # Season Avg Line
                axs[ax_num].axvline(chart_data.loc[chart_data['pitchername']==player,'PLV'].mean(),
                                    color=color_palette[pitch],
                                    linestyle='--',
                                    linewidth=2.5)

                # League Avg Line
                axs[ax_num].axvline(chart_data.loc[chart_data['p_hand'].isin(pitcher_hand),'PLV'].mean(), 
                                    color='w', 
                                    label='Lg. Avg.',
                                    alpha=0.5)

                # Format Axes Style
                axs[ax_num].get_xaxis().set_visible(False)
                axs[ax_num].get_yaxis().set_visible(False)
                axs[ax_num].set(xlim=(0,10))
                axs[ax_num].set_title('')
                if axs[ax_num].get_ylim()[1] > max_count:
                    max_count = axs[ax_num].get_ylim()[1]
                ax_num += 1
                if ax_num==len(pitch_list):
                    axs[ax_num-1].get_xaxis().set_visible(True)
                    axs[ax_num-1].set_xticks(range(0,11))
                    axs[ax_num-1].set(xlabel='')

            # Chart Styling & Add-Ons
            for axis in range(len(pitch_list)):
                # Fix Y-Axis size to most thrown pitch, for all pitches
                axs[axis].set(ylim=(0,max_count*1.025))

                num_pitches = plv_df.loc[(plv_df['pitchtype']==pitch_list[axis]) & 
                                         (plv_df['pitchername']==player) &
                                         plv_df['b_hand'].isin(hand_map[handedness])].shape[0]
                pitch_usage = round(num_pitches / plv_df.loc[(plv_df['pitchername']==player) &
                                                             plv_df['b_hand'].isin(hand_map[handedness])].shape[0] * 100,1)

                # Define the plot legend
                axs[axis].legend([pitch_names[pitch_list[axis]]+': {:.3}'.format(plv_df.loc[(plv_df['pitchtype']==pitch_list[axis]) & 
                                                                                            (plv_df['pitchername']==player) &
                                                                                            plv_df['b_hand'].isin(hand_map[handedness]),'PLV'].mean()),
                                  'Lg. Avg'+': {:.3}'.format(plv_df.loc[(plv_df['pitchtype']==pitch_list[axis]) &
                                                                         plv_df['b_hand'].isin(hand_map[handedness]) &
                                                                         plv_df['p_hand'].isin(pitcher_hand),'PLV'].mean())], 
                                 framealpha=0, edgecolor=pl_background, loc=(0,0.4), fontsize=14)

                # Pitch Totals
                axs[axis].text(9,max_count*0.425,'{:,} Pitches\n({}%)'.format(num_pitches,
                                                                              pitch_usage),
                               ha='center',va='bottom', fontsize=14)

            # Filler for Title
            hand_text = f'{pitcher_hand[0]}HP vs {hand_map[handedness][0]}HB, ' if handedness!='All' else ''

            fig.suptitle("{}'s {} PLV Distributions\n({}>=20 Pitches Thrown)".format(player,year,hand_text),x=0.4,fontsize=16)
            # Add PL logo
            pl_ax = fig.add_axes([0.75,0.8,0.2,0.2], anchor='NE', zorder=1)
            pl_ax.imshow(logo)
            pl_ax.axis('off')
            
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)
        arsenal_dist()
    else:
        st.write('Not enough pitches thrown in {} (<{})'.format(year,pitch_threshold))
elif chart=='Pitch Quality':
    # Hitter Handedness
    handedness = st.select_slider(
        'Hitter Handedness',
        options=['Left', 'All', 'Right'],
        value='All')
    # Pitcher Handedness
    if handedness=='All':
        pitcher_hand = ['L','R']
    else:
        pitcher_hand = list(plv_df.loc[(plv_df['pitchername']==player),'p_hand'].unique())

    hand_map = {
        'Left':['L'],
        'All':['L','R'],
        'Right':['R']
    }
    
    pla_data = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/pla_data.csv?raw=true', encoding='latin1')
    pq_df = (pla_data
             .loc[(pla_data['year_played']==year) &
                  pla_data['p_hand'].isin(pitcher_hand) &
                  pla_data['b_hand'].isin(hand_map[handedness])]
             .assign(total_plv = lambda x: x['num_pitches'] * x['plv'])
      .groupby(['pitchername','pitchtype','pitcher_mlb_id'])
      [['num_pitches','pitch_runs','total_plv','subset_ip']]
      .agg({
          'num_pitches':'sum',
          'subset_ip':'sum',
          'pitch_runs':'sum',
          'total_plv':'sum'
      })
      .sort_values('pitch_runs', ascending=False)
#       .query(f'num_pitches >={1}')
      .reset_index()
      )

    # Clean IP to actual fractions
    pq_df['season_IP'] = pq_df['subset_ip'].groupby(pq_df['pitcher_mlb_id']).transform('sum')
    pq_df['season_pitches'] = pq_df['num_pitches'].groupby(pq_df['pitcher_mlb_id']).transform('sum')

    # Calculate PLV, in general, and per-pitchtype
    pq_df['PLV'] = pq_df['total_plv'].groupby(pq_df['pitcher_mlb_id']).transform('sum').div(pq_df['season_pitches']).astype('float')
    pq_df['pitchtype_plv'] = pq_df['total_plv'].div(pq_df['num_pitches'])

    # Calculate PLA, in general, and per-pitchtype
    pq_df['PLA'] = pq_df['pitch_runs'].groupby(pq_df['pitcher_mlb_id']).transform('sum').mul(9).div(pq_df['season_IP']).astype('float')
    pq_df['pitchtype_pla'] = pq_df['pitch_runs'].mul(9).div(pq_df['subset_ip']) # ERA Scale)get_pla(year,pitch_threshold=25,p_hand=pitcher_hand,b_hand=hand_map[handedness]).reset_index().rename(columns={'Pitcher':'pitchername'})
    
    def plv_kde(df,name,num_pitches,ax,stat='PLV',pitchtype=''):
        pitch_color = 'w' if pitchtype=='' else marker_colors[pitchtype]
        df = df.query(f'season_pitches >= {pitch_threshold}').copy() if pitchtype=='' else df.loc[df['pitchtype']==pitchtype].query(f'num_pitches >= {int(pitch_threshold/20)}').copy()
        stat = 'PLV' if pitchtype=='' else 'pitchtype_plv'
        
        val = df.loc[df['pitchername']==name,stat].mean()
        val_percentile = np.clip(stats.percentileofscore(df[stat], val) / 100,0,1)

        sns.kdeplot(df[stat], ax=ax, color='w', legend=False, cut=0)

        x = ax.lines[-1].get_xdata()
        y = ax.lines[-1].get_ydata()

        quantiles = [1, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0]
        quant_colors = sns.color_palette(f'{diverging_palette}_r',n_colors=7001)[::1000]
        
        i = -1
        for quant in quantiles:
            if quant >= val_percentile:
                i += 1

        val_color = quant_colors[np.clip(i,0,7)]

        for quant in range(8):
            color = quant_colors[quant]
            thresh = 10 if quant==0 else df[stat].quantile(quantiles[quant])
            ax.fill_between(x, 0, y, 
                            where=x < thresh, 
                            color=quant_colors[quant], 
                            alpha=1)
        ax.vlines(df[stat].quantile(0.5), 
                0, 
                np.interp(df[stat].quantile(0.5), x, y), 
                linestyle='-', color='w', alpha=1, linewidth=2)
        ax.axvline(val, 
                 ymax=0.9,
                 linestyle='--', 
                 color='w', 
                 linewidth=2)
        props = dict(boxstyle='Round',
                   facecolor='k', 
                   alpha=1, 
                   edgecolor=val_color,
                   linewidth=2)
        y_max = ax.get_ylim()[1]
        ax.text(val+0.01,
              y_max*1.1,
              '{:.2f}'.format(val),
              ha='center',
              va='top',
              color=val_color,
              fontsize=16,
              fontweight='bold', 
              bbox=props)
        ax.set(xlim=(mean_plv-1.4,mean_plv+1.4),
             ylim=(0,y_max*1.2),
             xlabel=None,
             ylabel=None,
             )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False
                     )
        sns.despine(left=True,bottom=True)

    def percent_bar(ax):
        quantiles = [1, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0]
        quant_colors = [x for x in sns.color_palette(f'{diverging_palette}',n_colors=7001)[::1000]]

        prev_limit = 0
        for idx, lim in enumerate([x/8 for x in range(0,9)]):
            ax.barh([1], lim-prev_limit, left=prev_limit, height=10, color=quant_colors[idx-1])
            if idx in [0,8]:
                continue
            props = dict(boxstyle='Round',
                       facecolor='w',
                       alpha=0.75, 
                       edgecolor='#cccccc',
                       linewidth=2)
            ax.text(lim,
                  0,
                  '{:.0f}%'.format(quantiles[::-1][idx]*100),
                  color='k',
                  fontsize=10,
                  fontweight=500,
                  ha='center',
                  bbox=props
                  )
            prev_limit = lim

        ax.axvline(0.5,
                 ymin=0.08,
                 ymax=0.92,
                 color='w', 
                 linewidth=2)
        ax.set(xlim=(0,1.025))
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.tick_params(bottom=False)
        sns.despine()

    def plv_card():
        pitch_list = list(pq_df
                          .loc[(pq_df['pitchername']==player)]
                          .groupby('pitchtype',as_index=False)
                          ['num_pitches']
                          .sum()
                          .query(f'num_pitches >= {int(pitch_threshold/20)}')
                          .sort_values('num_pitches',
                                       ascending=False)
                          ['pitchtype']
                          .unique())

        fig = plt.figure(figsize=(8,8))

        # Parameters to divide card
        grid_height = len(pitch_list)+4
        pitch_feats = len(pitch_list)+1

        # Divide card into tiles
        grid = plt.GridSpec(grid_height, 3, wspace=0, hspace=0.2, width_ratios=[1,3,1],
                          height_ratios=[0.75,1]+[7.5/pitch_feats]*(pitch_feats)+[0.75])

        title_ax = plt.subplot(grid[0, :-1])
        title_ax.text(-0.15,0,"{}'s\n{} Pitch Quality{}".format(player,year,'' if handedness=='All' else f' (vs {hand_map[handedness][0]}HB)'), 
                      ha='center', va='center', fontsize=20,
               bbox=dict(facecolor='#162B50', alpha=0.6, edgecolor='#162B50'))
        title_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
        title_ax.set_xticklabels([])
        title_ax.set_yticklabels([])
        title_ax.tick_params(left=False, bottom=False)

        plv_desc_ax = plt.subplot(grid[1, 1])
        plv_desc_ax.text(0,-0.2,"PLV", ha='center', va='bottom', fontsize=18,
               bbox=dict(facecolor='#162B50', alpha=0.6, edgecolor='#162B50'))
        plv_desc_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
        plv_desc_ax.set_xticklabels([])
        plv_desc_ax.set_yticklabels([])
        plv_desc_ax.tick_params(left=False, bottom=False)

        pla_desc_ax = plt.subplot(grid[1, 2])
        pla_desc_ax.text(-0.25,-0.1,"PLA", ha='center', va='bottom', fontsize=18)
        pla_desc_ax.text(-0.25,-0.15,"(xRuns per 9IP*)", ha='center', va='top', fontsize=10)
        pla_desc_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
        pla_desc_ax.set_xticklabels([])
        pla_desc_ax.set_yticklabels([])
        pla_desc_ax.tick_params(left=False, bottom=False)

        ax_num = 2
        total_pitches = pq_df.loc[(pq_df['pitchername']==player),'num_pitches'].sum()
        for pitch in ['All']+pitch_list:
            type_ax = plt.subplot(grid[ax_num, 0])
            type_ax.text(0.25,-0.1, f'{pitch}', ha='center', va='bottom', 
                         fontsize=20, fontweight='bold',
                         color='w' if pitch=='All' else color_palette[pitch])
            if pitch!='All':
                usage = pq_df.loc[(pq_df['pitchername']==player) &
                                   (pq_df['pitchtype']==pitch),'num_pitches'].sum() / total_pitches * 100
                type_ax.text(0.25,-0.1,'({:.0f}%)'.format(usage), ha='center', va='top', fontsize=10)
            else:
                type_ax.text(0.25,-0.1,'(Usage%)', ha='center', va='top', fontsize=12)
            type_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
            type_ax.set_xticklabels([])
            type_ax.set_yticklabels([])
            type_ax.tick_params(left=False, bottom=False)
            ax_num+=1

        plv_dist_ax = plt.subplot(grid[2, 1])
        plv_kde(pq_df,
                player,
                len(pitch_list),
                plv_dist_ax)
        ax_num = 3
#         st.write(pq_df.columns)
        for pitch in pitch_list:
            pitch_ax = plt.subplot(grid[ax_num, 1])
            plv_kde(pq_df, 
                    player, 
                    len(pitch_list), 
                    pitch_ax, 
                    pitchtype=pitch)
            ax_num+=1

        ax_num = 2
        for pitch in ['PLA']+pitch_list:
            val = pq_df.loc[pq_df['pitchername']==player,'PLA'].mean() if pitch=='PLA' else pq_df.loc[(pq_df['pitchername']==player) &
                                                                                                      (pq_df['pitchtype']==pitch),'pitchtype_pla'].mean()
            pla_ax = plt.subplot(grid[ax_num, 2])
            pla_ax.text(-0.25,0,'{:.2f}'.format(val), ha='center', va='center', 
                        fontsize=20)
            pla_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
            pla_ax.set_xticklabels([])
            pla_ax.set_yticklabels([])
            pla_ax.tick_params(left=False, bottom=False)
            ax_num+=1

        league_ax = plt.subplot(grid[-1, 0])
        league_ax.text(0.8,0,"League\nPercentile:", ha='right', va='center', fontsize=14)
        league_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
        league_ax.set_xticklabels([])
        league_ax.set_yticklabels([])
        league_ax.tick_params(left=False, bottom=False)

        percent_bar_ax = plt.subplot(grid[-1, 1])
        percent_bar(percent_bar_ax)

        disclaimer_ax = plt.subplot(grid[-1, 2])
        disclaimer_ax.text(-0.25,0,"*IP based on \nUsage %", ha='center', va='center', fontsize=10)
        disclaimer_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
        disclaimer_ax.set_xticklabels([])
        disclaimer_ax.set_yticklabels([])
        disclaimer_ax.tick_params(left=False, bottom=False)
        
        # Add PL logo
        pl_ax = fig.add_axes([0.675,0.7,0.2,0.2], anchor='NE', zorder=1)
        pl_ax.imshow(logo)
        pl_ax.axis('off')

        sns.despine()
        st.pyplot(fig)

    plv_card()
    
else:
    def movement_chart():
        move_df = get_movement(year,player)
        pitch_list = list(move_df
                      .groupby('pitchtype')
                      ['pitch_id']
                      .count()
                      .reset_index()
                      .sort_values('pitch_id',ascending=False)
                      ['pitchtype']
                     )
        
        fig, ax = plt.subplots(figsize=(8,8))
        
        sns.scatterplot(data=move_df,
                        x='IHB',
                        y='IVB',
                        hue='pitchtype',
                        palette=color_palette)

        ax.axhline(0, color='w', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='w', linestyle='--', linewidth=1, alpha=0.5)
        
        sns.scatterplot(data=move_df.groupby('pitchtype')[['IVB','IHB']].mean().reset_index(),
                        x='IHB',
                        y='IVB',
                        hue='pitchtype',
                        palette=color_palette,
                        s=150,
                        legend=False,
                        linewidth=2
                       )
        
        ax.set(xlim=(-27,27),
               ylim=(-27,27),
               xlabel='Induced Horizontal Break (in)',
               ylabel='Induced Vertical Break (in)')
        
        handles, labels = ax.get_legend_handles_labels()
        pitchtype_order = []
        for x in pitch_list:
            pitchtype_order.append(labels.index(x))

        ax.legend([handles[idx] for idx in pitchtype_order],[labels[idx] for idx in pitchtype_order])

        fig.suptitle(f"{player}'s {year}\nPitch Movement Profile",x=0.4,
                     y=0.95, 
                     fontsize=18)
        
        # Add PL logo
        pl_ax = fig.add_axes([0.725,0.76,0.2,0.2], anchor='NE', zorder=1)
        pl_ax.imshow(logo)
        pl_ax.axis('off')
        
        sns.despine()
        st.pyplot(fig)
        
    movement_chart()
    
st.title("General Pitch Quality")
st.write('Under construction. Sorry!')
exit()
st.write('- ***Quality Pitch (QP%)***: Pitch with a PLV >= 5.5')
st.write('- ***Average Pitch (AP%)***: Pitch with 4.5 < PLV < 5.5')
st.write('- ***Bad Pitch (BP%)***: Pitch with a PLV <= 4.5')
st.write('- ***QP-BP%***: Difference between QP and BP. Avg is 7%')

# Num Pitches threshold
pitch_min_2 = st.number_input(f'Min # of Pitches:',
                              min_value=50 if year==2023 else 200, 
                              max_value=2000,
                              step=50, 
                              value=100 if year==2023 else 500)

class_df = (plv_df
             .rename(columns={
                 'pitchername':'Pitcher'
             })
             .groupby('Pitcher')
             [['Quality Pitch','Average Pitch','Bad Pitch','pitch_id']]
             .agg({
                 'Quality Pitch':'mean',
                 'Average Pitch':'mean',
                 'Bad Pitch':'mean',
                 'pitch_id':'count'
             })
             .query(f'pitch_id >={pitch_min_2}')
             .assign(QP_BP=lambda x: x['Quality Pitch'] - x['Bad Pitch'])
             .rename(columns={
                 'Quality Pitch':'QP%',
                 'Average Pitch':'AP%',
                 'Bad Pitch':'BP%',
                 'QP_BP':'QP-BP%',
                 'pitch_id':'# Pitches'
             })
             [['# Pitches','QP%','AP%','BP%','QP-BP%']]
             .mul([1,100,100,100,100])
             .sort_values('QP-BP%', ascending=False)
            .reset_index()
            .copy()
           )

st.dataframe(class_df
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, cmap=f"{diverging_palette}", subset=['QP%','QP-BP%'])
             .background_gradient(axis=0, cmap=f"{diverging_palette}_r", subset=['BP%'])
            )
 
