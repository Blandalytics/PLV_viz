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
st.image(logo)

# Year
years = [2022,2021,2020]
year = st.radio('Choose a year:', years)

seasonal_constants = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/plv_seasonal_constants.csv?raw=true').set_index('year')

@st.cache
# Load Data
def load_data(year):
    df = pd.DataFrame()
    for chunk in [1,2,3]:
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_App_Data-{chunk}.parquet?raw=true'
        df = pd.concat([df,
                        pd.read_parquet(file_name)
                       ])
    df = (df
          .sort_values('pitch_id')
          .astype({'pitch_id':'int',
                   'pitcher_mlb_id':'int'})
          .query(f'pitchtype not in {["KN","SC"]}')
          .reset_index(drop=True)
         )
    df['pitch_runs'] = df['PLV'].mul(seasonal_constants.loc[year]['run_plv_coef']).add(seasonal_constants.loc[year]['run_plv_constant'])
    df['season'] = year
    
    df['pitch_quality'] = 'Average'
    df.loc[df['PLV']>=5.5,'pitch_quality'] = 'Quality'
    df.loc[df['PLV']<4.5,'pitch_quality'] = 'Bad'

    for qual in df['pitch_quality'].unique():
      df[qual+' Pitch'] = 0
      df.loc[df['pitch_quality']==qual,qual+' Pitch'] = 1

    df['QP-BP'] = df['Quality Pitch'].sub(df['Bad Pitch'])
    
    return df

plv_df = load_data(year)

pitch_threshold = 300

## Selectors
# Player
players = list(plv_df
               .groupby('pitchername', as_index=False)
               [['pitch_id','PLV']]
               .agg({
                   'pitch_id':'count',
                   'PLV':'mean'
               })
               .query(f'pitch_id >={pitch_threshold}')
               .sort_values('PLV', ascending=False)
               ['pitchername']
              )
default_ix = players.index('Sandy Alcantara')
player = st.selectbox('Choose a player:', players, index=default_ix)

@st.cache
def get_ids():
    id_df = pd.DataFrame()
    for chunk in list(range(0,10))+['a','b','c','d','e','f']:
        chunk_df = pd.read_csv(f'https://github.com/chadwickbureau/register/blob/master/data/people-{chunk}.csv?raw=true')
        id_df = pd.concat([id_df,chunk_df])
    return id_df[['key_mlbam','key_fangraphs']].dropna().astype('int') 

id_df = get_ids()

# Season data
pla_df = pd.read_csv(f'https://github.com/Blandalytics/PLV_viz/blob/main/data/PLA_{year}.csv?raw=true', encoding='latin1')

st.title('Season Pitch Quality') 

def plv_kde(df,name,num_pitches,ax,stat='PLV',pitchtype=''):
    pitch_thresh = 500 if pitchtype=='' else 125
    pitch_color = 'w' if pitchtype=='' else marker_colors[pitchtype]

    df = df if pitchtype=='' else df.loc[df['pitchtype']==pitchtype]
    val = df.loc[df['pitchername']==name,stat].mean()
    df = df.query(f'pitch_id >= {pitch_thresh}').copy()
    val_percentile = stats.percentileofscore(df[stat], val) / 100

    sns.kdeplot(df[stat], ax=ax, color='w', legend=False, cut=0)

    x = ax.lines[-1].get_xdata()
    y = ax.lines[-1].get_ydata()

    quantiles = [1, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0]
    quant_colors = [x for x in sns.color_palette('vlag_r',n_colors=801)[::100]]

    val_color = quant_colors[sum(i >= val_percentile for i in quantiles)-1]

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
    ax.set(xlim=(3.6,6.4),
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
    quant_colors = [x for x in sns.color_palette('vlag',n_colors=7001)[::1000]]

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
    pla_dict = pla_df.loc[pla_df['Pitcher']==player,['PLA','FF','SI','SL','CH','CU','FC','FS']].to_dict(orient='list')

    pitch_list = list(plv_df
                    .loc[(plv_df['pitchername']==player)]
                    .groupby('pitchtype',as_index=False)
                    ['pitch_id']
                    .count()
                    .query('pitch_id >=20')
                    .sort_values('pitch_id',
                                ascending=False)
                    ['pitchtype'])

    fig = plt.figure(figsize=(8,8))

    # Parameters to divide card
    grid_height = len(pitch_list)+4
    pitch_feats = len(pitch_list)+1

    # Divide card into tiles
    grid = plt.GridSpec(grid_height, 3, wspace=0, hspace=0.2, width_ratios=[1,3,1],
                      height_ratios=[0.75,1]+[7.5/pitch_feats]*(pitch_feats)+[0.75])

    title_ax = plt.subplot(grid[0, :-1])
    title_ax.text(0,0,"{}\n{} Pitch Quality".format(player,year), ha='center', va='center', fontsize=24,
           bbox=dict(facecolor='#162B50', alpha=0.6, edgecolor='#162B50'))
    title_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    title_ax.set_xticklabels([])
    title_ax.set_yticklabels([])
    title_ax.tick_params(left=False, bottom=False)

    plv_desc_ax = plt.subplot(grid[1, 1])
    plv_desc_ax.text(0,0,"PLV", ha='center', va='center', fontsize=18,
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
    total_pitches = plv_df.loc[(plv_df['pitchername']==player)].shape[0]
    for pitch in ['All']+pitch_list:
        type_ax = plt.subplot(grid[ax_num, 0])
        type_ax.text(0.25,-0.1, f'{pitch}', ha='center', va='bottom', 
                     fontsize=20, fontweight='bold',
                     color='w' if pitch=='All' else marker_colors[pitch])
        if pitch!='All':
            usage = plv_df.loc[(plv_df['pitchername']==player) &
                               (plv_df['pitchtype']==pitch)].shape[0] / total_pitches * 100
            type_ax.text(0.25,-0.1,'({:.0f}%)'.format(usage), ha='center', va='top', fontsize=10)
        else:
            type_ax.text(0.25,-0.1,'(Usage%)', ha='center', va='top', fontsize=12)
        type_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
        type_ax.set_xticklabels([])
        type_ax.set_yticklabels([])
        type_ax.tick_params(left=False, bottom=False)
        ax_num+=1

    plv_dist_ax = plt.subplot(grid[2, 1])
    plv_kde((plv_df
             .groupby('pitchername')
             [['pitch_id','PLV']]
             .agg({
                 'pitch_id':'count',
                 'PLV':'mean'
             })
             .reset_index()
            ),
            player,
            len(pitch_list),
            plv_dist_ax)
    ax_num = 3
    for pitch in pitch_list:
        pitch_ax = plt.subplot(grid[ax_num, 1])
        plv_kde((plv_df
                 .groupby(['pitchername','pitchtype'])
                 [['pitch_id','PLV']]
                 .agg({
                     'pitch_id':'count',
                     'PLV':'mean'
                 })
                 .reset_index()
                ), 
                player, 
                len(pitch_list), 
                pitch_ax, 
                pitchtype=pitch)
        ax_num+=1

    ax_num = 2
    for pitch in ['PLA']+pitch_list:
        pla_ax = plt.subplot(grid[ax_num, 2])
        pla_ax.text(-0.25,0,'{:.2f}'.format(pla_dict[pitch][0]), ha='center', va='center', 
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

#     # Add PL logo
    pl_ax = plt.subplot(grid[0, -1])
    pl_ax.imshow(logo)
    pl_ax.axis('off')

    sns.despine()
    st.pyplot(fig)
    
plv_card()
