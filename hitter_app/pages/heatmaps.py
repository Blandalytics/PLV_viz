import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import urllib

from matplotlib import ticker
from matplotlib import colors
from PIL import Image
from scipy import stats
from statsmodels.nonparametric.kernel_regression import KernelReg

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

kde_min = '#236abe'
kde_mid = '#fefefe'
kde_max = '#a9373b'

kde_palette = (sns.color_palette(f'blend:{kde_min},{kde_mid}', n_colors=1001)[:-1] +
               sns.color_palette(f'blend:{kde_mid},{kde_max}', n_colors=1001)[:-1])

sns.set_theme(
    style={
        'axes.edgecolor': pl_white,
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

line_color = sns.color_palette('vlag', n_colors=100)[0]

st.title("PLV Hitter Heatmaps")

seasonal_constants = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/plv_seasonal_constants.csv?raw=true').set_index('year')

## Selectors
# Year
year = st.radio('Choose a year:', [2023,2022,2021,2020])

season_names = {
    'swing_agg':'Swing Agg (%)',
    'strike_zone_judgement':'SZ Judge',
    'decision_value':'Dec Value',
    'contact_over_expected':'Contact',
    'adj_power':'Power',
    'batter_wOBA':'HP'
}

# Load Data
@st.cache_data(ttl=2*3600,show_spinner=f"Loading {year} data")
def load_season_data(year):
    df = pd.DataFrame()
    for month in range(3,11):
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_App_Data-{month}.parquet?raw=true'
        df = pd.concat([df,
                        pd.read_parquet(file_name)[['hittername','p_hand','b_hand','pitch_id','balls','strikes','swing_agg',
                                                    'strike_zone_judgement','decision_value','contact_over_expected',
                                                    'adj_power','batter_wOBA','pitchtype','pitch_type_bucket',
                                                    'in_play_input','p_x','p_z','sz_z','strike_zone_top','strike_zone_bottom'
                                                   ]]
                       ])
    
    df = df.reset_index(drop=True)

    df.loc[df['p_x'].notna(),'kde_x'] = np.clip(df.loc[df['p_x'].notna(),'p_x'].astype('float').mul(12).round(0).astype('int').div(12),
                                                -20/12,
                                                20/12)
    df.loc[df['sz_z'].notna(),'kde_z'] = np.clip(df.loc[df['sz_z'].notna(),'sz_z'].astype('float').mul(24).round(0).astype('int').div(24),
                                                 -1.5,
                                                 1.25)
    
    df['base_decision_value'] = df['decision_value'].groupby([df['p_hand'],
                                                              df['b_hand'],
                                                              df['pitchtype'],
                                                              df['kde_x'],
                                                              df['kde_z'],
                                                              df['balls'],
                                                              df['strikes']]).transform('mean')
    df['base_power'] = df['adj_power'].groupby([df['p_hand'],
                                                df['b_hand'],
                                                df['pitchtype'],
                                                df['kde_x'],
                                                df['kde_z'],
                                                df['balls'],
                                                df['strikes']]).transform('mean')

    df['sa_oa'] = df['swing_agg'].copy()
    df['dv_oa'] = df['decision_value'].sub(df['base_decision_value'])
    df['ca_oa'] = df['contact_over_expected'].copy()
    df['pow_oa'] = df['adj_power'].sub(df['base_power'])

    
    df.loc[df['sz_z'].notna(),'kde_z'] = np.clip(df.loc[df['sz_z'].notna(),'p_z'].astype('float').mul(12).round(0).astype('int').div(12),
                                                 0,
                                                 4.5)
    
    df['count'] = df['balls'].astype('str')+'-'+df['strikes'].astype('str')
    
    return df

plv_df = load_season_data(year)

stat_names = {
    'swing_agg':'Swing Aggression',
    'strike_zone_judgement':'Strikezone Judgement',
    'decision_value':'Decision Value',
    'in_play_input':'Pitch Hittability',
    'contact_over_expected':'Contact Ability',
    'adj_power':'Power',
    'batter_wOBA':'Hitter Performance'
}

stat_values = {
    'swing_agg':'Swing Frequency, Above Expected',
    'strike_zone_judgement':'Ball/Strike Correctness',
    'decision_value':'Runs Added, per 100 Pitches',
    'in_play_input':'Batted Ball Likelihood of Pitches',
    'contact_over_expected':'Contact Frequency, Above Expected',
    'adj_power':'Expected Extra Bases Added, per BBE',
    'batter_wOBA':'Runs Added, per 100 Pitches'
}

plv_df = plv_df.rename(columns=stat_names)
# Player
players = list(plv_df.groupby('hittername', as_index=False)[['pitch_id','Hitter Performance']].agg({
    'pitch_id':'count',
    'Hitter Performance':'mean'}).query(f'pitch_id >=100').sort_values('Hitter Performance', ascending=False)['hittername'])
default_player = players.index('Juan Soto')
player = st.selectbox('Choose a hitter:', players, index=default_player)

# Pitchtype Selection
pitchtype_help = '''
**Fastballs**: 4-Seam, Sinkers, some Cutters\n
**Breaking Balls**: Sliders, Curveballs, most Cutters\n
**Offspeed**: Changeups, Splitters
'''
pitchtype_base = st.selectbox('Vs Pitchtype', 
                              ['All','Fastballs', 'Breaking Balls', 'Offspeed'],
                              index=0,
                              help=pitchtype_help
                                )
if pitchtype_base == 'All':
    pitchtype_select = ['Fastball', 'Breaking Ball', 'Offspeed', 'Other']
else:
    pitchtype_select = [pitchtype_base] if pitchtype_base=='Offspeed' else [pitchtype_base[:-1]] # remove the 's'

rolling_denom = {
    'Swing Aggression':'Pitches',
    'Strikezone Judgement':'Pitches',
    'Decision Value':'Pitches',
    'Pitch Hittability':'Pitches',
    'Contact Ability':'Swings',
    'Power': 'BBE',
    'Hitter Performance':'Pitches'
}

count_select = st.radio('Count Group', 
                        ['All','Hitter-Friendly','Pitcher-Friendly','Even','2-Strike','3-Ball','Custom'],
                        index=0,
                        horizontal=True
                       )
 
if count_select=='All':
    selected_options = ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2']
elif count_select=='Hitter-Friendly':
    selected_options = ['1-0', '2-0', '3-0', '2-1', '3-1']
elif count_select=='Pitcher-Friendly':
    selected_options = ['0-1','0-2','1-2']
elif count_select=='Even':
    selected_options = ['0-0','1-1','2-2']
elif count_select=='2-Strike':
    selected_options = ['0-2','1-2','2-2','3-2']
elif count_select=='3-Ball':
    selected_options = ['3-0','3-1','3-2']
else:
    selected_options = st.multiselect('Select the count(s):',
                                       ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2'],
                                       ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2'])
    
# Hitter Handedness
handedness = st.select_slider(
    'Pitcher Handedness',
    options=['Left', 'All', 'Right'],
    value='All')
# Pitcher Handedness
if handedness=='All':
    hitter_hand = ['L','R']
else:
    hitter_hand = list(plv_df.loc[(plv_df['hittername']==player),'b_hand'].unique())

hand_map = {
    'Left':['L'],
    'All':['L','R'],
    'Right':['R']
}

zone_df = pd.DataFrame(columns=['x','z'])
for x in range(-20,21):
    for y in range(0,55):
        zone_df.loc[len(zone_df)] = [x/12,y/12]

heatmap_df = plv_df.loc[plv_df['p_hand'].isin(hand_map[handedness]) &
                        plv_df['count'].isin(selected_options) &
                        plv_df['pitch_type_bucket'].isin(pitchtype_select)].copy()

def plv_hitter_heatmap(hitter=player,df=heatmap_df):
    b_hand = df.loc[(df['hittername']==hitter),'b_hand'].unique()[0]
    fig= plt.figure(figsize=(7,10))
    grid = plt.GridSpec(3, 4,height_ratios=[7,7,1],hspace=0.15,
                        width_ratios=[1,1,1.1,0.9],wspace=0.025)
    stat_dict = {
        0:['sa_oa',plt.subplot(grid[0, :2]),'Swing Aggression',0.175],
        1:['dv_oa',plt.subplot(grid[0, 2:]),'Decision Value',0.01],
        2:['ca_oa',plt.subplot(grid[1, :2]),'Contact Ability',0.1],
        3:['pow_oa',plt.subplot(grid[1, 2:]),'Power',0.1]
    }
    
    bandwidth = np.clip(df
                        .loc[(df['hittername']==hitter)]
                        .shape[0]/2000,
                        0.175,
                        0.25)
    
    sz_top = round(df.loc[df['hittername']==hitter,'strike_zone_top'].median()*12)
    sz_bot = round(df.loc[df['hittername']==hitter,'strike_zone_bottom'].median()*12)
    sz_range = sz_top-sz_bot
    sz_mid = sz_bot + sz_range/2
    
    for stat in range(len(stat_dict)):
        v_center = df[stat_dict[stat][0]].mean()
        kde_df = pd.merge(zone_df,
                          (df
                           .loc[(df['hittername']==hitter) &
                                (df['pitch_type_bucket'].isin(pitchtype_select))
                               ]
                           .dropna(subset=[stat_dict[stat][0],'p_x','sz_z'])
                           [['kde_x','kde_z',stat_dict[stat][0]]]
                          ),
                          how='left',
                          left_on=['x','z'],
                          right_on=['kde_x','kde_z']).fillna({stat_dict[stat][0]:v_center})
        
        kernel_regression = KernelReg(endog=kde_df[stat_dict[stat][0]], 
                                      exog= [kde_df['x'], kde_df['z']], 
                                      bw=[bandwidth,bandwidth],
                                      var_type='cc')
        kde_df['kernel_stat'] = kernel_regression.fit([kde_df['x'], kde_df['z']])[0]
        kde_df = kde_df.pivot_table(columns='x',index='z',values=['kernel_stat'], aggfunc='mean')

        sns.heatmap(data=kde_df['kernel_stat'].astype('float'),
                    cmap=kde_palette,
                    center=v_center,
                    vmin=v_center-stat_dict[stat][3],
                    vmax=v_center+stat_dict[stat][3],
                    ax=stat_dict[stat][1],
                    cbar=False
                   )

        stat_dict[stat][1].set(xlabel=None, ylabel=None)
        stat_dict[stat][1].set_xticklabels([])
        stat_dict[stat][1].set_yticklabels([])
        stat_dict[stat][1].tick_params(left=False, bottom=False)

        stat_dict[stat][1].set(xlim=(40,0),
                               ylim=(0,54),
                               aspect=1)

        # Strikezone
        stat_dict[stat][1].axhline(sz_bot, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        stat_dict[stat][1].axhline(sz_top, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        stat_dict[stat][1].axvline(10, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=2)
        stat_dict[stat][1].axvline(30, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=2)

        # Inner Strikezone
        stat_dict[stat][1].axhline(sz_bot+sz_range/3, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        stat_dict[stat][1].axhline(sz_bot+2*sz_range/3, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        stat_dict[stat][1].axvline(10+20/3, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=1)
        stat_dict[stat][1].axvline(30-20/3, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=1)

        # Plate
        stat_dict[stat][1].plot([11.27,27.73], [1,1], color='k', linewidth=1)
        stat_dict[stat][1].plot([11.25,11.5], [1,2], color='k', linewidth=1)
        stat_dict[stat][1].plot([27.75,27.5], [1,2], color='k', linewidth=1)
        stat_dict[stat][1].plot([27.43,20], [2,3], color='k', linewidth=1)
        stat_dict[stat][1].plot([11.57,20], [2,3], color='k', linewidth=1)
        stat_dict[stat][1].set_title(f"{stat_dict[stat][2]}")
        
        stat_dict[stat][1].text(37.5 if b_hand=='L' else 2.5,
                                sz_mid,
                                'Stands Here',
                                rotation=270 if b_hand=='L' else 90,
                                fontsize=14,
                                color='k',
                                ha='center',
                                va='center',
                                bbox=dict(boxstyle='round',
                                          color='w',
                                          alpha=0.5,
                                          pad=0.2))
        
    kde_thresh=0.05
    cb_ax = fig.add_axes([0.14,0.14,0.56,0.04], anchor='NE', zorder=1)
    norm = mpl.colors.Normalize(vmin=-kde_thresh, vmax=kde_thresh)
    cb1 = mpl.colorbar.ColorbarBase(cb_ax, 
                                    cmap=mpl.colors.ListedColormap(kde_palette),
                                    norm=norm,
                                    values=[x/100 for x in range(-int(kde_thresh*100),int(kde_thresh*100)+1)],
                                    orientation='horizontal'
                                   )

    cb1.outline.set_visible(False)
    cb_ax.set_xticklabels([])
    cb_ax.set_yticklabels([])
    cb_ax.tick_params(right=False, bottom=False)
    cb_ax.set(xlim=(-kde_thresh*1.5,kde_thresh*1.5))
    cb_ax.text(kde_thresh*1.31,0.5,'More/\nBetter',ha='center',va='center',
               color=sns.color_palette('vlag',n_colors=11)[-1],fontweight='bold',
              fontsize=10)
    cb_ax.text(0,0.5,'MLB\nAvg',ha='center',va='center',color='k',fontweight='bold',fontsize=8)
    cb_ax.text(-kde_thresh*1.31,0.5,'Less/\nWorse',ha='center',va='center',
               color=sns.color_palette('vlag',n_colors=11)[0],fontweight='bold',
              fontsize=10)
    # Add PL logo
    pl_ax = fig.add_axes([0.72,0.03,0.15,0.15], anchor='NE', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    pitchtype_text = '' if len(pitchtype_select)>1 else f' vs {pitchtype_select[0]}' + ('' if pitchtype_select[0]=='Offspeed' else 's')
  
    if (pitchtype_base == 'All') & (count_select=='All') & (handedness=='All'):
        context_text = ''
    else:
        comma_text = ',' if sum([pitchtype_base == 'All',count_select=='All',handedness=='All'])<2 else ''
        context_text = '\n({}{}{})'.format('' if pitchtype_base == 'All' else f'{pitchtype_text}{comma_text}',
                                         '' if count_select=='All' else f'In {selected_options} counts{comma_text}' if count_select=='Custom' else f'In {count_select} Counts{comma_text}',
                                         '' if handedness=='All' else f'; {hitter_hand[0]}HH vs {hand_map[handedness][0]}HP'
                                         )
    
    fig.suptitle(f"{hitter}'s {year}\nPLV Hitter Heatmaps{context_text}",y=0.95 if context_text=='' else 1,x=0.5)
    sns.despine(left=True,bottom=True)
    st.pyplot(fig)
    
plv_hitter_heatmap()

st.write("If you have questions or ideas on what you'd like to see, DM me! [@Blandalytics](https://twitter.com/blandalytics)")

st.write('- ***Swing Aggression***: How much more often a hitter swings at pitches, given the swing likelihoods of the pitches they face.')
st.write("- ***Decision Value***: Modeled value (runs per 100 pitches) of a hitter's decision to swing or take, minus the modeled value of the alternative.")
st.write("- ***Contact Ability***: A hitter's ability to make contact (foul strike or BIP), above the contact expectation of each pitch.")
st.write("- ***Power***: Modeled number of extra bases (xISO on contact) above a pitch's expectation, for each BBE.")
