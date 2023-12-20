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

st.title("Hitter Ability Metrics ")
st.write('- ***Swing Aggression***: How much more often a hitter swings at pitches, given the swing likelihoods of the pitches they face.')
st.write('''
- ***Strikezone Judgment***: The "correctness" of a hitter's swings and takes, using the likelihood of a pitch being a called strike (for swings) or a ball/HBP (for takes).
''')
st.write("- ***Decision Value***: Modeled value (runs per 100 pitches) of a hitter's decision to swing or take, minus the modeled value of the alternative.")
st.write("- ***Pitch Hittability***: Likelihood of the pitches a hitter faces becoming batted balls.")
st.write("- ***Contact Ability***: A hitter's ability to make contact (foul strike or BIP), above the contact expectation of each pitch.")
st.write("- ***Power***: Modeled number of extra bases (xISO on contact) above a pitch's expectation, for each BBE.")
st.write("- ***Hitter Performance (HP)***: Runs added per 100 pitches seen by the hitter (including swing/take decisions), after accounting for pitch quality.")

seasonal_constants = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/plv_seasonal_constants.csv?raw=true').set_index('year')

## Selectors
# Year
year = st.radio('Choose a year:', [2023,2022,2021,2020])

def z_score_scaler(series):
    return (series - series.mean()) / series.std()

season_names = {
    'swing_agg':'Swing Agg (%)',
    'strike_zone_judgement':'SZ Judge',
    'decision_value':'Dec Value',
    'contact_over_expected':'Contact',
    'adj_power':'Power',
    'batter_wOBA':'HP'
}

# Load Data
@st.cache_data(ttl=12*3600)
def load_season_data(year):
    df = pd.DataFrame()
    for month in range(3,11):
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_PLV_App_Data-{month}.parquet?raw=true'
        df = pd.concat([df,
                        pd.read_parquet(file_name)[['hittername','p_hand','b_hand','pitch_id','balls','strikes','swing_agg',
                                                    'strike_zone_judgement','decision_value','contact_over_expected',
                                                    'adj_power','batter_wOBA','pitchtype','pitch_type_bucket',
                                                   'in_play_input','p_x','p_z','sz_z'
                                                   ]]
                       ])
    
    df = df.reset_index(drop=True)
    for stat in ['swing_agg','strike_zone_judgement','contact_over_expected','in_play_input']:
        df[stat] = df[stat].mul(100).astype('float')
    
    # Convert to runs added
    df['decision_value'] = df['decision_value'].div(seasonal_constants.loc[year]['run_constant']).mul(100)
    df['batter_wOBA'] = df['batter_wOBA'].div(seasonal_constants.loc[year]['run_constant']).mul(100)
    
    df['count'] = df['balls'].astype('str')+'-'+df['strikes'].astype('str')
    
    return df

plv_df = load_season_data(year)

plv_df.loc[plv_df['p_x'].notna(),'kde_x'] = np.clip(plv_df.loc[plv_df['p_x'].notna(),'p_x'].astype('float').mul(12).round(0).astype('int').div(12),
                                                    -20/12,
                                                    20/12)
plv_df.loc[model_df['sz_z'].notna(),'kde_z'] = np.clip(plv_df.loc[plv_df['sz_z'].notna(),'sz_z'].astype('float').mul(24).round(0).astype('int').div(24),
                                                       -1.5,
                                                       1.25)

plv_df['base_decision_value'] = plv_df['decision_value'].groupby([plv_df['p_hand'],
                                                                  plv_df['b_hand'],
                                                                  plv_df['pitchtype'],
                                                                  plv_df['kde_x'],
                                                                  plv_df['kde_z'],
                                                                  plv_df['balls'],
                                                                  plv_df['strikes']]).transform('mean')
plv_df['base_power'] = plv_df['adj_power'].groupby([plv_df['p_hand'],
                                                    plv_df['b_hand'],
                                                    plv_df['pitchtype'],
                                                    plv_df['kde_x'],
                                                    plv_df['kde_z'],
                                                    plv_df['balls'],
                                                    plv_df['strikes']]).transform('mean')

plv_df['dv_oa'] = plv_df['decision_value'].sub(plv_df['base_decision_value'])
plv_df['pow_oa'] = plv_df['adj_power'].sub(plv_df['base_power'])

max_pitches = plv_df.groupby('hittername')['pitch_id'].count().max()
start_val = int(plv_df.groupby('hittername')['pitch_id'].count().quantile(0.4)/50)*50

# Num Pitches threshold
pitch_thresh = st.number_input(f'Min # of Pitches faced:',
                               min_value=min(100,start_val), 
                               max_value=2000,
                               step=50, 
                               value=500)

season_df = (plv_df
             .rename(columns=season_names)
             .rename(columns={'hittername':'Name',
                              'pitch_id':'Pitches'})
             .astype({'Name':'str'})
             .groupby('Name')
             [['Pitches']+list(season_names.values())]
             .agg({
                 'Pitches':'count',
                 'Swing Agg (%)':'mean',
                 'SZ Judge':'mean',
                 'Dec Value':'mean',
                 'Contact':'mean',
                 'Power':'mean',
                 'HP':'mean'
             })
             .query(f'Pitches >= {pitch_thresh}')
             .sort_values('HP', ascending=False)
            )

for stat in ['SZ Judge','Contact','Dec Value','Power','HP']:
    season_df[stat] = round(z_score_scaler(season_df[stat])*2+10,0)*5
    season_df[stat] = np.clip(season_df[stat].fillna(50), a_min=20, a_max=80).astype('int')

st.write(f'Metrics on a 20-80 scale. Table is sortable.')

st.dataframe(season_df
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=None, vmin=20, vmax=80, cmap="vlag",
                                  subset=['SZ Judge','Dec Value','Contact',
                                          'Power','HP']
                                 ) 
            )

### Rolling Charts
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
st.title("Rolling Ability Charts")

# Player
players = list(plv_df.groupby('hittername', as_index=False)[['pitch_id','Hitter Performance']].agg({
    'pitch_id':'count',
    'Hitter Performance':'mean'}).query(f'pitch_id >={pitch_thresh}').sort_values('Hitter Performance', ascending=False)['hittername'])
default_player = players.index('Juan Soto')
player = st.selectbox('Choose a hitter:', players, index=default_player)

col1, col2 = st.columns([0.5,0.5])

with col1:
    # Metric Selection
    metrics = list(stat_names.values())
    default_stat = metrics.index('Decision Value')
    metric = st.selectbox('Choose a metric:', metrics, index=default_stat)

with col2:
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

rolling_threshold = {
    'Swing Aggression':400,
    'Strikezone Judgement':400,
    'Decision Value':400,
    'Pitch Hittability':400,
    'Contact Ability':200,
    'Power': 75,
    'Hitter Performance':800
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
    
updated_threshold = int(round(rolling_threshold[metric]*len(selected_options)/12/5)*5 / (3 if year == 2023 else 1))

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

chart_thresh_list = (plv_df
                     .loc[plv_df['count'].astype('str').isin(selected_options) &
                          plv_df['pitch_type_bucket'].isin(pitchtype_select) &
                          plv_df['b_hand'].isin(hitter_hand) &
                          plv_df['p_hand'].isin(hand_map[handedness])
                         ]
                     .groupby('hittername')
                     [['pitch_id',metric]]
                     .agg({
                         'pitch_id':'count',
                         metric:'mean'
                     })
                     .query(f'pitch_id >= {updated_threshold}')
                     .copy()
                    )

chart_mean = plv_df.loc[plv_df['count'].isin(selected_options) & plv_df['pitch_type_bucket'].isin(pitchtype_select),metric].mean()
chart_90 = chart_thresh_list[metric].quantile(0.9)
chart_75 = chart_thresh_list[metric].quantile(0.75)
chart_25 = chart_thresh_list[metric].quantile(0.25)
chart_10 = chart_thresh_list[metric].quantile(0.1)

plv_df[metric] = plv_df[metric].replace([np.inf, -np.inf], np.nan)
rolling_df = (plv_df
              .sort_values('pitch_id')
              .loc[(plv_df['hittername']==player) &
                   plv_df['p_hand'].isin(hand_map[handedness]) &
                   plv_df['count'].isin(selected_options) &
                   plv_df['pitch_type_bucket'].isin(pitchtype_select),
                   ['hittername',metric]]
              .dropna()
              .reset_index(drop=True)
              .reset_index()
             )

window_max = max(rolling_threshold[metric],int(round(rolling_df.shape[0]/10)*7))

# Rolling Window
window = st.number_input(f'Choose a {rolling_denom[metric]} threshold:', 
                         min_value=25, 
                         max_value=window_max,
                         step=5, 
                         value=rolling_threshold[metric])

rolling_df['Rolling_Stat'] = rolling_df[metric].rolling(window).mean()
fixed_window = window if (rolling_df[metric].mean() < rolling_df['Rolling_Stat'].max()) and (rolling_df[metric].mean() > rolling_df['Rolling_Stat'].min()) else int(window*2/3)
rolling_df['Rolling_Stat'] = rolling_df[metric].rolling(window, min_periods=fixed_window).mean()

color_norm = colors.TwoSlopeNorm(vmin=chart_10, 
                                 vcenter=chart_mean,
                                 vmax=chart_90)

def rolling_chart():    
    rolling_df['index'] = rolling_df['index']+1 #Yay 0-based indexing
    fig, ax = plt.subplots(figsize=(6,6))
    sns.lineplot(data=rolling_df,
                 x='index',
                 y='Rolling_Stat',
                 color='w'
                   )
    
    line_text_loc = (rolling_df['index'].max() - fixed_window) * 1.05 + fixed_window
    
    ax.axhline(rolling_df[metric].mean(), 
               color='w',
               linestyle='--')
    ax.text(line_text_loc,
            rolling_df[metric].mean(),
            'Szn Avg',
            va='center',
            color='w')

    # Threshold Lines
    ax.axhline(chart_90,
               color=sns.color_palette('vlag', n_colors=100)[99],
               alpha=0.6)
    ax.axhline(chart_75,
               color=sns.color_palette('vlag', n_colors=100)[79],
               linestyle='--',
               alpha=0.5)
    ax.axhline(0 if (metric in ['Swing Aggression','Contact Ability']) and (count_select=='All') else chart_mean,
               color='w',
               alpha=0.5)
    ax.axhline(chart_25,
               color=sns.color_palette('vlag', n_colors=100)[19],
               linestyle='--',
               alpha=0.5)
    ax.axhline(chart_10,
               color=sns.color_palette('vlag', n_colors=100)[0],
               alpha=0.6)
    
    ax.text(line_text_loc,
            chart_90,
            '90th %' if abs(chart_90 - rolling_df[metric].mean()) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[99],
            alpha=1)
    ax.text(line_text_loc,
            chart_75,
            '75th %' if abs(chart_75 - rolling_df[metric].mean()) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[74],
            alpha=1)
    ax.text(line_text_loc,
            0 if (metric in ['Swing Aggression','Contact Ability']) and (count_select=='All') else chart_mean,
            'MLB Avg' if abs(chart_mean - rolling_df[metric].mean()) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color='w',
            alpha=0.75)
    ax.text(line_text_loc,
            chart_25,
            '25th %' if abs(chart_25 - rolling_df[metric].mean()) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[24],
            alpha=1)
    ax.text(line_text_loc,
            chart_10,
            '10th %' if abs(chart_10 - rolling_df[metric].mean()) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[9],
            alpha=1)
    
    chart_min = min(chart_10,
                    rolling_df['Rolling_Stat'].min()
                   )
    chart_max = max(chart_90,
                    rolling_df['Rolling_Stat'].max()
                   )
    
    ax.set(xlabel=rolling_denom[metric],
           ylabel=stat_values[list(stat_names.keys())[list(stat_names.values()).index(metric)]],
           ylim=(chart_min-(chart_max - chart_min)/25, 
                 chart_max+(chart_max - chart_min)/25)           
          )
    
    if metric in ['Swing Aggression','Contact Ability','Strikezone Judgement']:
        #ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.set_yticklabels([f'{int(x)}%' for x in ax.get_yticks()])

    if metric =='Pitch Hittability':
        #ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.set_yticklabels([f'{x:.1f}%' for x in ax.get_yticks()])

    pitch_text = f'; vs {pitchtype_select[0]}' if pitchtype_base == 'Offspeed' else f'; vs {pitchtype_select[0]}s'
    
    fig.suptitle("{}'s {} {}\n{}".format(player,
                                                 year,
                                                 metric,
                                                 '(Rolling {} {}{}{}{})'.format(window,
                                                                      rolling_denom[metric],
                                                                      '' if pitchtype_base == 'All' else pitch_text,
                                                                      '' if count_select=='All' else f'; in {selected_options} counts' if count_select=='Custom' else f'; in {count_select} Counts',
                                                                      '' if (handedness=='All') else f'; {hitter_hand[0]}HH vs {hand_map[handedness][0]}HP'
                                                                     )
                                                ),
                 fontsize=14
                )
    
    # Add PL logo
    pl_ax = fig.add_axes([0.8,-0.01,0.2,0.2], anchor='SE', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    
    sns.despine()
    st.pyplot(fig)
if window > rolling_df.shape[0]:
    st.write(f'Not enough {rolling_denom[metric]} ({rolling_df.shape[0]})')
else:
    rolling_chart()

st.title("PLV Heatmaps")

def plv_hitter_heatmap(hitter=player,df=plv_df,year=year):
    fig, ax = plt.subplots(figsize=(10,5))
    grid = plt.GridSpec(2, 3,height_ratios=[10,1])
    stat_dict = {
        'dv_oa':[plt.subplot(grid[0, 0]),'Decision Value',0.01],
        'contact_over_expected':[plt.subplot(grid[0, 1]),'Contact Ability',0.1],
        'pow_oa':[plt.subplot(grid[0, 2]),'Power',0.05]
    }
    
    sz_top = round(df.loc[df['hittername']==hitter,'strike_zone_top'].median()*12)
    sz_bot = round(df.loc[df['hittername']==hitter,'strike_zone_bottom'].median()*12)
    sz_range = sz_top-sz_bot
    
    for stat in stat_dict.keys():    
        v_center = df[stat].mean()
        kde_df = pd.merge(zone_df,
                          (df
                           .loc[(df['hittername']==hitter)]
                           .dropna(subset=[stat,'p_x','sz_z'])
                           [['kde_x','kde_z',stat]]
                          ),
                          how='left',
                          left_on=['x','z'],
                          right_on=['kde_x','kde_z']).fillna({stat:v_center})
        kernel_regression = KernelReg(endog=kde_df[stat], 
                                      exog= [kde_df['x'], kde_df['z']], 
                                      bw=[1/4,1/4],
                                      var_type='cc')
        kde_df['kernel_stat'] = kernel_regression.fit([kde_df['x'], kde_df['z']])[0]
        kde_df = kde_df.pivot_table(columns='x',index='z',values=['kernel_stat'], aggfunc='mean')

        sns.heatmap(data=kde_df['kernel_stat'].astype('float'),
                    cmap=kde_palette,
                    center=v_center,
                    vmin=v_center-stat_dict[stat][2],
                    vmax=v_center+stat_dict[stat][2],
                    ax=stat_dict[stat][0],
                    cbar=False
                   )

        stat_dict[stat][0].set(xlabel=None, ylabel=None)
        stat_dict[stat][0].set_xticklabels([])
        stat_dict[stat][0].set_yticklabels([])
        stat_dict[stat][0].tick_params(left=False, bottom=False)

        stat_dict[stat][0].set(xlim=(40,0),
                               ylim=(0,54))

        # Strikezone
        stat_dict[stat][0].axhline(sz_bot, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        stat_dict[stat][0].axhline(sz_top, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        stat_dict[stat][0].axvline(10, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=2)
        stat_dict[stat][0].axvline(30, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=2)

        # Inner Strikezone
        stat_dict[stat][0].axhline(sz_bot+sz_range/3, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        stat_dict[stat][0].axhline(sz_bot+2*sz_range/3, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        stat_dict[stat][0].axvline(10+20/3, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=1)
        stat_dict[stat][0].axvline(30-20/3, ymin=sz_bot/54, ymax=sz_top/54, color='black', linewidth=1)

        # Plate
        stat_dict[stat][0].plot([11.27,27.73], [1,1], color='k', linewidth=1)
        stat_dict[stat][0].plot([11.25,11.5], [1,2], color='k', linewidth=1)
        stat_dict[stat][0].plot([27.75,27.5], [1,2], color='k', linewidth=1)
        stat_dict[stat][0].plot([27.43,20], [2,3], color='k', linewidth=1)
        stat_dict[stat][0].plot([11.57,20], [2,3], color='k', linewidth=1)
        stat_dict[stat][0].set_title(f"{stat_dict[stat][1]}",pad=7)
    kde_thresh=0.1
    cb_ax = plt.subplot(grid[1, :])
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
    cb_ax.set(xlim=(-kde_thresh*1.25,kde_thresh*1.25))
    cb_ax.text(kde_thresh*1.07,0.5,'Better',ha='left',va='center',
               color=sns.color_palette('vlag',n_colors=11)[-1],fontweight='bold',
              fontsize=12)
    cb_ax.text(0,0.5,'Avg',ha='center',va='center',color='k',fontweight='bold')
    cb_ax.text(-kde_thresh*1.07,0.5,'Worse',ha='right',va='center',
               color=sns.color_palette('vlag',n_colors=11)[0],fontweight='bold',
              fontsize=12)
    # Add PL logo
    pl_ax = fig.add_axes([0.8,0.95,0.15,0.15], anchor='NE', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    fig.tight_layout()
    fig.tight_layout()
    fig.suptitle(f"{hitter}'s {year} PLV Hitter Heatmaps",y=1.05,x=0.5)
    sns.despine(left=True,bottom=True)
    st.pyplot(fig)
    
plv_hitter_heatmap()

st.write("If you have questions or ideas on what you'd like to see, DM me! [@Blandalytics](https://twitter.com/blandalytics)")
