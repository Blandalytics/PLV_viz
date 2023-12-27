import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import urllib

from PIL import Image
from collections import Counter
from scipy import stats

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

kde_min = '#236abe'
kde_max = '#a9373b'

kde_palette = (sns.color_palette(f'blend:{kde_min},{pl_white}', n_colors=1001)[:-1] +
               sns.color_palette(f'blend:{pl_white},{kde_max}', n_colors=1001)[:-1])

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

sz_bot = 1.5
sz_top = 3.5
x_ft = 2.5
y_bot = -0.5
y_lim = 6
plate_y = -.25

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

st.title("Pitchtype Cards")

# Year
years = [2023,
         2022,2021,2020
        ]
year = st.radio('Choose a year:', years)
# Load Data
@st.cache_data
def load_data(year):
    df = pd.DataFrame()
    for chunk in [1,2,3]:
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_Pitch_Analysis_Data-{chunk}.parquet?raw=true'
        load_cols = ['pitchername','pitchtype','pitch_id',
                                                    'p_hand','b_hand','IHB','IVB','called_strike_pred',
                                                    'ball_pred','PLV','velo','pitch_extension',
                                                    'adj_vaa','p_x','p_z']
        # if year == 2023:
        #     load_cols += ['b_hand']
        df = pd.concat([df,
                        pd.read_parquet(file_name)[load_cols]
                       ])
    df = (df
          .sort_values('pitch_id')
          .astype({'pitch_id':'int'})
          .query(f'pitchtype not in {["KN","SC","UN"]}')
          .reset_index(drop=True)
         )
    
    return df

def kde_calcs(df,pitcher,pitchtype,year=year):
    p_hand = df.loc[(df['pitchername']==pitcher),'p_hand'].iloc[0]
    kde_diffs = []
    for b_hand in ['L','R']:
        kde_df = (df
                  .loc[(df['pitchtype']==pitchtype) &
                       (df['b_hand']==b_hand) &
                       (df['p_hand']==p_hand)
                      ]
                  .assign(kde_x = lambda x: np.clip(x['p_x'].astype('float').mul(12).round(0).astype('int').div(12),-20/12,20/12),
                          kde_z = lambda x: np.clip(x['p_z'].astype('float').mul(12).round(0).astype('int').div(12),0,4.5))
                  .reset_index(drop=True)
                 )
        x_loc_league = kde_df['kde_x']
        y_loc_league = kde_df['kde_z']

        x_loc_pitcher = kde_df.loc[kde_df['pitchername']==pitcher,'kde_x']
        y_loc_pitcher = kde_df.loc[kde_df['pitchername']==pitcher,'kde_z']
        if x_loc_pitcher.empty:
          kde_diffs += [pd.DataFrame()]
          continue

        xmin = x_loc_league.min()
        xmax = x_loc_league.max()
        ymin = y_loc_league.min()
        ymax = y_loc_league.max()

        X, Y = np.mgrid[xmin:xmax:41j, ymin:ymax:55j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # league matrix
        values_league = np.vstack([x_loc_league, y_loc_league])
        kernel_league = sp.stats.gaussian_kde(values_league)
        f_league = np.reshape(kernel_league(positions).T, X.shape)

        # pitcher matrix
        values_pitcher = np.vstack([x_loc_pitcher, y_loc_pitcher])
        kernel_pitcher = sp.stats.gaussian_kde(values_pitcher)
        f_pitcher = np.reshape(kernel_pitcher(positions).T, X.shape)
        
        kde_diffs += [pd.DataFrame(f_pitcher-f_league).T]
    return kde_diffs

pitch_df = load_data(year)

pitch_thresh = 10

# Has at least 1 pitch with at least 50 thrown
pitcher_list = list(pitch_df.groupby(['pitchername','pitchtype'])['pitch_id'].count().reset_index().query(f'pitch_id >={pitch_thresh}')['pitchername'].sort_values().unique())

col1, col2, col3 = st.columns(3)

with col1:
    # Player
    default_ix = pitcher_list.index('Zack Wheeler')
    card_player = st.selectbox('Choose a player:', pitcher_list, index=default_ix)

with col2:
    # Pitch
    _pitches = list(pitch_df.loc[pitch_df['pitchername']==card_player].groupby('pitchtype')['pitch_id'].count().reset_index().sort_values('pitch_id',ascending=False).query(f'pitch_id>={pitch_thresh}')['pitchtype'])
    pitches = [pitch_names[x] for x in _pitches]
    pitch_ix = pitches.index('Four-Seamer') if 'Four-Seamer' in pitches else 0
    pitch_type = st.selectbox('Choose a pitch:', pitches, index=pitch_ix)
  
with col3:
    # Chart Type
    charts = ['Bar','Violin']
    chart_type = st.selectbox('Choose a chart style:', charts)

# st.write(_pitches)
pitch_type = {v: k for k, v in pitch_names.items()}[pitch_type]

def pitch_analysis_card(card_player,pitch_type,chart_type):
    pitches_thrown = int(pitch_df.loc[(pitch_df['pitchername']==card_player) & (pitch_df['pitchtype']==pitch_type)].shape[0]/100)*100
    pitch_num_thresh = max(pitch_thresh,
                           min(pitches_thrown,
                               int(pitch_df.loc[(pitch_df['pitchtype']==pitch_type)].groupby('pitchername')['pitch_id'].count().nlargest(75)[-1]/50)*50
                              )
                          )

    # model_df['zone_pred'] = model_df['called_strike_pred'].div(model_df[['called_strike_pred','ball_pred']].sum(axis=1))
    pitch_stats_df = (
        pitch_df
        .assign(IHB = lambda x: np.where(x['p_hand']=='R',x['IHB']*-1,x['IHB']),
                zone_pred = lambda x: x['called_strike_pred'] / x[['called_strike_pred','ball_pred']].sum(axis=1))
        .loc[(pitch_df['pitchtype']==pitch_type)]
        .groupby(['pitchername'])
        [['pitch_id','p_hand','PLV','velo','pitch_extension','IVB','IHB','adj_vaa','zone_pred']]
        .agg({
            'pitch_id':'count',
            'p_hand':pd.Series.mode,
            'PLV':'mean',
            'velo':'mean',
            'pitch_extension':'mean',
            'IVB':'mean',
            'IHB':'mean',
            'adj_vaa':'mean',
            'zone_pred':'mean'
        })
         .query(f'pitch_id>={pitch_num_thresh}')
        .reset_index()
        .sort_values('zone_pred', ascending=False)
    )

    def min_max_scaler(x):
        return ((x-x.min())/(x.max()-x.min()))

    for col in ['PLV','velo','pitch_extension','IVB','IHB','adj_vaa','zone_pred']:
        pitch_stats_df[col+'_scale'] = min_max_scaler(pitch_stats_df[col])
        pitch_stats_df[col+'_pct'] = pitch_stats_df[col].rank(pct=True)

    chart_stats = ['velo','pitch_extension','IVB','IHB','adj_vaa','zone_pred','PLV']
    fig = plt.figure(figsize=(10,10))

    stat_name_dict = {
        'velo':'Velocity',
        'pitch_extension':'Release\nExtension',
        'IVB':'Induced\nVertical\nBreak',
        'IHB':'Arm-Side\nBreak',
        'adj_vaa':'Adj. Vert.\nApproach\nAngle',
        'zone_pred':'xZone%',
        'PLV':'PLV',
    }

    stat_tops = {
        'velo':'Faster',
        'pitch_extension':'Longer',
        'IVB':'Rise',
        'IHB':'Arm',
        'adj_vaa':'Flatter',
        'zone_pred':'In',
        'PLV':'Good',
    }
    stat_bottoms = {
        'velo':'Slower',
        'pitch_extension':'Shorter',
        'IVB':'Drop',
        'IHB':'Glove',
        'adj_vaa':'Steeper',
        'zone_pred':'Out',
        'PLV':'Bad',
    }

    # Divide card into tiles
    grid = plt.GridSpec(2, len(chart_stats),height_ratios=[5,5],hspace=0.2)
    ax = plt.subplot(grid[0, :3])
    sns.scatterplot(data=(pitch_df
                          .loc[(pitch_df['pitchername']==card_player) &
                               (pitch_df['pitchtype']==pitch_type)]
                          .assign(p_x = lambda x: x['p_x']*-1)),
                    x='p_x',
                    y='p_z',
                    color=marker_colors[pitch_type],
                    alpha=1)

    # Strike zone outline
    ax.plot([-10/12,10/12], [sz_bot,sz_bot], color='w', linewidth=2)
    ax.plot([-10/12,10/12], [sz_top,sz_top], color='w', linewidth=2)
    ax.plot([-10/12,-10/12], [sz_bot,sz_top], color='w', linewidth=2)
    ax.plot([10/12,10/12], [sz_bot,sz_top], color='w', linewidth=2)

    # Inner Strike zone
    ax.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color='w', linewidth=1)
    ax.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color='w', linewidth=1)
    ax.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='w', linewidth=1)
    ax.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='w', linewidth=1)

    # Plate
    ax.plot([-8.5/12,8.5/12], [plate_y,plate_y], color='w', linewidth=2)
    ax.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color='w', linewidth=2)
    ax.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color='w', linewidth=2)
    ax.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color='w', linewidth=2)
    ax.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color='w', linewidth=2)

    ax.set(xlim=(-x_ft,x_ft),
           ylim=(y_bot,y_lim),
           aspect=1)
    fig.text(0.23,0.89,'Locations',fontsize=18,bbox=dict(facecolor=pl_background, alpha=0.75, edgecolor=pl_background))
    ax.axis('off')
    sns.despine()

    hand = pitch_df.loc[(pitch_df['pitchername']==card_player),'p_hand'].values[0]
    ax = plt.subplot(grid[0, 3:])
    sns.scatterplot(data=pitch_df.loc[(pitch_df['pitchername']==card_player) &
                                      (pitch_df['pitchtype']==pitch_type)],
                    x='IHB',
                    y='IVB',
                    color=marker_colors[pitch_type],
                    s=25,
                    alpha=1)

    ax.axhline(0, color='w', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='w', linestyle='--', linewidth=1, alpha=0.5)
    ax.set(aspect=1)

    sns.scatterplot(data=(pitch_df
                          .loc[(pitch_df['pitchername']==card_player) &
                               (pitch_df['pitchtype']==pitch_type)]
                          .groupby('pitchtype')
                          [['IVB','IHB']]
                          .mean()
                          .reset_index()
                         ),
                    x='IHB',
                    y='IVB',
                    color=marker_colors[pitch_type],
                    s=200,
                    legend=False,
                    linewidth=2
                   )

    ax_lim = max(25,
                 pitch_df.loc[(pitch_df['pitchername']==card_player) &
                              (pitch_df['pitchtype']==pitch_type),
                              ['IHB','IVB']].abs().quantile(0.999).max()+1
                )
    ax.set(xlim=(ax_lim,-ax_lim),
           ylim=(-ax_lim,ax_lim))
    plt.xlabel('Arm-Side Break', fontsize=12)
    plt.ylabel('Induced Vertical Break', fontsize=12,labelpad=-1)
    ax.set_xticks([x*10 for x in range(-int(ax_lim/10),int(ax_lim/10)+1)][::-1])
    if hand=='R':
        ax.set_xticklabels([x*-1 for x in ax.get_xticks()])
    fig.text(0.62,0.89,'Movement',fontsize=18)
    sns.despine(left=True,bottom=True)

    fig.text(0.5,0.45,'Pitch Characteristics',ha='center',fontsize=18)
    fig.text(0.5,0.43,f'(Compared to MLB {pitch_names[pitch_type]}s; Min {pitch_num_thresh} Thrown; - - - is MLB Median)',ha='center',fontsize=12)
    for stat in chart_stats:
        if chart_type=='Violin':
            val = pitch_stats_df.loc[(pitch_stats_df['pitchername']==card_player),
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
            ax.collections[0].set_edgecolor('w')
    
            top = ax.get_ylim()[1]
            bot = ax.get_ylim()[0]
            plot_height = top - bot
    
            format_dict = {
                'PLV':f'{val:.2f}',
                'velo':f'{val:.1f}mph',
                'pitch_extension':f'{val:.1f}ft',
                'IVB':f'{val:.1f}"',
                'IHB':f'{val:.1f}"',
                'adj_vaa':f'{val:.1f}°',
                'zone_pred':f'{val*100:.1f}%'
            }
            ax.axhline(pitch_stats_df[stat+'_scale'].median(),
                       linestyle='--',
                       color='w')
            ax.axhline(top + (0.25 * plot_height),
                       xmin=0.1,
                       xmax=0.9,
                       color='w')
            ax.text(0,
                    pitch_stats_df.loc[(pitch_stats_df['pitchername']==card_player),
                                       stat+'_scale'],
                    format_dict[stat],
                    va='center',
                    ha='center',
                    fontsize=12 if stat=='velo' else 14,
                    bbox=dict(facecolor=pl_background, alpha=0.75, edgecolor='w'))
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
        else:
            plot_val = pitch_stats_df.loc[(pitch_stats_df['pitchername']==card_player),stat+'_pct'].item()
            text_val = pitch_stats_df.loc[(pitch_stats_df['pitchername']==card_player),stat].item()

            format_dict = {
                'PLV':f'{text_val:.2f}',
                'velo':f'{text_val:.1f}mph',
                'pitch_extension':f'{text_val:.1f}ft',
                'IVB':f'{text_val:.1f}"',
                'IHB':f'{text_val:.1f}"',
                'adj_vaa':f'{text_val:.1f}°',
                'zone_pred':f'{text_val*100:.1f}%'
            }
            
            ax = plt.subplot(grid[1, chart_stats.index(stat)])
            ax.axhline(1.25,
                       xmin=0.1,
                       xmax=0.9,
                       color='w')
            ax.bar(1, 1, color='w',alpha=0.1)
            ax.bar(1, plot_val, color=marker_colors[pitch_type])
            ax.axhline(0.5,
                       linestyle='--',
                       color='w')
            ax.text(1, plot_val+0.01,
                    format_dict[stat],
                    va='bottom',
                    ha='center',
                    fontsize=12 if stat=='velo' else 14,
                    bbox=dict(facecolor='#2d4061', alpha=1, edgecolor='#2d4061', pad=0.5))
            ax.text(1,
                    1.5,
                    stat_name_dict[stat],
                    va='center',
                    ha='center',
                    fontsize=14)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set(ylim=(0,2))
            ax.tick_params(left=False, bottom=False)

    # Add PL logo
    pl_ax = fig.add_axes([0.41,0.025,0.2,0.2], anchor='S', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')

    apostrophe_text = "'" if card_player[-1]=='s' else "'s"
    fig.suptitle(f"{card_player}{apostrophe_text} {year} {pitch_names[pitch_type]}",y=0.97,fontsize=20,x=0.525)
    fig.text(0.525,0.925,"(From Pitcher's Perspective)",ha='center',fontsize=12)
    fig.text(0.77,0.07,"@Blandalytics",ha='center',fontsize=10)
    fig.text(0.77,0.05,"pitch-analysis-card.streamlit.app",ha='center',fontsize=10)
    sns.despine(left=True,bottom=True)
    st.pyplot(fig)
pitch_analysis_card(card_player,pitch_type,chart_type)

p_hand = pitch_df.loc[(pitch_df['pitchername']==card_player),'p_hand'].iloc[0]
def kde_chart(kde_data,p_hand=p_hand,kde_thresh=0.1):
    fig = plt.figure(figsize=(11,7))
    grid = plt.GridSpec(2, 3,height_ratios=[50,1],width_ratios=[5,1,5],hspace=0,wspace=0.05)
    for hand in ['L','R']:
        hand_index = 0 if hand=='L' else 1
        ax = plt.subplot(grid[0, 0]) if hand=='L' else plt.subplot(grid[0, 2])
        ax.set(xlabel=None, ylabel=None)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        if kde_diffs[hand_index].empty:
            ax.text(0.5,0.5,f'None thrown\nto {hand}HH',va='center',ha='center',fontsize=18)
            continue
        sns.heatmap(kde_diffs[hand_index],
                    cmap=kde_palette,
                    center=0,
                    vmin=-kde_thresh,
                    vmax=kde_thresh,
                    cbar=False,
                    ax=ax
                   )

        # Strikezone
        ax.axhline(18, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        ax.axhline(42, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        ax.axvline(10, ymin=1/3, ymax=7/9, color='black', linewidth=2)
        ax.axvline(30, ymin=1/3, ymax=7/9, color='black', linewidth=2)
    
        # Inner Strikezone
        ax.axhline(26, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        ax.axhline(34, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        ax.axvline(10+20/3, ymin=1/3, ymax=7/9, color='black', linewidth=1)
        ax.axvline(30-20/3, ymin=1/3, ymax=7/9, color='black', linewidth=1)
    
        # Plate
        ax.plot([11.52,27.48], [1,1], color='k', linewidth=1)
        ax.plot([11.5,11.75], [1,2], color='k', linewidth=1)
        ax.plot([27.5,27.25], [1,2], color='k', linewidth=1)
        ax.plot([27.3,20], [2,3], color='k', linewidth=1)
        ax.plot([11.7,20], [2,3], color='k', linewidth=1)
    
        ax.set(xlim=(40,0),
               ylim=(0,54),
               aspect=1)
    
        ax.text(20,55,f"{p_hand[0]}HP vs {hand}HH",ha='center',fontsize=16)
        ax.axis('off')
    ax = plt.subplot(grid[0, 1])
    norm = mpl.colors.Normalize(vmin=-kde_thresh, vmax=kde_thresh)
    cb1 = mpl.colorbar.ColorbarBase(ax, 
                                    cmap=mpl.colors.ListedColormap(kde_palette),
                                    norm=norm,
                                    values=[x/100 for x in range(-int(kde_thresh*100),int(kde_thresh*100)+1)],
                                   )
    
    cb1.outline.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(right=False, bottom=False)
    ax.set(ylim=(-kde_thresh*1.5,kde_thresh*1.5))
    ax.text(0.5,kde_thresh,f'+{int(kde_thresh*100)}%\n',ha='center',va='bottom',color=kde_palette[-150],fontweight='bold')
    ax.text(0.5,kde_thresh*1.25,'More\nOften',ha='center',va='bottom',color=kde_palette[-150],fontweight='bold')
    ax.text(0.5,-kde_thresh/100,'0%',ha='center',va='center',color='k',fontweight='bold')
    ax.text(0.5,-kde_thresh,f'\n-{int(kde_thresh*100)}%',ha='center',va='top',color=kde_palette[150],fontweight='bold')
    ax.text(0.5,-kde_thresh*1.25,'Less\nOften',ha='center',va='top',color=kde_palette[150],fontweight='bold')
    ax.axis('off')
  
    apostrophe_text = "'" if card_player[-1]=='s' else "'s"
    fig.suptitle(f"{card_player}{apostrophe_text} {year} {pitch_names[pitch_type]} Locations",ha='center',y=1, fontsize=18)
    fig.text(0.5,0.88,"(From Pitcher's Perspective; Relative to MLB)\n\n",ha='center',va='bottom')
    sns.despine(left=True,bottom=True)

    # Add PL logo
    pl_ax = fig.add_axes([0.41,0.015,0.2,0.2], anchor='S', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    fig.text(0.77,0.08,"@Blandalytics",ha='center',fontsize=10)
    fig.text(0.77,0.05,"pitch-analysis-card.streamlit.app",ha='center',fontsize=10)
    st.pyplot(fig)

heatmap_thresh = 100
if pitch_df.loc[(pitch_df['pitchername']==card_player) & (pitch_df['pitchtype']==pitch_type)].shape[0] < heatmap_thresh :
    st.write(f'Not enough pitches (<{heatmap_thresh}) to generate heatmaps')
else:
    kde_diffs = kde_calcs(pitch_df,pitcher=card_player,pitchtype=pitch_type,year=year)
    kde_chart(kde_diffs)

st.title("Metric Definitions")
st.write("- ***Velocity***: Release speed of the pitch, out of the pitcher's hand (in miles per hour).")
st.write('- ***Release Extension***: Distance towards the plate when the pitcher releases the pitch (in feet).')
st.write('- ***Induced Vertical Break (IVB)***: Vertical break of the pitch, controlling for the effect of gravity (in inches).')
st.write("- ***Arm-Side Break***: Horizontal break of the pitch, relative to the pitcher's handedness (in inches).")
st.write("- ***Adjusted Vertical Approach Angle (VAA)***: Vertical angle at which the pitch approaches home plate, controlling for its vertical location at the plate (in degrees).")
st.write("- ***xZone%***: Predicted likelihood of the pitch being in the strike zone (as is called), assuming a swing isn't made.")
st.write('- ***Pitch Level Value (PLV)***: Estimated value of the pitch, based on the predicted outcomes of the pitch (0-10 scale. 5 is league average pitch value. PLV is not adjusted for pitch type.).')
