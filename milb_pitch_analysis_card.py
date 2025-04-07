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
    'FT':'#c57a02',
    'FS':'#00a1c5',  
    'FC':'#933f2c', 
    'SL':'#9300c7',  
    'ST':'#C95EBE',
    'CU':'#3c44cd',
    'CH':'#07b526', 
    'KN':'#999999',
    'CS':'#999999', 
    'SC':'#999999', 
    'UN':'#999999', 
}

cb_colors = {
    'FF':'#920000', 
    'SI':'#ffdf4d',
    'FS':'#006ddb',  
    'FC':'#ff6db6', 
    'SL':'#b66dff',  
    'ST':'#DB4B93',
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
    'ST':'Sweeper',
    'CU':'Curveball',
    'CH':'Changeup', 
    'KN':'Knuckleball',
    'SC':'Screwball', 
    'UN':'Unknown', 
}

mlb_stat_averages = {'FF': {'velo': 94.08736842105263,
  'pitch_extension': 6.434037897782064,
  'IVB': 15.488106816423368,
  'IHB': 7.860111826342976,
  'adj_vaa': 0.9542526876861707,
  'zone_pred': 0.5524634628096649,
  'PLV': 4.959100994300099},
 'SI': {'velo': 93.37165354330709,
  'pitch_extension': 6.421503314121037,
  'IVB': 7.1894812567087465,
  'IHB': 16.11683109762078,
  'adj_vaa': 0.5547257941492305,
  'zone_pred': 0.5641941277842722,
  'PLV': 4.940891148091489},
 'FS': {'velo': 86.37046979865772,
  'pitch_extension': 6.4438092219020175,
  'IVB': 1.579503883985698,
  'IHB': 12.366853622592707,
  'adj_vaa': -0.22828997970648665,
  'zone_pred': 0.40843780249945844,
  'PLV': 4.98044945975445},
 'FC': {'velo': 88.73486005089059,
  'pitch_extension': 6.348863327370304,
  'IVB': 7.0253350087713855,
  'IHB': -2.651600606856387,
  'adj_vaa': 0.04028590670661612,
  'zone_pred': 0.5417138511743801,
  'PLV': 5.036561520860988},
 'SL': {'velo': 85.38645320197044,
  'pitch_extension': 6.339913089005235,
  'IVB': 0.1900596288890174,
  'IHB': -4.895031088405101,
  'adj_vaa': -0.5685276887366394,
  'zone_pred': 0.5187218074222507,
  'PLV': 5.090979841319493},
 'ST': {'velo': 81.75238095238096,
  'pitch_extension': 6.466326666666666,
  'IVB': -0.7414287370085063,
  'IHB': -15.137090657135802,
  'adj_vaa': -0.7479210122281921,
  'zone_pred': 0.5080484211063543,
  'PLV': 5.193385876049742},
 'CU': {'velo': 79.40216183292995,
  'pitch_extension': 6.321732090034413,
  'IVB': -12.42331509058707,
  'IHB': -9.444564155605065,
  'adj_vaa': -2.529186648853596,
  'zone_pred': 0.5093645938521,
  'PLV': 4.9908054964123645},
 'CH': {'velo': 85.04103092783505,
  'pitch_extension': 6.410149210526315,
  'IVB': 4.610948378394975,
  'IHB': 15.019310112008792,
  'adj_vaa': -0.15015396609012815,
  'zone_pred': 0.4445410306119117,
  'PLV': 4.906497086537889}}

sz_bot = 1.5
sz_top = 3.5
x_ft = 2.5
y_bot = -0.5
y_lim = 6
plate_y = -.25

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

st.title("MiLB Pitchtype Cards")

#Year
years = [2025,
         2024
        ]
year = st.radio('Choose a year:', years)
# Load Data
@st.cache_data(ttl=60*30,show_spinner=f"Loading {year} data")
def load_data(year):
    df = pd.DataFrame()
    for month in range(3,11):
        file_name = f'https://github.com/Blandalytics/PLV_viz/blob/main/data/{year}_MiLB_Analysis_Data-{month}.parquet?raw=true'
        load_cols = ['pitchername','pitchtype','pitch_id','game_played','level',
                                                    'p_hand','b_hand','IHB','IVB','called_strike_pred',
                                                    'ball_pred','PLV','velo','pitch_extension',
                                                    'adj_vaa','p_x','p_z']
        df = pd.concat([df,
                        pd.read_parquet(file_name)[load_cols]
                       ])
    df = (df
          .sort_values('pitch_id')
          .astype({'pitch_id':'str'})
          .query(f'pitchtype not in {["KN","SC","UN"]}')
          .reset_index(drop=True)
         )
    df['game_played'] = pd.to_datetime(df['game_played']).dt.date
  
    return df

base_df = load_data(year)
pitch_thresh = 25

# Has at least 1 pitch with at least 50 thrown
pitcher_list = list(base_df.groupby(['pitchername','pitchtype'])['pitch_id'].count().reset_index().query(f'pitch_id >={pitch_thresh}')['pitchername'].sort_values().unique())

col1, col2, col3 = st.columns([0.4,0.35,0.25])

with col1:
    # Player
    default_ix = pitcher_list.index('David Festa')
    card_player = st.selectbox('Choose a player:', pitcher_list, index=default_ix)

with col2:
    # Pitch
    pitches = (base_df
     .loc[base_df['pitchername']==card_player,'pitchtype']
     .map(pitch_names)
     .value_counts(normalize=True)
     .where(lambda x : x>0.005)
     .dropna()
     .to_dict()
    )
    
    select_list = []
    for pitch in pitches.keys():
        select_list += [f'{pitch} ({pitches[pitch]:.1%})']
    pitch_type = st.selectbox('Choose a pitch (season usage):', select_list)
    pitch_type = pitch_type.split('(')[0][:-1]
  
with col3:
    # Chart Type
    charts = ['Bar','Violin']
    chart_type = st.selectbox('Chart style:', charts)

season_start = base_df.loc[base_df['pitchername']==card_player,'game_played'].min()
season_end = base_df.loc[base_df['pitchername']==card_player,'game_played'].max()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(f"Start Date (Season started: {season_start:%b %d})", 
                               season_start,
                               min_value=season_start,
                               max_value=season_end,
                               format="MM/DD/YYYY")
with col2:
    end_date = st.date_input(f"End Date (Season ended: {season_end:%b %d})", 
                             season_end,
                             min_value=season_start,
                             max_value=season_end,
                             format="MM/DD/YYYY")

pitch_type = {v: k for k, v in pitch_names.items()}[pitch_type]

pitch_df = base_df.loc[(base_df['game_played']>=start_date) &
                        (base_df['game_played']<=end_date)].copy()

pitcher_level = ' ('+pitch_df.loc[(pitch_df['pitchername']==card_player),'level'].mode()[0]+')' if len(pitch_df.loc[(pitch_df['pitchername']==card_player),'level'].value_counts())==1 else ''

def pitch_analysis_card(card_player,pitch_type,chart_type):
    pitches_thrown = int(pitch_df.loc[(pitch_df['pitchername']==card_player) & (pitch_df['pitchtype']==pitch_type)].shape[0]/100)*100
    pitch_num_thresh = max(pitch_thresh,
                           min(pitches_thrown,
                               int(pitch_df.loc[(pitch_df['pitchtype']==pitch_type)].groupby('pitchername')['pitch_id'].count().nlargest(75)[-1]/50)*50
                              )
                          )

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
    # ax = plt.subplot(grid[0, :3])
    # sns.scatterplot(data=(pitch_df
    #                       .loc[(pitch_df['pitchername']==card_player) &
    #                            (pitch_df['pitchtype']==pitch_type)]
    #                       .assign(p_x = lambda x: x['p_x']*-1)),
    #                 x='p_x',
    #                 y='p_z',
    #                 color=marker_colors[pitch_type],
    #                 alpha=1)

    # # Strike zone outline
    # ax.plot([-10/12,10/12], [sz_bot,sz_bot], color='w', linewidth=2)
    # ax.plot([-10/12,10/12], [sz_top,sz_top], color='w', linewidth=2)
    # ax.plot([-10/12,-10/12], [sz_bot,sz_top], color='w', linewidth=2)
    # ax.plot([10/12,10/12], [sz_bot,sz_top], color='w', linewidth=2)

    # # Inner Strike zone
    # ax.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color='w', linewidth=1)
    # ax.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color='w', linewidth=1)
    # ax.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='w', linewidth=1)
    # ax.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-y_bot), ymax=(sz_top-y_bot)/(y_lim-y_bot), color='w', linewidth=1)

    # # Plate
    # ax.plot([-8.5/12,8.5/12], [plate_y,plate_y], color='w', linewidth=2)
    # ax.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color='w', linewidth=2)
    # ax.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color='w', linewidth=2)
    # ax.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color='w', linewidth=2)
    # ax.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color='w', linewidth=2)

    # ax.set(xlim=(-x_ft,x_ft),
    #        ylim=(y_bot,y_lim),
    #        aspect=1)
    # fig.text(0.23,0.89,'Locations',fontsize=18,bbox=dict(facecolor=pl_background, alpha=0.75, edgecolor=pl_background))
    # ax.axis('off')
    # sns.despine()

    hand = pitch_df.loc[(pitch_df['pitchername']==card_player),'p_hand'].values[0]
    ax1 = plt.subplot(grid[0, 2:5])
    circle1 = plt.Circle((0, 0), 6, color=pl_white,fill=False,alpha=0.2,linestyle='--')
    ax1.add_patch(circle1)
    circle2 = plt.Circle((0, 0), 12, color=pl_white,fill=False,alpha=0.5)
    ax1.add_patch(circle2)
    circle3 = plt.Circle((0, 0), 18, color=pl_white,fill=False,alpha=0.2,linestyle='--')
    ax1.add_patch(circle3)
    circle4 = plt.Circle((0, 0), 24, color=pl_white,fill=False,alpha=0.5)
    ax1.add_patch(circle4)
    ax1.axvline(0,ymin=4/58,ymax=54/58,color=pl_white,alpha=0.5,zorder=1)
    ax1.axhline(0,xmin=4/58,xmax=54/58,color=pl_white,alpha=0.5,zorder=1)
    
    for dist in [12,24]:
        label_dist = dist-0.25
        ax1.text(label_dist,-0.3,f'{dist}"',ha='right',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(-label_dist,-0.3,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,label_dist-0.25,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,-label_dist,f'{dist}"',ha='left',va='bottom',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
    
    if hand=='R':
        ax1.text(28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    else:
        ax1.text(28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    
    ax1.text(0,27,'Rise',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    ax1.text(0,-27,'Drop',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    
    sns.scatterplot((pitch_df
                     .loc[(pitch_df['pitchername']==card_player) &
                           (pitch_df['pitchtype']==pitch_type)]
                     .assign(IHB = lambda x: np.where(hand=='R',x['IHB'].astype('float').mul(-1),x['IHB'].astype('float')))
                    ),
                    x='IHB',
                    y='IVB',
                   color=marker_colors[pitch_type],
                   # palette=marker_colors,
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    zorder=10,
                   ax=ax1,
                   legend=False)    
    
    # handles, labels = ax1.get_legend_handles_labels()
    # pitch_type_names = [pitch_names[x] for x in labels]
    # # pitch_type_names = [pitch_names[x].ljust(15, " ") for x in labels]
    # ax1.legend(handles,[pitch_names[x] for x in labels], ncols=len(labels),
    #          loc='lower center', 
    #            fontsize=min(52/len(labels),14),
    #           framealpha=0,bbox_to_anchor=(0.5, -0.23+len(labels)/100,
    #                                        0,0))
    
    ax1.set(xlim=(-29,29),
           ylim=(-29,29),
           aspect=1)
    ax1.set_title('Movement',fontsize=18)
    ax1.axis('off')
    sns.despine(left=True,bottom=True)

    sz_bot = 1.5
    sz_top = 3.5
    x_ft = 2.5
    y_bot = -0.5
    y_lim = 6.5
    plate_y = -.25
    alpha_val = 0.5
    title_y = 0.95
    
    ax2 = plt.subplot(grid[0, :2])
    # Outer Strike Zone
    zone_outline = plt.Rectangle((-10/12, sz_bot), 20/12, 2, color=pl_white,fill=False,alpha=alpha_val)
    ax2.add_patch(zone_outline)

    # Inner Strike zone
    ax2.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    
    # Plate
    ax2.plot([-8.5/12,8.5/12], [plate_y,plate_y], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    
    sns.scatterplot(data=(pitch_df
                          .loc[(pitch_df['pitchername']==card_player) &
                               (pitch_df['pitchtype']==pitch_type) &
                                (pitch_df['b_hand']=='L')].assign(p_x = lambda x: x['p_x']*-1)),
                    x='p_x',
                    y='p_z',
                    color=marker_colors[pitch_type],
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    legend=False,
                   zorder=10,
                   ax=ax2)
    
    ax2.set(xlim=(-2,2),
           ylim=(y_bot,y_lim-1),
           aspect=1,
           title='Locations\nvs LHH')
    ax2.set_title('Locations\nvs LHH',fontsize=18,y=title_y)
    ax2.axis('off')
    
    ax3 = plt.subplot(grid[0,5:])
    # Outer Strike Zone
    zone_outline = plt.Rectangle((-10/12, sz_bot), 20/12, 2, color=pl_white,fill=False,alpha=alpha_val)
    ax3.add_patch(zone_outline)
    
    # Inner Strike zone
    ax3.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    
    # Plate
    ax3.plot([-8.5/12,8.5/12], [plate_y,plate_y], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    
    sns.scatterplot(data=(pitch_df
                          .loc[(pitch_df['pitchername']==card_player) &
                               (pitch_df['pitchtype']==pitch_type) &
                                (pitch_df['b_hand']=='R')].assign(p_x = lambda x: x['p_x']*-1)),
                    x='p_x',
                    y='p_z',
                    color=marker_colors[pitch_type],
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    legend=False,
                   zorder=10,
                   ax=ax3)
    
    ax3.set(xlim=(-2,2),
           ylim=(y_bot,y_lim-1),
           aspect=1)
    ax3.set_title('Locations\nvs RHH',fontsize=18,y=title_y)
    ax3.axis('off')
  
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
            ax.axhline((mlb_stat_averages[pitch_type][stat] - pitch_stats_df[stat].min())/(pitch_stats_df[stat].max()-pitch_stats_df[stat].min()),
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
            ax.axhline(1.15,
                       xmin=0.1,
                       xmax=0.9,
                       color='w')
            ax.bar(1, 1, color='w',alpha=0.1)
            ax.bar(1, plot_val, color=marker_colors[pitch_type])
            ax.axhline((mlb_stat_averages[pitch_type][stat] - pitch_stats_df[stat].min())/(pitch_stats_df[stat].max()-pitch_stats_df[stat].min()),
                   linestyle='--',
                   color='w')
            ax.text(1, plot_val+0.01,
                    format_dict[stat],
                    va='bottom',
                    ha='center',
                    fontsize=12 if stat=='velo' else 14,
                    bbox=dict(facecolor='#2d4061', alpha=0.75 if plot_val<0.5 else 0, linewidth=0, pad=1))
            ax.text(1,
                    1.4,
                    stat_name_dict[stat],
                    va='center',
                    ha='center',
                    fontsize=14)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set(ylim=(0,1.9))
            ax.tick_params(left=False, bottom=False)

    adjusted_pitch_name = pitch_names[pitch_type] if (card_player != 'Kutter Crawford') | (pitch_names[pitch_type] != 'Cutter') else 'Kutter'
    fig.text(0.525,0.45,'Pitch Characteristics',ha='center',fontsize=18)
    fig.text(0.525,0.43,f'(Compared to AAA {adjusted_pitch_name}s; Min {pitch_num_thresh} Thrown; - - - is MLB Median)',ha='center',fontsize=12)
    
    # Add PL logo
    pl_ax = fig.add_axes([0.41,0.475,0.2,0.2], anchor='S', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')

    apostrophe_text = "'" if card_player[-1]=='s' else "'s"
    
    fig.suptitle(f"{card_player}{apostrophe_text} {year} MiLB {adjusted_pitch_name}{pitcher_level}",y=0.97,fontsize=20,x=0.5)
    date_text = '' if (start_date==season_start) & (end_date==season_end) else f'{start_date:%b %-d} - {end_date:%b %-d}; '
    fig.text(0.5,0.925,f"({date_text}From Pitcher's Perspective)",ha='center',fontsize=12)
    # fig.text(0.77,0.07,"@Blandalytics",ha='center',fontsize=10)
    # fig.text(0.77,0.05,"pitch-analysis-card.streamlit.app",ha='center',fontsize=10)
    sns.despine(left=True,bottom=True)
    st.pyplot(fig)


def movement_chart(player):
    hand = pitch_df.loc[(pitch_df['pitchername']==player),'p_hand'].values[0]
    move_df = pitch_df.loc[(pitch_df['pitchername']==player)].copy()
    
    pitch_list = [x[0] for x in Counter(move_df['pitchtype']).most_common() if (x[0] != 'UN')]
    
    fig, ax1 = plt.subplots(figsize=(8,8))

    circle1 = plt.Circle((0, 0), 6, color=pl_white,fill=False,alpha=0.2,linestyle='--')
    ax1.add_patch(circle1)
    circle2 = plt.Circle((0, 0), 12, color=pl_white,fill=False,alpha=0.5)
    ax1.add_patch(circle2)
    circle3 = plt.Circle((0, 0), 18, color=pl_white,fill=False,alpha=0.2,linestyle='--')
    ax1.add_patch(circle3)
    circle4 = plt.Circle((0, 0), 24, color=pl_white,fill=False,alpha=0.5)
    ax1.add_patch(circle4)
    ax1.axvline(0,ymin=4/58,ymax=54/58,color=pl_white,alpha=0.5,zorder=1)
    ax1.axhline(0,xmin=4/58,xmax=54/58,color=pl_white,alpha=0.5,zorder=1)
    
    for dist in [12,24]:
        label_dist = dist-0.25
        ax1.text(label_dist,-0.3,f'{dist}"',ha='right',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(-label_dist,-0.3,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,label_dist-0.25,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,-label_dist,f'{dist}"',ha='left',va='bottom',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
    
    if move_df['P Hand'].value_counts().index[0]=='R':
        ax1.text(28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    else:
        ax1.text(28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    
    ax1.text(0,27,'Rise',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    ax1.text(0,-27,'Drop',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    
    sns.scatterplot(move_df.assign(IHB = lambda x: np.where(hand=='R',x['IHB'].astype('float').mul(-1),x['IHB'].astype('float'))),
                    x='IHB',
                    y='IVB',
                   hue='pitchtype',
                   palette=marker_colors,
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    zorder=10,
                   ax=ax1)    
       
    ax1.set(xlim=(-29,29),
           ylim=(-29,29),
           aspect=1)
    ax1.axis('off')

    handles, labels = ax.get_legend_handles_labels()
    pitchtype_order = []
    pitch_velos = {}
    for x in pitch_list:
        pitchtype_order.append(labels.index(x))
        
        pitch_velo = move_df.loc[move_df['pitchtype']==x,'velo'].mean()
        pitch_velos[x] = f' ({pitch_velo:.1f})'
    ax.legend([handles[idx] for idx in pitchtype_order],
              [pitch_names[labels[idx]]+pitch_velos[labels[idx]] for idx in pitchtype_order],
              title='Pitchtype (velo)',
              loc='upper right' if hand =='L' else 'upper left')
    
    fig.text(0.83,0.0425,'Glove' if hand == 'L' else 'Arm',ha='left')
    fig.text(0.185,0.0425,'Arm' if hand == 'L' else 'Glove',ha='right')
    fig.text(0.05,0.84,'Rise',ha='center')
    fig.text(0.05,0.11,'Drop',ha='center')
    
    ax.annotate('', xy=(0.65, -0.08), xycoords='axes fraction', xytext=(0.9, -0.08), 
                arrowprops=dict(arrowstyle="<-", color='w'))
    ax.annotate('', xy=(0.35, -0.08), xycoords='axes fraction', xytext=(0.09, -0.08), 
                arrowprops=dict(arrowstyle="<-", color='w'))
    ax.annotate('', xy=(-0.1, 0.64), xycoords='axes fraction', xytext=(-0.1, 0.93), 
                arrowprops=dict(arrowstyle="<-", color='w'))
    ax.annotate('', xy=(-0.1, 0.35), xycoords='axes fraction', xytext=(-0.1, 0.05), 
                arrowprops=dict(arrowstyle="<-", color='w'))
    
    fig.suptitle(f"{player}'s {year}\nInduced Movement Profile",x=0.45,
                 y=0.95, 
                 fontsize=18)
    
    # Add PL logo
    pl_ax = fig.add_axes([0.725,0.76,0.2,0.2], anchor='NE', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    
    sns.despine()
    st.pyplot(fig)
  
if st.button("Generate Data Viz"):
    pitch_analysis_card(card_player,pitch_type,chart_type)
    movement_chart(card_player)

st.title("Metric Definitions")
st.write("- ***Velocity***: Release speed of the pitch, out of the pitcher's hand (in miles per hour).")
st.write('- ***Release Extension***: Distance towards the plate when the pitcher releases the pitch (in feet).')
st.write('- ***Induced Vertical Break (IVB)***: Vertical break of the pitch, controlling for the effect of gravity (in inches).')
st.write("- ***Arm-Side Break***: Horizontal break of the pitch, relative to the pitcher's handedness (in inches).")
st.write("- ***Adjusted Vertical Approach Angle (VAA)***: Vertical angle at which the pitch approaches home plate, controlling for its vertical location at the plate (in degrees).")
st.write("- ***xZone%***: Predicted likelihood of the pitch being in the strike zone (as is called), assuming a swing isn't made.")
st.write('- ***Pitch Level Value (PLV)***: Estimated value of the pitch, based on the predicted outcomes of the pitch (0-10 scale. 5 is league average pitch value. PLV is not adjusted for pitch type.).')
