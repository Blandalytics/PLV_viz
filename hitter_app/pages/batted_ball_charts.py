import streamlit as st
st.set_page_config(page_title='Batted Ball Charts')
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import urllib

from PIL import Image
from scipy import stats
from statsmodels.nonparametric.kernel_regression import KernelReg

st.title('Batted Ball Charts')
st.write("These charts compare a hitter's batted ball distribution to the distribution of all MLB batted balls")

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

@st.cache_resource()
def load_logo():
    logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    img_url = urllib.request.urlopen(logo_loc)
    logo = Image.open(img_url)
    return logo
    
logo = load_logo()
st.image(logo, width=200)

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

years = [2025,2024,2023,2022,2021]
year = st.radio('Choose a year:', years)

@st.cache_data(ttl=2*3600,show_spinner=f"Loading {year} data")
def load_data(year):
    pitch_data = pd.read_parquet(f'https://github.com/Blandalytics/PLV_viz/blob/main/hitter_app/pages/batted_ball_df_{year}.parquet?raw=true')
    bbe_df = (
      pitch_data
      # .loc[(pitch_data['spray_deg']>=0) &
      #        (pitch_data['spray_deg']<=90) &
      #        (pitch_data['launch_angle']>=-30) &
      #        (pitch_data['launch_angle']<=60)]
        [['hittername','stand','spray_deg','launch_angle']]
        .astype({'spray_deg':'float',
                 'launch_angle':'float'})
        .dropna(subset=['spray_deg','launch_angle'])
        .copy()
    )
    prior_year = year-1
    prior_data = pd.read_parquet(f'https://github.com/Blandalytics/PLV_viz/blob/main/hitter_app/pages/batted_ball_df_{prior_year}.parquet?raw=true')
    year_before_df = (
      prior_data
      # .loc[(prior_data['spray_deg']>=0) &
      #        (prior_data['spray_deg']<=90) &
      #        (prior_data['launch_angle']>=-30) &
      #        (prior_data['launch_angle']<=60)]
      [['hittername','stand','spray_deg','launch_angle']]
      .astype({'spray_deg':'float',
               'launch_angle':'int'})
      .dropna(subset=['spray_deg','launch_angle'])
      .copy()
      )

    x_loc_league = (
      bbe_df.loc[(bbe_df['spray_deg']>=0) &
      (bbe_df['spray_deg']<=90) &
      (bbe_df['launch_angle']>=-30) &
      (bbe_df['launch_angle']<=60)]
      ['spray_deg']
    )
    y_loc_league = (
      bbe_df
      .loc[(bbe_df['spray_deg']>=0) &
      (bbe_df['spray_deg']<=90) &
      (bbe_df['launch_angle']>=-30) &
      (bbe_df['launch_angle']<=60)]
      ['launch_angle']
    )
    
    xmin = x_loc_league.min()
    xmax = x_loc_league.max()
    ymin = y_loc_league.min()
    ymax = y_loc_league.max()
    
    X, Y = np.mgrid[xmin:xmax:91j, ymin:ymax:91j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # league matrix
    values_league = np.vstack([x_loc_league, y_loc_league])
    kernel_league = sp.stats.gaussian_kde(values_league)
    f_league = np.reshape(kernel_league(positions).T, X.shape)
    f_league = f_league * (100/f_league.sum())
  
    return bbe_df, f_league, year_before_df

bbe_df, f_league, year_before_df = load_data(year)

X, Y = np.mgrid[0:90:91j, -30:60:91j]

col1, col2, col3 = st.columns([0.5,0.25,0.25])

with col1:
    # Player
    players = list(bbe_df
                   .reset_index()
                   .sort_values('hittername')
                   ['hittername'].unique()
                  )
    default_ix = players.index('Juan Soto')
    player = st.selectbox('Choose a player:', players, index=default_ix)
with col2:
    # Color Scale
    color_scales = ['Discrete','Continuous']
    color_scale_type = st.selectbox('Choose a color scale:', color_scales)
with col3:
    # Comparison
    comparisons = ['League','Self (prior year)']
    comparison = st.selectbox('Compared to:', comparisons)
    if comparison=='Self (prior year)':
        comparison = 'Self'

def kde_calc(df,hitter,year=year,league_vals=f_league):
    x_loc_player = (
      df
      .loc[
      (df['spray_deg']>=0) &
      (df['spray_deg']<=90) &
      (df['launch_angle']>=-30) &
      (df['launch_angle']<=60) &
      (df['hittername']==hitter),
      'spray_deg']
    )
    y_loc_player = (
      df
      .loc[
      (df['spray_deg']>=0) &
      (df['spray_deg']<=90) &
      (df['launch_angle']>=-30) &
      (df['launch_angle']<=60) &
      (df['hittername']==hitter),
      'launch_angle']
    )

    xmin = 0
    xmax = 90
    ymin = -30
    ymax = 60

    X, Y = np.mgrid[xmin:xmax:91j, ymin:ymax:91j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # pitcher matrix
    values_player = np.vstack([x_loc_player, y_loc_player])
    kernel_player = sp.stats.gaussian_kde(values_player)
    f_player = np.reshape(kernel_player(positions).T, X.shape)
    f_player = f_player * (100/f_player.sum())

    return f_player - league_vals

def kde_chart(kde_data,hitter,chart_type='Discrete',comparison='League'):
    if (year==2020) & (comparison=='Self'):
        st.write("No data for comparison year (2019).\nPlease select a year above 2020.")
    levels=13
    b_hand = bbe_df.loc[bbe_df['hittername']==hitter,'stand'].value_counts().index[0]
    fig, ax = plt.subplots(figsize=(7,7))
    if color_scale_type=='Discrete':
        cfset = ax.contourf(X, Y, kde_data*1000, list(range(-levels+1,levels-1))[::2], 
                            cmap='vlag',extend='both')
        ax.set(xlim=(0,90) if b_hand=='R' else (90,0),
               xlabel='',
               ylim=(-30,60),
               ylabel='',
               aspect=1)
        ax.axhline(y=10, color='k',linewidth=1,alpha=0.25)
        ax.axhline(y=25, color='k',linewidth=1,alpha=0.25)
        ax.axhline(y=50, color='k',linewidth=1,alpha=0.25)
        ax.axvline(x=30, color='k',linewidth=1,alpha=0.25)
        ax.axvline(x=60, color='k',linewidth=1,alpha=0.25)
    else:
        kde_thresh=0.01
        kde_data = pd.DataFrame(kde_data).T
        sns.heatmap(kde_data,
                    cmap=kde_palette,
                    center=0,
                    vmin=-kde_thresh,
                    vmax=kde_thresh,
                    cbar=False,
                    ax=ax
                   )
        
        ax.set(xlim=(0,90) if b_hand=='R' else (90,0),
           xlabel='',
           ylim=(0,90),
           ylabel='',
           aspect=1)
        ax.axhline(y=40, color='k', linewidth=1, alpha=0.25)
        ax.axhline(y=50, color='k', linewidth=1, alpha=0.25)
        ax.axhline(y=80, color='k', linewidth=1, alpha=0.25)
        ax.axvline(x=30, color='k', linewidth=1, alpha=0.25)
        ax.axvline(x=60, color='k', linewidth=1, alpha=0.25)

    ax.set_xticks([])
    ax.set_yticks([])

    x_ticks = [0,30,60,90]
    x_labels = ['Pull','Center','Oppo']
    # labels at the center of their range
    for label, pos0, pos1 in zip(x_labels, x_ticks[:-1], x_ticks[1:]):
        ax.text((pos0 + pos1) / 2, -0.02, label, ha='center', va='top', 
                fontsize=15, clip_on=False, transform=ax.get_xaxis_transform())
      
    if comparison=='League':
        pull_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['spray_deg']<30)].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        center_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['spray_deg'].between(30,60))].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        oppo_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['spray_deg']>60)].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        x_label_vals = [f'({pull_val:.1%})',
                        f'({center_val:.1%})',
                        f'({oppo_val:.1%})']
        
        for label, pos0, pos1 in zip(x_label_vals, x_ticks[:-1], x_ticks[1:]):
            ax.text((pos0 + pos1) / 2, -0.08, label, ha='center', va='top', 
                    fontsize=10, clip_on=False, transform=ax.get_xaxis_transform())

    y_ticks = [-30,10,25,50,60] if color_scale_type=='Discrete' else [0,40,55,80,90]
    y_labels = ['Ground\nBall','Line Drive','Fly Ball','Pop Up']
    # labels at the center of their range
    for label, pos0, pos1 in zip(y_labels, y_ticks[:-1], y_ticks[1:]):
        ax.text(-0.14, (pos0 + pos1) / 2 + 1, label, ha='center', va='center', 
                fontsize=15, clip_on=False, transform=ax.get_yaxis_transform())
      
    if comparison=='League':
        gb_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['launch_angle']<10)].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        ld_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['launch_angle'].between(10,25,inclusive='left'))].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        fb_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['launch_angle'].between(25,50))].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        pu_val = bbe_df.loc[(bbe_df['hittername']==hitter) & (bbe_df['launch_angle']>50)].shape[0] / bbe_df.loc[bbe_df['hittername']==hitter].shape[0]
        y_label_vals = [f'({gb_val:.1%})',
                        f'({ld_val:.1%})',
                        f'({fb_val:.1%})',
                        f'({pu_val:.1%})']
        
        for label, pos0, pos1 in zip(y_label_vals, y_ticks[:-1], y_ticks[1:]):
            adj_val = 3 if label != y_label_vals[0] else 6
            ax.text(-0.14, (pos0 + pos1) / 2 - adj_val, label, ha='center', va='center', 
                    fontsize=10, clip_on=False, transform=ax.get_yaxis_transform())

    bounds = [x/levels for x in range(levels)]+[1]
    if color_scale_type=='Discrete':
        norm = mpl.colors.BoundaryNorm(bounds, sns.color_palette('vlag', as_cmap=True).N)
    else:
        norm = mpl.colors.CenteredNorm()

    colorbar_scale = 0.8
    sm = plt.cm.ScalarMappable(norm=norm, cmap='vlag')
    sm.set_array([])
    cb = fig.colorbar(sm,
                      ax=[ax],
                      pad=0.02,
                      shrink=colorbar_scale, 
                      aspect=5*colorbar_scale,
                      ticks=[]
                     )
    cb.ax.axis('off')
    label_colors = [sns.color_palette('vlag',n_colors=25)[0],'k',sns.color_palette('vlag',n_colors=25)[-1]]
    for label, deg, color in zip(['Less\nOften','Same','More\nOften'], 
                                 [-24,15,53.5] if color_scale_type=='Discrete' else [6,45,83.5], 
                                 label_colors):
        cb.ax.text(1.115, deg, label, ha='center', va='center', color=color, 
                   fontsize=15, fontweight='medium', clip_on=False, 
                   transform=ax.get_yaxis_transform())

    # Add PL logo
    pl_ax = fig.add_axes([-0.035,0.12,0.15,0.15], anchor='SW', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')

    apostrophe_text = "'" if hitter[-1]=='s' else "'s"
    fig.suptitle(f"{hitter}{apostrophe_text} {year} Batted Ball Profile" if comparison=='League' else f'{hitter}{apostrophe_text} Batted Ball Difference',
                 ha='center',x=0.45,y=0.88,fontsize=16)
    fig.text(0.45,0.827,'(Compared to rest of MLB)' if comparison=='League' else f'({year}, compared to {year-1})',
             ha='center',fontsize=12)
    fig.text(-0.06,0.116,'batted-ball-charts.streamlit.app',ha='left',fontsize=6)
    fig.text(0.83,0.115,'@blandalytics',ha='center',fontsize=10)
    fig.text(-0.065,0.1,'Data: Baseball Savant/pybaseball',ha='left',fontsize=6)

    sns.despine()
    st.pyplot(fig)

if comparison=='Self':
    if year_before_df.loc[year_before_df['hittername']==player].shape[0]==0:
        st.write(f'No data on {player} for {year-1}')
    else:
        xmin = 0
        xmax = 90
        ymin = -30
        ymax = 60
    
        X, Y = np.mgrid[xmin:xmax:91j, ymin:ymax:91j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        x_loc_before = year_before_df.loc[year_before_df['hittername']==player,'spray_deg']
        y_loc_before = year_before_df.loc[year_before_df['hittername']==player,'launch_angle']
    
        # league matrix
        values_before = np.vstack([x_loc_before, y_loc_before])
        kernel_before = sp.stats.gaussian_kde(values_before)
        f_before = np.reshape(kernel_before(positions).T, X.shape)
        f_before = f_before * (100/f_before.sum())
    
        kde_chart(kde_calc(bbe_df,player,
                            league_vals=f_before),
                  player,
                  color_scale_type,
                  comparison)
else:
    kde_chart(kde_calc(bbe_df,player),
              player,
              color_scale_type)
st.write("If you have questions or ideas on what you'd like to see, DM me! [@Blandalytics](https://twitter.com/blandalytics)")
