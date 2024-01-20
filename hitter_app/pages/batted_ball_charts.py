import streamlit as st
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
st.write("Charts compare a hitter's batted ball distribution against the MLB distribution")

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

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))

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

years = [2023,2022,2021,2020]
year = st.radio('Choose a year:', years)

@st.cache_data(ttl=2*3600,show_spinner=f"Loading {year} data")
def load_data(year):
    pitch_data = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/hitter_app/pages/batted_ball_df.csv?raw=true', encoding='latin1')
    return (
      pitch_data
      .loc[(pitch_data['spray_deg']>=0) &
             (pitch_data['spray_deg']<=90) &
             (pitch_data['launch_angle']>=-30) &
             (pitch_data['launch_angle']<=60) &
             (pitch_data['game_year']==year)]
        [['hittername','stand','spray_deg','launch_angle']]
        .astype({'spray_deg':'float',
                 'launch_angle':'float'})
        .dropna(subset=['spray_deg','launch_angle'])
        .copy()
    )

bbe_df = load_data(year)

players = list(bbe_df
               .reset_index()
               .sort_values('hittername')
               ['hittername'].unique()
              )
default_ix = players.index('Ronald Acuña Jr.')
player = st.selectbox('Choose a player:', players, index=default_ix)

x_loc_league = bbe_df['spray_deg']
y_loc_league = bbe_df['launch_angle']

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

def kde_calc(df,hitter,year=year,league_vals=f_league):
    x_loc_player = df.loc[df['hittername']==hitter,'spray_deg']
    y_loc_player = df.loc[df['hittername']==hitter,'launch_angle']

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

def kde_chart(kde_data,hitter,levels=13):
    b_hand = bbe_df.loc[bbe_df['hittername']==hitter,'stand'].value_counts().index[0]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, 90)
    ax.set_ylim(-30, 60)
    cfset = ax.contourf(X, Y, kde_data*1000, list(range(-levels+1,levels-1))[::2], 
                        cmap='vlag',extend='both')
    ax.set(xlim=(0,90) if b_hand=='R' else (90,0),
           xlabel='',
           ylim=(-30,60),
           ylabel='',
           aspect=1)
    ax.axhline(y=10, color='k',linewidth=1,alpha=0.25)
    ax.axhline(y=20, color='k',linewidth=1,alpha=0.25)
    ax.axhline(y=50, color='k',linewidth=1,alpha=0.25)
    ax.axvline(x=30, color='k',linewidth=1,alpha=0.25)
    ax.axvline(x=60, color='k',linewidth=1,alpha=0.25)

    ax.set_xticks([])
    ax.set_yticks([])

    x_ticks = [0,30,60,90]
    x_labels = ['Pull','Center','Oppo']
    # labels at the center of their range
    for label, pos0, pos1 in zip(x_labels, x_ticks[:-1], x_ticks[1:]):
        ax.text((pos0 + pos1) / 2, -0.02, label, ha='center', va='top', 
                fontsize=14, clip_on=False, transform=ax.get_xaxis_transform())

    y_ticks = [-30,10,20,50,60]
    y_labels = ['Ground\nBall','Line Drive','Fly Ball','Pop Up']
    # labels at the center of their range
    for label, pos0, pos1 in zip(y_labels, y_ticks[:-1], y_ticks[1:]):
        ax.text(-0.14, (pos0 + pos1) / 2, label, ha='center', va='center', 
                fontsize=14, clip_on=False, transform=ax.get_yaxis_transform())

    bounds = [x/levels for x in range(levels)]+[1]
    norm = mpl.colors.BoundaryNorm(bounds, sns.color_palette('vlag', as_cmap=True).N)

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
    for label, deg, color in zip(['Less\nOften','Same','More\nOften'], [-24,15,53.5], label_colors):
        cb.ax.text(1.115, deg, label, ha='center', va='center', color=color, 
                   fontsize=15, fontweight='medium', clip_on=False, 
                   transform=ax.get_yaxis_transform())

    # Add PL logo
    pl_ax = fig.add_axes([-0.05,0.09,0.2,0.2], anchor='SW', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')

    apostrophe_text = "'" if card_player[-1]=='s' else "'s"
    fig.suptitle(f"{hitter}{apostrophe_text}\n{year} Batted Ball Profile",ha='center',x=0.45,y=0.95,fontsize=18)
    fig.text(0.45,0.84,'(Compared to rest of MLB)',ha='center',fontsize=12)

    sns.despine()
    st.pyplot(fig)

kde_chart(kde_calc(bbe_df,player),player)
