import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

st.title("PLV Distributions")

# Load Data
def load_data():
    file_name = r'https://github.com/Blandalytics/PLV_viz/blob/main/2020-2022_PLV.parquet?raw=true'
    df = pd.read_parquet(file_name).sort_values('pitch_id')
    return df
plv_df = load_data()

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
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

## Selectors
# Player
players = list(plv_df['pitchername'].unique())
default_ix = players.index('Sandy Alcantara')
player = st.selectbox('Choose a player:', players, index=default_ix)

# Year
years = plv_df.loc[plv_df['pitchername']==player,'year_played'].sort_values(ascending=False).unique()
year = st.radio('Choose a year:', years)

pitch_threshold = 200
pitches_thrown = plv_df.loc[(plv_df['pitchername']==player) &
                            (plv_df['year_played']==year)].shape[0]
st.write('Pitches Thrown: {:,}'.format(pitches_thrown))
if pitches_thrown >= pitch_threshold:
    pitch_list = list(plv_df
                .loc[(plv_df['year_played']==year) &
                    (plv_df['pitchername']==player)]
                .groupby('pitchtype',as_index=False)
                ['pitch_id']
                .count()
                .dropna()
                .sort_values('pitch_id', ascending=False)
                .query('pitch_id > 50')
                ['pitchtype']
                )

    def arsenal_dist():
        fig, axs = plt.subplots(len(pitch_list),1,figsize=(8,8), sharex='row', sharey='row', constrained_layout=True)
        ax_num = 0
        max_count = 0
        for pitch in pitch_list:
            chart_data = plv_df.loc[(plv_df['year_played']==year) &
                                    (plv_df['pitchtype']==pitch)].copy()
            chart_data['PLV'] = np.clip(chart_data['PLV'], a_min=0, a_max=10)
            num_pitches = chart_data.loc[chart_data['pitchername']==player].shape[0]

            sns.histplot(data=chart_data.loc[chart_data['pitchername']==player],
                        x='PLV',
                        hue='pitchtype',
                        palette=marker_colors,
                        binwidth=0.5,
                        binrange=(0,10),
                        alpha=1,
                        ax=axs[ax_num],
                        legend=False
                        )

            axs[ax_num].axvline(chart_data.loc[chart_data['pitchername']==player]['PLV'].mean(),
                                color=marker_colors[pitch],
                                linestyle='--',
                                linewidth=2.5)
            axs[ax_num].axvline(chart_data['PLV'].mean(), 
                                color='w', 
                                label='Lg. Avg.',
                                alpha=0.5)
            axs[ax_num].get_xaxis().set_visible(False)
            axs[ax_num].get_yaxis().set_visible(False)
            axs[ax_num].set(xlim=(0,10))
            axs[ax_num].set_title(None)
            if axs[ax_num].get_ylim()[1] > max_count:
                max_count = axs[ax_num].get_ylim()[1]
            ax_num += 1
            if ax_num==len(pitch_list):
                axs[ax_num-1].get_xaxis().set_visible(True)
                axs[ax_num-1].set_xticks(range(0,11))
                axs[ax_num-1].set(xlabel='')

        for axis in range(len(pitch_list)):
            axs[axis].set(ylim=(0,max_count*1.025))
            axs[axis].legend([pitch_names[pitch_list[axis]]+' - {:.3}'.format(plv_df.loc[(plv_df['pitchername']==player) &
                                                                                       (plv_df['year_played']==year) &
                                                                                       (plv_df['pitchtype']==pitch_list[axis]),'PLV'].mean()),
                              'Lg. Avg.'], 
                             edgecolor='#162B50', loc=(0,0.4), fontsize=14)
            axs[axis].text(9,max_count*0.425,'{:,}\nPitches'.format(plv_df.loc[(plv_df['pitchername']==player) &
                                                                               (plv_df['year_played']==year) &
                                                                               (plv_df['pitchtype']==pitch_list[axis])].shape[0]),
                           ha='center',va='bottom', fontsize=14)

        fig.suptitle("{}'s {} PLV Distributions".format(player,year),fontsize=16)
        sns.despine(left=True)
        st.pyplot(fig)
    arsenal_dist()
else:
    st.write('Not enough pitches thrown in {} (<{})'.format(year,pitch_threshold))
