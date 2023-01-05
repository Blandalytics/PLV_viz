import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import data
plv_data = pd.read_csv('2020-2022 PLV.csv', encoding='latin1')

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

# Color Style
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

def arsenal_dist(player,year):
  pitch_list = list(plv_data
                    .loc[(plv_data['year_played']==year) &
                        (plv_data['pitchername']==player)]
                    .groupby('pitchtype',as_index=False)
                    ['pitch_id']
                    .count()
                    .dropna()
                    .sort_values('pitch_id', ascending=False)
                    .query('pitch_id > 50')
                    ['pitchtype']
                    )

  fig, axs = plt.subplots(len(pitch_list),1,figsize=(8,8), sharex='row', sharey='row', constrained_layout=True)
  ax_num = 0
  max_count = 0
  for pitch in pitch_list:
    chart_data = plv_data.loc[(plv_data['year_played']==year) &
                            (plv_data['pitchtype']==pitch)].copy()
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
    axs[ax_num].legend([pitch,'Lg. Avg.'], 
                       edgecolor='w', loc='upper left', fontsize=14)
    axs[ax_num].text(8.5,max_count*0.75,'{:,}\nPitches'.format(num_pitches),
                     ha='center',va='center', fontsize=14)
    ax_num += 1
    if ax_num==len(pitch_list):
      axs[ax_num-1].get_xaxis().set_visible(True)
      axs[ax_num-1].set_xticks(range(0,11))
      axs[ax_num-1].set(xlabel='')

  for axis in range(len(pitch_list)):
    axs[axis].set(ylim=(0,max_count))

  fig.suptitle("{}'s {} PLV Distributions".format(player,year),fontsize=16)
  sns.despine(left=True)
