import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import urllib

from PIL import Image
import os

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

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
st.set_page_config(page_title='Hitter Skill Similarities', page_icon='📊',layout='wide')
st.title("Hitter Skill Similarities")

@st.cache_data(ttl=2*3600,show_spinner=f"Loading similarity data")
def load_data():
    combined_df = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/data/hitter_similarity_distances.parquet?raw=true')
    sim_df = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/data/hitter_similarity_metrics.parquet?raw=true')
    similarity_stats = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/data/hitter_similarity_stats.parquet?raw=true')
    return combined_df, sim_df, similarity_stats

combined_df, sim_df, similarity_stats = load_data()

def generate_comp_values(season_id,distance_df=combined_df,season_df=sim_df,similarity_stats=similarity_stats,top=True):
    top_comps = distance_df.loc[season_id].astype('float').fillna(100).sort_values(ascending=False)
    comp_players = []
    for player in list(top_comps.index):
        comp_players += [season_df.loc[season_df['season_id']==player,['Name','player_season','season_id','Pitches','usage_vsR','process_vsR','process_vsL','Aggression','zDV','oDV','Contact','Power']]]
    comp_df = pd.merge(pd.concat(comp_players),top_comps,how='inner',left_on='season_id',right_index=True).rename(columns={season_id:'Sim Score'})
    comp_df[['Season','MLBAMID']] = comp_df['season_id'].str.split('_', n=1, expand=True).astype('int')
    sim_player_season = f"{sim_season} {comp_df.iloc[0]['Name']}"
    
    sim_stats = pd.merge(
        comp_df.loc[comp_df['season_id']!=id_season].head(10),
        similarity_stats,
        how='inner',
        on=['Season','MLBAMID']
    )
    for stat in ['HR_600','ISO','BB%','K%','AVG','OBP','SLG','wOBA','xwOBA','wRC+']:
        sim_stats[stat] = sim_stats[stat].mul(sim_stats['Sim Score'].pow(2)).div(sim_stats['Sim Score'].pow(2).sum())
    
    player_stats = (
        similarity_stats
        .loc[(similarity_stats['Season']==int(season_id[:4])) & (similarity_stats['MLBAMID']==int(season_id[-6:])),
                ['HR_600','ISO','BB%','K%','AVG','OBP','SLG','wOBA','xwOBA','wRC+']]
        .to_dict(orient='records')
        [0]
    )
    sim_stats = sim_stats[['HR_600','BB%','K%','AVG','OBP','SLG','wOBA','wRC+']].sum()

    if top:
        comp_df = pd.concat([comp_df.iloc[[0]],comp_df.loc[comp_df['MLBAMID']!=int(season_id[-6:])].head(5)])
    else:
        comp_df = pd.concat([comp_df.iloc[[0]],comp_df.iloc[-5:]])

    return player_stats, sim_stats, comp_df


player_list = sim_df.sort_values('Process',ascending=False)['Name'].unique()
pad1, col1, col2, pad2 = st.columns([0.2,0.3,0.3,0.2])
with col1:
    player_name = st.selectbox('Select a player',
                               player_list, 
                               index=1)
    sim_player_id = sim_df.loc[sim_df['Name']==player_name,'season_id'].str[-6:].unique()
    if len(mlbamid)>1:
        sim_player_id = st.selectbox('Select an MLBAMID',
                                     sim_player_id, 
                                     index=0)
    else:
        sim_player_id = sim_player_id[0]
with col2:
    sim_season = st.selectbox('Select a season',
                              sim_df.loc[sim_df['season_id'].str[-6:]==sim_player_id,'season_id'].str[:4].sort_values(ascending=False).unique(),
                              index=0)
# sim_season = int(sim_season)
# sim_player_id = int(sim_player_id)
season_id = f'{sim_season}_{sim_player_id}'

player_stats, sim_stats, top_comps = generate_comp_values(season_id)

def generate_comp_card(player_stats, sim_stats, top_comps,top=True):
    fig, axs = plt.subplots(1,2,figsize=(10,5),width_ratios=[5,1])
    
    chart_df = (
        top_comps
        [[
            'Name','player_season','Sim Score',
            'Aggression','zDV','oDV','Contact','Power']]
        .round(0)
        .astype({
            'Aggression':'int',
            'zDV':'int',
            'oDV':'int',
            'Contact':'int',
            'Power':'int',
            'Sim Score':'int'
        })
        .assign(label_text = lambda x: x['player_season'].astype('str')+': '+x['Sim Score'].astype('str'))
        .sort_values('Sim Score',ascending=False)
        .round(1)
        .reset_index()
        .melt(id_vars=['label_text'],
              value_vars=['Aggression','zDV','oDV','Contact','Power'])
    )
    
    sns.barplot(chart_df,
                x='variable',y='value',hue='label_text',palette='Set1',
               edgecolor=pl_background,linewidth=1, alpha=1,ax=axs[0]
               )
    axs[0].axhline(100,alpha=1,linestyle='--',color=pl_white,xmax=0.98)
    y_diff_max = max(abs(chart_df['value'].max()-100),abs(100-chart_df['value'].min()))+2
    axs[0].set(xlabel='',ylabel='',
          # ylim=(100-y_diff_max,100+y_diff_max)
          ylim=(25,175)
          )
    axs[0].set_xticks(range(5))
    axs[0].set_xticklabels(['Aggression+','In-Zone\nDecision Value+','Out-of-Zone\nDecision Value+','Contact+','Power+'],fontsize=9)
    axs[0].set_yticks([50,75,100,125,150])
    handles, labels = axs[0].get_legend_handles_labels() 
      
    # specify order 
    order = [0,3,1,4,2,5] 
    
    # pass handle & labels lists along with order as below 
    axs[0].legend()
    least_text = '' if top else ' (Least)'
    legend = axs[0].legend([handles[i] for i in order], [labels[i] for i in order],
                        ncol=3,
                        loc='upper center',
                        bbox_to_anchor=(0.1,0.925,1.125,0.2),
                        title='',
                        fontsize=11,
                        framealpha=1,
                        edgecolor=pl_background)
    legend._legend_box.sep = 10
    
    axs[1].text(0.3,0.875,'Results',ha='center',va='center',fontsize=16)
    axs[1].text(1.1,0.875,'Top-10\nComps',ha='center',va='center',fontsize=13)
    axs[1].text(-0.05,0.775,'AVG:',ha='right',va='center')
    axs[1].text(0.3,0.775,f'{player_stats['AVG']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(1.1,0.775,f'{sim_stats['AVG']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(-0.05,0.675,'OBP:',ha='right',va='center')
    axs[1].text(0.3,0.675,f'{player_stats['OBP']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(1.1,0.675,f'{sim_stats['OBP']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(-0.05,0.575,'SLG:',ha='right',va='center')
    axs[1].text(0.3,0.575,f'{player_stats['SLG']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(1.1,0.575,f'{sim_stats['SLG']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(-0.05,0.475,'wOBA:',ha='right',va='center')
    axs[1].text(0.3,0.475,f'{player_stats['wOBA']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(1.1,0.475,f'{sim_stats['wOBA']:.3f}'.lstrip('0'),ha='center',va='center')
    axs[1].text(-0.05,0.375,'HR/600:',ha='right',va='center')
    axs[1].text(0.3,0.375,f'{player_stats['HR_600']:.1f}',ha='center',va='center')
    axs[1].text(1.1,0.375,f'{sim_stats['HR_600']:.1f}',ha='center',va='center')
    axs[1].text(-0.05,0.275,'BB%:',ha='right',va='center')
    axs[1].text(0.3,0.275,f'{player_stats['BB%']*100:.1f}',ha='center',va='center')
    axs[1].text(1.1,0.275,f'{sim_stats['BB%']*100:.1f}',ha='center',va='center')
    axs[1].text(-0.05,0.175,'K%:',ha='right',va='center')
    axs[1].text(0.3,0.175,f'{player_stats['K%']*100:.1f}',ha='center',va='center')
    axs[1].text(1.1,0.175,f'{sim_stats['K%']*100:.1f}',ha='center',va='center')
    axs[1].text(-0.05,0.075,'wRC+:',ha='right',va='center')
    axs[1].text(0.3,0.075,f'{player_stats['wRC+']:.0f}',ha='center',va='center')
    axs[1].text(1.1,0.075,f'{sim_stats['wRC+']:.0f}',ha='center',va='center')
    
    axs[1].axis('off')
    
    fig.suptitle(f"{top_comps.iloc[0]['Name']}'s {sim_season} Skill Similarity Scores{least_text}",y=1.05,fontsize=20)
    # Add PL logo
    pl_ax = fig.add_axes([0.8,-0.01,0.2,0.2], anchor='SE', zorder=1)
    pl_ax.imshow(logo)
    pl_ax.axis('off')
    
    sns.despine(left=True,bottom=True)
    st.pyplot(fig)
generate_comp_card(player_stats, sim_stats, top_comps)
