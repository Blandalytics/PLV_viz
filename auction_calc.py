import streamlit as st
import pandas as pd
import numpy as np

import urllib
from PIL import Image

st.set_page_config(page_title='PL Auction Draft Calculator', page_icon='📊',layout="wide")

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=400)

st.title('PL Auction Draft Calculator')

# Settings
st.header('Team Settings')
col1, col2, col3, col4 = st.columns(4)
with col1:
    num_hitters = st.number_input('Starting hitters:',min_value=4,max_value=20,value=10)
with col2:
    num_pitchers = st.number_input('Starting pitchers:',min_value=4,max_value=20,value=8)
with col3:
    raw_bench = st.number_input('Number of bench spots:',min_value=0,max_value=20,value=5)
with col4:
    num_catchers = st.number_input('Starting catchers:',min_value=0,max_value=3,value=1)

col1, col2, col3, col4 = st.columns(4)
with col3:
    bench_suppress = st.checkbox("Minimize bench value",value=True,
                                 help="""
                                 Does not consider bench players  
                                 when calculating replacement level
                                 """)
    num_bench = 1 if bench_suppress else raw_bench


st.header('League Settings')
col1, col2, col3, col4 = st.columns(4)
with col1:
    num_teams = st.number_input('Number of teams:',min_value=4,max_value=30,value=12)
with col2:
    min_bid = st.number_input('Minimum bid:',min_value=0,value=1)
with col3:
    team_budget = st.number_input('Per-Team Budget:',min_value=(min_bid+1)*(num_hitters+num_pitchers+num_bench),value=260)
with col4:
    hitter_split = st.number_input('Hitter Split of budget (%):',min_value=0.,max_value=100.,value=65.0, format="%0.1f")
    hitter_split = hitter_split/100

st.header('Scoring Categories')
col1, col2 = st.columns(2)
with col1:
    hitter_cats = st.multiselect('Hitter categories',
                                 ['G', 'AB','PA', 'R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'ISO', 'SLG', 'OPS',
                                  'wOBA', 'BB%', 'K%', 'H', '1B', '2B', '3B', 'XBH',
                                  'TB', 'K', 'BB', 'HBP', 'SF', 'CS'],
                                 default=['R','HR','RBI','SB','AVG'])
    rate_cats_h = ['AVG','OBP','ISO','SLG','OPS','wOBA','BB%','K%']
    rate_scoring_cats_h = [x for x in hitter_cats if x in rate_cats_h]
    volume_scoring_cats_h = [x for x in hitter_cats if x not in rate_scoring_cats_h]
    inverted_categories_h = ['K','CS','SF']
with col2:
    pitcher_cats = st.multiselect('Pitcher categories',
                                  ['IP', 'TBF','G', 'GS', 'W', 'L', 'QS', 'SV', 'HD', 'SV+H', 'K', 'ERA', 
                                   'WHIP','K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'HR/9', 'H', 'ER', 'HBP',
                                   'HR', 'BB', 'BS','K/BB'],
                                  default=['W','SV','K','ERA','WHIP'])
    rate_cats_p = ['ERA', 'WHIP','K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'HR/9']
    rate_scoring_cats_p = [x for x in pitcher_cats if x in rate_cats_p]
    volume_scoring_cats_p = [x for x in pitcher_cats if x not in rate_scoring_cats_p]
    inverted_categories_p = ['BB','H','ER','BS','ERA','WHIP','L','HBP','HR','BB/9','HR/9','BB%']
  
# Values derived from settings
hitters_above_replacement = int(round(num_teams * (num_hitters + num_bench/2) * 1.1,0))
pitchers_above_replacement = int(round(num_teams * (num_pitchers + num_bench/2) * 1.1,0))
non_replacement_dollars = (num_teams * team_budget) - (num_teams * (num_hitters + num_pitchers + raw_bench) * min_bid)
total_hitter_dollars = non_replacement_dollars * hitter_split
total_pitcher_dollars = non_replacement_dollars * (1-hitter_split)

# Value functions
def volume_z_score(feat_col, population, sample):
    return (population[feat_col] - sample[feat_col].mean())/sample[feat_col].std()

def rate_z_score(feat_col, playing_time_col, population, sample, n_players):
    population_playing_time = population[playing_time_col]
    sample_playing_time = sample[playing_time_col]
    league_playing_time = sample[playing_time_col].mean()

    population_volume = population[feat_col].mul(population_playing_time)
    sample_volume = sample[feat_col].mul(sample_playing_time)
    league_volume = sample[feat_col].mean() * league_playing_time

    sample_val = (sample_volume + (n_players-1) * league_volume) / (sample_playing_time + (n_players-1) * league_playing_time)
    population_val = (population_volume + (n_players-1) * league_volume) / (population_playing_time + (n_players-1) * league_playing_time)

    return (population_val - sample_val.mean())/sample_val.std()

def unadjusted_value(position_df,rate_stats,volume_stats,invert_stats,sample_pop,num_players,pos='h'):
    position_df[[x+'_val' for x in volume_stats]] = volume_z_score(volume_stats,
                                                                   position_df,
                                                                   sample_pop)
    # Z-Scores for Rate Stats
    for x in rate_stats:
        position_df[x+'_val'] = rate_z_score(x,'PA' if pos=='h' else 'IP',
                                             position_df,
                                             sample_pop,
                                             num_players)
    
    # Invert relevant scoring categories
    for x in list(set(rate_stats+volume_stats) & set(invert_stats)):
        position_df[x+'_val'] = position_df[x+'_val'].mul(-1)
      
    # Total unadjusted Value: sum of all scoring category Z-Scores
    return position_df[[x+'_val' for x in rate_stats+volume_stats]].sum(axis=1)

# Load projections
projections_hitters = pd.read_csv('https://docs.google.com/spreadsheets/d/1nnH9bABVxgD28KVj9Oa67bn9Kp5x2dD0nFiZ7jIfvmQ/export?gid=1029181665&format=csv')
projections_pitchers = pd.read_csv('https://docs.google.com/spreadsheets/d/1nnH9bABVxgD28KVj9Oa67bn9Kp5x2dD0nFiZ7jIfvmQ/export?gid=354379391&format=csv')
projections_pitchers['K/BB'] = projections_pitchers['K'].div(np.clip(projections_pitchers['BB'],1,1000))

if st.button("Generate Auction Values:  📊 -> 💲"):
    st.header('Auction Values')
    ## Hitters
    sample_hitters  = projections_hitters.nlargest(hitters_above_replacement, 'PA')
    projections_hitters['unadjusted_value'] = unadjusted_value(projections_hitters,
                                                               rate_scoring_cats_h,
                                                               volume_scoring_cats_h,
                                                               inverted_categories_h,
                                                               sample_hitters,num_hitters)
    
    projections_hitters['is_C'] = projections_hitters['Y! Pos'].fillna('UT').str.replace('CF','').str.contains('C')
    c_adj = projections_hitters.loc[projections_hitters['is_C'],'unadjusted_value'].nlargest(num_teams * num_catchers).min()
    non_c_adj = projections_hitters.loc[~projections_hitters['is_C'],'unadjusted_value'].nlargest(int(num_teams * (num_hitters - num_catchers + num_bench/2))).min()
    projections_hitters['ADJ'] = np.where(projections_hitters['is_C'],c_adj,non_c_adj)
    projections_hitters['adjusted_value'] = projections_hitters['unadjusted_value'].sub(projections_hitters['ADJ'])
    # Convert hitter value to Dollars 
    total_hitter_value = projections_hitters.loc[projections_hitters['adjusted_value']>0,'adjusted_value'].sum()
    hitter_dollars_per_value = total_hitter_dollars / total_hitter_value
    
    
    ## Pitchers
    ip_thresh = min(50,projections_pitchers['IP'].nlargest(num_teams * num_pitchers).min())
    sample_pitchers  = projections_pitchers.loc[projections_pitchers['IP'] >= ip_thresh]
    projections_pitchers['unadjusted_value'] = unadjusted_value(projections_pitchers,
                                                                rate_scoring_cats_p,
                                                                volume_scoring_cats_p,
                                                                inverted_categories_p,
                                                                sample_pitchers,
                                                                num_pitchers,
                                                                pos='p')
    
    projections_pitchers['ADJ'] = projections_pitchers['unadjusted_value'].nlargest(int(num_teams * (num_pitchers + num_bench/2))).min()
    projections_pitchers['adjusted_value'] = projections_pitchers['unadjusted_value'].sub(projections_pitchers['ADJ'])
    # Convert hitter value to Dollars 
    total_pitcher_value = projections_pitchers.loc[projections_pitchers['adjusted_value']>0,'adjusted_value'].sum()
    pitcher_dollars_per_value = total_pitcher_dollars / total_pitcher_value
    
    # Merge position dfs
    combined_value_df = (
        pd.concat(
            [
                projections_hitters[['Player','MLBAMID','Y! Pos','PA']+[x for x in hitter_cats if x!='PA']+['adjusted_value']],
                projections_pitchers.rename(columns={'Name':'Player'})[['Player','MLBAMID','IP']+[x for x  in pitcher_cats if x!='IP']+['adjusted_value']]
                ],
            ignore_index=True)
        [['Player','MLBAMID','Y! Pos','adjusted_value','PA']+[x for x  in hitter_cats if x!='PA']+['IP']+[x for x  in pitcher_cats if x!='IP']]
    )
    combined_value_df['Y! Pos'] = combined_value_df['Y! Pos'].fillna('P')
    combined_value_df['Auction $'] = min_bid + np.where(
        combined_value_df['Y! Pos']=='P',
        combined_value_df['adjusted_value'].mul(pitcher_dollars_per_value),
        combined_value_df['adjusted_value'].mul(hitter_dollars_per_value)
    )
    
    # Level the dollars out
    projected_auction_dollars = combined_value_df.loc[combined_value_df['Auction $']>0,'Auction $'].sum()
    fudge_factor = (num_teams * team_budget) / projected_auction_dollars
    combined_value_df['Auction $'] = combined_value_df['Auction $'].mul(fudge_factor)
    combined_value_df['Rank'] = combined_value_df['Auction $'].rank(ascending=False)
    st.dataframe(combined_value_df[['Rank','Player','Y! Pos','Auction $','PA']+[x for x  in hitter_cats if x!='PA']+['IP']+[x for x  in pitcher_cats if x!='IP']]
                 .sort_values('Auction $',ascending=False),
                 # .fillna('')
                 # .style
                 # .map(lambda x: 'color: transparent; background-color: transparent' if x==0 else ''),
                 use_container_width=True,
                 hide_index=True,
                 #height=(25 + 1) * 35 + 3,
                 column_config={
                         "Auction $": st.column_config.NumberColumn(
                             format="$ %.2f",
                             ),
                 }
                 )
