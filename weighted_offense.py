import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

team_map = {
    'Arizona Diamondbacks':'ARI', 
    'Atlanta Braves':'ATL',
    'Baltimore Orioles':'BAL', 
    'Boston Red Sox':'BOS', 
    'Chicago Cubs':'CHC', 
    'Chicago White Sox':'CHW', 
    'Cincinnati Reds':'CIN', 
    'Cleveland Guardians':'CLE', 
    'Cleveland Indians':'CLE', 
    'Colorado Rockies':'COL', 
    'Detroit Tigers':'DET', 
    'Houston Astros':'HOU', 
    'Kansas City Royals':'KCR', 
    'Los Angeles Angels':'LAA', 
    'Los Angeles Dodgers':'LAD', 
    'Miami Marlins':'MIA', 
    'Milwaukee Brewers':'MIL', 
    'Minnesota Twins':'MIN', 
    'New York Mets':'NYM',
    'New York Yankees':'NYY',
    'Oakland Athletics':'OAK',
    'Philadelphia Phillies':'PHI',
    'Pittsburgh Pirates':'PIT',
    'San Diego Padres':'SDP',
    'San Francisco Giants':'SFG',
    'Seattle Mariners':'SEA',
    'St. Louis Cardinals':'STL',
    'Tampa Bay Rays':'TBR',
    'Texas Rangers':'TEX',
    'Toronto Blue Jays':'TOR',
    'Washington Nationals':'WSH'
}
# col1, col2, col3 = st.columns([0.2,0.6,0.2])

# with col1:
#     st.write(' ')

# with col2:
st.title('MLB Offense Ranks')

pa_df = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/2023_PAs.csv?raw=true')
pa_df['game_played'] = pd.to_datetime(pa_df['game_played'])

stat_dict = {
    'wOBA (Actual Results)':'wOBA',
    'Hitter Performance (Context Adjusted)':'hitter_perf'
}
stat_string = st.radio('Choose a measurement:', list(stat_dict.keys()))
stat = stat_dict[stat_string]

time_string = st.radio('Choose a time frame:', ['Season','Last 60 Days','Last 30 Days','Last 15 Days'])
time_thresh = 365 if time_string=='Season' else int(time_string[5:-5])
time_df = pa_df[pa_df['game_played'] > (pa_df['game_played'].max() - pd.Timedelta(days=time_thresh))].copy()

# @st.cache_data(ttl=12*3600)
def calc_wOBA_ranks(df=time_df,time_frame='Season', stat='wOBA'):   
    thresh= 0.075 if stat=='wOBA' else 0.02
    thresh = thresh if df.shape[0] >= 60000 else thresh*1.75 if df.shape[0] >= 45000 else thresh*2 if df.shape[0] >= 30000 else thresh*3 if df.shape[0] >= 15000 else thresh*4

    test_df = pd.DataFrame(index=df['hitterteam'].sort_values().unique())
    test_df['wOBA'] = df.groupby('hitterteam')[stat].mean()
    test_df['wOBA_lhp'] = df.loc[df['pitcherside']=='L'].groupby('hitterteam')[stat].mean()
    test_df['wOBA_rhp'] = df.loc[df['pitcherside']=='R'].groupby('hitterteam')[stat].mean()
    test_df['wOBA_home'] = df.loc[df['is_home']==1].groupby('hitterteam')[stat].mean()
    test_df['wOBA_away'] = df.loc[df['is_home']==0].groupby('hitterteam')[stat].mean()
    test_df['wOBA_lhp_home'] = df.loc[(df['pitcherside']=='L') & (df['is_home']==1)].groupby('hitterteam')[stat].mean()
    test_df['wOBA_lhp_away'] = df.loc[(df['pitcherside']=='L') & (df['is_home']==0)].groupby('hitterteam')[stat].mean()
    test_df['wOBA_rhp_home'] = df.loc[(df['pitcherside']=='R') & (df['is_home']==1)].groupby('hitterteam')[stat].mean()
    test_df['wOBA_rhp_away'] = df.loc[(df['pitcherside']=='R') & (df['is_home']==0)].groupby('hitterteam')[stat].mean()
    if stat=='wOBA':
        test_df['hand_stdev'] = test_df[['wOBA_lhp','wOBA_rhp']].std(axis=1).div(test_df['wOBA'])
        test_df['location_stdev'] = test_df[['wOBA_home','wOBA_away']].std(axis=1).div(test_df['wOBA'])
    else:
        test_df['hand_stdev'] = test_df[['wOBA_lhp','wOBA_rhp']].std(axis=1)
        test_df['location_stdev'] = test_df[['wOBA_home','wOBA_away']].std(axis=1)
    test_df['val'] = 'season'
    test_df.loc[test_df['hand_stdev']>=thresh,'val'] = 'hand'
    test_df.loc[test_df['location_stdev']>=thresh,'val'] = 'location'
    test_df.loc[(test_df['hand_stdev']>=thresh) & 
                   (test_df['location_stdev']>=thresh),'val'] = 'hand_location'
    test_df = test_df.reset_index().replace(team_map)

    test_ranks = {}
    for team in test_df['index']:
        subset = test_df.loc[test_df['index']==team,'val'].item()
        if subset=='season':
            test_ranks.update({team:test_df.loc[test_df['index']==team,'wOBA'].item()})   
        elif subset=='location':
            test_ranks.update({team+' (Home)':test_df.loc[test_df['index']==team,'wOBA_home'].item()})
            test_ranks.update({team+' (Away)':test_df.loc[test_df['index']==team,'wOBA_away'].item()})
        elif subset=='hand':
            test_ranks.update({team+' (vs RHP)':test_df.loc[test_df['index']==team,'wOBA_rhp'].item()})
            test_ranks.update({team+' (vs LHP)':test_df.loc[test_df['index']==team,'wOBA_lhp'].item()})
        else:
            test_ranks.update({team+' (Home; vs RHP)':test_df.loc[test_df['index']==team,'wOBA_rhp_home'].item()})
            test_ranks.update({team+' (Home; vs LHP)':test_df.loc[test_df['index']==team,'wOBA_lhp_home'].item()})
            test_ranks.update({team+' (Away; vs RHP)':test_df.loc[test_df['index']==team,'wOBA_rhp_away'].item()})
            test_ranks.update({team+' (Away; vs LHP)':test_df.loc[test_df['index']==team,'wOBA_lhp_away'].item()})

    weighted_test_df = pd.DataFrame.from_dict(test_ranks, orient='index').rename(columns={0:'wOBA'}).sort_values('wOBA',ascending=False)
    weighted_test_df['z_wOBA'] = weighted_test_df['wOBA'].sub(weighted_test_df['wOBA'].mean()).div(weighted_test_df['wOBA'].std())
    offense_tiers = ['Poor', 'Weak', 'Average', 'Solid', 'Top']
    weighted_test_df['bucket'] = pd.cut(weighted_test_df['wOBA'].rank(), 5,labels=offense_tiers)
    weighted_test_df['count'] = weighted_test_df.groupby((weighted_test_df['bucket'] != weighted_test_df['bucket'].shift(1)).cumsum()).cumcount()+1
    
    return pd.pivot(weighted_test_df.reset_index().round(3).astype({'wOBA':'str'}),
                    values='index', index=['count'], columns=['bucket'])[offense_tiers[::-1]]

rank_df = calc_wOBA_ranks(df=time_df,time_frame=time_string, stat=stat)

st.dataframe(rank_df
             .style
             .set_properties(**{'color': 'black'})
             .set_properties(**{'background-color': '#eda1a1'}, subset='Top')
             .set_properties(**{'background-color': '#f9dddc'}, subset='Solid')
             .set_properties(**{'background-color': '#e6dbcf'}, subset='Average')
             .set_properties(**{'background-color': '#e2f3e3'}, subset='Weak')
             .set_properties(**{'background-color': '#acdcb2'}, subset='Poor'),
             # .format(precision=4)
             # .background_gradient(axis=0,gmap=(rank_df['wOBA']-time_df['wOBA'].mean())/time_df.groupby('hitterteam')['wOBA'].mean().std(), 
             #                      vmin=-2,vmax=2.5,
             #                      cmap='vlag'),
              width=600,
              height=300,
             hide_index=True
            )

# with col3:
#     st.write(' ')
