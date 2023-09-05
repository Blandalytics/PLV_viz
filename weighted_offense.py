import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

colors = {
  'Top':['white','#bb5f5d'],
  'Solid':['black','#dbaba8'],
  'Average':['black','#faf5f5'],
  'Weak':['black','#aebcd1'],
  'Poor':['white','#5a84bd']
}

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

def highlight_cols(x):
    df = x.copy()
    #select all values
    for col in ['Team','wOBA','Tier']:
      df[col] = df['Tier'].apply(lambda x: f'color: {colors[x][0]}; border: 1.5px solid white; background-color: {colors[x][1]}')
    #return color df
    return df

pa_df = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/data/2023_PAs.csv?raw=true')
pa_df['game_played'] = pd.to_datetime(pa_df['game_played'])

# @st.cache_data(ttl=12*3600)
def calc_wOBA_ranks(df=pa_df,time_frame='Season',thresh=0.075):
    if time_frame=='Season':
        time_thresh = 365
    else:
        time_thresh = int(time_frame[-2:])
    
    test_pa_df = df[df['game_played'] > (df['game_played'].max() - pd.Timedelta(days=time_thresh))]
    
    thresh = thresh if test_pa_df.shape[0] >= 60000 else thresh*2 if test_pa_df.shape[0] >= 30000 else thresh*3 if test_pa_df.shape[0] >= 15000 else thresh*4

    test_df = pd.DataFrame(index=test_pa_df['hitterteam'].sort_values().unique())
    test_df['wOBA'] = test_pa_df.groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_lhp'] = test_pa_df.loc[test_pa_df['pitcherside']=='L'].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_rhp'] = test_pa_df.loc[test_pa_df['pitcherside']=='R'].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_home'] = test_pa_df.loc[test_pa_df['is_home']==1].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_away'] = test_pa_df.loc[test_pa_df['is_home']==0].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_lhp_home'] = test_pa_df.loc[(test_pa_df['pitcherside']=='L') & (test_pa_df['is_home']==1)].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_lhp_away'] = test_pa_df.loc[(test_pa_df['pitcherside']=='L') & (test_pa_df['is_home']==0)].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_rhp_home'] = test_pa_df.loc[(test_pa_df['pitcherside']=='R') & (test_pa_df['is_home']==1)].groupby('hitterteam')['wOBA'].mean()
    test_df['wOBA_rhp_away'] = test_pa_df.loc[(test_pa_df['pitcherside']=='R') & (test_pa_df['is_home']==0)].groupby('hitterteam')['wOBA'].mean()
    test_df['hand_stdev'] = test_df[['wOBA_lhp','wOBA_rhp']].std(axis=1).div(test_df['wOBA'])
    test_df['location_stdev'] = test_df[['wOBA_home','wOBA_away']].std(axis=1).div(test_df['wOBA'])
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
    return weighted_test_df

time_string = st.radio('Choose a time frame:', ['Season','Last 30','Last 15'])
rank_df = calc_wOBA_ranks(df=pa_df,time_frame=time_string,thresh=0.075)

time_frame_deviation = {
  'Season':0.0125,
  'Last 30':0.02,
  'Last 15':0.025
}

st.title('MLB Offense Ranks')
st.dataframe(rank_df
             .style
             .format(precision=4)
             .background_gradient(axis=0,gmap=(rank_df['wOBA']-pa_df['wOBA'].mean())/time_frame_deviation[time_string], 
                                  vmin=-2,vmax=2,
                                  cmap='vlag'),
             width=400,
             height=800,
#              hide_index=True
            )
