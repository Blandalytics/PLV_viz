import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np

import urllib
from PIL import Image

@st.cache_data(ttl=3600)
def load_logo():
    logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

logo = load_logo()
# st.image(logo, width=400)

@st.cache_data(ttl=3600)
def letter_logo():
    logo_loc = 'https://github.com/Blandalytics/baseball_snippets/blob/main/teal_letter_logo.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

letter_logo = letter_logo()

st.set_page_config(page_title='PL Auction Draft Calculator', page_icon=letter_logo,layout="wide")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 310px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

new_title = '<p style="color:#72CBFD; font-weight: bold; font-size: 42px;">PL Auction Draft Calculator</p>'
st.markdown(new_title, unsafe_allow_html=True)

team_leagues = {
    'LAA':'AL',
    'NYY':'AL',
    'SDP':'NL',
    'CLE':'AL',
    'LAD':'NL',
    'TOR':'AL',
    'ATL':'NL',
    'HOU':'AL',
    'NYM':'NL',
    'PHI':'NL',
    'STL':'NL',
    'SEA':'AL',
    'BOS':'AL',
    'TEX':'AL',
    'KCR':'AL',
    'PIT':'NL',
    'TBR':'AL',
    'CHC':'NL',
    'MIL':'NL',
    'BAL':'AL',
    'ARI':'NL',
    'MIN':'AL',
    'MIA':'NL',
    'COL':'NL',
    'CHW':'AL',
    'DET':'AL',
    'SFG':'NL',
    'CIN':'NL',
    'ATH':'AL',
    'OAK':'AL',
    'WAS':'NL',
    'WSN':'NL',
    # 'FA':''
}

with st.sidebar:
    pad1, col1, pad2 = st.columns([0.25,0.5,0.25])
    with col1:
        st.image(letter_logo)
    
    # Settings
    # st.header('Team Settings')
    team_header = '<p style="color:#72CBFD; font-weight: bold; text-align: center; font-size: 21px;">Team Settings</p>'
    st.markdown(team_header, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        num_hitters = st.number_input('Hitters',min_value=4,max_value=20,value=10)
    with col2:
        num_pitchers = st.number_input('Pitchers',min_value=4,max_value=20,value=8)
    
    col1, col2 = st.columns(2)
    with col1:
        num_catchers = st.number_input('Catchers',min_value=0,max_value=3,value=1)
    with col2:
        raw_bench = st.number_input('Bench spots',min_value=0,max_value=20,value=5)
    
    bench_suppress = st.checkbox("Minimize bench value",value=True,
                                 help="""
                                 Does not consider bench players  
                                 when calculating replacement level
                                 """)
    
    num_bench = 1 if bench_suppress else raw_bench
    
    st.write('')
    # st.header('League Settings')
    league_header = '<p style="color:#72CBFD; font-weight: bold; text-align: center; font-size: 21px;">League Settings</p>'
    st.markdown(league_header, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        scoring_style = st.radio(
            "League Type",
            ["Categories", "Points"],
        )
    with col2:
        num_teams = st.number_input('Number of Teams',min_value=4,max_value=30,value=12)
    if scoring_style=='Categories':
        col1, col2 = st.columns(2)
        with col1:
            min_bid = st.number_input('Min bid',min_value=0,value=1)
        with col2:
            team_budget = st.number_input('Team Budget',min_value=(min_bid+1)*(num_hitters+num_pitchers+num_bench),value=260)
    ### Filler until I implement points
    else:
        min_bid=1
        team_budget=260
    col1, col2 = st.columns(2)
    with col1:
        league_select = st.selectbox('Player pool',['All','NL-Only','AL-Only'])
        league_pool = ['NL','AL'] if league_select=='All' else [league_select[:2]]
    with col2:
        hitter_split = st.number_input('Hitter Split (%)',min_value=0,max_value=100,value=65)
        hitter_split = hitter_split/100
    
    include_fa = st.checkbox("Include FA?",value=True,
                             help="Include free agents in layer pool")
    if include_fa:
        team_leagues.update({'FA':league_select[:2].upper()})
        
    st.write('')
    # st.header('Scoring')
    scoring_header = '<p style="color:#72CBFD; font-weight: bold; text-align: center; font-size: 21px;">Scoring</p>'
    st.markdown(scoring_header, unsafe_allow_html=True)
    if scoring_style=='Categories':
        hitter_cats = st.multiselect('Hitter categories',
                                     ['G', 'AB','PA', 'R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'ISO', 'SLG', 'OPS',
                                      'wOBA', 'BB%', 'K%', 'H', '1B', '2B', '3B', 'XBH',
                                      'TB', 'K', 'BB', 'HBP', 'SF', 'CS'],
                                     default=['R','HR','RBI','SB','AVG'])
        rate_cats_h = ['AVG','OBP','ISO','SLG','OPS','wOBA','BB%','K%']
        rate_scoring_cats_h = [x for x in hitter_cats if x in rate_cats_h]
        volume_scoring_cats_h = [x for x in hitter_cats if x not in rate_scoring_cats_h]
        inverted_categories_h = ['K','CS','SF','K%']
        pitcher_cats = st.multiselect('Pitcher categories',
                                      ['IP', 'TBF','G', 'GS', 'W', 'L', 'QS', 'SV', 'HD', 'SV+H', 'K', 'ERA', 
                                       'WHIP','K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'HR/9', 'H', 'ER', 'HBP',
                                       'HR', 'BB', 'BS','K/BB','W+QS'],
                                      default=['W','SV','K','ERA','WHIP'])
        rate_cats_p = ['ERA', 'WHIP','K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'HR/9']
        rate_scoring_cats_p = [x for x in pitcher_cats if x in rate_cats_p]
        volume_scoring_cats_p = [x for x in pitcher_cats if x not in rate_scoring_cats_p]
        inverted_categories_p = ['BB','H','ER','BS','ERA','WHIP','L','HBP','HR','BB/9','HR/9','BB%']

        hitter_renames = {x:x+'_h' for x in hitter_cats if x in pitcher_cats}
        pitcher_renames = {x:x+'_p' for x in pitcher_cats if x in hitter_cats}

    else:
        hitter_start = ['AB','H','2B','3B','HR','BB','HBP','SB','CS']
        pitcher_start = ["IP","K","H","BB",'HBP','HR','SV','HD']
    
        hitter_point_cats = ['G', 'AB','PA', 'R', 'HR', 'RBI', 'SB', 'H', '1B', '2B', '3B', 'K', 'BB', 'HBP', 'SF', 'CS']
        pitcher_point_cats = ['IP', 'TBF', 'G', 'GS', 'W', 'L', 'QS', 'SV', 'HD', 'K','H', 'ER', 'HBP', 'HR', 'BB', 'BS']
        st.write('Hitting Points')
        hitter_cat_df = pd.DataFrame(
            {
                "Category":hitter_start,
                "Points": [
                    -1.0,
                    5.6,
                    2.9,
                    5.7,
                    9.4,
                    3.0,
                    3.0,
                    1.9,
                    -2.8
                ]
            }
            )
        edited_hitter_df = st.data_editor(
            hitter_cat_df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    # help="The category of the app",
                    # width="medium",
                    options=hitter_point_cats,
                    required=True,
                ),
                "Points": st.column_config.NumberColumn(
                    "Points",
                    min_value=-1000,
                    max_value=1000,
                    step=0.1,
                    required=True,
                )
            },
            hide_index=True,
            height=(5 + 1) * 35 + 3,
            num_rows="dynamic"
        )
        st.write('Pitching Points')
        pitcher_cat_df = pd.DataFrame(
            {
                "Category": pitcher_start,
                "Points": [
                    7.4,
                    2.0,
                    -2.6,
                    -3.0,
                    -3.0,
                    -12.3,
                    5.0,
                    4.0
                ]
            }
            )
        
        edited_pitcher_df = st.data_editor(
            pitcher_cat_df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    # help="The category of the app",
                    # width="medium",
                    options=pitcher_point_cats,
                    required=True,
                ),
                "Points": st.column_config.NumberColumn(
                    "Points",
                    min_value=-1000,
                    max_value=1000,
                    step=0.1,
                    required=True,
                )
            },
            hide_index=True,
            height=(5 + 1) * 35 + 3,
            num_rows="dynamic"
        )
        hitter_cats = edited_hitter_df['Category'].to_list()
        pitcher_cats = edited_pitcher_df['Category'].to_list()
        
        volume_scoring_cats_h = hitter_cats
        rate_scoring_cats_h = []
        inverted_categories_h = ['K','CS','SF','K%']
        volume_scoring_cats_p = pitcher_cats
        rate_scoring_cats_p = []
        inverted_categories_p = ['BB','H','ER','BS','ERA','WHIP','L','HBP','HR','BB/9','HR/9','BB%']

        hitter_renames = {x:x+'_h' for x in hitter_cats if x in pitcher_cats}
        pitcher_renames = {x:x+'_p' for x in pitcher_cats if x in hitter_cats}
        point_values = edited_hitter_df.assign(Category = lambda x: x['Category'].replace(hitter_renames)).set_index('Category').to_dict()['Points']
        point_values.update(edited_pitcher_df.assign(Category = lambda x: x['Category'].replace(pitcher_renames)).set_index('Category').to_dict()['Points'])


adj_hitter_cats = [x+'_h' if x in pitcher_cats else x for x in hitter_cats]
adj_pitcher_cats = [x+'_p' if x in hitter_cats else x for x in pitcher_cats]
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

@st.cache_data(ttl=3600)
def load_data(team_leagues,league_pool):
    # Load projections
    projections_hitters = pd.read_csv('https://docs.google.com/spreadsheets/d/17r2LFFyd3cJVDviOCUSSYEe6wgejtdAukOKT4XH50n4/export?gid=1029181665&format=csv')
    projections_hitters['League'] = projections_hitters['Team'].fillna('FA').map(team_leagues)
    projections_hitters = projections_hitters.loc[projections_hitters['League'].isin(league_pool)].reset_index(drop=True).copy()
    for stat in ['K%','BB%']:
        projections_hitters[stat] = projections_hitters[stat].str[:-1].astype('float')
    
    projections_pitchers = pd.read_csv('https://docs.google.com/spreadsheets/d/17r2LFFyd3cJVDviOCUSSYEe6wgejtdAukOKT4XH50n4/export?gid=354379391&format=csv')
    projections_pitchers['League'] = projections_pitchers['Team'].fillna('FA').map(team_leagues)
    projections_pitchers = projections_pitchers.loc[projections_pitchers['League'].isin(league_pool)].reset_index(drop=True).copy()
    for stat in ['K%','BB%','K-BB%']:
        projections_pitchers[stat] = projections_pitchers[stat].str[:-1].astype('float')
    projections_pitchers['W+QS'] = projections_pitchers['W'].add(projections_pitchers['QS'])
    return projections_hitters, projections_pitchers

projections_hitters, projections_pitchers = load_data(team_leagues,league_pool)

# if st.button("Generate Auction Values:  📊 -> 💲"):
# st.header('Auction Values')
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
            projections_hitters[['Name','MLBAMID','Team','Y! Pos','PA']+[x for x in hitter_cats if x!='PA']+['adjusted_value']].rename(columns=hitter_renames),
            projections_pitchers[['Name','MLBAMID','Team','IP']+[x for x  in pitcher_cats if x!='IP']+['adjusted_value']].rename(columns=pitcher_renames)
            ],
        ignore_index=True)
    [['Name','MLBAMID','Team','Y! Pos','adjusted_value','PA']+[x for x in adj_hitter_cats if x!='PA']+['IP']+[x for x in adj_pitcher_cats if x!='IP']]
)
combined_value_df['Y! Pos'] = combined_value_df['Y! Pos'].fillna('P')
if scoring_style=='Categories':
    combined_value_df['Value'] = min_bid + np.where(
        combined_value_df['Y! Pos']=='P',
        combined_value_df['adjusted_value'].mul(pitcher_dollars_per_value),
        combined_value_df['adjusted_value'].mul(hitter_dollars_per_value)
    )
    
    # Level the dollars out
    projected_auction_dollars = combined_value_df.loc[combined_value_df['Value']>0,'Value'].sum()
    fudge_factor = (num_teams * team_budget) / projected_auction_dollars
    combined_value_df['Value'] = combined_value_df['Value'].mul(fudge_factor)
else:
    combined_value_df['Value'] = combined_value_df[list(point_values.keys())].mul(point_values).sum(axis=1)
combined_value_df['Rank'] = combined_value_df['Value'].rank(ascending=False)
display_df = combined_value_df[['Rank','Name','Team','Y! Pos','Value','PA']+[x for x in adj_hitter_cats if x!='PA']+['IP']+[x for x in adj_pitcher_cats if x!='IP']].sort_values('Value',ascending=False).copy()

# col1, col2 = st.columns([0.8,0.2])
# with col1:
st.write('To change settings, tap the >> in the upper left of the page')
# with col2:
st.download_button(label='Download CSV',
                  data=display_df.to_csv(index=False),
                  file_name='pitcher_list_auction_values.csv',
                   mime='text/csv',
                   icon=":material/download:",
                   on_click='ignore')

st.dataframe(display_df,
             width='content',
             hide_index=True,
             height=(25 + 1) * 35 + 3,
             # height='stretch',
             column_config={
                     "Value": st.column_config.NumberColumn(
                         label = 'Auction $' if scoring_style=='Categories' else 'Points',
                         format="$ %.2f" if scoring_style=='Categories' else "%.1f",
                         ),
                     "Name": st.column_config.TextColumn(
                         width=220,
                         ),
                 },
             placeholder='',
             )
