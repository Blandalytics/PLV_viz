import streamlit as st
import datetime
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import urllib
import pickle
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image

# Convenience Functions
def adjusted_vaa(dataframe):
    ## Physical characteristics of pitch
    # Pitch velocity (to plate) at plate
    dataframe['vyf'] = -1 * (dataframe['vy0']**2 - (2 * dataframe['ay']*(50-17/12)))**0.5
    # Pitch time in air (50ft to home plate)
    dataframe['pitch_time_50ft'] = (dataframe['vyf'] - dataframe['vy0'])/dataframe['ay']
    # Pitch velocity (vertical) at plate
    dataframe['vzf'] = dataframe['vz0'] + dataframe['az'] * dataframe['pitch_time_50ft']

    ## raw and height-adjusted VAA
    # Raw VAA 
    dataframe['raw_vaa'] = -1 * np.arctan(dataframe['vzf']/dataframe['vyf']) * (180/np.pi)
    # VAA of all pitches at that height
    dataframe['vaa_z_adj'] = np.where(dataframe['pz']<3.5,
                                      dataframe['pz'].mul(1.5635).add(-10.092),
                                      dataframe['pz'].pow(2).mul(-0.1996).add(dataframe['pz'].mul(2.704)).add(-11.69))
    dataframe['adj_vaa'] = dataframe['raw_vaa'].sub(dataframe['vaa_z_adj'])
    # Adjusted VAA, based on height
    return dataframe[['raw_vaa','adj_vaa']]

st.set_page_config(page_title='PL Live Spring Training Stats', page_icon='⚾',layout="wide")

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

st.title('PL Live Spring Training Stats')
col1, col2, col3 = st.columns(3)

with col1:
    today = datetime.date.today()
    date = st.date_input("Select a game date:", today, min_value=datetime.date(2024, 2, 19), max_value=datetime.date(2025, 3, 30))
    
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}')
    x = r.json()
    if x['totalGames']==0:
        st.write(f'No games on {date}')
    game_list = {}
    for game in range(len(x['dates'][0]['games'])):
        if x['dates'][0]['games'][game]['gamedayType'] == 'E':
            game_list.update({x['dates'][0]['games'][game]['teams']['away']['team']['name']+' @ '+x['dates'][0]['games'][game]['teams']['home']['team']['name']:x['dates'][0]['games'][game]['gamePk']})

with col2:
    game_select = st.selectbox('Choose a game:',list(game_list.keys()))
    
    game_id = game_list[game_select]
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    pitcher_list = {}
    for home_away_pitcher in ['home','away']:
        if f'{home_away_pitcher}_pitchers' not in x.keys():
            continue
        for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
            pitcher_list.update({x[f'{home_away_pitcher}_pitchers'][pitcher_id][0]['pitcher_name']:[pitcher_id,x['scoreboard']['teams']['home' if home_away_pitcher=='away' else 'away']['abbreviation']]})

with col3:
    player_select = st.selectbox('Choose a pitcher:',list(pitcher_list.keys()))

def load_season_avgs():
    return pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/season_avgs_2024.parquet?raw=true')

season_avgs = load_season_avgs()
with open('2025_3d_xwoba_model.pkl', 'rb') as f:
    xwOBAcon_model = pickle.load(f)

def scrape_savant_data(player_name, game_id):
    game_ids = []
    game_date = []
    pitcher_id_list = []
    pitcher_name = []
    throws = []
    stands = []
    pitch_id = []
    pitch_type = []
    velo = []
    extension = []
    called_strikes = []
    swinging_strikes = []
    ivb = []
    ihb = []
    vy0 = []
    vz0 = []
    ay = []
    az = []
    pz = []
    hit_x = []
    hit_y = []
    hit_speed = []
    hit_angle = []
    games = 0
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()

    if ('game_status_code' in x.keys()):
        if (x['game_status_code'] in ['P','S']):
        # if (x['game_status_code'] != 'E'):
            st.write('Non-Spring Training game_status_code')
    elif ('code' in x.keys()):
        if (x['code']=='UNCERTAIN_STATE'):
            st.write('Unexpected code')
    games+=1
    for home_away_pitcher in ['home','away']:
        if f'{home_away_pitcher}_pitchers' not in x.keys():
            continue
        for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
            if pitcher_id != pitcher_list[player_select][0]:
                continue
            for pitch in range(len(x[f'{home_away_pitcher}_pitchers'][pitcher_id])):
                game_ids += [game_id]
                game_date += [x['gameDate']]
                pitcher_id_list += [pitcher_id]
                p_name = x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitcher_name']
                pitcher_name += [p_name]
                throws += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['p_throws']]
                stands += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['stand']]
                called_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call']=='called_strike' else 0]
                swinging_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call'] in ['swinging_strike','foul_tip','swinging_strike_blocked'] else 0]
                pitch_id += [pitch]
                try:
                    pitch_type += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_type']]
                    velo += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['start_speed']]
                    extension += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['extension'] if 'extension' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else None]
                    ivb += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pfxZWithGravity']]
                    ihb += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pfxXNoAbs']]
                    # x0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['x0']]
                    # z0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['z0']]
                    # vx0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vx0']]
                    vy0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vy0']]
                    vz0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vz0']]
                    # ax += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ax']]
                    ay += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ay']]
                    az += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['az']]
                    # px += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['px']]
                    pz += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pz']]
                            
                except KeyError:
                    pitch_type += ['UN']
                    velo += [None]
                    extension += [None]
                    ivb += [None]
                    ihb += [None]
                    # x0 += [None]
                    # z0 += [None]
                    # vx0 += [None]
                    vy0 += [None]
                    vz0 += [None]
                    # ax += [None]
                    ay += [None]
                    az += [None]
                    # px += [None]
                    pz += [None]
                if all(i in list(x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys()) for i in ['hc_x_ft','hc_y_ft','hit_speed','hit_angle']):
                    hit_x += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['hc_x_ft']]
                    hit_y += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['hc_y_ft']]
                    hit_speed += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['hit_speed']]
                    hit_angle += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['hit_angle']]
                else:
                    hit_x += [None]
                    hit_y += [None]
                    hit_speed += [None]
                    hit_angle += [None]
    if games == 0:
        print('No Games Played')
        exit()
        
    df = pd.DataFrame()
    df['game_pk'] = game_ids
    df['game_date'] = game_date
    df['Opp'] = pitcher_list[player_select][1]
    df['MLBAMID'] = pitcher_id_list
    df['MLBAMID'] = df['MLBAMID'].astype('int')
    df['Pitcher'] = pitcher_name
    df['P Hand'] = throws
    df['hitterside'] = stands
    df['CS'] = called_strikes
    df['Whiffs'] = swinging_strikes
    df['Num Pitches'] = pitch_id
    df['pitch_type'] = pitch_type
    df['Velo'] = velo
    df['Ext'] = extension
    df['vert_break'] = ivb
    df['vy0'] = vy0
    df['vz0'] = vz0
    df['ay'] = ay
    df['az'] = az
    df['pz'] = pz
    df['IVB'] = df['vert_break'].add((523/df['Velo'])**2).astype('float')
    df['IHB'] = ihb
    df['IHB'] = np.where(df['P Hand']=='R',df['IHB'].astype('float').mul(-1),df['IHB'].astype('float'))
    df['hit_x'] = hit_x
    df['hit_y'] = hit_y
    df['Launch Speed'] = hit_speed
    df['Launch Angle'] = hit_angle
    df['spray_deg_base'] = np.degrees(np.arctan(df['hit_y'].astype('float').div(df['hit_x'].astype('float').abs())))
    df['Spray Angle'] = np.where((((df['hit_x']>0) & (df['hitterside']=='R')) | 
                                ((df['hit_x']<0) & (df['hitterside']=='L'))),
                               135-df['spray_deg_base'],
                               df['spray_deg_base']-45)
    df[['VAA','HAVAA']] = adjusted_vaa(df)
    
    def xwOBA_model(df_):
        return 0
    
    df['3D wOBAcon'] = [None if any(np.isnan([x,y,z])) else sum(np.multiply(xwOBAcon_model.predict_proba([[x,y,z]])[0],np.array([0,0.9,1.25,1.6,2]))) for x,y,z in zip(df['Spray Angle'].astype('float'),df['Launch Angle'].astype('float'),df['Launch Speed'].astype('float'))]

    game_df = df.assign(vs_rhh = lambda x: np.where(x['hitterside']=='R',1,0)).groupby(['game_date','Opp','MLBAMID','Pitcher','pitch_type'])[['Num Pitches','Velo','IVB','IHB','Ext','vs_rhh','CS','Whiffs','3D wOBAcon','HAVAA']].agg({
        'Num Pitches':'count',
        'Velo':'mean',
        'IVB':'mean',
        'IHB':'mean',
        'Ext':'mean',
        'vs_rhh':'sum',
        'CS':'sum',
        'Whiffs':'sum',
        '3D wOBAcon':'mean',
        'HAVAA':'mean'
    }).assign(CSW = lambda x: x['CS'].add(x['Whiffs']).div(x['Num Pitches']).mul(100),
              vs_lhh = lambda x: x['Num Pitches'].sub(x['vs_rhh'])).reset_index()

    merge_df = (
        pd.merge(game_df.loc[game_df['Pitcher']==player_name].assign(Usage = lambda x: x['Num Pitches'].div(x['Num Pitches'].sum()).mul(100)).copy(),
                 season_avgs.rename(columns={
                     'release_speed':'Velo','pfx_x':'IHB'
                     }),
                 how='left',
                 left_on=['MLBAMID','pitch_type'],
                 right_on=['pitcher','pitch_type'],
                 suffixes=['','_2024'])
        .assign(vs_rhh = lambda x: x['vs_rhh'].div(x['vs_rhh'].sum()).fillna(0),
                vs_lhh = lambda x: x['vs_lhh'].div(x['vs_lhh'].sum()).fillna(0),
                usage_diff = lambda x: x['Usage'].sub(x['Usage_2024']),
                velo_diff = lambda x: x['Velo'].sub(x['Velo_2024']),
                ivb_diff = lambda x: x['IVB'].sub(x['IVB_2024']),
                ihb_diff = lambda x: x['IHB'].sub(x['IHB_2024']))
        .rename(columns={'game_date':'Date',
                         'pitch_type':'Type',
                         'usage_diff':'Usage Diff',
                         'velo_diff':'Velo Diff',
                         'ivb_diff':'IVB Diff',
                         'ihb_diff':'IHB Diff'})
        .sort_values('Num Pitches',ascending=False)
        )
    merge_df['CSW'] = [f'{x:.1f}%' for x in merge_df['CSW']]
    merge_df['vs R'] = [f'{x:.1%}' for x in merge_df['vs_rhh']]
    merge_df['vs L'] = [f'{x:.1%}' for x in merge_df['vs_lhh']]
    merge_df['Ext'] = merge_df['Ext'].round(1)
    merge_df['3D wOBAcon'] = merge_df['3D wOBAcon'].round(3)
    merge_df['HAVAA'] = [f'{x:.1f}°' for x in merge_df['HAVAA']]

    merge_df['Usage'] = [f'{x:.1f}% ({y:+.1f}%)' for x,y in zip(merge_df['Usage'],merge_df['Usage Diff'].fillna(merge_df['Usage']))]
    
    merge_df['Velo'] = np.where(merge_df['Velo Diff'].isna(),
                                 [f'{x:.1f}' for x in merge_df['Velo']],
                                 [f'{x:.1f} ({y:+.1f})' for x,y in zip(merge_df['Velo'],merge_df['Velo Diff'].fillna(0))])
    merge_df['IVB'] = np.where(merge_df['IVB Diff'].isna(),
                                 [f'{x:.1f}"' for x in merge_df['IVB']],
                                 [f'{x:.1f}" ({y:+.1f}")' for x,y in zip(merge_df['IVB'],merge_df['IVB Diff'].fillna(0))])
    merge_df['IHB'] = np.where(merge_df['IHB Diff'].isna(),
                                 [f'{x:.1f}"' for x in merge_df['IHB']],
                                 [f'{x:.1f}" ({y:+.1f}")' for x,y in zip(merge_df['IHB'],merge_df['IHB Diff'].fillna(0))])

    return merge_df[['Date','Opp','Pitcher','Type','Num Pitches','Velo','Usage','vs R','vs L','Ext','IVB','IHB','HAVAA','CS','Whiffs','CSW','3D wOBAcon']]#.rename(columns={'Num Pitches':'#'})
if pitcher_list == {}:
    st.write('No pitches thrown yet')
elif st.button("Generate Player Table"):
    table_df = scrape_savant_data(player_select,game_id)
    st.dataframe(table_df,#.style.background_gradient(axis=0, vmin=0, vmax=0.755,cmap="vlag_r", subset=['3D wOBAcon']),
                 column_config={
                     "Num Pitches": st.column_config.NumberColumn(
                         "#",
                         ),
                     "3D wOBAcon": st.column_config.NumberColumn(
                         help="""
                         xwOBA on contact, using Launch Speed, Launch Angle, and Spray Angle
                         League Average is ~.378
                         """,
                         ),
                     "HAVAA": st.column_config.Column(
                         help="""
                         Height-Adjusted Vertical Approach Angle
                         >0 means flatter than other pitches at that location
                         <0 means steeper than pitches at that location
                         """,
                         ),
                     "vs R": st.column_config.Column(
                         help="% of pitches thrown vs Right-Handed Hitters",
                         ),
                     "vs L": st.column_config.Column(
                         help="% of pitches thrown vs Left-Handed Hitters",
                         ),
                     },
                 # use_container_width=True,
                 hide_index=True)
