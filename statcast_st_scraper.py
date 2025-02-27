import streamlit as st
import datetime
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib
import pickle
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier

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
    dataframe['vaa_z_adj'] = np.where(dataframe['p_z']<3.5,
                                      dataframe['p_z'].mul(1.5635).add(-10.092),
                                      dataframe['p_z'].pow(2).mul(-0.1996).add(dataframe['p_z'].mul(2.704)).add(-11.69))
    dataframe['adj_vaa'] = dataframe['raw_vaa'].sub(dataframe['vaa_z_adj'])
    # Adjusted VAA, based on height
    return dataframe[['raw_vaa','adj_vaa']]

### Standardized Strikezone (z-location, in 'strikezones')
def strikezone_z(dataframe,top_column,bottom_column):
    dataframe[['p_z',top_column,bottom_column]] = dataframe[['p_z',top_column,bottom_column]].astype('float')
    
    # Ratio of 'strikezones' above/below midpoint of strikezone
    dataframe['sz_mid'] = dataframe[[top_column,bottom_column]].mean(axis=1)
    dataframe['sz_height'] = dataframe[top_column].sub(dataframe[bottom_column])
    
    return dataframe['p_z'].sub(dataframe['sz_mid']).div(dataframe['sz_height'])

def loc_model(df,year=2024):
    df['balls_before_pitch'] = np.clip(df['balls'], 0, 3)
    df['strikes_before_pitch'] = np.clip(df['strikes'], 0, 2)
    df['pitcherside'] = df['P Hand'].copy()

    df = pd.get_dummies(df, columns=['pitcherside','hitterside','balls_before_pitch','strikes_before_pitch'])
    for hand in ['L','R']:
        if f'pitcherside_{hand}' not in df.columns.values:
            df[f'pitcherside_{hand}'] = 0

    df[['take_input','swing_input','called_strike_raw','ball_raw',
                'hit_by_pitch_raw','swinging_strike_raw','contact_raw',
                'foul_strike_raw','in_play_raw','10deg_raw','10-20deg_raw',
                '20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw',
                'called_strike_pred','ball_pred','hit_by_pitch_pred','contact_input',
                'swinging_strike_pred','foul_strike_pred','in_play_input','50+deg_pred',
                'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']] = None

    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        df[[launch_angle+'_input',launch_angle+': <90mph_raw',
                 launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',
                 launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw',
                 launch_angle+': <90mph_pred',launch_angle+': 90-95mph_pred',
                 launch_angle+': 95-100mph_pred',launch_angle+': 100-105mph_pred',
                 launch_angle+': 105+mph_pred']] = None

    # Swing Decision
    with open('model_files/2024_pl_swing_model_Fastball_loc.pkl', 'rb') as f:
        decision_model = pickle.load(f)

    df[['take_input','swing_input']] = decision_model.predict_proba(df[decision_model.feature_names_in_])

    # Take Result
    with open('model_files/2024_pl_take_model_Fastball_loc.pkl', 'rb') as f:
        take_model = pickle.load(f)

    df[['called_strike_raw','ball_raw','hit_by_pitch_raw']] = take_model.predict_proba(df[take_model.feature_names_in_])
    df['called_strike_pred'] = df['called_strike_raw'].mul(df['take_input'])
    df['ball_pred'] = df['ball_raw'].mul(df['take_input'])
    df['hit_by_pitch_pred'] = df['hit_by_pitch_raw'].mul(df['take_input'])

    # Swing Result
    with open('model_files/2024_pl_contact_model_Fastball_loc.pkl', 'rb') as f:
        swing_result_model = pickle.load(f)

    df[['swinging_strike_raw','contact_raw']] = swing_result_model.predict_proba(df[swing_result_model.feature_names_in_])
    df['contact_input'] = df['contact_raw'].mul(df['swing_input'])
    df['swinging_strike_pred'] = df['swinging_strike_raw'].mul(df['swing_input'])

    # Contact Result
    with open('model_files/2024_pl_in_play_model_Fastball_loc.pkl', 'rb') as f:
        contact_model = pickle.load(f)

    df[['foul_strike_raw','in_play_raw']] = contact_model.predict_proba(df[contact_model.feature_names_in_])
    df['foul_strike_pred'] = df['foul_strike_raw'].mul(df['contact_input'])
    df['in_play_input'] = df['in_play_raw'].mul(df['contact_input'])

    # Launch Angle Result
    with open('model_files/2024_pl_launch_angle_model_Fastball_loc.pkl', 'rb') as f:
        launch_angle_model = pickle.load(f)

    df[['10deg_raw','10-20deg_raw','20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw']] = launch_angle_model.predict_proba(df[launch_angle_model.feature_names_in_])
    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        df[launch_angle+'_input'] = df[launch_angle+'_raw'].mul(df['in_play_input'])
    df['50+deg_pred'] = df['50+deg_raw'].mul(df['in_play_input'])

    # Launch Velo Result
    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        with open('model_files/2024_pl_{}_model_Fastball_loc.pkl'.format(launch_angle), 'rb') as f:
            launch_velo_model = pickle.load(f)

        df[[launch_angle+': <90mph_raw',launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw']] = launch_velo_model.predict_proba(df[launch_velo_model.feature_names_in_])
        for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
            df[bucket+'_pred'] = df[bucket+'_raw'].mul(df[launch_angle+'_input'])

    bip_result_dict = (
        pd.read_csv('model_files/data_bip_result.csv')
        .set_index(['year_played','bb_bucket'])
        .to_dict(orient='index')
    )

    # Apply averages to each predicted grouping
    for outcome in ['out', 'single', 'double', 'triple', 'home_run']:
        # Start with 50+ degrees (popups)
        df[outcome+'_pred'] = df['50+deg_pred']*bip_result_dict[(year,'50+deg')][outcome]

        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                df[outcome+'_pred'] += df[bucket+'_pred']*bip_result_dict[(year,bucket)][outcome]

    ### Find the estimated change in wOBA/runs for each pitch
    # wOBA value of an outcome, based on the count that it came in
    outcome_wOBAs = pd.read_csv('model_files/data_woba_outcome.csv').set_index(['year_played','balls','strikes'])

    df = df.merge(outcome_wOBAs,
                  how='left',
                  on=['year_played','balls','strikes'])

    # wOBA_effect is how the pitch is expected to affect wOBA
    # (either by moving the count, or by ending the PA)
    df['wOBA_effect'] = 0

    for stat in [x[:-5] for x in list(outcome_wOBAs.columns)]:
        df['wOBA_effect'] = df['wOBA_effect'].add(df[stat+'_pred'].fillna(df[stat+'_pred'].median()).mul(df[stat+'_wOBA'].fillna(df[stat+'_wOBA'].median())))

    return df['wOBA_effect'].sub(-0.004253050593194383).div(0.05179234832326223).mul(-50).add(100)


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

sz_bot = 1.5
sz_top = 3.5
x_ft = 2.5
y_bot = -0.5
y_lim = 6
plate_y = -.25
alpha_val = 0.5

pitchtype_map = {
    'FF':'FF','FA':'FF',
    'SI':'SI','FT':'SI',
    'FC':'FC',
    'SL':'SL',
    'ST':'ST',
    'CH':'CH',
    'CU':'CU','KC':'CU','CS':'CU','SV':'CU',
    'FS':'FS','FO':'FS',
    'KN':'KN',
    'UN':'UN','EP':'UN'
}

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

marker_colors = {
    'FF':'#d22d49', 
    'SI':'#c57a02',
    'FS':'#00a1c5',  
    'FC':'#933f2c', 
    'SL':'#9300c7',  
    'ST':'#C95EBE',
    'CU':'#3c44cd',
    'CH':'#07b526', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
}

pitch_names = {
    'FF':'Four-Seam', 
    'SI':'Sinker',
    'FS':'Splitter',  
    'FC':'Cutter', 
    'SL':'Slider', 
    'ST':'Sweeper',
    'CU':'Curveball',
    'CH':'Changeup', 
    'KN':'Knuckleball',
    'SC':'Screwball', 
    'UN':'Unknown', 
}

def scrape_savant_data(player_name, game_id):
    game_ids = []
    game_date = []
    pitcher_id_list = []
    pitcher_name = []
    throws = []
    stands = []
    balls = []
    strikes = []
    pitch_id = []
    pitch_type = []
    velo = []
    extension = []
    called_strikes = []
    swinging_strikes = []
    total_strikes = []
    ivb = []
    ihb = []
    vy0 = []
    vz0 = []
    ay = []
    az = []
    px = []
    pz = []
    sz_top = []
    sz_bot = []
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
                total_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call'] in ['swinging_strike','foul_tip','swinging_strike_blocked','called_strike','foul','hit_into_play'] else 0]
                balls += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['balls']]
                strikes += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['strikes']]
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
                    px += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['px']]
                    pz += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pz']]
                    sz_top += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['sz_top']]
                    sz_bot += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['sz_bot']]
                            
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
                    px += [None]
                    pz += [None]
                    sz_top += [None]
                    sz_bot += [None]
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
    df['balls'] = balls
    df['strikes'] = strikes
    df['Pitcher'] = pitcher_name
    df['P Hand'] = throws
    df['hitterside'] = stands
    df['CS'] = called_strikes
    df['Whiffs'] = swinging_strikes
    df['total_strikes'] = total_strikes
    df['Num Pitches'] = pitch_id
    df['pitch_type'] = pitch_type
    df['pitch_type'] = df['pitch_type'].map(pitchtype_map)
    df['Velo'] = velo
    df['Ext'] = extension
    df['vert_break'] = ivb
    df['vy0'] = vy0
    df['vz0'] = vz0
    df['ay'] = ay
    df['az'] = az
    df['p_x'] = px
    df['p_z'] = pz
    df['sz_top'] = sz_top
    df['sz_bot'] = sz_bot
    df['sz_z'] = strikezone_z(df,'sz_top','sz_bot')
    # if df.shape[0]==0:
        # df['plvLoc+'] = None
    # else:
        # df['plvLoc+'] = loc_model(df)
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
    
    df['3D wOBAcon'] = [None if any(np.isnan([x,y,z])) else sum(np.multiply(xwOBAcon_model.predict_proba([[x,y,z]])[0],np.array([0,0.9,1.25,1.6,2]))) for x,y,z in zip(df['Spray Angle'].astype('float'),df['Launch Angle'].astype('float'),df['Launch Speed'].astype('float'))]

    game_df = df.assign(vs_rhh = lambda x: np.where(x['hitterside']=='R',1,0)).groupby(['game_date','Opp','MLBAMID','Pitcher','pitch_type'])[['Num Pitches','Velo','IVB','IHB','Ext','vs_rhh','CS','Whiffs','total_strikes','3D wOBAcon','HAVAA',
                                                                                                                                              # 'plvLoc+'
                                                                                                                                             ]].agg({
        'Num Pitches':'count',
        'Velo':'mean',
        'IVB':'mean',
        'IHB':'mean',
        'Ext':'mean',
        'vs_rhh':'sum',
        'CS':'sum',
        'Whiffs':'sum',
        'total_strikes':'sum',
        '3D wOBAcon':'mean',
        'HAVAA':'mean',
        # 'plvLoc+':'mean'
    }).assign(CSW = lambda x: x['CS'].add(x['Whiffs']).div(x['Num Pitches']).mul(100),
              strike_rate = lambda x: x['total_strikes'].div(x['Num Pitches']).mul(100),
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
    merge_df['Strike%'] = [f'{x:.1f}%' for x in merge_df['strike_rate']]
    merge_df['vs R'] = [f'{x:.1%}' for x in merge_df['vs_rhh']]
    merge_df['vs L'] = [f'{x:.1%}' for x in merge_df['vs_lhh']]
    merge_df['Ext'] = [f'{x:.1f}ft' for x in merge_df['Ext']]
    merge_df['3D wOBAcon'] = merge_df['3D wOBAcon'].round(3)
    merge_df['HAVAA'] = [f'{x:.1f}°' for x in merge_df['HAVAA']]
    # merge_df['plvLoc+'] = merge_df['plvLoc+'].round(0).astype('int')

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

    return merge_df[['Date','Opp','Pitcher','Type','Num Pitches','Velo','Usage','vs R','vs L','Ext','IVB','IHB','HAVAA','Strike%','CS','Whiffs','CSW','3D wOBAcon']], df

def game_charts(move_df):
    fig = plt.figure(figsize=(8,8))
    grid = plt.GridSpec(1, 3, width_ratios=[1,2,1],wspace=0.15)
    ax1 = plt.subplot(grid[1])
    circle1 = plt.Circle((0, 0), 6, color=pl_white,fill=False,alpha=0.2,linestyle='--')
    ax1.add_patch(circle1)
    circle2 = plt.Circle((0, 0), 12, color=pl_white,fill=False,alpha=0.5)
    ax1.add_patch(circle2)
    circle3 = plt.Circle((0, 0), 18, color=pl_white,fill=False,alpha=0.2,linestyle='--')
    ax1.add_patch(circle3)
    circle4 = plt.Circle((0, 0), 24, color=pl_white,fill=False,alpha=0.5)
    ax1.add_patch(circle4)
    ax1.axvline(0,ymin=4/58,ymax=54/58,color=pl_white,alpha=0.5,zorder=1)
    ax1.axhline(0,xmin=4/58,xmax=54/58,color=pl_white,alpha=0.5,zorder=1)
    
    for dist in [12,24]:
        label_dist = dist-0.25
        ax1.text(label_dist,-0.3,f'{dist}"',ha='right',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(-label_dist,-0.3,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,label_dist-0.25,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,-label_dist,f'{dist}"',ha='left',va='bottom',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
    
    if move_df['P Hand'].value_counts().index[0]=='R':
        ax1.text(28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    else:
        ax1.text(28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
        ax1.text(-28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    
    ax1.text(0,27,'Rise',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    ax1.text(0,-27,'Drop',ha='center',va='center',fontsize=8,color=pl_white,alpha=0.75,zorder=1)
    
    sns.scatterplot(move_df.assign(IHB = lambda x: np.where(x['P Hand']=='L',x['IHB'].astype('float').mul(-1),x['IHB'].astype('float'))),
                    x='IHB',
                    y='IVB',
                   hue='pitch_type',
                   palette=marker_colors,
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    zorder=10,
                   ax=ax1)    
    
    handles, labels = ax1.get_legend_handles_labels()
    pitch_type_names = [pitch_names[x] for x in labels]
    # pitch_type_names = [pitch_names[x].ljust(15, " ") for x in labels]
    ax1.legend(handles,[pitch_names[x] for x in labels], ncols=len(labels),
             loc='lower center', 
               fontsize=min(52/len(labels),14),
              framealpha=0,bbox_to_anchor=(0.5, -0.23+len(labels)/100,
                                           0,0))
    
    ax1.set(xlim=(-29,29),
           ylim=(-29,29),
           aspect=1,
           title='Movement')
    ax1.axis('off')
    
    ax2 = plt.subplot(grid[0])
    # Outer Strike Zone
    zone_outline = plt.Rectangle((-10/12, sz_bot), 20/12, 2, color=pl_white,fill=False,alpha=alpha_val)
    ax2.add_patch(zone_outline)
    
    # Inner Strike zone
    ax2.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    
    # Plate
    ax2.plot([-8.5/12,8.5/12], [plate_y,plate_y], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    ax2.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    
    sns.scatterplot(data=move_df.loc[move_df['hitterside']=='L'].assign(p_x = lambda x: x['p_x']*-1),
                    x='p_x',
                    y='p_z',
                    hue='pitch_type',
                    palette=marker_colors,
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    legend=False,
                   zorder=10,
                   ax=ax2)
    
    ax2.set(xlim=(-1.5,1.5),
           ylim=(y_bot,y_lim-1),
           aspect=1,
           title='Locations\nvs LHH')
    ax2.axis('off')
    
    ax3 = plt.subplot(grid[2])
    # Outer Strike Zone
    zone_outline = plt.Rectangle((-10/12, sz_bot), 20/12, 2, color=pl_white,fill=False,alpha=alpha_val)
    ax3.add_patch(zone_outline)
    
    # Inner Strike zone
    ax3.plot([-10/12,10/12], [1.5+2/3,1.5+2/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([-10/12,10/12], [1.5+4/3,1.5+4/3], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.axvline(10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.axvline(-10/36, ymin=(sz_bot-y_bot)/(y_lim-1-y_bot), ymax=(sz_top-y_bot)/(y_lim-1-y_bot), color=pl_white, linewidth=1, alpha=alpha_val)
    
    # Plate
    ax3.plot([-8.5/12,8.5/12], [plate_y,plate_y], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([-8.5/12,-8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([8.5/12,8.25/12], [plate_y,plate_y+0.15], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    ax3.plot([-8.28/12,0], [plate_y+0.15,plate_y+0.25], color=pl_white, linewidth=1, alpha=alpha_val)
    
    sns.scatterplot(data=move_df.loc[move_df['hitterside']=='R'].assign(p_x = lambda x: x['p_x']*-1),
                    x='p_x',
                    y='p_z',
                    hue='pitch_type',
                    palette=marker_colors,
                    edgecolor=pl_white,
                    s=85,
                    linewidth=0.3,
                    alpha=1,
                    legend=False,
                   zorder=10,
                   ax=ax3)
    
    ax3.set(xlim=(-1.5,1.5),
           ylim=(y_bot,y_lim-1),
           aspect=1,
           title='Locations\nvs RHH')
    ax3.axis('off')
    
    logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    pl_ax = fig.add_axes([0.4,0.175,0.2,0.1], anchor='NE', zorder=1)
    width, height = logo.size
    pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    pl_ax.axis('off')

    fig.suptitle(f"{player_select}'s Pitch Charts ({date.strftime('%m/%d/%y')})",y=0.75)
    sns.despine()
    st.pyplot(fig,use_container_width=False)
    
if len(list(pitcher_list.keys()))==0:
    st.write('No pitches thrown yet')
elif st.button("Generate Player Table"):
    table_df, chart_df = scrape_savant_data(player_select,game_id)
    st.dataframe(table_df,
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
                         <0 means steeper than other pitches at that location
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

    game_charts(chart_df)
