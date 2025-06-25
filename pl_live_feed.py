import streamlit as st
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pytz
import requests
import seaborn as sns
import urllib

import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py

from sklearn.neighbors import KNeighborsClassifier
from datetime import timedelta

from PIL import Image

#st.write()

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

### Calculate the differences between each pitch and their avg fastball
def fastball_differences(dataframe,stat):
    dataframe[stat] = dataframe[stat].astype('float')
    temp_df = dataframe.loc[dataframe['pitch_type']==dataframe['fastball_type']].groupby(['MLBAMID','game_date','pitch_type'], as_index=False)[stat].mean().rename(columns={stat:'fb_'+stat})
    dataframe = dataframe.merge(temp_df,
                                left_on=['MLBAMID','game_date','fastball_type'],
                                right_on=['MLBAMID','game_date','pitch_type']).drop(columns=['pitch_type_y']).rename(columns={'pitch_type_x':'pitch_type'})
    return dataframe[stat].sub(dataframe['fb_'+stat])

def feature_engineer(dataframe):
    category_feats = ['P Hand',#'pitcherside',
                      'hitterside',
                      'balls',#'balls_before_pitch',
                      'strikes',#'strikes_before_pitch'
                     ]
    dataframe['stand'] = dataframe['hitterside'].copy()
    dataframe['throw'] = dataframe['P Hand'].copy()
    dataframe['balls_before_pitch'] = dataframe['balls'].copy()
    dataframe['strikes_before_pitch'] = dataframe['strikes'].copy()
    dataframe = pd.get_dummies(dataframe, columns=category_feats)

    for hand in ['L','R']:
        if f'P Hand_{hand}' not in dataframe:
            dataframe[f'P Hand_{hand}'] = False
        if f'hitterside_{hand}' not in dataframe:
            dataframe[f'hitterside_{hand}'] = False
    for balls in [0,1,2,3]:
        if f'balls_{balls}' not in dataframe:
            dataframe[f'balls_{balls}'] = False
    for strikes in [0,1,2]:
        if f'strikes_{strikes}' not in dataframe:
            dataframe[f'strikes_{strikes}'] = False
    # Pythagorean movement
    dataframe['total_IB'] = (dataframe['IHB'].astype('float')**2+dataframe['IVB'].astype('float')**2)**0.5
    
    # df of most common fastballs for each pitcher, in each appearance
    fastballs = ['FF','FC','FT','SI']
    fastball_df = (dataframe
                   .loc[dataframe['pitch_type'].isin(fastballs)]
                   .groupby(['MLBAMID','game_date'], as_index=False)
                   ['pitch_type']
                   .agg(pd.Series.mode)
                   .rename(columns={'pitch_type':'fastball_type'})
                   .copy()
                  )
  
    # Add most common Fastball type
    dataframe = dataframe.merge(fastball_df,on=['MLBAMID','game_date'], how='left')
    dataframe['fastball_type'] = dataframe['fastball_type'].fillna('NA').apply(lambda x: x if len(x[0])==1 else x[0])

    # Add comparison stats to fastball
    for stat in ['IHB','IVB','Velo']:
        dataframe[stat+'_diff'] = fastball_differences(dataframe,stat)
    dataframe['total_IB_diff'] = (dataframe['IHB_diff'].astype('float')**2+dataframe['IVB_diff'].astype('float')**2)**0.5
    
    return dataframe.rename(columns={'stand':'hitterside','throw':'P Hand',
                                     'balls_before_pitch':'balls','strikes_before_pitch':'strikes'})

bip_result_dict = {
    '10-20deg: 100-105mph': {'out': 0.3336331744175587,'single': 0.34973464064046056,'double': 0.2876675362058109,'triple': 0.023207699919042906,'home_run': 0.005756948817126923},
    '10-20deg: 105+mph': {'out': 0.23763796909492274,'single': 0.27759381898454744,'double': 0.4052980132450331,'triple': 0.02240618101545254,'home_run': 0.05706401766004415},
    '10-20deg: 90-95mph': {'out': 0.47004786166435075,'single': 0.4350779681951521,'double': 0.08537903350316504,'triple': 0.00895476300756523,'home_run': 0.0005403736297668674},
    '10-20deg: 95-100mph': {'out': 0.3468848460209371,'single': 0.44598028254903954,'double': 0.18904360199207237,'triple': 0.017786360402479925,'home_run': 0.0003049090354710845},
    '10-20deg: <90mph': {'out': 0.3034200193478632,'single': 0.6208956922551642,'double': 0.071074944517157,'triple': 0.0045524383998179025,'home_run': 5.690547999772378e-05},
    '10deg: 100-105mph': {'out': 0.5532356374011153,'single': 0.3946310465568668,'double': 0.048675052954653526,'triple': 0.0034150347987723165,'home_run': 4.322828859205464e-05},
    '10deg: 105+mph': {'out': 0.472506256703611,'single': 0.4556310332499106,'double': 0.06828745084018592,'triple': 0.003575259206292456,'home_run': 0.0},
    '10deg: 90-95mph': {'out': 0.7286569556404092,'single': 0.2379672842125112,'double': 0.03125441945976524,'triple': 0.0021213406873143827,'home_run': 0.0},
    '10deg: 95-100mph': {'out': 0.6431076885860582,'single': 0.3165746079103334,'double': 0.03749546425835584,'triple': 0.00282223924525259,'home_run': 0.0},
    '10deg: <90mph': {'out': 0.8404299549990217,'single': 0.1465711211113285,'double': 0.012509782821365683,'triple': 0.0004891410682840931,'home_run': 0.0},
    '20-30deg: 100-105mph': {'out': 0.3110007397231322,'single': 0.013526365845926239,'double': 0.22519285638803763,'triple': 0.02832082848990806,'home_run': 0.4219592095529959},
    '20-30deg: 105+mph': {'out': 0.051047120418848166,'single': 0.01030759162303665,'double': 0.11861910994764398,'triple': 0.012107329842931938,'home_run': 0.8079188481675392},
    '20-30deg: 90-95mph': {'out': 0.8553492021972273,'single': 0.020533612346324875,'double': 0.10044467695527073,'triple': 0.009678263144127649,'home_run': 0.013994245357049438},
    '20-30deg: 95-100mph': {'out': 0.6610534962521819,'single': 0.013040353219016327,'double': 0.19108738063456207,'triple': 0.019201150015401992,'home_run': 0.11561761987883766},
    '20-30deg: <90mph': {'out': 0.5885616242493565,'single': 0.3451529882756649,'double': 0.06050900772090363,'triple': 0.005661995996568487,'home_run': 0.00011438375750643409},
    '30-40deg: 100-105mph': {'out': 0.5065131206343213,'single': 0.0013215027373985274,'double': 0.030960921276194073,'triple': 0.012648669057957335,'home_run': 0.44855578629412873},
    '30-40deg: 105+mph': {'out': 0.17124939700916547,'single': 0.001447178002894356,'double': 0.013989387361312108,'triple': 0.003376748673420164,'home_run': 0.8099372889532079},
    '30-40deg: 90-95mph': {'out': 0.9417808219178082,'single': 0.0008933889219773674,'double': 0.024865991661703394,'triple': 0.004169148302561048,'home_run': 0.02829064919594997},
    '30-40deg: 95-100mph': {'out': 0.8001646994235521,'single': 0.0015097447158934944,'double': 0.031155640955256657,'triple': 0.00947021685424101,'home_run': 0.15769969805105682},
    '30-40deg: <90mph': {'out': 0.8385996524872901,'single': 0.131218225111011,'double': 0.02741489156316365,'triple': 0.0020593345775146406,'home_run': 0.0007078962610206577},
    '40-50deg: 100-105mph': {'out': 0.9092651757188498,'single': 0.0012779552715654952,'double': 0.01597444089456869,'triple': 0.0019169329073482429,'home_run': 0.07156549520766774},
    '40-50deg: 105+mph': {'out': 0.6349693251533742,'single': 0.006134969325153374,'double': 0.015337423312883436,'triple': 0.003067484662576687,'home_run': 0.34049079754601225},
    '40-50deg: 90-95mph': {'out': 0.9949664429530202,'single': 0.0006291946308724832,'double': 0.002936241610738255,'triple': 0.0006291946308724832,'home_run': 0.0008389261744966443},
    '40-50deg: 95-100mph': {'out': 0.9731580388934539,'single': 0.0008216926869350862,'double': 0.008216926869350863,'triple': 0.003286770747740345,'home_run': 0.014516570802519857},
    '40-50deg: <90mph': {'out': 0.9243428936646161,'single': 0.057361987696732446,'double': 0.018055444595350325,'triple': 0.00015978269553407366,'home_run': 7.989134776703683e-05},
    '50+deg': {'out': 0.9862377429898503,'single': 0.008515396525030104,'double': 0.005132175010034979,'triple': 0.00011468547508458054,'home_run': 0.0}
}

def apply_plv_outcomes(model_df):
    model_df[['take_input','swing_input','ooz_raw','called_strike_raw','ball_raw',
                'hit_by_pitch_raw','swinging_strike_raw','contact_raw',
                'foul_strike_raw','in_play_raw','10deg_raw','10-20deg_raw','20-30deg_raw',
              '30-40deg_raw','40-50deg_raw','50+deg_raw','50+deg_pred',
                'called_strike_pred','ball_pred','hit_by_pitch_pred','ooz_input','contact_input',
                'swinging_strike_pred','foul_strike_pred','in_play_input','50+deg_pred',
                'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']] = None
    
    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        model_df[[launch_angle+'_input',launch_angle+': <90mph_raw',
                 launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',
                 launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw',
                 launch_angle+': <90mph_pred',launch_angle+': 90-95mph_pred',
                 launch_angle+': 95-100mph_pred',launch_angle+': 100-105mph_pred',
                 launch_angle+': 105+mph_pred']] = None

    for pitch_type in ['Fastball','Breaking_Ball','Offspeed']:
        if model_df.loc[model_df['pitch_group']==pitch_type].shape[0]==0:
            continue
        # Swing Decision
        with open('model_files/live_swing_model_{}.pkl'.format(pitch_type), 'rb') as f:
            decision_model = pickle.load(f)
    
        model_df.loc[model_df['pitch_group']==pitch_type,['take_input','swing_input']] = decision_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,decision_model.feature_names_in_])
        
        # Out-of-Zone Take Result
        with open('model_files/live_ca_strike_model_{}.pkl'.format(pitch_type), 'rb') as f:
            called_strike_model = pickle.load(f)
    
        model_df.loc[model_df['pitch_group']==pitch_type,['called_strike_raw','ooz_raw']] = called_strike_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,called_strike_model.feature_names_in_])
        model_df.loc[model_df['pitch_group']==pitch_type,'called_strike_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,'called_strike_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'take_input'])
        model_df.loc[model_df['pitch_group']==pitch_type,'ooz_input'] = model_df.loc[model_df['pitch_group']==pitch_type,'ooz_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'take_input'])
    
        # HBP Result
        with open('model_files/live_hbp_model_{}.pkl'.format(pitch_type), 'rb') as f:
            hbp_model = pickle.load(f)
    
        model_df.loc[model_df['pitch_group']==pitch_type,['ball_raw','hit_by_pitch_raw']] = hbp_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,hbp_model.feature_names_in_])
        model_df.loc[model_df['pitch_group']==pitch_type,'ball_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,'ball_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'ooz_input'])
        model_df.loc[model_df['pitch_group']==pitch_type,'hit_by_pitch_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,'hit_by_pitch_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'ooz_input'])
    
        # Swing Result
        with open('model_files/live_contact_model_{}.pkl'.format(pitch_type), 'rb') as f:
            swing_result_model = pickle.load(f)
    
        model_df.loc[model_df['pitch_group']==pitch_type,['swinging_strike_raw','contact_raw']] = swing_result_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,swing_result_model.feature_names_in_])
        model_df.loc[model_df['pitch_group']==pitch_type,'contact_input'] = model_df.loc[model_df['pitch_group']==pitch_type,'contact_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'swing_input'])
        model_df.loc[model_df['pitch_group']==pitch_type,'swinging_strike_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,'swinging_strike_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'swing_input'])
    
        # Contact Result
        with open('model_files/live_in_play_model_{}.pkl'.format(pitch_type), 'rb') as f:
            contact_model = pickle.load(f)
    
        model_df.loc[model_df['pitch_group']==pitch_type,['foul_strike_raw','in_play_raw']] = contact_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,contact_model.feature_names_in_])
        model_df.loc[model_df['pitch_group']==pitch_type,'foul_strike_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,'foul_strike_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'contact_input'])
        model_df.loc[model_df['pitch_group']==pitch_type,'in_play_input'] = model_df.loc[model_df['pitch_group']==pitch_type,'in_play_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'contact_input'])
    
        # Launch Angle Result
        with open('model_files/live_launch_angle_model_{}.pkl'.format(pitch_type), 'rb') as f:
            launch_angle_model = pickle.load(f)
    
        model_df.loc[model_df['pitch_group']==pitch_type,['10deg_raw','10-20deg_raw','20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw']] = launch_angle_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,launch_angle_model.feature_names_in_])
        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            model_df.loc[model_df['pitch_group']==pitch_type,launch_angle+'_input'] = model_df.loc[model_df['pitch_group']==pitch_type,launch_angle+'_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'in_play_input'])
        model_df.loc[model_df['pitch_group']==pitch_type,'50+deg_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,'50+deg_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,'in_play_input'])
    
        # Launch Velo Result
        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            with open('model_files/live_{}_model_{}.pkl'.format(launch_angle,pitch_type), 'rb') as f:
                launch_velo_model = pickle.load(f)
    
            model_df.loc[model_df['pitch_group']==pitch_type,[launch_angle+': <90mph_raw',launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw']] = launch_velo_model.predict_proba(model_df.loc[model_df['pitch_group']==pitch_type,launch_velo_model.feature_names_in_])
            for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                model_df.loc[model_df['pitch_group']==pitch_type,bucket+'_pred'] = model_df.loc[model_df['pitch_group']==pitch_type,bucket+'_raw'].mul(model_df.loc[model_df['pitch_group']==pitch_type,launch_angle+'_input'])

    for outcome in ['out', 'single', 'double', 'triple', 'home_run']:
        # Start with 50+ degrees (popups)
        model_df[outcome+'_pred'] = model_df['50+deg_pred']*bip_result_dict['50+deg'][outcome]
        
        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                model_df[outcome+'_pred'] += model_df[bucket+'_pred']*bip_result_dict[bucket][outcome]
    return model_df[['called_strike_pred','ball_pred','hit_by_pitch_pred',
                     'swinging_strike_pred','foul_strike_pred',
                     'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']]

def generate_games(games_today):
    game_dict = {}
    code_dict = {
        'F':0,
        'U':0,
        'O':1,
        'I':1,
        'N':1,
        'P':2,
        'S':2,
        'D':2
    }
    for game in games_today:
        r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game}')
        x = r.json()
        game_hour = int(x['scoreboard']['datetime']['dateTime'][11:13])
        game_hour = game_hour-4 if game_hour >3 else game_hour+20
        game_minutes = int(x['scoreboard']['datetime']['dateTime'][14:16])
        raw_time = game_hour*60+game_minutes
        am_pm = 'AM' if game_hour <12 else 'PM'
        game_time = f'{game_hour-12}:{game_minutes:>02}{am_pm}' if (am_pm=='PM') & (game_hour!=12) else f'{game_hour}:{game_minutes:>02}{am_pm}'
        ppd = 0 if x['scoreboard']['datetime']['originalDate']==x['scoreboard']['datetime']['officialDate'] else 1
        
        away_team = x['scoreboard']['teams']['away']['abbreviation']
        home_team = x['scoreboard']['teams']['home']['abbreviation']
        game_status_code = x['game_status_code']
        code_map = code_dict[game_status_code]
        if game_status_code  in ['P','S','D']:
            game_info = f'{away_team} @ {home_team}: {game_time}'
            inning_sort = None
        # elif game_status_code =='S':
        #     game_info = f'PPD: {away_team} @ {home_team}'
        #     inning_sort = None
        else:
            game_info = f'{away_team} @ {home_team}'
            home_runs = x['scoreboard']['linescore']['teams']['home']['runs']
            away_runs = x['scoreboard']['linescore']['teams']['away']['runs']
            inning = x['scoreboard']['linescore']['currentInning']
            top_bot = x['scoreboard']['linescore']['inningHalf'][0]
            inning_sort = int(inning)*2 - (0 if top_bot=='Bottom' else 1)
            if game_status_code == 'F':
                if home_runs>away_runs:
                    game_info = f'FINAL: {away_team} {away_runs} @ **:green[{home_team} {home_runs}]**'
                else:
                    game_info = f'FINAL: **:green[{away_team} {away_runs}]** @ {home_team} {home_runs}'
            else:
                game_info = f'{top_bot}{inning}: {away_team} {away_runs} @ {home_team} {home_runs}'
        game_dict.update({game_info:[game,game_time,raw_time,inning_sort,code_map]})
    game_df = pd.DataFrame.from_dict(game_dict, orient='index',columns=['Game ID','Time','Sort Time','Sort Inning','Sort Code'])
    return game_df.sort_values(['Sort Code','Sort Time','Game ID','Sort Inning'])['Game ID'].to_dict()

st.set_page_config(page_title='PL Live Pitching Stats', page_icon='⚾',layout="wide")

@st.cache_resource()
def load_logo():
    logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    img_url = urllib.request.urlopen(logo_loc)
    logo = Image.open(img_url)
    return logo
    
logo = load_logo()
st.image(logo, width=200)

st.title('PL Live Pitching Stats')
st.write('Data (especially pitch types) are subject to change.')
col1, col2, col3 = st.columns([0.25,0.5,0.25])

with col1:
    today = (datetime.datetime.now(pytz.utc)-timedelta(hours=16)).date()
    input_date = st.date_input("Select a game date:", today, 
                         min_value=datetime.date(2024, 2, 19), max_value=today+timedelta(days=2))

    if 'date' not in st.session_state:
        st.session_state['date'] = input_date

    date = input_date#st.session_state['date']
    
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}')
    x = r.json()
    if x['totalGames']==0:
        st.write(f'No games on {date}')
    # game_list = {}
    # for game in range(len(x['dates'][0]['games'])):
    #     if x['dates'][0]['games'][game]['gamedayType'] in ['E','P']:
    #         game_list.update({x['dates'][0]['games'][game]['teams']['away']['team']['name']+' @ '+x['dates'][0]['games'][game]['teams']['home']['team']['name']:x['dates'][0]['games'][game]['gamePk']})
    games_today = []
    for game in range(len(x['dates'][0]['games'])):
        if x['dates'][0]['games'][game]['gamedayType'] in ['E','P']:
            games_today += [x['dates'][0]['games'][game]['gamePk']]
    game_list = generate_games(games_today)
with col2:
    input_game = st.pills('Choose a game (all times EST):',list(game_list.keys()),default=list(game_list.keys())[0],
                           key='game')
    
    if 'game' not in st.session_state:
        st.session_state['game'] = input_game

    game_select = st.session_state['game']
    
    game_id = game_list[game_select]
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    game_code = x['game_status_code']
    if (len(x['home_pitcher_lineup'])>0) | (len(x['away_pitcher_lineup'])>0):
        pitcher_lineup = [x['home_pitcher_lineup'][0]]+[x['away_pitcher_lineup'][0]]+([] if len(x['home_pitcher_lineup'])==1 else x['home_pitcher_lineup'][1:])+([] if len(x['away_pitcher_lineup'])==1 else x['away_pitcher_lineup'][1:])
        home_team = [1]+[0]+([] if len(x['home_pitcher_lineup'])==1 else [1]*(len(x['home_pitcher_lineup'])-1))+([] if len(x['away_pitcher_lineup'])==1 else [0]*(len(x['away_pitcher_lineup'])-1))
        test_list = {}
        for home_away_pitcher in ['home','away']:
            if f'{home_away_pitcher}_pitchers' not in x.keys():
                continue
            for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
                test_list.update({pitcher_id:x[f'{home_away_pitcher}_pitchers'][pitcher_id][0]['pitcher_name']})
        pitcher_lineup = [x for x in pitcher_lineup if str(x) in test_list.keys()]
        if len(test_list.keys())>0:
            pitcher_list = {test_list[str(x)]:[str(x),y] for x,y in zip(pitcher_lineup,home_team)}
        else:
            pitcher_list = {}
    else:
        pitcher_list = {}
        
#@st.cache_data()
def load_season_avgs(timeframe):
    if timeframe=='2025':
        df = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/season_to_date.parquet?raw=true')
    else:
        df = pd.read_parquet('https://github.com/Blandalytics/PLV_viz/blob/main/season_avgs_2024.parquet?raw=true').rename(columns={
            'pitcher':'MLBAMID',
            'release_speed':'Velo',
            'pfx_x':'IHB'
            })
    return df
    
with col3:
    # Game Line
    if len(list(pitcher_list.keys()))>0:
        input_player = st.selectbox('Choose a pitcher:',list(pitcher_list.keys()),
                                     key='pitcher')
        if 'pitcher' not in st.session_state:
            st.session_state['pitcher'] = input_player
    
        player_select = st.session_state['pitcher']
        home=pitcher_list[player_select][1]
        stat_base = x['boxscore']['teams']['home' if home==1 else 'away']['players'][f'ID{pitcher_list[player_select][0]}']['stats']['pitching']
        game_summary = stat_base['summary']
        team = x['scoreboard']['teams']['home' if home==1 else 'away']['abbreviation']
        opp = x['scoreboard']['teams']['home' if home==0 else 'away']['abbreviation']
        starter = stat_base['gamesStarted']
        innings = stat_base['inningsPitched']
        outs = stat_base['outs']
        earned_runs = stat_base['earnedRuns']
        tbf = stat_base['battersFaced']
        hits = stat_base['hits']
        strikeouts = stat_base['strikeOuts']
        walks = stat_base['baseOnBalls']
        win = stat_base['wins']
        loss = stat_base['losses']
        save = stat_base['saves']
        hold = stat_base['holds']
        blown_save = stat_base['blownSaves']
        home_away = 'vs' if home==1 else '@'
        supplemental_decision  = ', QS' if (int(innings[0])>=6) and (int(earned_runs)<=3) else ', BS' if blown_save==1 else ''
        decision = f'(ND{supplemental_decision})' if (win+loss==0) and (starter==1) else f'(W{supplemental_decision})' if win==1 else f'(L{supplemental_decision})' if loss==1 else '(SV)' if save==1 else '(HD)' if hold==1 else ''
        decision = decision if game_code == 'F' else ''
        timeframe = st.radio('Select a timeframe for comparison:',['2025','2024'],horizontal=True)
        comp_data = load_season_avgs(timeframe)
    else:
        away_pitcher = x['scoreboard']['probablePitchers']['away']['fullName']
        home_pitcher = x['scoreboard']['probablePitchers']['home']['fullName']
        st.write(f'Probable Pitchers: {away_pitcher} @ {home_pitcher}')

if len(list(pitcher_list.keys()))>0:
    if timeframe=='2025':
        comp_data['game_date'] = pd.to_datetime(comp_data['game_date'])
        season_avgs = (
            comp_data.loc[comp_data['game_date']!=datetime.datetime(date.year, date.month, date.day)]
            .groupby(['MLBAMID','Pitcher','pitch_type'])
            [['game_pk','Velo','IVB','IHB']]
            .agg({
                'game_pk':'count',
                'Velo':'mean',
                'IVB':'mean',
                'IHB':'mean'
                })
            .reset_index()
            )
        season_avgs['Usage'] = season_avgs['game_pk'].div(season_avgs['game_pk'].groupby(season_avgs['MLBAMID']).transform('sum')).mul(100)
    else:
        season_avgs = comp_data
    st.subheader(f'{date.strftime('%-m/%-d/%y')}: {player_select} {home_away} {opp} {decision} - {innings} IP, {earned_runs} ER, {hits} Hits, {walks} BBs, {strikeouts} Ks')
    col1, col2, col3 = st.columns(3)
    with col1:
        count_select = st.radio('Count Group', 
                                ['All','Hitter-Friendly','Pitcher-Friendly','Even','2-Strike','3-Ball','Custom'],
                                index=0,
                                horizontal=True
                               )
         
        if count_select=='All':
            counts = ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2']
        elif count_select=='Hitter-Friendly':
            counts = ['1-0', '2-0', '3-0', '2-1', '3-1']
        elif count_select=='Pitcher-Friendly':
            counts = ['0-1','0-2','1-2']
        elif count_select=='Even':
            counts = ['0-0','1-1','2-2']
        elif count_select=='2-Strike':
            counts = ['0-2','1-2','2-2','3-2']
        elif count_select=='3-Ball':
            counts = ['3-0','3-1','3-2']
        else:
            counts = st.multiselect('Select the count(s):',
                                    ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2'],
                                    ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2'])
    with col2:
        home_away = 'home' if home==1 else 'away'
        
        inning_min = x[f'{home_away}_pitchers'][pitcher_list[player_select][0]][0]['inning']
        inning_max = x[f'{home_away}_pitchers'][pitcher_list[player_select][0]][-1]['inning']
        if inning_min!=inning_max:
            start_inning, end_inning = st.select_slider('Innings',
                                                        options=list(range(inning_min,inning_max+1)),
                                                        value=(inning_min,inning_max))
        else:
            start_inning, end_inning = inning_min, inning_max

with open('2025_3d_xwoba_model.pkl', 'rb') as f:
    xwOBAcon_model = pickle.load(f)

# with open('arm_angle_model.pkl', 'rb') as f:
#     arm_angle_knn = pickle.load(f)

sz_bot = 1.5
sz_top = 3.5
x_ft = 2.5
y_bot = -0.5
y_lim = 6
plate_y = -.25
alpha_val = 1

pitchtype_map = {
    'FF':'FF',
    'FA':'FF',
    'SI':'SI',
    'FT':'SI',
    'FC':'FC',
    'SL':'SL',
    'ST':'ST',
    'CH':'CH',
    'CU':'CU',
    'KC':'CU',
    'CS':'CU',
    'SV':'CU',
    'FS':'FS',
    'FO':'FS',
    'SC':'CH',
    'KN':'KN',
    'UN':'UN',
    'EP':'UN'
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
    'CS':'#3c44cd',
    'SV':'#3c44cd',
    'CH':'#07b526', 
    'SC':'#07b526', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
}

pitch_names = {
    'FF':'Four-Seam', 
    'FA':'Fastball',
    'SI':'Sinker',
    'FT':'Two-Seam',
    'FS':'Splitter',  
    'FO':'Forkball',
    'FC':'Cutter', 
    'SL':'Slider', 
    'ST':'Sweeper',
    'CU':'Curveball',
    'KC':'Knuckle Curve',
    'CS':'Slow Curve',
    'SV':'Slurve',
    'CH':'Changeup', 
    'KN':'Knuckleball',
    'SC':'Screwball', 
    'UN':'Unknown', 
    'EP':'Eephus'
}

pitchgroup_map = {
    'FF':'Fastball',
    'FA':'Fastball',
    'SI':'Fastball',
    'FT':'Fastball',
    'FC':'Breaking_Ball',
    'SL':'Breaking_Ball',
    'ST':'Breaking_Ball',
    'CH':'Offspeed',
    'CU':'Breaking_Ball',
    'KC':'Breaking_Ball',
    'CS':'Breaking_Ball',
    'SV':'Breaking_Ball',
    'FS':'Offspeed',
    'FO':'Offspeed',
    'KN':'Offspeed',
    'SC':'Offspeed',
    'UN':'UN',
    'EP':'Offspeed'
}

def player_height(mlbamid):
    mlbamid = int(mlbamid)
    url = f'https://statsapi.mlb.com/api/v1/people?personIds={mlbamid}&fields=people,id,height,weight'
    response = requests.get(url)
    # raise an HTTPError if the request was unsuccessful
    response.raise_for_status()
    return sum(list(map(lambda x, y: int(x)*y, response.json()['people'][0]['height'].replace("\'","").replace('"',"").split(' '),[1,1/12])))

def arm_angle(x0,z0,extension,height):
    return -43 - 33.1 * (abs(x0)/height) + 94 * (z0/height) + 4.4 * (extension/height)
    
@st.cache_data(ttl='1m',show_spinner='Loading player data')
def scrape_savant_data(player_name, game_id, counts, start_inning, end_inning):
    game_ids = []
    game_date = []
    pitcher_id_list = []
    pitcher_name = []
    pitcher_height = []
    hitter_name = []
    throws = []
    stands = []
    inning = []
    pitch_call = []
    events = []
    result_code = []
    pa = []
    zone = []
    balls = []
    strikes = []
    pitch_id = []
    pitch_type = []
    velo = []
    extension = []
    called_strikes = []
    swinging_strikes = []
    foul_strikes = []
    swing = []
    total_strikes = []
    total_balls = []
    ivb = []
    ihb = []
    x0 = []
    z0 = []
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
        if (x['game_status_code'] in ['S']):
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
            height = player_height(pitcher_id)
            if pitcher_id != pitcher_list[player_select][0]:
                continue
            for pitch in range(len(x[f'{home_away_pitcher}_pitchers'][pitcher_id])):
                game_ids += [game_id]
                game_date += [x['gameDate']]
                pitcher_id_list += [pitcher_id]
                p_name = x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitcher_name']
                pitcher_name += [p_name]
                pitcher_height += [height]
                hitter_name += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['batter_name']]
                throws += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['p_throws']]
                stands += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['stand']]
                inning += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inning']]
                pitch_call += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call']]
                try:
                    events += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['events']]
                except KeyError:
                    events += [None]
                result_code += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['result_code']]
                pa += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ab_number']]
                if 'zone' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys():
                    zone += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['zone'] <10 else 0]
                called_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call']=='called_strike' else 0]
                swinging_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call'] in ['swinging_strike','foul_tip','swinging_strike_blocked'] else 0]
                swing += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['is_strike_swinging']]
                foul_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call']=='foul' else 0]
                total_strikes += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call'] in ['swinging_strike','foul_tip','swinging_strike_blocked','missed_bunt',
                                                                                                               'called_strike','foul','foul_bunt','foul_pitchout','hit_into_play'] else 0]
                total_balls += [1 if x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_call'] in ['ball','pitchout','ball_in_dirt','hit_by_pitch','blocked_ball','intentional_ball'] else 0]
                balls += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['balls']]
                strikes += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['strikes']]
                pitch_id += [pitch+1]
                try:
                    pitch_type += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_type']]
                    velo += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['start_speed']]
                    extension += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['extension'] if 'extension' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else None]
                    ivb += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pfxZWithGravity']]
                    ihb += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pfxXNoAbs']]
                    x0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['x0']]
                    z0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['z0']]
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
                    if 'zone' not in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys():
                        zone += [1 if (abs(float(px))<10/12) & (float(pz) >= float(sz_bot)) & (float(pz) <= float(sz_top)) else 0]
                except KeyError:
                    pitch_type += ['UN']
                    velo += [None]
                    extension += [None]
                    ivb += [None]
                    ihb += [None]
                    x0 += [None]
                    z0 += [None]
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
                    if 'zone' not in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys():
                        zone += [None]
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
    df['MLBAMID'] = pitcher_id_list
    df['MLBAMID'] = df['MLBAMID'].astype('int')
    df['balls'] = balls
    df['strikes'] = strikes
    df['count'] = df['balls'].astype('str')+'-'+df['strikes'].astype('str')
    df['Pitcher'] = pitcher_name
    df['Height'] = pitcher_height
    df['Hitter'] = hitter_name
    df['P Hand'] = throws
    df['hitterside'] = stands
    df['pitch_call'] = pitch_call
    df['event'] = events
    df['result_code'] = result_code
    df['PA'] = pa
    df['PA'] = np.where(df['PA']!=df['PA'].shift(1).fillna(0),1,0)
    df['inning'] = inning
    df['inning'] = df['inning'].astype('int')
    df['zone'] = zone
    df['CS'] = called_strikes
    df['Whiffs'] = swinging_strikes
    df['Swings'] = swing
    df['chase'] = np.where(df['pitch_call'].isin(['swinging_strike','foul_tip','swinging_strike_blocked','foul','hit_into_play']) & (df['zone']==0),1,0)
    df['Fouls'] = foul_strikes
    df['Strikes'] = total_strikes
    df['Balls'] = total_balls
    df['K'] = np.where(df['pitch_call'].isin(['called_strike','foul_tip','swinging_strike','swinging_strike_blocked']) & (df['strikes']==2),1,0)
    df['BB'] = np.where((df['pitch_call']=='ball') & (df['balls']==3),1,0)
    df['Num Pitches'] = pitch_id
    df['pitch_type'] = [pitchtype_map[x] for x in pitch_type]
    df['sub_type'] = pitch_type
    df['pitch_group'] = [pitchgroup_map[x] for x in pitch_type]
    # df['pitch_type'] = df['pitch_type'].map(pitchtype_map)
    df['Velo'] = velo
    df['Ext'] = extension
    df['vert_break'] = ivb
    df['x0'] = x0
    df['z0'] = z0
    df['vy0'] = vy0
    df['vz0'] = vz0
    df['ay'] = ay
    df['az'] = az
    df['p_x'] = px
    df['p_z'] = pz
    df['pos_z_scale'] = df['z0'].div(df['Height'])
    df['pos_x_scale'] = df['x0'].abs().div(df['Height'])
    df['extension_scale'] = df['Ext'].div(df['Height'])
    df['sz_top'] = sz_top
    df['sz_bot'] = sz_bot
    df['sz_z'] = strikezone_z(df,'sz_top','sz_bot')
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
    
    df['BIP'] = np.where((df['pitch_call']=='hit_into_play'),1,0)
    in_play_outs = ['Sac Fly', 'Groundout', 'Flyout', 'Pop Out',
                    'Lineout', 'GIDP', 'Forceout', 'Sac Bunt',
                    'Fielders Choice', 'Bunt Groundout','Double Play', 
                    'Fielders Choice Out', 'Bunt Lineout',
                    'Bunt Pop Out', 'Triple Play','Sac Fly Double Play']
    df['In Play Out'] = np.where((df['pitch_call']=='hit_into_play') & (df['event'].isin(in_play_outs)),1,0)
    df['Error'] = np.where((df['pitch_call']=='hit_into_play') & (df['event']=='Field Error'),1,0)
    df['HB'] = np.where((df['pitch_call']=='hit_by_pitch'),1,0)
    df['Hit'] = np.where((df['pitch_call']=='hit_into_play') & df['event'].isin(['Single','Double','Triple','Home Run']),1,0)
    df['1B'] = np.where((df['pitch_call']=='hit_into_play') & (df['event']=='Single'),1,0)
    df['2B'] = np.where((df['pitch_call']=='hit_into_play') & (df['event']=='Double'),1,0)
    df['3B'] = np.where((df['pitch_call']=='hit_into_play') & (df['event']=='Triple'),1,0)
    df['HR'] = np.where((df['pitch_call']=='hit_into_play') & (df['event']=='Home Run'),1,0)
    df['xDamage'] = [None if any(np.isnan([x,y,z])) else sum(np.multiply(xwOBAcon_model.predict_proba([[x,y,z]])[0],np.array([0,1,2,3,4]))) for x,y,z in zip(df['Spray Angle'].astype('float'),df['Launch Angle'].astype('float'),df['Launch Speed'].astype('float'))]
    df = feature_engineer(df)
    # df[['plvCS','plvBall','plvHBP',
    #     'plvWhiff','plvFoul',
    #     'plvOut', 'plv1B', 'plv2B', 'plv3B', 'plvHR']] = apply_plv_outcomes(df)
    # df['plvCSW'] = df[['plvCS','plvWhiff']].sum(axis=1)
    # df['plvDamage'] = df[['plv1B', 'plv2B', 'plv3B', 'plvHR']].mul([1,2,3,4]).sum(axis=1) / df[['plvOut', 'plv1B', 'plv2B', 'plv3B', 'plvHR']].sum(axis=1)

    df = df.loc[df['count'].isin(counts) & df['inning'].between(start_inning, end_inning)].reset_index().copy()
    
    agg_dict = {
        'Num Pitches':'count',
        # 'Arm Angle':'median',
        'PA':'sum',
        'Strikes':'sum',
        'Balls':'sum',
        'zone':'sum',
        'K':'sum',
        'vs_rhh':'sum',
        'Velo':'mean',
        'IVB':'mean',
        'IHB':'mean',
        'Ext':'mean',
        'HAVAA':'mean',
        'zone':'sum',
        'BB':'sum',
        'CS':'sum',
        'Whiffs':'sum',
        'chase':'sum',
        'Fouls':'sum',
        'K':'sum',
        'BIP':'sum',
        'In Play Out':'sum',
        'Error':'sum',
        'Hit':'sum',
        '1B':'sum',
        '2B':'sum',
        '3B':'sum',
        'HR':'sum',
        'HB':'sum',
        'xDamage':'mean',
        # 'plvCS':'mean',
        # 'plvBall':'mean',
        # 'plvHBP':'mean',
        # 'plvWhiff':'mean',
        # 'plvFoul':'mean',
        # 'plvOut':'mean', 
        # 'plv1B':'mean', 
        # 'plv2B':'mean', 
        # 'plv3B':'mean', 
        # 'plvHR':'mean',
        # 'plvDamage':'mean',
    }
    game_df = (
        df
        .assign(vs_rhh = lambda x: np.where(x['hitterside']=='R',1,0))
        .groupby(['game_date','MLBAMID','Pitcher','P Hand','pitch_type'])
        [list(agg_dict.keys())]
        .agg(agg_dict)
        .assign(CSW = lambda x: x['CS'].add(x['Whiffs']).div(x['Num Pitches']).mul(100),
                # plvCSW = lambda x: x['plvCS'].add(x['plvWhiff']).mul(100),
                strike_rate = lambda x: x['Strikes'].div(x['Num Pitches']).mul(100),
                zone_rate = lambda x: x['zone'].div(x['Num Pitches']).mul(100),
                chase_rate = lambda x: x['chase'].div(x['Num Pitches'].sub(x['zone'])).mul(100),
                vs_lhh = lambda x: x['Num Pitches'].sub(x['vs_rhh']))
        .reset_index()
    )

    merge_df = (
        pd.merge(game_df.loc[game_df['Pitcher']==player_name].assign(Usage = lambda x: x['Num Pitches'].div(x['Num Pitches'].sum()).mul(100)).copy(),
                 season_avgs,
                 how='left',
                 on=['MLBAMID','pitch_type'],
                 suffixes=['','_comp'])
        .assign(vs_rhh = lambda x: x['vs_rhh'].div(x['vs_rhh'].sum()).fillna(0),
                vs_lhh = lambda x: x['vs_lhh'].div(x['vs_lhh'].sum()).fillna(0),
                usage_diff = lambda x: x['Usage'].sub(x['Usage_comp']),
                velo_diff = lambda x: x['Velo'].sub(x['Velo_comp']),
                ivb_diff = lambda x: x['IVB'].sub(x['IVB_comp']),
                ihb_diff = lambda x: x['IHB'].sub(x['IHB_comp']))
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
    merge_df['Zone%'] = [f'{x:.1f}%' for x in merge_df['zone_rate']]
    merge_df['Chase%'] = [f'{x:.1f}%' for x in merge_df['chase_rate'].fillna(0)]
    merge_df['vs R'] = [f'{x:.1%}' for x in merge_df['vs_rhh']]
    merge_df['vs L'] = [f'{x:.1%}' for x in merge_df['vs_lhh']]
    merge_df['Ext'] = [f'{x:.1f} ft' for x in merge_df['Ext']]
    merge_df['xDamage'] = merge_df['xDamage'].astype('float').round(3)
    merge_df['HAVAA'] = [f'{x:.1f}°' for x in merge_df['HAVAA']]
    # merge_df['plvCSW'] = [f'{x:.1f}%' for x in merge_df['plvCSW']]
    # merge_df['plvDamage'] = merge_df['plvDamage'].astype('float').round(3)
    # merge_df['plvCS'] = [f'{x:.1f}%' for x in merge_df['plvCS'].mul(100)]
    # merge_df['plvBall'] = [f'{x:.1f}%' for x in merge_df['plvBall'].mul(100)]
    # merge_df['plvHBP'] = [f'{x:.1f}%' for x in merge_df['plvHBP'].mul(100)]
    # merge_df['plvWhiff'] = [f'{x:.1f}%' for x in merge_df['plvWhiff'].mul(100)]
    # merge_df['plvFoul'] = [f'{x:.1f}%' for x in merge_df['plvFoul'].mul(100)]
    # merge_df['plvOut'] = [f'{x:.1f}%' for x in merge_df['plvOut'].mul(100)]
    # merge_df['plv1B'] = [f'{x:.1f}%' for x in merge_df['plv1B'].mul(100)]
    # merge_df['plv2B'] = [f'{x:.1f}%' for x in merge_df['plv2B'].mul(100)]
    # merge_df['plv3B'] = [f'{x:.1f}%' for x in merge_df['plv3B'].mul(100)]
    # merge_df['plvHR'] = [f'{x:.1f}%' for x in merge_df['plvHR'].mul(100)]
    # merge_df['Arm Angle'] = [f'{x:.0f}°' for x in merge_df['Arm Angle']]

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
    
    merge_df.loc['Total'] = ['-']*len(merge_df.columns)
    merge_df.loc['Total','P Hand'] = '-'
    merge_df.loc['Total','Type'] = 'Total'
    # Usage
    merge_df.loc['Total','Num Pitches'] = game_df['Num Pitches'].sum()
    v_rhh_val = game_df['vs_rhh'].sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','vs R'] = f'{v_rhh_val:.1%}'
    v_lhh_val = 1-v_rhh_val
    merge_df.loc['Total','vs L'] = f'{v_lhh_val:.1%}'
    merge_df.loc['Total','PA'] = df['PA'].sum()
    # Strikes
    merge_df.loc['Total','Strikes'] = game_df['Strikes'].sum()
    strike_val = df['Strikes'].sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','Strike%'] = f'{strike_val:.1%}'
    merge_df.loc['Total','CS'] = game_df['CS'].sum()
    merge_df.loc['Total','Whiffs'] = game_df['Whiffs'].sum()
    merge_df.loc['Total','Fouls'] = game_df['Fouls'].sum()
    csw_val = df[['CS','Whiffs']].sum(axis=1).sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','CSW'] = f'{csw_val:.1%}'
    merge_df.loc['Total','K'] = game_df['K'].sum()
    # Location
    merge_df.loc['Total','Balls'] = game_df['Balls'].sum()
    zone_val = df['zone'].sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','Zone%'] = f'{zone_val:.1%}'
    chase_val = df['chase'].sum() / (game_df['Num Pitches'].sum() - df['zone'].sum())
    merge_df.loc['Total','Chase%'] = f'{chase_val:.1%}'
    merge_df.loc['Total','BB'] = game_df['BB'].sum()
    # Batted Ball
    merge_df.loc['Total','BIP'] = game_df['BIP'].sum()
    merge_df.loc['Total','In Play Out'] = game_df['In Play Out'].sum()
    merge_df.loc['Total','Error'] = game_df['Error'].sum()
    merge_df.loc['Total','Hit'] = game_df['Hit'].sum()
    merge_df.loc['Total','1B'] = game_df['1B'].sum()
    merge_df.loc['Total','2B'] = game_df['2B'].sum()
    merge_df.loc['Total','3B'] = game_df['3B'].sum()
    merge_df.loc['Total','HR'] = game_df['HR'].sum()
    merge_df.loc['Total','HB'] = game_df['HB'].sum()
    merge_df.loc['Total','xDamage'] = round(df['xDamage'].astype('float').mean(),3)

    #PLV
    # for stat in ['plvCS','plvBall','plvHBP','plvWhiff','plvFoul',
    #              'plvOut', 'plv1B', 'plv2B', 'plv3B', 'plvHR']:
    #     stat_val = df[stat].sum() / game_df['Num Pitches'].sum()
    #     merge_df.loc['Total',stat] = f'{stat_val:.1%}'
    # plv_csw_val = df[['plvCS','plvWhiff']].sum(axis=1).sum() / game_df['Num Pitches'].sum()
    # merge_df.loc['Total','plvCSW'] = f'{plv_csw_val:.1%}'
    # merge_df.loc['Total','plvDamage'] = round(df['plvDamage'].mean(),3)
    return merge_df, df

def game_charts(move_df):
    fig = plt.figure(figsize=(8,8))
    grid = plt.GridSpec(1, 3, width_ratios=[1,2,1],wspace=0.15)
    ax1 = plt.subplot(grid[1])
    circle1 = plt.Circle((0, 0), 6, color=pl_white,fill=False,alpha=alpha_val/2,linestyle='--')
    ax1.add_patch(circle1)
    circle2 = plt.Circle((0, 0), 12, color=pl_white,fill=False,alpha=alpha_val)
    ax1.add_patch(circle2)
    circle3 = plt.Circle((0, 0), 18, color=pl_white,fill=False,alpha=alpha_val/2,linestyle='--')
    ax1.add_patch(circle3)
    circle4 = plt.Circle((0, 0), 24, color=pl_white,fill=False,alpha=alpha_val)
    ax1.add_patch(circle4)
    ax1.axvline(0,ymin=4/58,ymax=54/58,color=pl_white,alpha=alpha_val,zorder=1)
    ax1.axhline(0,xmin=4/58,xmax=54/58,color=pl_white,alpha=alpha_val,zorder=1)
    
    for dist in [12,24]:
        label_dist = dist-0.25
        ax1.text(label_dist,-0.3,f'{dist}"',ha='right',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(-label_dist,-0.3,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,label_dist-0.25,f'{dist}"',ha='left',va='top',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
        ax1.text(0.25,-label_dist,f'{dist}"',ha='left',va='bottom',fontsize=6,color=pl_white,alpha=0.5,zorder=1)
    
    if move_df['P Hand'].value_counts().index[0]=='R':
        ax1.text(28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=alpha_val,zorder=1)
        ax1.text(-28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=alpha_val,zorder=1)
    else:
        ax1.text(28.5,0,'Glove\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=alpha_val,zorder=1)
        ax1.text(-28.5,0,'Arm\nSide',ha='center',va='center',fontsize=8,color=pl_white,alpha=alpha_val,zorder=1)
    
    ax1.text(0,27,'Rise',ha='center',va='center',fontsize=8,color=pl_white,alpha=alpha_val,zorder=1)
    ax1.text(0,-27,'Drop',ha='center',va='center',fontsize=8,color=pl_white,alpha=alpha_val,zorder=1)
    
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
    
    # logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
    # logo = Image.open(urllib.request.urlopen(logo_loc))
    pl_ax = fig.add_axes([0.4,0.175,0.2,0.1], anchor='NE', zorder=1)
    width, height = logo.size
    pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    pl_ax.axis('off')

    fig.suptitle(f"{player_select}'s Pitch Charts ({date.strftime('%-m/%-d/%y')})",y=0.75)
    sns.despine()
    st.pyplot(fig,use_container_width=False)

def loc_charts(df):
    sz_bot = 1.5
    sz_top = 3.5
    y_mid = 2.5
    
    aspect_ratio = 20/24
    
    x_dist = 1.75
    y_bot = y_mid-(x_dist/aspect_ratio)
    y_top = y_mid+(x_dist/aspect_ratio)

    fig, ax = plt.subplots(figsize=(5,5))
    zone_outline = plt.Rectangle((-10/12, sz_bot), 
                                 20/12, sz_top-sz_bot, 
                                 color=pl_white, fill=False,
                                 linewidth=2,alpha=0.5)
    ax.add_patch(zone_outline)
    
    sns.scatterplot(df.loc[(df['p_x'].abs()<=x_dist) & (df['p_z'].sub(y_mid)<=x_dist/aspect_ratio)].assign(p_x = lambda x: x['p_x'].mul(-1)),
                    x='p_x',
                    y='p_z',
                    hue='pitch_type',
                    palette=marker_colors,
                    s=225,
                    linewidth=0,
                    alpha=1,
                    legend=False
                   )
    
    
    ax.set(xlim=(-x_dist,x_dist),
           ylim=(y_bot,y_top),
           aspect=aspect_ratio)
    
    ax.axis('off')
    fig.patch.set_alpha(0)
    sns.despine()
    st.pyplot(fig)

def hextriplet(color):
    return f"#{''.join(f'{hex(int(c*255))[2:].upper():0>2}' for c in color)}"

marker_colors = {
    'FF':'#d22d49', 
    'FA':'#d22d49', 
    'SI':'#c57a02',
    'FT':'#c57a02',
    'FS':'#00a1c5',  
    'FO':'#00a1c5',  
    'FC':'#933f2c', 
    'SL':'#9300c7', 
    'ST':'#C95EBE',
    'CU':'#3c44cd',
    'CS':'#3c44cd',
    'KC':'#3c44cd',
    'SV':'#3c44cd',
    'CH':'#07b526', 
    'KN':'#999999',
    'SC':'#999999', 
    'UN':'#999999', 
    'Total':'#999999', 
}

marker_names = {
    'FF':'Four-Seamer', 
    'FA':'Fastball', 
    'SI':'Sinker',
    'FT':'Two-Seamer',
    'FS':'Splitter',  
    'FO':'Forkball',  
    'FC':'Cutter', 
    'SL':'Slider', 
    'ST':'Sweeper',
    'CU':'Curveball',
    'KC':'Knuckle Curve',
    'SV':'Slurve',
    'CH':'Changeup', 
    'KN':'Knuckleball',
    'SC':'Screwball', 
    'UN':'Unknown', 
}

highlight_dict = {k:hextriplet(sns.dark_palette(v,n_colors=20)[2]) for k, v in marker_colors.items()}
highlight_dict.update({'Total':'#20232c'})
type_dict = {k:hextriplet(sns.dark_palette(v,n_colors=20)[5]) for k, v in marker_colors.items()}
type_dict.update({'Total':'#20232c'})

# def highlight_cols(s, coldict):
#     return ['background-color: {}'.format(highlight_dict[v]) if v else '' for v in list(test_df.index).isin(highlight_dict.keys())*list(test_df.index).values]

# def index_style(s):
#     return [f"background-color: {type_dict[i]};" for i in s]

def highlight_cols(s, coldict, stat_tab):
    if stat_tab=='Default':
        col_format = ['background-color: {}'.format(highlight_dict[v]) if v else '' for v in table_df[('','Type')].isin(highlight_dict.keys())*table_df[('','Type')].values]
    else:
        col_format = ['background-color: {}'.format(highlight_dict[v]) if v else '' for v in table_df['Type'].isin(highlight_dict.keys())*table_df['Type'].values]
    return col_format

default_groups = {
    '':['Type','#'],
    'Usage':['Usage','vs R','vs L'],
    'Stuff':['Velo','Ext','IVB','IHB','HAVAA'],
    'Strikes':['Strike%','Fouls','CS','Whiffs','CSW','K'],
    'Locations':['Zone%','Chase%','BB'],
    'Batted Ball':['BIP','In Play Out','Hit','HR','Error','xDamage'],
    # 'PLV':['plvCSW','plvDamage']
    }

stat_tabs = {
    'Default':'',
    'Standard':['Strikes','Balls','PA','Hit','1B','2B','3B','HR','K','BB','HB'],
    # 'PLV':['plvCS','plvBall','plvHBP','plvWhiff','plvFoul','plvOut', 'plv1B', 'plv2B', 'plv3B', 'plvHR','plvCSW','plvDamage']
}

if len(list(pitcher_list.keys()))==0:
    st.write('No pitches thrown yet')
else:
    idx = pd.IndexSlice
    slice_ = idx['Total',:]
    table_df, chart_df = scrape_savant_data(player_select,game_id, counts, start_inning, end_inning)
    tab_select = st.segmented_control('',list(stat_tabs.keys()),default='Default')
    chart_df['pitch_type'] = chart_df['pitch_type'].map(pitchtype_map)
    chart_df['Description'] = np.where(chart_df['pitch_call']=='hit_into_play',
             chart_df['event'],
             chart_df['pitch_call'])
    chart_df['Description'] = chart_df['Description'].str.replace('_',' ').str.title()
    # table_df = table_df.set_index('Type')
    if tab_select=='Default':
        col_names = [(k,v) for k, l in default_groups.items() for v in l ]
        table_df = table_df.rename(columns={'Num Pitches':'#'})[sum(list(default_groups.values()),[])]
        table_df.columns = pd.MultiIndex.from_tuples(col_names)
    else:
        table_df = table_df.rename(columns={'Num Pitches':'#'})[['Type','#']+stat_tabs[tab_select]]
    # plv_cols = [x for x in ['plvCS','plvBall','plvHBP','plvWhiff','plvFoul','plvOut', 'plv1B', 'plv2B', 'plv3B', 'plvHR'] if x in list(table_df.columns.values)]
    st.dataframe((table_df
                  .style
                  .format(precision=3)
                  # .format(precision=2,subset=plv_cols)
                  # .apply(lambda r: [f"background-color:{highlight_dict.get(r.name)}" for i in r], axis=1)
                  # .apply_index(index_style)
                  .apply(highlight_cols,coldict=highlight_dict,stat_tab=tab_select)
                  # .apply(lambda r: [f"background-color:{type_dict.get(r[('','Type')],'')}"]+[f"background-color:{highlight_dict.get(r[('','Type')],'')}"]*(len(r)-1), axis=1)
                  # .set_properties(**{'background-color': '#20232c'}, subset=slice_)
                 ),
                 use_container_width=False, hide_index=True)

    game_charts(chart_df)
    if st.button('Location Charts'):
        loc_charts(chart_df)

def plotly_charts(chart_df):
    chart_df['Pitch Name'] = chart_df['pitch_type'].map(marker_names)
    chart_df['pitch_type'] = chart_df['pitch_type'].map(pitchtype_map)
    chart_df['sub_type_name'] = chart_df['sub_type'].map(pitch_names)
    chart_df['Description'] = np.where(chart_df['pitch_call']=='hit_into_play',
             chart_df['event'],
             chart_df['pitch_call'])
    chart_df['Description'] = chart_df['Description'].str.replace('_',' ').str.title()
    chart_df['pa_count'] = chart_df['PA'].expanding().sum()
    chart_df['ev_la_text'] = np.where(chart_df['Launch Angle'].isna() | chart_df['Launch Speed'].isna(),
                                      '',
                                      'EV/LA: '+chart_df['Launch Speed'].round(1).astype('str')+'mph @ '+chart_df['Launch Angle'].astype('str')+'°')
    lhh_df = chart_df.loc[chart_df['hitterside']=='L'].copy()
    rhh_df = chart_df.loc[chart_df['hitterside']=='R'].copy()
    pitcher_hand = chart_df['P Hand'][0]
    faded_label_color = hextriplet(sns.light_palette(pl_background,n_colors=20)[10])
    move_df = chart_df.assign(IHB = lambda x: np.where(pitcher_hand=='L',x['IHB'].astype('float').mul(-1),x['IHB'].astype('float'))).copy()
    fig = make_subplots(rows=5, cols=3, column_widths=[.3,.4,.3],row_heights=[0.5,0.3,.05,.05,.1],
                        specs = [[{}, {}, {}],
                                 [{"colspan": 3}, None, None],
                                 [{"colspan": 3}, None, None],
                                 [{"colspan": 3}, None, None],
                                 [{"colspan": 3}, None, None]], 
                        horizontal_spacing = 0,
                        vertical_spacing = 0,
                        subplot_titles=("Locations<br>vs LHH","Movement<br> ","Locations<br>vs RHH"))
    plate_y = -1.25
    
    # layout = go.Layout(height = 600,width = 1500,xaxis_range=[-2.5,2.5], yaxis_range=[-1,6])
    # LHH Plot
    fig.update_xaxes(title_text="", range=[-2.5,2.5], row=1, col=1)
    fig.update_yaxes(title_text="", range=[-1.75,1.5], row=1, col=1)
    labels = lhh_df['pitch_type'].map(marker_colors)
    hover_text = '<b>%{customdata[2]}: %{customdata[3]}</b><br>Hitter: %{customdata[5]}<br>Count: %{customdata[0]}-%{customdata[1]}<br>Velo: %{customdata[4]:.1f}mph<br>%{customdata[6]}<extra></extra>'
    marker_dict = dict(color=labels,
                       size=25,
                       line=dict(width=1,color='white'))
    
    # fig = go.Figure(layout = layout)
    fig.add_trace(go.Scatter(x=[10/36,10/36], y=[-0.5,0.5],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[-10/36,-10/36], y=[-0.5,0.5],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[-1/6,-1/6],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[1/6,1/6],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    
    fig.add_shape(type="rect",
        x0=-10/12, y0=-0.5, x1=10/12, y1=0.5,
        line=dict(color="white"),
                  layer='below',
                  row=1, col=1
    )
    
    # Plate
    fig.add_trace(go.Scatter(x=[-8.5/12,8.5/12], y=[plate_y,plate_y],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[-8.5/12,-8.15/12], y=[plate_y,plate_y+0.075],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[8.5/12,8.15/12], y=[plate_y,plate_y+0.075],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[8.18/12,0], y=[plate_y+0.075,plate_y+0.125],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    fig.add_trace(go.Scatter(x=[-8.18/12,0], y=[plate_y+0.075,plate_y+0.125],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=1)
    
    bonus_text = lhh_df['hitterside']
    fig.add_trace(go.Scatter(x=lhh_df['p_x'].mul(-1), y=lhh_df['sz_z'], mode='markers', 
                       marker=marker_dict, text=bonus_text,
                       customdata=lhh_df[['balls','strikes','Pitch Name','Description','Velo','Hitter','ev_la_text','sub_type_name']],
                       hovertemplate=hover_text,
                        showlegend=False),
                            row=1, col=1)
    
    # RHH Plot
    fig.update_xaxes(title_text="", range=[-2.5,2.5], row=1, col=3)
    fig.update_yaxes(title_text="", range=[-1.75,1.5], row=1, col=3)
    labels = rhh_df['pitch_type'].map(marker_colors)
    # hover_text = '<b>%{customdata[2]}: %{customdata[3]}</b><br>Count: %{customdata[0]}-%{customdata[1]}<br>Hitter Hand: %{text}<br>X Loc: %{x:.1f}ft<br>Y Loc: %{y:.1f}ft<extra></extra>'
    marker_dict = dict(color=labels,
                       size=25,
                       line=dict(width=1,color='white'))
    
    # fig = go.Figure(layout = layout)
    fig.add_trace(go.Scatter(x=[10/36,10/36], y=[-0.5,0.5],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[-10/36,-10/36], y=[-0.5,0.5],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[-1/6,-1/6],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[1/6,1/6],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    
    fig.add_shape(type="rect",
        x0=-10/12, y0=-0.5, x1=10/12, y1=0.5,
        line=dict(color="white"),
                  layer='below',
                  row=1, col=3
    )
    
    # Plate
    fig.add_trace(go.Scatter(x=[-8.5/12,8.5/12], y=[plate_y,plate_y],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[-8.5/12,-8.15/12], y=[plate_y,plate_y+0.075],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[8.5/12,8.15/12], y=[plate_y,plate_y+0.075],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[8.18/12,0], y=[plate_y+0.075,plate_y+0.125],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    fig.add_trace(go.Scatter(x=[-8.18/12,0], y=[plate_y+0.075,plate_y+0.125],
                             mode='lines',
                             line=dict(color='white', width=2),
                             showlegend=False,
                            hoverinfo='skip',
                            ),row=1, col=3)
    
    bonus_text = rhh_df['hitterside']
    fig.add_trace(go.Scatter(x=rhh_df['p_x'].mul(-1), y=rhh_df['sz_z'], mode='markers', 
                       marker=marker_dict, text=bonus_text,
                       customdata=rhh_df[['balls','strikes','Pitch Name','Description','Velo','Hitter','ev_la_text','sub_type_name']],
                       hovertemplate=hover_text,
                        showlegend=False),
                            row=1, col=3)
    
    # Movement
    ax_lim = max(25,move_df[['IVB','IHB']].abs().max().max())+12
    labels = move_df['pitch_type'].map(marker_colors)
    hover_text = '<b>%{customdata[0]}: %{customdata[2]}</b><br>Hitter: %{customdata[4]}<br>Velo: %{customdata[1]}mph<br>IVB: %{y:.1f}"<br>IHB: %{x:.0f}"<extra></extra>'
    marker_dict = dict(color=labels,
                       size=25,
                       line=dict(width=0.5,color='white'))
    fig.add_trace(go.Scatter(x=[0,0], y=[12-ax_lim,ax_lim-12],
                             mode='lines',
                             line=dict(color='white', width=3),
                             showlegend=False,
                            hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[12-ax_lim,ax_lim-12], y=[0,0],
                             mode='lines',
                             line=dict(color='white', width=3),
                             showlegend=False,
                            hoverinfo='skip'), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[-21.75,21.75,2,2,-9.75,9.75,2,2],
        y=[-1.5,-1.5,-22.5,22,-1.5,-1.5,-10.5,10],
        text=['24"','24"','24"','24"','12"','12"','12"','12"'],
        mode="text",
        textfont=dict(
            color=faded_label_color,
            size=15,
        ),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[0,0,28 if pitcher_hand=='R' else -28,-28 if pitcher_hand=='R' else 28],
        y=[27,-27,0,0],
        text=['Rise','Drop','Arm<br>Side','Glove<br>Side'],
        mode="text",
        textfont=dict(
            color="white",
            size=20,
        ),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=2)
    
    
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-24, y0=-24, x1=24, y1=24,
                  line=dict(
                      color="white",
                      width=3,
                      ),
                  layer='below', row=1, col=2
                  )
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-12, y0=-12, x1=12, y1=12,
                  line=dict(
                      color="white",
                      width=3,
                      ),
                  layer='below', row=1, col=2
                  )
    
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-18, y0=-18, x1=18, y1=18,
                  line=dict(
                      color=faded_label_color,
                      width=1,
                      dash='dash',
                      ),
                  layer='below', row=1, col=2
                  )
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-6, y0=-6, x1=6, y1=6,
                  line=dict(
                      color=faded_label_color,
                      width=1,
                      dash='dash'
                      ),
                  layer='below', row=1, col=2
                  )
    
    fig.add_trace(go.Scatter(x=move_df['IHB'], y=move_df['IVB'], mode='markers', 
                       marker=marker_dict, text=bonus_text,
                       customdata=move_df[['Pitch Name','Velo','Description','sub_type_name','Hitter']],
                       hovertemplate=hover_text,
                        showlegend=False), row=1, col=2)

     ### Sequencing Charts
    pa_df = chart_df.assign(Description = lambda x: x['Pitch Name'].astype('str')+': '+x['Description'].astype('str')).groupby(['pa_count','Hitter','event'])[['PA','Description']].agg({
        'PA':'count',
        'Description':lambda x: '<br>- '.join([a for a in list(x) if a is not None])
    }).reset_index().rename(columns={'PA':'count'})
    inning_df = chart_df.assign(event = lambda x: np.where(x['PA']==1,x['Hitter'].astype('str')+': '+x['event'].astype('str'),None)).groupby(['inning'])[['PA','event']].agg({
        'PA':'count',
        'event':lambda x: '<br>- '.join([a for a in list(x) if a is not None])
    }).reset_index().rename(columns={'PA':'count'})
    
    data_len = chart_df['pitch_type'].shape[0]
    colors = list(chart_df['pitch_type'].map(marker_colors))
    data_fill_x = [1]*data_len
    data_fill_y = [1]*data_len
    
    pa_count = list(pa_df['count'])
    pa_num = list(pa_df['pa_count'].astype('int'))
    pa_fill = [1]*len(pa_count)
    
    inning_count = list(inning_df['count'])
    inning_num = list(inning_df['inning'].astype('int'))
    inning_fill = [1]*len(inning_count)

    data_len = chart_df['pitch_type'].shape[0]
    colors = list(chart_df['pitch_type'].map(marker_colors))
    data_fill_x = [1]*data_len
    data_fill_y = [1]*data_len
    
    pa_count = list(pa_df['count'])
    pa_num = list(pa_df['pa_count'].astype('int'))
    pa_fill = [1]*len(pa_count)
    
    inning_count = list(inning_df['count'])
    inning_num = list(inning_df['inning'].astype('int'))
    inning_fill = [1]*len(inning_count)

    hover_text = '<b>Inning %{customdata[0]}</b><br>Events:<br>- %{customdata[1]}<extra></extra>'
    fig.add_trace(
        go.Bar(
            x=inning_count, y=inning_fill,
            orientation='h',
             marker=dict(
                 color='white',
                 line=dict(color=pl_background, width=1)
                 ),
            text=inning_num,
            insidetextanchor ="middle",
            textfont=dict(
                size=16,
                color="black"
                ),
            textangle=0,
            customdata=inning_df[['inning','event']],
            hovertemplate=hover_text
        ),
        row=3, col=1
    )
    fig.add_trace(go.Scatter(
            x=[-data_len/40],
            y=[1],
            text=['IP'],
            mode="text",
            textfont=dict(
                color="white",
                size=16,
            ),
            showlegend=False,
            hoverinfo='skip',
        ), row=3, col=1
                 )

    hover_text = '<b>%{customdata[0]}</b>: %{customdata[1]}<br>- %{customdata[2]}</b><extra></extra>'
    fig.add_trace(
        go.Bar(
            x=pa_count, y=pa_fill,
            orientation='h',
             marker=dict(
                 color='white',
                 line=dict(color=pl_background, width=1)
                 ),
            text=pa_num,
            insidetextanchor ="middle",
            textposition ="inside",
            textfont=dict(
                size=16,
                color="black"
                ),
            textangle=0,
            customdata=pa_df[['Hitter','event','Description']],
            hovertemplate=hover_text
        ),
        row=4, col=1
    )
    fig.add_trace(go.Scatter(
            x=[-data_len/40],
            y=[1],
            text=['PA'],
            mode="text",
            textfont=dict(
                color="white",
                size=16,
            ),
            showlegend=False,
            hoverinfo='skip',
        ), row=4, col=1
                 )
    
    hover_text = '<b>%{customdata[4]}: %{customdata[0]}</b>%{customdata[3]}<br>- %{customdata[2]}<br>- Velo: %{customdata[1]}mph<extra></extra>'
    fig.add_trace(
        go.Bar(
            x=data_fill_x, y=data_fill_y,
            orientation='h',
             marker=dict(
                 
                 color=colors,
                 line=dict(color=pl_background, width=1)
                 ),
            customdata=(
                chart_df
                .assign(pitch_type = lambda x: x['pitch_type'].map(pitch_names),
                        sub_type_name = lambda x: np.where(x['pitch_type']==x['sub_type_name'],
                                                           '',
                                                           '<br>Sub-Type: '+x['sub_type_name']))
                [['pitch_type','Velo','Description','sub_type_name','Num Pitches']]
            ),
            hovertemplate=hover_text
            ),
        row=5, col=1
    )
    fig.add_trace(go.Scatter(
            x=[-data_len/40],
            y=[1],
            text=['Type'],
            mode="text",
            textfont=dict(
                color="white",
                size=16,
            ),
            showlegend=False,
            hoverinfo='skip',
        ), row=5, col=1
                 )

    ### Pitch Types
    sub_count = (
        chart_df
        .groupby('sub_type')
        ['game_pk']
        .count()
        .sort_values(ascending=False)
        .to_dict()
    )
    
    label_df = (
        chart_df
        .groupby('pitch_type')
        [['game_pk','Velo','IVB','IHB']]
        .agg({
            'game_pk':'count',
            'Velo':'mean',
            'IVB':'mean',
            'IHB':'mean'
        })
        .round(1)
        .rename(columns={'game_pk':'count'})
        .reset_index()
        .sort_values(['count'], ascending=False)
        .assign(sub_type = lambda x: x['pitch_type'].map(chart_df.groupby(['pitch_type'])[['sub_type']].agg({
            'sub_type':lambda x: '' if len(list(set(x))) ==1 else '<br>Sub-Types Thrown:<br>'+'<br>'.join(['- '+pitch_names[a]+': '+str(sub_count[a]) for a in list(set(sorted(list(x),key=list(x).count,reverse=True))) if a is not None])
        }).to_dict()['sub_type'])
               )
    )
    
    pitches_thrown = list(label_df['pitch_type'])
    num_thrown = list(label_df['count'])
    perc_thrown = list(label_df['count'].div(label_df['count'].sum()).mul(100).round(0).astype('int'))
    pitch_colors = [marker_colors[x] for x in pitches_thrown]

    hover_text = 'Velo: %{customdata[2]}mph<br>IVB: %{customdata[3]}"<br>IHB: %{customdata[4]}"<br>%{customdata[5]}<br><extra></extra>'
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pitches_thrown))), 
            y=[0] * len(pitches_thrown),
            mode='markers+text',
            marker=dict(size=[2*min(50,x)+20 for x in perc_thrown],
                        opacity=1,
                        color=pitch_colors,
                        line=dict(color='white',
                                 width=2)),
            text=num_thrown,
            textfont=dict(
                color="white",
                size=20
            ),
            customdata=label_df.assign(full_name = lambda x: x['pitch_type'].map(pitch_names)),
            hovertemplate=hover_text
        ), row=2, col=1
    )
    
    for pitchtype in pitches_thrown:
        fig.add_annotation(x=pitches_thrown.index(pitchtype), y=-0.25,
                           text=pitch_names[pitchtype],
                           showarrow=False,
                           font=dict(
                               size=20,
                               color="#ffffff"
                               ), row=2, col=1)
    fig.add_annotation(x=(len(pitches_thrown)-1)/2, y=-0.65,
                       text='Sequencing',
                       showarrow=False,
                       font=dict(
                           size=24,
                           color="#ffffff"
                           ), row=2, col=1)
    
    fig.update_xaxes(title_text="", range=[-1,len(pitches_thrown)], row=2, col=1)
    fig.update_yaxes(title_text="", range=[-1,0.5], row=2, col=1)
    fig.update_traces(textposition='middle center', row=2, col=1)
    
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    
    fig.update_layout(plot_bgcolor=pl_background,
                     paper_bgcolor=pl_background)
    
    fig.update_layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=100,
                pad=4
                ),
        )
    fig.update_annotations(yshift=-35,
                           font=dict(size=30, color="white")
                          )
    fig.update_layout(height=960, width=1200,
                      hoverlabel={
                          'font':{'color':'white',
                                 'size':16}
                          },
                      title={
                'text': f"{player_select}'s Pitch Charts ({date.strftime('%-m/%-d/%y')})",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            'font':{'color':'white',
                   'size':40}},
                      showlegend=False,
                      hoverlabel_align = 'left'
                     )
    fig.update_traces(hoverlabel={
                          'font':{'color':'black'}
                          }, row=3, col=1)
    fig.update_traces(hoverlabel={
                          'font':{'color':'black'}
                          }, row=4, col=1)
    
    # fig.show()
    st.plotly_chart(fig,use_container_width=False,theme=None)

# if st.button('Experimental test charts (NOT mobile-friendly)'):
    # plotly_charts(chart_df)
