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

from sklearn.neighbors import KNeighborsClassifier
from datetime import timedelta

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

def generate_games(games_today):
    game_dict = {}
    code_dict = {
        'F':0,
        'O':1,
        'I':1,
        'P':2,
        'S':3
    }
    for game in games_today:
        r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game}')
        x = r.json()
        game_hour = int(x['scoreboard']['datetime']['dateTime'][11:13])
        game_hour = game_hour-4 if game_hour >3 else game_hour+20
        game_minutes = int(x['scoreboard']['datetime']['dateTime'][14:16])
        raw_time = game_hour*60+game_minutes
        am_pm = 'AM' if game_hour <12 else 'PM'
        game_time = f'{game_hour-12}:{game_minutes:>02}{am_pm}' if am_pm=='PM' else f'{game_hour}:{game_minutes:>02}{am_pm}'
        ppd = 0 if x['scoreboard']['datetime']['originalDate']==x['scoreboard']['datetime']['officialDate'] else 1
        
        away_team = x['scoreboard']['teams']['away']['abbreviation']
        home_team = x['scoreboard']['teams']['home']['abbreviation']
        game_status_code = x['game_status_code']
        code_map = code_dict[game_status_code]
        if game_status_code =='P':
            game_info = f'{away_team} @ {home_team}: {game_time}'
            inning_sort = None
        elif game_status_code =='S':
            game_info = f'PPD: {away_team} @ {home_team}'
            inning_sort = None
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
st.write('Click the `R` key on your keyboard to reload the page with your selection. Data (especially pitch types) are subject to change.')
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
                test_list.update({x[f'{home_away_pitcher}_pitchers'][pitcher_id][0]['pitcher_name']:pitcher_id})
        test_list = {v: k for k, v in test_list.items()}
        if len(test_list.keys())>0:
            pitcher_list = {test_list[str(x)]:[str(x),y] for x,y in zip(pitcher_lineup,home_team)}
        else:
            pitcher_list = {}
    else:
        pitcher_list = {}
        
@st.cache_data()
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
    

with open('2025_3d_xwoba_model.pkl', 'rb') as f:
    xwOBAcon_model = pickle.load(f)

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
    # df['pitch_type'] = df['pitch_type'].map(pitchtype_map)
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
    
    df['xHits'] = [None if any(np.isnan([x,y,z])) else sum(np.multiply(xwOBAcon_model.predict_proba([[x,y,z]])[0],np.array([0,1,1,1,1]))) for x,y,z in zip(df['Spray Angle'].astype('float'),df['Launch Angle'].astype('float'),df['Launch Speed'].astype('float'))]
    df['3D wOBAcon'] = [None if any(np.isnan([x,y,z])) else sum(np.multiply(xwOBAcon_model.predict_proba([[x,y,z]])[0],np.array([0,0.9,1.25,1.6,2]))) for x,y,z in zip(df['Spray Angle'].astype('float'),df['Launch Angle'].astype('float'),df['Launch Speed'].astype('float'))]

    game_df = df.assign(vs_rhh = lambda x: np.where(x['hitterside']=='R',1,0)).groupby(['game_date','MLBAMID','Pitcher','P Hand','pitch_type'])[['Num Pitches','Velo','IVB','IHB','Ext','vs_rhh','CS','Whiffs','total_strikes','xHits','3D wOBAcon','HAVAA']].agg({
        'Num Pitches':'count',
        'Velo':'mean',
        'IVB':'mean',
        'IHB':'mean',
        'Ext':'mean',
        'vs_rhh':'sum',
        'CS':'sum',
        'Whiffs':'sum',
        'total_strikes':'sum',
        'xHits':'sum',
        '3D wOBAcon':'mean',
        'HAVAA':'mean'
    }).assign(CSW = lambda x: x['CS'].add(x['Whiffs']).div(x['Num Pitches']).mul(100),
              strike_rate = lambda x: x['total_strikes'].div(x['Num Pitches']).mul(100),
              vs_lhh = lambda x: x['Num Pitches'].sub(x['vs_rhh'])).reset_index()

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
    merge_df['vs R'] = [f'{x:.1%}' for x in merge_df['vs_rhh']]
    merge_df['vs L'] = [f'{x:.1%}' for x in merge_df['vs_lhh']]
    merge_df['Ext'] = [f'{x:.1f} ft' for x in merge_df['Ext']]
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
    
    merge_df.loc['Total'] = ['-']*len(merge_df.columns)
    merge_df.loc['Total','P Hand'] = '-'
    merge_df.loc['Total','Type'] = 'Total'
    merge_df.loc['Total','Num Pitches'] = game_df['Num Pitches'].sum()
    v_rhh_val = game_df['vs_rhh'].sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','vs R'] = f'{v_rhh_val:.1%}'
    v_lhh_val = 1-v_rhh_val
    merge_df.loc['Total','vs L'] = f'{v_lhh_val:.1%}'
    strike_val = df['total_strikes'].sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','Strike%'] = f'{strike_val:.1%}'
    merge_df.loc['Total','CS'] = game_df['CS'].sum()
    merge_df.loc['Total','Whiffs'] = game_df['Whiffs'].sum()
    csw_val = df[['CS','Whiffs']].sum(axis=1).sum() / game_df['Num Pitches'].sum()
    merge_df.loc['Total','CSW'] = f'{csw_val:.1%}'
    merge_df.loc['Total','xHits'] = df['xHits'].sum()
    merge_df.loc['Total','3D wOBAcon'] = round(df['3D wOBAcon'].mean(),3)

    return merge_df[['Type','Num Pitches','Velo','Usage','vs R','vs L','Ext','IVB','IHB','HAVAA','Strike%','CS','Whiffs','CSW','3D wOBAcon']], df

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
    # ax.plot([-10/12,10/12], [sz_bot,sz_bot], color='w', linewidth=2,zorder=0, alpha=0.5)
    # ax.plot([-10/12,10/12], [sz_top,sz_top], color='w', linewidth=2,zorder=0, alpha=0.5)
    # ax.plot([-10/12,-10/12], [sz_bot,sz_top], color='w', linewidth=2,zorder=0, alpha=0.5)
    # ax.plot([10/12,10/12], [sz_bot,sz_top], color='w', linewidth=2,zorder=0, alpha=0.5)
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

if len(list(pitcher_list.keys()))==0:
    st.write('No pitches thrown yet')
else:
    idx = pd.IndexSlice
    slice_ = idx['Total',:]
    table_df, chart_df = scrape_savant_data(player_select,game_id)
    chart_df['pitch_type'] = chart_df['pitch_type'].map(pitchtype_map)
    st.dataframe((table_df
                  .style
                  .format(precision=3)
                  .set_properties(**{'background-color': '#20232c'}, subset=slice_)
                 ),
                 column_config={
                     "Num Pitches": st.column_config.NumberColumn(
                         "#"
                         ),
                     "3D wOBAcon": st.column_config.NumberColumn(
                         "xDamage",
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
                 use_container_width=False,
                 hide_index=True)

    game_charts(chart_df)
    loc_charts(chart_df)
