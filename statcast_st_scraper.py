import streamlit as st
import datetime
import requests
import numpy as np
import pandas as pd
import urllib
import pickle
# from sklearn.neighbors import KNeighborsClassifier

from PIL import Image

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
# with open('2025_3d_xwoba_model.pkl', 'rb') as f:
#     xwOBAcon_model = pickle.load(f)

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
                except KeyError:
                    pitch_type += ['UN']
                    velo += [None]
                    extension += [None]
                    ivb += [None]
                    ihb += [None]
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
    
    # def xwOBA_model(df_):
        
    #     df_[['out_pred','1B_pred','2B_pred','3B_pred','HR_pred']] = np.where(df_['Spray Angle'].isna(),
    #                                                                          None,
    #                                                                          xwOBAcon_model.predict_proba(df_[xwOBAcon_model.feature_names_in_].astype('float')))
    #     return df_[['out_pred','1B_pred','2B_pred','3B_pred','HR_pred']].mul([0,0.9,1.25,1.6,2]).sum(axis=1)

    df['3D wOBAcon'] = None#[sum(np.multiply(xwOBA_model.predict_proba([[x,y,z]])[0],np.array([0,0.9,1.25,1.6,2]))) for x,y,z in zip(df['Spray Angle'],df['Launch Angle'],df['Launch Speed']) if x != None else None]

    game_df = df.assign(vs_rhh = lambda x: np.where(x['hitterside']=='R',1,0)).groupby(['game_date','Opp','MLBAMID','Pitcher','pitch_type'])[['Num Pitches','Velo','IVB','IHB','Ext','vs_rhh','CS','Whiffs','3D wOBAcon']].agg({
        'Num Pitches':'count',
        'Velo':'mean',
        'IVB':'mean',
        'IHB':'mean',
        'Ext':'mean',
        'vs_rhh':'sum',
        'CS':'sum',
        'Whiffs':'sum',
        '3D wOBAcon':'mean'
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
    merge_df['Ext'] = merge_df['Ext'].round(2)

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

    return merge_df[['Date','Opp','Pitcher','Type','Num Pitches','Usage','vs R','vs L','Velo','Ext','IVB','IHB','CS','Whiffs','CSW','3D wOBAcon']].rename(columns={'Num Pitches':'#'})
if pitcher_list == {}:
    st.write('No pitches thrown yet')
elif st.button("Generate Player Table"):
    table_df = scrape_savant_data(player_select,game_id)
    st.dataframe(table_df,use_container_width=True,hide_index=True)
