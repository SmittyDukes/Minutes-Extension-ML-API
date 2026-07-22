from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time

pd.set_option('display.max_columns', None)


def build_features(df):

    # Convert GAME_DATE from text to real datetime objects so sorting works chronologically
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')

    # NOW sort — this sorts by actual date, oldest first
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    # Previous game's minutes: shift MIN down by 1, so each row shows the PRIOR game's minutes
    df['prev_min'] = df['MIN'].shift(1)



    df['days_rest'] = df['GAME_DATE'] - df['GAME_DATE'].shift(1)
    df['days_rest'] = df['days_rest'].dt.days

    df['is_away'] = df['MATCHUP'].str.contains("@").astype(int)

    WINDOW = 10
    df['rolling_avg_min'] = df['MIN'].shift(1).rolling(WINDOW).mean()

    df['label'] = (df['rolling_avg_min'] * 1.10 < df['MIN']).astype(int)
    df = df[10:].reset_index(drop=True)


    return df



player_names = ['Alex Caruso', 'Payton Pritchard', 'Naz Reid', 'Davion Mitchell', 'Rui Hachimura', 'Austin Reaves', 'Anfernee Simons', 'Obi Toppin','Gary Trent Jr.']
all_players =[]
seasons =['2023','2024','2025']
for name in player_names:
    player = [p for p in players.get_players() if p['full_name'] == name][0]
    for season in seasons:
        gamelog = playergamelog.PlayerGameLog(player_id=player['id'], season= season)
        raw = gamelog.get_data_frames()[0]
        all_players.append(build_features(raw))
        time.sleep(1)
df = pd.concat(all_players).reset_index(drop=True)


df.to_csv('features.csv', index=False)