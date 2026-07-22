from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd


pd.set_option('display.max_columns', None) #show all columns, dont truncate

# Find a player's ID by name

player = [p for p in players.get_players() if p['full_name'] == 'Stephen Curry'][0]
print("Player ID:", player['id'])

# Pull their game log for one season
gamelog = playergamelog.PlayerGameLog(player_id=player['id'], season = '2022')
df = gamelog.get_data_frames()[0]

print("\nShape (rows, cols): ", df.shape)
print("\nColumn names :")
print(df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head())

