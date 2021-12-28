import pandas as pd
import numpy as np
import torch


def get_data(path):
    #read the csv file
    df = pd.read_csv(path)
    df = df[['WhitePlayer', 'BlackPlayer', 'WhiteScore']]

    #for testing
    print(df.head())

    df.loc[df['WhiteScore'] == 0.5, 'WhiteScore'] \
         = np.random.randint(0, 2, size=len(df[df['WhiteScore'] == 0.5]))
    
    max_whiteplayer_index = np.max(df['WhitePlayer'])
    max_blackplayer_index = np.max(df['WhitePlayer'])
    assert max_whiteplayer_index == max_blackplayer_index

    num_players = max_blackplayer_index
    num_games = len(df)

    print("Number of players:", num_players)
    print("Number of games", num_games)
  
    indices = np.array(df.index).reshape(num_games, 1)
    white_player_idx = np.array(df['WhitePlayer'].values).reshape(num_games, 1)
    black_player_idx = np.array(df['BlackPlayer'].values).reshape(num_games, 1)

    white_player_idx -= 1
    black_player_idx -= 1
   
    white_player_sparse_indices = np.concatenate([indices, white_player_idx], axis=-1).astype(int)
    black_player_sparse_indices = np.concatenate([indices, black_player_idx], axis=-1).astype(int)

    sparse_game_indices = np.concatenate([white_player_sparse_indices, black_player_sparse_indices], axis=0)
    sparse_game_values = np.concatenate([np.ones((num_games, )), np.ones((num_games, ))* -1], axis=0)
    
    sparse_game_indices = sparse_game_indices.T
    
    #Game matrix has shape [G, P]
    sparse_game_matrix = torch.sparse_coo_tensor(sparse_game_indices,\
         sparse_game_values, torch.Size([num_games, num_players]))
    
    outcomes = torch.tensor(df['WhiteScore'].values.reshape(num_games, 1))

    return sparse_game_matrix, outcomes
