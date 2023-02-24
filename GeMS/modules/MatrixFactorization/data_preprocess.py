GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import pandas as pd

def load_data(file, names, delimiter):
    load_data = pd.read_csv(file, names=names, delimiter=delimiter)
    return load_data

def process_interactions(data):
    user_interactions = {}
    for _, row in data.iterrows():
        user_id = row['user_id']
        if user_id not in user_interactions:
            user_interactions[user_id] = []
        user_interactions[user_id].append(int(row['item_id']))
    return user_interactions
