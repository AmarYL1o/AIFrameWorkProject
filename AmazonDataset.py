import paddle
import numpy as np
from paddle.io import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import random

class RecDataset(Dataset):
    def __init__(self, user_item_pairs, num_users, num_items, user_pos_items):
        self.user_neg_items = {} 
        self.user_item_pairs = user_item_pairs
        self.num_users = num_users
        self.num_items = num_items
        # 记录每个用户的正样本
        self.user_pos_items = user_pos_items  
        all_items = set(range(num_items))
        for user, pos_list in self.user_pos_items.items():
            pos_set = set(pos_list)
            self.user_neg_items[user] = list(all_items - pos_set)

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.user_item_pairs[idx]
        # 随机负样本采样
        neg_item = random.choice(self.user_neg_items[user])
        return user, pos_item, neg_item

def load_and_split_data(csv_path, min_rating=4.0, n_splits=5):
    df = pd.read_csv(csv_path, names=["item", "user", "rate", "timestamp"])
    df["rate"]=pd.to_numeric(df["rate"],errors='coerce')
    df = df[df["rate"] >= min_rating]
    
    
    # 编码用户和物品ID
    user_ids = df["user"].astype("category").cat.codes.values
    item_ids = df["item"].astype("category").cat.codes.values
    num_users, num_items = len(df["user"].unique()), len(df["item"].unique())
    
    # 记录每个用户的所有正样本
    user_pos_items = {}
    for u, i in zip(user_ids, item_ids):
        if u not in user_pos_items:
            user_pos_items[u] = []
        user_pos_items[u].append(i)
    
    # 5折分割：按用户分组，确保同一用户的所有交互在同一fold中
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in kf.split(df):
        train_pairs = list(zip(user_ids[train_idx], item_ids[train_idx]))
        test_pairs = list(zip(user_ids[test_idx], item_ids[test_idx]))
        folds.append((train_pairs, test_pairs))
    
    return folds, num_users, num_items, user_pos_items