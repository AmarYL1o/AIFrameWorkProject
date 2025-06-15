import paddle
import paddle.nn as nn
import numpy as np
from model import PMLAM
from AmazonDataset import RecDataset
from paddle.io import DataLoader
import pandas
from sklearn.model_selection import KFold
from AmazonDataset import load_and_split_data
from trainer import train_and_evaluate

csv_path = "/irip/yueziben_2024/project/AIFramework/dataset/CDs_and_Vinyl.csv"  # 格式：item,user,rate,timestamp
folds, num_users, num_items, user_pos_items = load_and_split_data(csv_path)

# 5折训练与验证
train_and_evaluate(folds, num_users, num_items, user_pos_items, embed_dim=50, epochs=40)