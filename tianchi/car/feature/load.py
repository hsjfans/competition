import numpy as np
import pandas as pd


train_path = '../data/used_car_train_20200313.csv'
test_path = '../data/used_car_testB_20200421.csv'


def load_data():
    ori_data = pd.read_csv(train_path)
    return ori_data
