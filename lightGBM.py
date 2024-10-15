import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb

window_size = 15
industry_list = ['Chemical Industry','Electronic Technology','Information Technology','Lifestyle Services','Manufacturing and Transportation','Pharmaceuticals and Biotechnology','Finance','Construction']
def Mape(true_y, pred_y):
    n = len(true_y)
    mape = 100 * sum([abs(true_y[i] - pred_y[i]) / true_y[i] for i in range(n)]) / n
    return mape

def evaluate(true_y, pred_y):
    mae = mean_absolute_error(true_y, pred_y)
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    mape = Mape(true_y, pred_y)
    return mae, rmse, mape


def get_test_data(df):
    df_train = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2023-11-01')]
    df_test = df[(df['date'] >= '2023-11-01') & (df['date'] <= '2024-04-01')]
    close_min = df['close'].min()
    close_max = df['close'].max()
    df_train = df_train.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)
    for i in range(len(df_train)):
        df_train.loc[i, 'close'] = (df_train.loc[i, 'close'] - close_min) / (close_max - close_min)
        df_train.loc[i, 'high'] = (df_train.loc[i, 'high'] - df['high'].min()) / (df['high'].max() - df['high'].min())
        df_train.loc[i, 'low'] = (df_train.loc[i, 'low'] - df['low'].min()) / (df['low'].max() - df['low'].min())
        df_train.loc[i, 'open'] = (df_train.loc[i, 'open'] - df['open'].min()) / (df['open'].max() - df['open'].min())
        df_train.loc[i, 'volume'] = (df_train.loc[i, 'volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())
        df_train.loc[i, 'amount'] = (df_train.loc[i, 'amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())
        df_train.loc[i, 'amplitude'] = (df_train.loc[i, 'amplitude'] - df['amplitude'].min()) / (df['amplitude'].max() - df['amplitude'].min())
        df_train.loc[i, 'pct_chg'] = (df_train.loc[i, 'pct_chg'] - df['pct_chg'].min()) / (df['pct_chg'].max() - df['pct_chg'].min())
        df_train.loc[i, 'rfs'] = (df_train.loc[i, 'rfs'] - df['rfs'].min()) / (df['rfs'].max() - df['rfs'].min())
        df_train.loc[i, 'change'] = (df_train.loc[i, 'change'] - df['change'].min()) / (df['change'].max() - df['change'].min())
        df_train.loc[i, 'pivot'] = (df_train.loc[i, 'pivot'] - df['pivot'].min()) / (df['pivot'].max() - df['pivot'].min())


    for i in range(len(df_test)):
        df_test.loc[i, 'close'] = (df_test.loc[i, 'close'] - close_min) / (close_max - close_min)
        df_test.loc[i, 'high'] = (df_test.loc[i, 'high'] - df['high'].min()) / (df['high'].max() - df['high'].min())
        df_test.loc[i, 'low'] = (df_test.loc[i, 'low'] - df['low'].min()) / (df['low'].max() - df['low'].min())
        df_test.loc[i, 'open'] = (df_test.loc[i, 'open'] - df['open'].min()) / (df['open'].max() - df['open'].min())
        df_test.loc[i, 'volume'] = (df_test.loc[i, 'volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())
        df_test.loc[i, 'amount'] = (df_test.loc[i, 'amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())
        df_test.loc[i, 'amplitude'] = (df_test.loc[i, 'amplitude'] - df['amplitude'].min()) / (df['amplitude'].max() - df['amplitude'].min())
        df_test.loc[i, 'pct_chg'] = (df_test.loc[i, 'pct_chg'] - df['pct_chg'].min()) / (df['pct_chg'].max() - df['pct_chg'].min())
        df_test.loc[i, 'rfs'] = (df_test.loc[i, 'rfs'] - df['rfs'].min()) / (df['rfs'].max() - df['rfs'].min())
        df_test.loc[i, 'change'] = (df_test.loc[i, 'change'] - df['change'].min()) / (df['change'].max() - df['change'].min())
        df_test.loc[i, 'pivot'] = (df_test.loc[i, 'pivot'] - df['pivot'].min()) / (df['pivot'].max() - df['pivot'].min())

    X_train = []
    y_train = []
    for i in range(window_size, len(df_train)):
        train_list = []
        train_list.extend(list(df_train.loc[i-window_size:i, 'close']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'high']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'low']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'open']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'volume']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'amount']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'amplitude']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'pct_chg']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'rfs']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'change']))
        train_list.extend(list(df_train.loc[i-window_size:i, 'pivot']))
        X_train.append(train_list)
        y_train.append(df_train.loc[i, 'close'])
        
    X_test = []
    y_test = []
    for i in range(window_size, len(df_test)):
        test_list = []
        test_list.extend(list(df_test.loc[i-window_size:i, 'close']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'high']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'low']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'open']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'volume']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'amount']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'amplitude']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'pct_chg']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'rfs']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'change']))
        test_list.extend(list(df_test.loc[i-window_size:i, 'pivot']))

        X_test.append(test_list)
        y_test.append(df_test.loc[i, 'close'])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    lgb_model = lgb.LGBMRegressor(random_state=52,n_estimators=100)
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    return y_pred,y_test,close_min,close_max   

def get_data(industry):
    true_y = []
    pred_y = []
    # 读入./industry/中所有的文件
    path = './'+industry+'/'
    files = os.listdir(path)
    for file in files:
        df = pd.read_csv(path+file)
        y_pred,y_test,close_min,close_max = get_test_data(df)
        true_y.extend(y_test*(close_max - close_min) + close_min)
        pred_y.extend(y_pred*(close_max - close_min) + close_min)
    return true_y,pred_y


for industry in industry_list:
    true_y, pred_y = get_data(industry)
    print(industry)
    print('-------------------------------------')
    mae, rmse, mape = evaluate(true_y, pred_y)
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('MAPE:', mape)
    # 将以上结果写入result.txt文件中
    with open('result_lightGBM.txt', 'a') as f:
        f.write(industry+'\n')
        f.write('MAE:'+str(mae)+'\n')
        f.write('RMSE:'+str(rmse)+'\n')
        f.write('MAPE:'+str(mape)+'\n')
        f.write('-------------------------------------\n')