import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
# 计算两个list的MAPE
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

industry_list = ['Chemical Industry','Electronic Technology','Information Technology','Lifestyle Services','Manufacturing and Transportation','Pharmaceuticals and Biotechnology','Finance','Construction']
window_size  = 15
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
    df_test = df[(df['date'] >= '2023-11-01') & (df['date'] <= '2024-04-01')]
    # 对df_test中的close列进行min-max归一化
    close_min = df_test['close'].min()
    close_max = df_test['close'].max()
    df_test['close'] = (df_test['close'] - df_test['close'].min()) / (df_test['close'].max() - df_test['close'].min())
    df_test = df_test.reset_index(drop = True)
    return df_test, close_min, close_max

def get_data(industry):
    true_y = []
    pred_y = []
    path = './'+industry+'/'
    files = os.listdir(path)
    for file in files:
        df = pd.read_csv(path+file)
        df_test,close_min,close_high = get_test_data(df)
        for i in range(0, len(df_test)-window_size):
            true_y.append(df_test.iloc[i+window_size]['close']*(close_high-close_min)+close_min)
            df_train = df_test.iloc[i:i+window_size]
            model = ExponentialSmoothing(df_train['close'], trend='add', seasonal=None)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            pred_y.append(forecast.values[0]* (close_high - close_min) + close_min)
    return true_y, pred_y

for industry in industry_list:
    true_y, pred_y = get_data(industry)
    print(industry)
    print('-------------------------------------')
    mae, rmse, mape = evaluate(true_y, pred_y)
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('MAPE:', mape)
    with open('result_ES.txt', 'a') as f:
        f.write(industry+'\n')
        f.write('MAE:'+str(mae)+'\n')
        f.write('RMSE:'+str(rmse)+'\n')
        f.write('MAPE:'+str(mape)+'\n')
        f.write('-------------------------------------\n')