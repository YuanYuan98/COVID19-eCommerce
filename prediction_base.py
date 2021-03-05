# -*- coding:utf-8 -*-

import pandas as pd
import torch.utils.data as Data
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import xgboost as xgb
import time
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,mean_squared_error,mean_absolute_error,mean_squared_log_error


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

def time_trans(x):  
    x = time.strptime(x,'%Y/%m/%d')
    return time.strftime("%Y-%m-%d", x) 


def prediction_base(args,Config,m=1):

    if args.city==0:

        if args.flag==0:
            data = pd.read_csv(file1,index_col=0)
            
            for c in ['mask','disinfectant','hand_sanitizer','vitamin','thermometer']:
                data[c] = data[c].apply(lambda x:eval(x))

            data = data[data['date']<'2020-04-01']

            l =[[float(i[args.p_type][args.b_type]),float(i[5])] for i in data[['mask','hand_sanitizer','disinfectant','vitamin','thermometer','cases_pdf']].values.tolist()]

        else:
            data = pd.read_csv(file2)

            data = data[data['time']<'2020-04-01']

            l = [[float(i[args.p_type]),float(i[13])] for i in data[['milk', 'daily', 'vegtable', 'chocalate', 'disposable', 'small_bottal', 'drink', 'nut', 'cotton', 'present', 'china_wine', 'yogurt', 'apple', 'cases_pdf']].values.tolist()]
        

    
    else:
        data = pd.read_csv(file3,index_col=0).reset_index()

        data = data[data['province']==args.province]

        data['time'] = data['time'].apply(lambda x:time_trans(x))

        data = data[data['time']<'2020-04-01']
        
        l =[[float(i[0]),float(i[1])] for i in data[['mask','cases_pdf']].values.tolist()]

        
    d = l

    data_all = []
    for index,i in enumerate(d):
        if index>=args.src and index<len(d)-args.trg:
            before = d[index-args.src+1:index+1]
            future = d[index+1:index+1+args.trg]
            data_all.append([[i[0] for i in before],[i[1] for i in before],[i[0] for i in future],[i[1] for i in future]])
        

    train_data = data_all[:int(len(data_all)*Config.train_test_split)]
    test_data = data_all[int(len(data_all)*Config.train_test_split):]

    # AR, ARIMA
    
    y_pred = []
    y_test = []
    if m in [1,2]:
        for i in test_data:
            train,test = i[0],i[2]
            if m==1:
                model = AutoReg(train,lags = 1)
                model_fit = model.fit()
            elif m==2:
                model = ARIMA(train,order=[0,0,1])
                model_fit = model.fit(disp=0)
            
            predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
            y_pred += list(predictions)
            y_test += test
        
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        print('mape:',mape(y_test,y_pred))
        print('mae:',mean_absolute_error(y_test,y_pred))
        print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
        print('nrmse:',np.sqrt(mean_squared_error(y_test,y_pred))/(max(y_test)-min(y_test)))

    

    if m in [3,4]:
        # Xgboost

        if m==3:
            X_train = np.array([i[0]+i[1] for i in train_data])
            X_test = np.array([i[0]+i[1] for i in test_data])

        if m==4:
            X_train = np.array([i[0] for i in train_data])
            X_test = np.array([i[0] for i in test_data])

        y_train = np.array([i[2][0] for i in train_data])

        
        reg = xgb.XGBRegressor(n_estimators=1000)
        reg.fit(X_train, y_train,verbose=False)
        y_pred = []
        for u in X_test:
            u = u[np.newaxis, :]
            temp = []
            input = u
            for d in range(args.trg):
                pred = reg.predict(input)
                input = np.array(list(u[0,1:args.src])+list(pred)+list(u[0,args.src:]))
                input = input[np.newaxis,:]
                temp.append(pred[0])
            y_pred += temp
        
        test = [i[2] for i in test_data]
        y_test = []
        for u in test:
            y_test += u

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        print('mape:',mape(y_test,y_pred))
        print('mae:',mean_absolute_error(y_test,y_pred))
        print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
        print('nrmse:',np.sqrt(mean_squared_error(y_test,y_pred))/(max(y_test)-min(y_test)))


    return (mean_absolute_error(y_test,y_pred),np.sqrt(mean_squared_error(y_test,y_pred)),np.sqrt(mean_squared_error(y_test,y_pred))/(max(y_test)-min(y_test)))