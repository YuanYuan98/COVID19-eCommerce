# -*- coding:utf-8 -*-

import pandas as pd
import torch.utils.data as Data
import torch
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def data_loader(args,Config):
    data = pd.read_csv('{}'.format(args.file),index_col=0)
    for c in ['mask','disinfectant']:
        data[c] = data[c].apply(lambda x:eval(x))
    l =[[float(i[3][3]),float(i[6]),i[7]] for i in data.values.tolist()]

    scaler_x1 = MinMaxScaler()
    scaler_x2 = MinMaxScaler()

    x1 = scaler_x1.fit_transform([[i[0]] for i in l])
    x2 = scaler_x2.fit_transform([[i[1]] for i in l])
    x_future = [i[2][:1] for i in l]

    print(x_future)

    x3 = x_future
    x3 = []
    for index,j in enumerate(x_future):
        trans = scaler_x2.transform([[m] for m in j])
        trans = [i[0] for i in trans]
        x3.append(trans)

    d = []

    for i in range(len(l)):
        d.append([x1[i][0],x2[i][0],x3[i]])

    data_temp = []
    for index,i in enumerate(d):
        if index>=args.days_before and index<len(d)-args.days_predction:
            f = d[index-days_before:index]
            y = d[index+1:index+1+days_predction]
            data_temp.append([torch.tensor([[j[0] for j in f],[j[1] for j in f]]),torch.tensor(d[index][2]),torch.tensor([j[0] for j in y])])
    
    data_all = data_temp

    #print(len(data_all))
    train_data = data_all[:int(len(data_all)*Config.train_test_split)]
    test_data = data_all[int(len(data_all)*Config.train_test_split):]

    train_data = Data.DataLoader(train_data, batch_size = args.batch_size,shuffle=True)
    test_data = Data.DataLoader(test_data, batch_size = 500)

    return 0

    return train_data,test_data, scaler_y

def time_trans(x):  
    x = time.strptime(x,'%Y/%m/%d')
    return time.strftime("%Y-%m-%d", x) 


def data_loader_seq2seq(args,Config):

    if args.city==0:
        if args.flag==0:
            data = pd.read_csv(file1,index_col=0).reset_index()
            
            for c in ['mask','disinfectant','hand_sanitizer','vitamin','thermometer']:
                data[c] = data[c].apply(lambda x:eval(x))

            data = data[data['date']<'2020-04-01']

            l =[[float(i[args.p_type][args.b_type]),float(i[5])] for i in data[['mask','hand_sanitizer','disinfectant','vitamin','thermometer','cases_pdf']].values.tolist()]

        if args.flag==1:
            data = pd.read_csv(file2,index_col=0).reset_index()

            data = data[data['time']<'2020-04-01']

            l = [[float(i[args.p_type]),float(i[13])] for i in data[['milk', 'daily', 'vegtable', 'chocalate', 'disposable', 'small_bottal', 'drink', 'nut', 'cotton', 'present', 'china_wine', 'yogurt', 'apple', 'cases_pdf']].values.tolist()]
        
    else:
        data = pd.read_csv(file3,index_col=0).reset_index()

        data = data[data['province']==args.province]

        data['time'] = data['time'].apply(lambda x:time_trans(x))

        data = data[data['time']<'2020-04-01']
        
        l =[[float(i[0]),float(i[1])] for i in data[['mask','cases_pdf']].values.tolist()]     


    scaler_x1 = MinMaxScaler()
    scaler_x2 = MinMaxScaler()

    x1 = scaler_x1.fit_transform([[i[0]] for i in l])
    x2 = scaler_x2.fit_transform([[i[1]] for i in l])

    d = []
    for i in range(len(l)):
        d.append([x1[i][0],x2[i][0]])

    data_all = []
    for index,i in enumerate(d):
        if index>=args.src and index<len(d)-args.trg:
            before = d[index-args.src+1:index+1]
            future = d[index+1:index+1+args.trg]
            data_all.append([torch.tensor([[i[0] for i in before],[i[1] for i in before]]),torch.tensor([[-1]+[i[0] for i in future],[-1]+[i[1] for i in future]])])

    train_data = data_all[:int(len(data_all)*Config.train_test_split)]
    test_data = data_all[int(len(data_all)*Config.train_test_split):]

    train_data = Data.DataLoader(train_data, batch_size = args.batch_size)
    test_data = Data.DataLoader(test_data, batch_size = 1000)

    return train_data,test_data, scaler_x1,scaler_x2
