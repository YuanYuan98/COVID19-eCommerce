import torch
import pandas as pd
import json
import numpy as np
import copy
import time
from config import Config
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,mean_squared_error,mean_absolute_error,mean_squared_log_error
from data_loader import *
from model import *
import random

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

class Trainer(object):
    def __init__(self,args,logging):
        self.config = Config
        self.args = args

    def data_split(self):
        self.train_data,self.test_data = data_loader(self.args,self.config)

    def evaluate(self,test_iter,model,myloss):
        model.eval()
        for X,x_future, y in test_iter:
            X = X.float().cuda()
            y = y.float().cuda()
            x_future = x_future.cuda()
            y_pred = model(X,x_future)

            y_pred, y= [i for i in y_pred.cpu().detach().numpy()], [i for i in y.cpu().detach().numpy()]

            y_pred, y = torch.tensor(self.scaler.inverse_transform(y_pred)).cuda(),torch.tensor(self.scaler.inverse_transform(y)).cuda()

            
            l = myloss(y_pred,y).item()
            mse = F.mse_loss(y_pred,y)
            rmse = np.sqrt(mse.cpu().detach().numpy())
            nrmse = rmse/(max(y).item()-min(y).item())

            y_pred,y_test = y_pred.cpu().detach().numpy(),y.cpu().detach().numpy()
            print('mape:',mape(y_test,y_pred))
            print('smape:',smape(y_test,y_pred))
            print('mae:',mean_absolute_error(y_test,y_pred))
            print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
            print('nrmse:',np.sqrt(mean_squared_error(y_test,y_pred))/(max(y_test)-min(y_test)))
            y_pred = [i+1 for i in y_pred]
            y_test = [i+1 for i in y_test]
            #print('rmsle',np.sqrt(mean_squared_log_error(y_test,y_pred)))

        return l,rmse,nrmse

    def init_model_se2seq(self):

        if self.args.model==1:
            enc = Encoder_GRU(self.config,self.args).cuda()
            dec = Decoder_GRU(self.config,self.args).cuda()
            model = Seq2Seq(self.config,self.args,enc,dec).cuda()

        if self.args.model==2:
            enc = Encoder_GRU(self.config,self.args,2).cuda()
            dec = Decoder_GRU(self.config,self.args).cuda()
            model = EnCovid(self.config,self.args,enc,dec).cuda()

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.01)
                        
        model.apply(init_weights)

        optimizer = torch.optim.Adam(model.parameters(),lr=self.args.lr,weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience = self.args.patience, min_lr=1e-5)
        train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        print('Trainable Parameters:', np.sum([p.numel() for p in train_params]))
        return model,optimizer,scheduler

    
    def train_seq2seq(self, model, iterator, optimizer, criterion, clip):

        model.train()
    
        epoch_loss = 0

        for src,trg in iterator:
            src = src.cuda()
            trg = trg.cuda()

            src_b = src[:,0].squeeze(dim=1).permute(1,0).unsqueeze(dim=2)
            src_covid = src[:,1].squeeze(dim=1).permute(1,0).unsqueeze(dim=2)

            trg_b = trg[:,0].permute(1,0)
            trg_covid = trg[:,1].permute(1,0)

            optimizer.zero_grad()

            output = model(src_b,trg_b,src_covid,trg_covid)


            output = output[1:].view(-1)
            trg_b = trg_b[1:].reshape(-1)

            loss = criterion(output, trg_b)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)



    def evaluate_seq2seq(self,model, iterator, criterion):
    
        model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
        
            for src,trg in iterator:

                src = src.cuda()
                trg = trg.cuda()

                src_b = src[:,0].squeeze().permute(1,0).unsqueeze(dim=2)
                src_covid = src[:,1].squeeze().permute(1,0).unsqueeze(dim=2)

                trg_b = trg[:,0].permute(1,0)
                trg_covid = trg[:,1].permute(1,0)

                output = model(src_b, trg_covid, src_covid,trg_covid,0) #turn off teacher forcing

                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]

                output = output[1:].view(-1,1)
                trg_b = trg_b[1:].reshape(-1,1)

                output, trg_b = output.cpu().numpy(), trg_b.cpu().numpy()

                output, trg_b = self.scaler_x1.inverse_transform(output),self.scaler_x1.inverse_transform(trg_b)

                print('mape:',mape(output, trg_b ))
                print('smape:',smape(output, trg_b ))
                print('mae:',mean_absolute_error(output, trg_b ))
                print('rmse:',np.sqrt(mean_squared_error(output, trg_b )))
                print('nrmse:',np.sqrt(mean_squared_error(output, trg_b ))/(max(trg_b )-min(trg_b)))
            
                epoch_loss += mean_absolute_error(output, trg_b)

                print(epoch_loss)
            
        #return epoch_loss / len(iterator)
        return (mean_absolute_error(output, trg_b),np.sqrt(mean_squared_error(output, trg_b)),np.sqrt(mean_squared_error(output, trg_b ))/(max(trg_b )-min(trg_b))[0])


    def train_step(self):
        CLIP = 1
        train_iterator,valid_iterator,scaler_x1,scaler_x2 = data_loader_seq2seq(self.args,self.config)
        self.scaler_x1 = scaler_x1
        self.scaler_x2 = scaler_x2
        best_valid_loss = float('inf')
        stop_num=0
        file_place = 'data'

        # #print('\n------------------------- Initialize Model -------------------------')
        model,optimizer,scheduler = self.init_model_se2seq()
        criterion = nn.L1Loss(reduction = 'mean')

        print('\n------------------------- Training -------------------------')
        for epoch in range(self.args.epoch_num):

            print('learning rate:',optimizer.param_groups[0]['lr'])   
            
            train_loss = self.train_seq2seq(model, train_iterator, optimizer,criterion, CLIP)
            valid_loss = self.evaluate_seq2seq(model, valid_iterator, criterion)[0]
            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                stop_num = 0
                torch.save(model.state_dict(),file_place)
                print("!!!!!!!!!! Model Saved !!!!!!!!!!")

            else:
                if stop_num>=self.args.early_stop:
                    break
                stop_num += 1
            
            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

        print('\n------------------------- Testing -------------------------')
        model.load_state_dict(torch.load(file_place))
        test_loss = self.evaluate_seq2seq(model,valid_iterator,criterion)
        info = 'province:{},product:{},model:{},lr:{},batch_size:{},teaching_force_ratio:{},src:{},trg:{}'.format(self.args.province,self.args.p_type,self.args.model,self.args.lr,self.args.batch_size,self.args.teacher_forcing_ratio,self.args.trg,self.args.src)
        print(info)
        print('-------------------------------------------------------------')
        return test_loss,info
    