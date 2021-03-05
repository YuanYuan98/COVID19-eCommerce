import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import copy
import random


class Encoder_GRU(nn.Module):
    def __init__(self,config,args,input_size = 1):
        super(Encoder_GRU, self).__init__()
        self.config = config
        self.args = args
        self.gru = nn.GRU(input_size, self.args.hidden_size, dropout=self.args.dropout)
        
    def forward(self,src):
        output,hidden = self.gru(src)
        return output,hidden

class Decoder_GRU(nn.Module):
    def __init__(self,config,args):
        super(Decoder_GRU, self).__init__()
        self.config = config
        self.args = args
        self.gru = nn.GRU(1, self.args.hidden_size, dropout=self.args.dropout)
        self.fc_out = nn.Linear(2*self.args.hidden_size, self.config.output_size)
        
    def forward(self,input,hidden,context):

        input = input.unsqueeze(0).unsqueeze(2)
        
        output, hidden = self.gru(input, hidden)

        output = torch.cat([output.squeeze(0),context.squeeze(0)],dim=1)
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self,config,args,encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, src_covid,trg_covid, flag=1):

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_size = self.config.output_size
        
        outputs = torch.zeros(trg_len, batch_size, trg_size).cuda()
        
        output, context = self.encoder(src)
        
        input = trg[0,:]
        
        for t in range(1, trg_len):
            if t==1:
                hidden = context
            output, hidden= self.decoder(input,hidden,context)
            
            outputs[t] = output
            if flag==1:
                teacher_forcing_ratio = self.args.teacher_forcing_ratio
            else:
                teacher_forcing_ratio = 0.0

            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.squeeze(1)
            
            input = trg[t] if teacher_force else top1
    
        return outputs

class EnCovid(nn.Module):
    def __init__(self,config,args,encoder, decoder):
        super(EnCovid, self).__init__()
        self.config = config
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, src_covid,trg_covid, flag=1):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_size = self.config.output_size

        src = torch.cat([src,src_covid],dim=2)
        outputs = torch.zeros(trg_len, batch_size, trg_size).cuda()
    
        output, context = self.encoder(src)
        
        input = trg[0,:]
        
        for t in range(1, trg_len):

            if t==1:
                hidden = context
            
            output, hidden= self.decoder(input, hidden, context)
            
            outputs[t] = output

            if flag==1:
                teacher_forcing_ratio = self.args.teacher_forcing_ratio
            else:
                teacher_forcing_ratio = 0.0

            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.squeeze(1)
            
            input = trg[t] if teacher_force else top1
        
        return outputs

