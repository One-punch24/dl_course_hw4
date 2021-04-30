import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary

class LMModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # TODO

    def logits(self, source, **unused):
        # TODO
        return logits
    
    def get_loss(self, source, target, reduce=True, **unused):
        logits = self.logits(source)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs, 
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, prefix, max_len=100, beam_size=None):
        '''
        prefix: The initial words, like "白"
        
        output a string like "白日依山尽，黄河入海流，欲穷千里目，更上一层楼。"
        '''
        # TODO 
        outputs = ""
        return outputs


class Seq2SeqModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # TODO
        l = len(self.dictionary)
        self.emb = nn.Embedding(l, 10)
        self.enc = nn.LSTM(10,l,2,batch_first =True, bidirectional=True)
        self.dec = nn.LSTM(10,l,2,batch_first =True)

    def logits(self, source, prev_outputs, **unused):
        # TODO
        x = self.emb(source)
        y = self.emb(prev_outputs)
        output, hidden = self.enc(x)
        logits, hidden = self.dec(y,hidden)
        return logits
    
    def get_loss(self, source, prev_outputs, target, reduce=True, **unused):
        # print(source.shape)
        # print(prev_outputs.shape)
        # print(target.shape)
        # for j in range(source.shape[1]):
        #     print(self.dictionary[source[0,j]], self.dictionary[prev_outputs[0,j]], self.dictionary[target[0,j]])
        # exit()
        logits = self.logits(source, prev_outputs)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs, 
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, inputs, max_len=100, beam_size=None):
        '''
        inputs, 上联: "改革春风吹满地"
        
        output 下联: "复兴政策暖万家"
        '''
        # TODO 
        outputs = ""
        return outputs
