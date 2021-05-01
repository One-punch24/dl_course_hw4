import copy
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchtext

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

def get2Vec(di):
    d = torchtext.vocab.Vectors('../sgns.literature.bigram-char')
    vec = []
    for s in di.symbols:
        vec.append(d.get_vecs_by_tokens(s))
    vec = torch.stack(vec)
    emb = nn.Embedding(len(di),300)
    emb.from_pretrained(vec)
    return emb

# get2Vec()

class Seq2SeqModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # TODO
        l = len(self.dictionary)
        # self.emb = nn.Embedding(l, 32)
        self.vec = get2Vec(self.dictionary)
        self.enc = nn.LSTM(300,64,2,batch_first =True)
        self.dec = nn.LSTM(300,64,2,batch_first =True)
        self.fc = nn.Linear(64,l)

    def logits(self, source, prev_outputs, **unused):
        # TODO
        x = self.vec(source)
        y = self.vec(prev_outputs)

        output, hidden = self.enc(x)
        logits, hidden = self.dec(y,hidden)
        logits = self.fc(logits)
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
        if beam_size == None:
            beam_size = 5
        x = self.dictionary.index(inputs)
        for x in range(max_len):
            logits = 0
        return outputs
