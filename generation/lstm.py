import copy
from math import inf
from unicodedata import bidirectional
from numpy.core.numeric import Inf
import torch
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
    d = torchtext.vocab.Vectors('../sgns.merge.word')
    vec = []
    for s in di.symbols:
        v = d.stoi.get(s,-1)
        if v == -1:
            v = torch.randn(300)
        else:
            v = d.vectors[v]
        # v = torch.randn(300)
        vec.append(v)
    vec = torch.stack(vec)
    emb = nn.Embedding(len(di),300)
    emb.from_pretrained(vec,freeze=True)
    
    return emb


class Seq2SeqModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # TODO
        l = len(self.dictionary)
        # self.emb = nn.Embedding(l, 32)
        self.vec = get2Vec(self.dictionary)
        sz = 512
        self.enc = nn.LSTM(300,sz,1, batch_first = True, dropout=.5, bidirectional=True)
        self.dec = nn.LSTM(300,sz * 2,1, batch_first = True, dropout=.5)
        self.fc1 = nn.Linear(sz * 4,sz * 2)
        self.fc2 = nn.Linear(sz * 2,l)
        self.dropout = nn.Dropout(0.5)

        self.k_proj = nn.Linear(sz * 2,sz * 2)
        self.v_proj = nn.Linear(sz * 2,sz * 2)
        
    def change(self, h):
        a,b,c = h.shape
        h = h.reshape((a//2,2,b,c))
        h = h.permute((0,2,3,1))
        h = h.reshape((a//2,b,c*2))
        return h.contiguous()

    def logits(self, source, prev_outputs, **unused):
        # TODO
        output, hidden = self.enc(self.vec(source))
        hidden = (self.change(hidden[0]), self.change(hidden[1]))
        Q, hidden = self.dec(self.vec(prev_outputs),hidden)
        batch, SeqLen_q, Channel = Q.shape
        SeqLen_k = output.shape[1]
        K = self.k_proj(output)
        V = self.v_proj(output)
        # V = self.dropout(V)
        M = torch.bmm(Q,K.permute(0,2,1))
        pad = self.dictionary.pad()
        key_padding_mask = (source == pad).reshape((batch,1,-1))
        # print(M.shape, key_padding_mask.shape)
        M = M.masked_fill(key_padding_mask,-inf)
        M = F.softmax(M,2)
        # print(M.shape, V.shape)
        M = (M.unsqueeze(3)) * (V.unsqueeze(1))
        M = M.sum(2)
        # print(M.shape)
        # print(Q.shape)
        x = self.fc1(torch.cat([M,Q],2))
        x = F.tanh(self.dropout(x))
        logits = self.fc2(x)
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
        ) / source.shape[0]

    @torch.no_grad()
    def generate(self, inputs, max_len=100, beam_size=None):
        '''
        inputs, 上联: "改革春风吹满地"
        
        output 下联: "复兴政策暖万家"
        '''
        # TODO 
        device = self.fc2.weight.device
        if beam_size == None:
            beam_size = 1
        bos = self.dictionary.bos()
        eos = self.dictionary.eos()
        source = [bos,] + [self.dictionary.index(s) for s in inputs]
        source = torch.tensor(source,dtype=torch.int32).reshape((1,-1)).to(device)
        
        final = []
        rklist = [ {"str":[bos,],"lprob":0}, ]
        for l in range(max_len):
            tmp = []
            for sample in rklist:
                s = sample["str"]
                lp = sample["lprob"]
                prev = torch.tensor(s).reshape((1,-1)).to(device)
                logits = self.logits(source,prev)[0,-1,:]
                
                # print(source.shape)
                # print(prev.shape)
                # print(logits.shape)
                lprobs = F.log_softmax(logits, dim=0).view(-1)
                topk = torch.topk(logits,beam_size).indices
                topk = list(topk.reshape(-1))
                
                if len(s) == len(inputs)+1:
                    final.append({"str":s+[eos,], "lprob": lp + lprobs[eos]})
                
                for t in topk:
                    x = t.item()
                    tmp.append({"str":s+[x,], "lprob": lp + lprobs[x]})
                if len(final) >= beam_size:
                    break
                    
            tmp.sort(key = lambda x: -x["lprob"])
            rklist = tmp[:beam_size]
            if len(final) == beam_size:
                break
        final.sort(key = lambda x: -x["lprob"])
        # for x in final:
        #     t = x["str"]
        #     outputs = "" 
        #     for i in t:
        #         outputs += self.dictionary.symbols[i]    
        #     print(outputs, x["lprob"])
        print(final[0]['lprob'])
        final = final[0]["str"]
        outputs = "" 
        for i in final:
            outputs += self.dictionary.symbols[i]
        
        return outputs
