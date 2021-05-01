import copy
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
        self.enc = nn.LSTM(300,64,4, batch_first = True, dropout=.5, bidirectional=True)
        self.fc1 = nn.Linear(128,64)
        self.dec = nn.LSTM(300,64,4, batch_first = True, dropout=.5)
        self.fc2 = nn.Linear(64,l)

    def logits(self, source, prev_outputs, **unused):
        # TODO
        output, hidden = self.enc(self.vec(source))
        hidden = self.fc1(hidden)
        logits, hidden = self.dec(self.vec(prev_outputs),hidden)
        logits = self.fc2(logits)
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
        device = self.vec.weight.device
        if beam_size == None:
            beam_size = 1
        source = [self.dictionary.index(s) for s in inputs]
        source = torch.tensor(source,dtype=torch.int32).reshape((1,-1)).to(device)
        out, hidden = self.enc(self.vec(source))
        hidden = self.fc1(hidden)
        
        bos = self.dictionary.bos()
        eos = self.dictionary.eos()
        final = []
        rklist = [ {"str":[bos,],"lprob":0, "hidden":hidden}, ]
        for l in range(max_len):
            tmp = []
            for sample in rklist:
                prev = torch.tensor([sample["str"][-1],]).reshape((1,1)).to(device)
                logits, hidden = self.dec(self.vec(prev),sample["hidden"])
                logits = self.fc2(logits).reshape(-1)
                lprobs = F.log_softmax(logits)
                for k in range(beam_size):
                    x = torch.argmax(lprobs).item()
                    s = sample["str"].copy()
                    p = sample["lprob"]
                    # print(x,s,lprobs,lprobs[x])
                    if x == eos:
                        # if len(s) == len(inputs)+1:
                        final.append({"str":s+[x,], "lprob": l + lprobs[x], "hidden":hidden})
                    else:
                        tmp.append({"str":s+[x,], "lprob": l + lprobs[x], "hidden":hidden})
                    lprobs[x] = -Inf
            tmp.sort(key = lambda x: -x["lprob"])
            rklist = tmp[:beam_size]
            if len(final) == beam_size:
                break
        final.sort(key = lambda x: -x["lprob"])
        final = final[0]["str"]
        outputs = "" 
        for i in final:
            outputs += self.dictionary.symbols[i]
        return outputs
