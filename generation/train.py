import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import LMDataset, Seq2SeqDataset
from evaluation import evaluate
import argparse
import numpy as np
import os
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=512, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--seq2seq", default=False, action="store_true")
    parser.add_argument("--model-type", default="transformer", choices=["lstm", "transformer"])
    parser.add_argument("--load-dir", default='')
    args = parser.parse_args()
    return args


def train(args):
    args.save_dir += "_" + args.model_type + "_lm" if not args.seq2seq else "_seq2seq"
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == "lstm":
        from lstm import LMModel, Seq2SeqModel
    elif args.model_type == "transformer":
        from transformer import LMModel, Seq2SeqModel

    if args.seq2seq:
        train_set = Seq2SeqDataset(device=device)
        valid_set = Seq2SeqDataset(split="valid", device=device)
        model = Seq2SeqModel(args, train_set.dictionary).to(device)
    else:
        train_set = LMDataset(device=device)
        valid_set = LMDataset(split="valid", device=device)
        model = LMModel(args, train_set.dictionary).to(device)
    
    if args.load_dir != '':
        _model = torch.load(args.load_dir)
        model.load_state_dict(_model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6], gamma=0.3)

    # evaluate(model, valid_set)
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for samples in pbar:
                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
                if len(losses)>100:
                    now_loss = np.mean(losses[-100:])
                else:
                    now_loss = np.mean(losses)
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f, now loss: %0.8f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr'], now_loss))
        scheduler.step()
        
        if epoch % args.save_interval == 0:
            torch.save(model, args.save_dir + "/{}_{}.pt".format(args.model_type, epoch + 1))
            torch.save(optimizer.state_dict(), args.save_dir + "/{}opt_{}.pt".format(args.model_type, epoch + 1))
        evaluate(model,valid_set)
        
        print("改革春风吹满地-->", model.generate("改革春风吹满地", beam_size=5))
        print("苟利国家生死以-->", model.generate("苟利国家生死以", beam_size=5))

if __name__ == "__main__":
    args = get_args()
    train(args)
