import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule


from model import BERTDataset,BERTClassifier,calc_accuracy

if __name__ = '__main__' :
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()
    dataset_train = nlp.data.TSVDataset("ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
## Train parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5
    valid_split = 0.2
    train_size = len(dataset_train)
    
    data_train = BERTDataset(dataset_train[int(train_size*valid_split):], 0, 1, tok, max_len, True, False)
    data_valid = BERTDataset(dataset_train[:int(train_size*valid_split)], 0, 1, tok, max_len, True, False)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    valid_dataloader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, num_workers=5)
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(valid_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} valid acc {}".format(e+1, test_acc / (batch_id+1)))
        
    torch.save(model.state_dict(), "trained_model.pth")
