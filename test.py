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
import csv

if __name__ = '__main__' :
    device = torch.device("cuda:0")
    bertmodel, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))
    
    test_path = '/ko_data.csv'
    f = open(test_path, 'r', encoding='CP949')
    rdr = csv.reader(f)
    test_data = []
    for line in rdr:
      test_data.append(line)
      
    test_data_inversed=[]
    for x in test_data[1:]:
      test_data_inversed.append([x[1],x[0]])
    data_test = BERTDataset(test_data_inversed, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, num_workers=5)    
    
    answer = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            with torch.no_grad():
              out = model(token_ids, valid_length, segment_ids)
            answer.append(out)
    
    ans = []
    for x in answer:
      max_vals, max_indices = torch.max(x, 1)
      ans.append(int(max_indices))
      
    f = open('/answer_data.csv', 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['Id','Predicted'])
    for i,x in enumerate(ans):
      wr.writerow([i,x])
    f.close()