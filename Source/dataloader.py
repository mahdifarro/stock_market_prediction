from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
from textwrap import wrap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 512
BATCH_SIZE = 16
pre_trained_model_ckpt = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

class StockNewsDataset(Dataset):
    def __init__(self, news, targets, tokenizer, max_len, include_raw_text=False):
        super(StockNewsDataset, self).__init__()
        self.news = news
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_raw_text = include_raw_text
        
        
    def get_my_raw_text(self, index):
        return str(self.news[index])
    
    def __len__(self):
        return len(self.news)
    
    def __getitem__(self, item):
        new = str(self.news[item])
        target = self.targets[item]
        
        encoding = self.tokenizer.encode_plus(
            new,
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = False,
            return_attention_mask = True,
            truncation = True,
            padding = True, 
            return_tensors = 'pt'
        )
        
        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
       
        
        if self.include_raw_text:
            output['review_text'] = item
            
            
        return output




def create_data_loader(df, tokenizer, max_len = MAX_LEN, batch_size = BATCH_SIZE, include_raw_text=False):
    ds = StockNewsDataset(
        news=df.headline.to_list(),
        targets=df.label.to_list(),
        tokenizer=tokenizer,
        max_len=max_len,
        include_raw_text=include_raw_text
    )
    return DataLoader(ds, batch_size=batch_size, collate_fn=collator)

    