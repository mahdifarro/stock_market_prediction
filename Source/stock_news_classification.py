from dataloader import *
from modeling import *
from trainer import *
import os
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

from preprocess import load

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams

import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
from textwrap import wrap
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted Sentiment')
    
    

def tokenize_datasets(tokenizer):
    class_names = ['Negative', 'Neutral', 'Positive']
    df_train, df_test, df_val = load()
    # print(df_train.headline.to_list())
    
    train_data_loader = create_data_loader(df_train, tokenizer)
    val_train_loader = create_data_loader(df_val, tokenizer, include_raw_text=True)
    test_data_loader = create_data_loader(df_test, tokenizer, include_raw_text=True)
    
    # for value in test_data_loader:
    #     print(value['review_text'])
    #     print(np.array(test_data_loader.dataset.news)[value['review_text'].tolist()])
    #     break
    
    data = next(iter(train_data_loader))
    
    model = SentimentClassifier(3)
    model = model.to(device)
    
    EPOCHS = 10
    
    optimizer = optim.Adam(model.parameters(), lr= 1e-5)
    
    total_steps = len(train_data_loader) * EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1} / {EPOCHS}')
        train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        val_acc, val_loss = eval_model(model, val_train_loader, loss_fn, device, len(df_val))
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'Source/Model/best_model.bin'))
            best_accuracy = val_acc
    y_news_texts, y_pred, y_pred_porbs, y_test = get_predictions(model, test_data_loader)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)

tokenize_datasets(tokenizer)  

# sample_text = "Text Processing with Machine learning is moving at a rapid speed"

# tokens = tokenizer.tokenize(sample_text)

# token_ids = tokenizer.convert_tokens_to_ids(tokens)

# print(f'Sentence: {sample_text}')

# print(f'Tokens: {tokens}')

# print(f'Token_ids: {token_ids}')

# encoding = tokenizer.encode_plus(
#     sample_text,
#     max_length=32,
#     truncation = True,
#     add_special_tokens=True,
#     return_token_type_ids=False,
#     padding=True,
#     return_attention_mask=True,
#     return_tensors='pt'
# )

# print(f'Encoding keys: {encoding.keys()}')
# print(len(encoding['input_ids'][0]))
# print(encoding['input_ids'][0])
# print(len(encoding['attention_mask'][0]))
# print(encoding['attention_mask'])
# print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))


