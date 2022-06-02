# Import packages

import unicodedata
import sys
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Primary data cleaning
STOP_WORDS = ["a", "an", "the", "this", "that", "of", "for", "or",
              "and", "on", "to", "be", "if", "we", "you", "in", "is",
              "at", "it", "rt", "mt", "with"]
STOP_PREFIXES = ("@", "#", "http", "&amp")
punctuations = string.punctuation + '–' + '…'

###########################
# PART 1: DATA PROCESSING #
###########################

# Import and merge data
true_data = pd.read_csv('data/true.csv')
fake_data = pd.read_csv('data/fake.csv')
true_data["label"] = 1
fake_data["label"] = 0
data = true_data.append(fake_data).sample(frac=1).reset_index().drop(columns=['index', 'subject', 'date'])

train_text, temp_text, train_labels, temp_labels = train_test_split(data['title'], data['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=data['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)


###########################
#   PART 2: BERT MODEL    #
###########################

class NewsDetect():

    def __init__(self, L=15, batch_size=32):
        self.L = L
        self.batch_size = batch_size
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    def tokenize(self, text):
        '''
        A method to turn English text into a token by the max length
        '''
        tk = tokenizer.batch_encode_plus(
            text.tolist(),
            max_length = self.L,
            pad_to_max_length=True,
            truncation=True)
        return tk
    
    def data_prep(self, seq, mask, y):
        '''
        '''
        my_data = TensorDataset(seq, mask, y)
        my_sampler = RandomSampler(my_data)
        my_dataloader = DataLoader(my_data, sampler=my_sampler, batch_size=self.batch_size)


    def train():





