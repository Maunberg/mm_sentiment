import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from utils_preprocess import Tokeniser
import re

mode = ['tfidf', 'bert', 'token'] #tfidf bert token
input_dir = '/home/dutov@ad.speechpro.com/work/course/'
train = '/home/dutov@ad.speechpro.com/work/course/Data_Train_modified.csv'
test = '/home/dutov@ad.speechpro.com/work/course/Data_Test_original.csv'
vec = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/rub/vectoriser.pickle'
bert = 'cointegrated/rubert-tiny2-cedr-emotion-detection'
vocab = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/rub/vocab.json'
n = 2

# 0 Gherman/bert-base-NER-Russian
# 1 DeepPavlov/rubert-base-cased
# 2 cointegrated/rubert-tiny2-cedr-emotion-detection

def process_data(input_file, input_dir='/home/dutov@ad.speechpro.com/work/course/', mode=['tfidf', 'bert', 'token'], 
                 vec='vectoriser.pickle', text_column = 'ASR', bert='DeepPavlov/rubert-base-cased', vocab='vocab.json'):
    print('Loading...')
    data = pd.read_csv(input_file)
    data['id'] = data.index
    data['id'] = data['id'].astype(str)
    def transform_sentiment(x):
        if x > 1:
            return 2
        elif x == 0:
            return 1
        elif x < 0:
            return 0
    
    def transform_emotion(x):
        if x > 0:
            return 1
        else:
            return 0

    for emo in ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
        data[emo] = data[emo].apply(transform_emotion)
    data['sentiment'] = data['sentiment'].apply(transform_sentiment)
    
    print('Converting...')
    features = {}
    if 'tfidf' in mode:
        print('TF-IDF converting...')
        if os.path.exists(vec):
            vectorizer = joblib.load(vec)
        else:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(data[text_column])
            joblib.dump(vectorizer, vec)
        features['tfidf'] = vectorizer.transform(data[text_column]).toarray()
    if 'bert' in mode:
        print('BERT converting...')
        tokenizer = BertTokenizer.from_pretrained(bert)
        features['bert'] = tokenizer(data[text_column].to_list(), padding='max_length', truncation=True, return_tensors="pt", max_length=256)['input_ids']
    if 'token' in mode:
        print('Token converting...')
        if os.path.exists(vocab):
            tokeniser = Tokeniser()
            tokeniser.generate_vocab(
                        'Test',
                        use_saved=True,
                        way=vocab
                        )
        else:
            text = ' <EOS> <BOS> '.join([str(i) for i in data[text_column].to_list()])
            text = re.sub(' +', ' ', re.sub('([^\w<>@])', r' \1 ', text)).strip()
            tokeniser = Tokeniser()
            tokeniser.generate_vocab(
                        text, 
                        use_saved=False,
                        way=vocab
                        )                        
        text = data[text_column].astype(str).apply(lambda x: '<BOS> '+re.sub(' +', ' ', re.sub(r'([^\w<>@])', r' \1 ', x)).strip()+' <EOS>').to_list()
        tokenized_sequences = [tokeniser.tokenize(i) for i in text]
        padded_sequences = []
        max_length = 256
        for seq in tokenized_sequences:
            if len(seq) < max_length:
                padded_seq = seq + [0] * (max_length - len(seq))
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)

        features['token'] = padded_sequences
        print('Vocab size:', len(tokeniser))
    return features, data

features, target = process_data(input_file=train, vec=vec, bert=bert, vocab=vocab)

if not os.path.exists('ex'):
    os.mkdir('ex')
name = f'ex/features_train_{n}.pickle'
with open(name, 'wb') as f:
    pickle.dump(features, f)

name = f'ex/targets_train_{n}.pickle'
with open(name, 'wb') as f:
    pickle.dump(target, f)

features, target = process_data(input_file=test, vec=vec, bert=bert, vocab=vocab)

name = f'ex/features_test_{n}.pickle'
with open(name, 'wb') as f:
    pickle.dump(features, f)

name = f'ex/targets_test_{n}.pickle'
with open(name, 'wb') as f:
    pickle.dump(target, f)
