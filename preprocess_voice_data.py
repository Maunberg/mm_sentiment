import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import torch
from tqdm import tqdm
import pickle
import os

input_dir = '/home/dutov@ad.speechpro.com/work/course/'
train = '/home/dutov@ad.speechpro.com/work/course/Data_Train_modified.csv'
test = '/home/dutov@ad.speechpro.com/work/course/Data_Test_original.csv'
window = 2
overlap = 0.5

def process_data(input_file, window, input_dir='/home/dutov@ad.speechpro.com/work/course/', overlap=0, mode=['spectr', 'mel', 'mfcc'], ):
    print('Loading...')
    
    df = pd.read_csv(input_file)
    df['len_time'] = df['end_time'] - df['start_time']
    df['id'] = df.index
    df['id'] = df['id'].astype(str)
    
    def split_intervals(row, window=3, overlap=1.5):
        segments = []
        start, end = row['start_time'], row['end_time']
        duration = row['len_time']
    
        if duration <= window:
            segments.append(row)
        else:
            current_time = start
            last_current = 0
            n = 100
            while current_time < end and round(current_time, 1) != round(last_current, 1):
                last_current = current_time
                next_time = min(current_time + window, end)
                segment = row.copy()
                segment['start_time'] = current_time
                segment['end_time'] = next_time
                segment['len_time'] = next_time - current_time
                segments.append(segment)
    
                # Перекрытие, если overlap задан
                if next_time != end:
                    current_time = next_time - overlap
    
        return segments
    
    expanded_data = []
    for _, row in df.iterrows():
        expanded_data.extend(split_intervals(row, window=window, overlap=overlap))
    
    print('Length of expanded data:', len(expanded_data))
    
    
    df_expanded = pd.DataFrame(expanded_data)
    
    agg_columns = ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    df_expanded = df_expanded.groupby(['video', 'start_time'], as_index=False).agg({
        **{col: 'mean' for col in agg_columns},
        'id':'|'.join,
        'ASR': ' '.join,
        'text': ' '.join,
        'end_time': 'max',
        'len_time': 'sum'
    })
    
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
    
    df_expanded['sentiment_classes'] = df_expanded['sentiment'].apply(transform_sentiment)
    for emo in ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
        df_expanded[emo] = df_expanded[emo].apply(transform_emotion)
    df_expanded['sentiment_classes'] = df_expanded['sentiment'].apply(transform_sentiment)
    
    
    df_expanded['emotions_concat'] = (
        df_expanded['happy'].astype(str) +
        df_expanded['sad'].astype(str) +
        df_expanded['anger'].astype(str) +
        df_expanded['surprise'].astype(str) +
        df_expanded['disgust'].astype(str) +
        df_expanded['fear'].astype(str)
    )
    
    full_info = {}
    uniq_values = {}
    for i in range(len(df_expanded)):
        data = dict(df_expanded.iloc[i])
        key = data['video']
        key_count = data['emotions_concat']
        if key not in full_info:
            full_info[key] = []
        if key_count not in uniq_values:
            uniq_values[key_count] = 0
        uniq_values[key_count] += 1
        if uniq_values[key_count] < 10_000:
            full_info[key].append({
                'start_time': data['start_time'],
                'end_time':data['end_time'],
                'happy':data['happy'],
                'sad':data['sad'],
                'anger':data['anger'],
                'surprise':data['surprise'],
                'disgust':data['disgust'],
                'fear':data['fear'],
                'sentiment':data['sentiment_classes'],
                'id':data['id']
            })
    
    n = 0
    for key in full_info:
        for i in full_info[key]:
            n += 1
    print('Result length:', n)
    
    print('Extract features...')
    
    features = {}
    for feat in mode:
        features[feat] = []
    target = {'happy':[], 'sad':[], 'anger':[], 'surprise':[],'disgust':[], 'fear':[], 'sentiment':[], 'id':[]}
    for audio in tqdm(sorted(full_info.keys())[0:]):
        path = input_dir+'/Audio/Audio/WAV_16000/'+audio+'.wav'
        audio_get, file_sr = sf.read(path)
        for sample in full_info[audio]:
            start_sample = int(sample['start_time'] * file_sr)
            end_sample = int(sample['end_time'] * file_sr)
            curr_audio = audio_get[start_sample:end_sample]
            if len(curr_audio) == 16000*window:
                for f in ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
                    target[f].append(sample[f])
                target['id'].append(sample['id'])
                n_fft = min(2048, len(curr_audio))
                # print(len(curr_audio))
                if 'spectr' in mode:
                    features['spectr'].append(np.abs(np.fft.rfft(curr_audio, n=n_fft))),
                if 'mel' in mode:
                    features['mel'].append(librosa.feature.melspectrogram(y=curr_audio, sr=file_sr, n_mels=64, n_fft=n_fft))
                if 'mfcc' in mode:
                    features['mfcc'].append(librosa.feature.mfcc(y=curr_audio, sr=file_sr, n_mfcc=13))
    return features, target
    
features, target = process_data(input_file=train, window=window, overlap=overlap)

if not os.path.exists('ex'):
    os.mkdir('ex')
name = f'ex/features_train_w{window}_o{overlap}.pickle'
with open(name, 'wb') as f:
    pickle.dump(features, f)

name = f'ex/targets_train_w{window}_o{overlap}.pickle'
with open(name, 'wb') as f:
    pickle.dump(target, f)

features, target = process_data(input_file=test, window=window, overlap=overlap)

name = f'ex/features_test_w{window}_o{overlap}.pickle'
with open(name, 'wb') as f:
    pickle.dump(features, f)

name = f'ex/targets_test_w{window}_o{overlap}.pickle'
with open(name, 'wb') as f:
    pickle.dump(target, f)
