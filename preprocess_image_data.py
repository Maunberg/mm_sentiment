import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
import joblib
import cv2
import mediapipe as mp
import threading


input_dir = '/home/dutov@ad.speechpro.com/work/course/'
train = '/home/dutov@ad.speechpro.com/work/course/Data_Train_modified.csv'
test = '/home/dutov@ad.speechpro.com/work/course/Data_Test_original.csv'
video = '/home/dutov@ad.speechpro.com/work/course/Video/Combined/'


def process_data(input_file, video_dir='/home/dutov@ad.speechpro.com/work/course/Video//Combined/', input_dir='/home/dutov@ad.speechpro.com/work/course/', mode=['image', 'landmarks']):
    videos = [i.replace('.mp4', '') for i in video_dir]
    df = pd.read_csv(input_file)
    df[df['video'].isin(videos)]
    df['len_time'] = df['end_time'] - df['start_time']
    df['id'] = df.index
    df['id'] = df['id'].astype(str)
    
    def split_intervals(row, window=1, overlap=0):
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

                if next_time != end:
                    current_time = next_time - overlap

        return segments

    expanded_data = []
    for _, row in df.iterrows():
        expanded_data.extend(split_intervals(row))

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
    
    features = {'landmarks':[], 'rgb':[], 'wb':[]}
    target = {'happy':[], 'sad':[], 'anger':[], 'surprise':[],'disgust':[], 'fear':[], 'sentiment':[], 'id':[]}

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    num_per_one = 5

    def process_frame(rgb_frame, results):
        results[0] = face_mesh.process(rgb_frame)

    for video in tqdm(sorted(full_info.keys())[0:]):
        path = video_dir + video + '.mp4'
        if not os.path.exists(path):
            continue

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        for sample in full_info[video]:
            start = sample['start_time']
            end = sample['end_time']
            frame_time = 0
            while frame_time <= float(start):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_time = frame_count / fps
                frame_count += 1

            n = 0
            while frame_time <= float(end) and num_per_one >= n:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_time = frame_count / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #thread = threading.Thread(target=process_frame, args=(rgb_frame, results))
                #thread.start()
                #thread.join(timeout=15)
                #if thread.is_alive():
                #    thread.join()
                #else:
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]
                        features['landmarks'].append(points)
                        #features['rgb'].append(rgb_frame)
                        #features['wb'].append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        for f in ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
                            target[f].append(sample[f])
                        break
                frame_count += 1
                n += 1
        cap.release()
    return features, target

features, target = process_data(input_file=train)

if not os.path.exists('ex'):
    os.mkdir('ex')
name = f'ex/features_train.pickle'
with open(name, 'wb') as f:
    pickle.dump(features, f)

name = f'ex/targets_train.pickle'
with open(name, 'wb') as f:
    pickle.dump(target, f)


features, target = process_data(input_file=test)
name = f'ex/features_test.pickle'
with open(name, 'wb') as f:
    pickle.dump(features, f)

name = f'ex/targets_test.pickle'
with open(name, 'wb') as f:
    pickle.dump(target, f)
