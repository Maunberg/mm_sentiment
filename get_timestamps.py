import pandas as pd
import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from ctc_forced_aligner import AlignmentTorch
import json
from pydub import AudioSegment

name = '/home/dutov@ad.speechpro.com/work/course/' + 'Data_Train_modified.csv'
name_save = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/ex_data/' + 'train.json'

dataset = pd.read_csv(name)

if os.path.exists(name_save):
    with open(name_save) as f:
        results = json.load(f)
else:
    results = {}

n = 0
for i in tqdm(range(0, len(dataset))):
    try:
        info = dict(dataset.iloc[i])
        audio = info['video']
        input_audio_path = '/home/dutov@ad.speechpro.com/work/course/Audio/Audio/WAV_16000/'+audio+'.wav'
        input_text_path = 'rub/test.txt'
        output_srt_path = 'rub/test_srt'
        output_vtt_path = 'rub/test_vtt'
        start_ms = int(info['start_time'] * 1000)
        end_ms = int(info['end_time'] * 1000)
        cropped_audio_path = 'rub/test_wav.wav'

        audio_data = AudioSegment.from_wav(input_audio_path)
        cropped_audio = audio_data[start_ms:end_ms]
        cropped_audio.export(cropped_audio_path, format="wav")
        with open(input_text_path, 'w') as f:
            f.write(info['text'])
        key = audio+'|'+str(info['start_time'])
        if key not in results:
            results[key] = []

            at = AlignmentTorch()
            ret = at.generate_srt(cropped_audio_path, input_text_path, output_srt_path)
            ret = at.generate_webvtt(cropped_audio_path, input_text_path, output_vtt_path)
            results[key].extend(at.word_timestamps)
            n += 1
            if n % 10 == 0:
                with open(name_save, 'w') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            torch.cuda.empty_cache()
    except:
        print(i)
