import deepspeech
from audio2numpy import open_audio
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

def process_audio(filename, new_rate=16000):
    signal, sampling_rate = open_audio(filename)
    down_sample = librosa.resample(y=signal, orig_sr=sampling_rate, target_sr=new_rate)
    down_sample = np.int16(down_sample*32767.0)
    return down_sample
    
def main():
    model = deepspeech.Model('deepspeech-0.9.3-models.pbmm')
    new_rate = model.sampleRate()
    
    df = pd.read_csv('../cv-corpus-6.1-2020-12-11/en/train.tsv', sep='\t')
    filelist = df['path'].tolist()

    pred_dict = dict()
    
    for i in tqdm(range(2000)):
        filename = filelist[i]
        path = '../cv-corpus-6.1-2020-12-11/en/clips/' + filename

        audio_signal = process_audio(path, new_rate)
        pred_dict[filename] = model.stt(audio_signal)

#         if i%1000==0:
#             jname = str(i) + ".json"
#             with open(jname, "w") as outfile: 
#                 json.dump(pred_dict, outfile)
    jname = "testfromtrain.json"
    with open(jname, "w") as outfile: 
        json.dump(pred_dict, outfile)
                
if __name__=="__main__":
    main()