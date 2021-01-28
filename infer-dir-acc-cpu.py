import os 
import json
import GPUtil
import argparse 
import torch 
from tqdm import tqdm 
import torch.nn as nn
import kaldiio
import dataset_infer as dataset 
import dataloader 
import module.model as module_model
from utils.parse_config import ConfigParser
import shutil
import os
import audiofile as af
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from python_speech_features import logfbank, fbank, mfcc
import time
# parse args
def parse_args():
    desc="parse model info"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_bad', type=str, required=True, help='/path/to/model')
    parser.add_argument('--model_clf', type=str, required=True, help='path to classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='/path/to/test/audio')
    parser.add_argument('--seg_len', type=int, default=5, help='length of each segment')
    parser.add_argument('--index', type=str, help='/dir/to/index/directory')
#     parser.add_argument('--win_shift', type=int, default=2, help='window shift between adjacent audio')
    parser.add_argument('--gpuid', type=str, default='0', help='assign gpu by id')
    parser.add_argument('--noise_threshold', type=float, default=0.5, help='threshold to detect noise')
    parser.add_argument('--topk', type=int, default=3, help='display top k possible predictions')
    parser.add_argument('--label', type=str, default=None, help="the label of audios in the dir")
    return parser.parse_args()

# util funcs
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def wav2spec(audioDir, sampleRate):
    sr, y = wavfile.read(audioDir)
    logFeat = python_speech_features.base.logfbank(y, samplerate=44100, winlen=0.046, winstep=0.014, nfilt=80, nfft=2048, lowfreq=50, highfreq=12000, preemph=0.97)
    logFeat = np.resize(logFeat, (700,80))
    logFeat = NormalizeData(logFeat)
    return logFeat


# define a trimmer
def trimmer(clipIn, sampleRate, segLen, clipOut, win_shift):
    audioLen = sox.file_info.duration(clipIn)
    label = 0
    frameStart = 0
    frameWin = win_shift
    frameLen = segLen
    frameEnd = frameLen
    frames = []
    while frameEnd <= audioLen:
        nameTmp = '_'.join(clipIn.split('/')[-2:])[:-4] + '_seg_%d.wav'%label
        os.system('sox %s %s trim %d %d'%(clipIn, clipOut+nameTmp, frameStart, segLen))
        label += 1
        frameStart += frameWin
        frameEnd = frameStart + frameLen
        
def featExtractWriter(wavPath, cmn=True):
    y, sr = af.read(wavPath)
    featLogfbank = logfbank(y, sr, **kwargs)
    if cmn:
        featLogfbank -= np.mean(featLogfbank, axis=0, keepdims=True)
    return featLogfbank

def get_instance(module, cfs, *args):
    return getattr(module, cfs['type'])(*args, **cfs['args'])

args = parse_args()
model_path  = args.model_bad
model_clf = args.model_clf
audio_dir = args.data_dir
seg_len = args.seg_len
# win_shift = args.win_shift
gpuid = args.gpuid
noise_thres = args.noise_threshold
topk = args.topk
dir_label = args.label
top1_count = 0
top3_count = 0
tik = 1

audios = [audio_dir + x for x in os.listdir(audio_dir) if ('.wav' in x or '.mp3' in x)]
clips_num = len(audios)

# apply VAD to detect noise
import os
import sys
import sox
import h5py
import numpy as np
import random
import argparse
from scipy.io import wavfile
import soundfile as sf
import python_speech_features
from tqdm import tqdm

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, AveragePooling2D
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.regularizers import l2

# build model
print('... loading VAD model ...')
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
except:
    pass
model = Sequential()
# convolution layers
model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(700, 80, 1), ))  # low: try different kernel_initializer
model.add(BatchNormalization())  # explore order of Batchnorm and activation
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 3)))  # experiment with using smaller pooling along frequency axis
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 1)))
model.add(Conv2D(16, (3, 3), padding='valid', kernel_regularizer=l2(0.01)))  # drfault 0.01. Try 0.001 and 0.001
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 1)))

# dense layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))  # leaky relu value is very small experiment with bigger ones
model.add(Dropout(0.5))  # experiment with removing this dropout
model.add(Dense(1, activation='sigmoid'))

# load trained model
modelDir = model_path
modelTest = load_model(modelDir)
print('... finished loading ...')


# check sample rate
import sox
# if not sox.file_info.sample_rate(audio) == 44100:
tfm = sox.Transformer()
tfm.set_output_format(file_type='wav', rate=44100, channels=1)

start_time = time.time()
print('######## Total %d test cases ########'%clips_num)

for audio in audios:
    print('######## Test %d ########'%tik)
    tik+=1
    print('... identifying audio %s ...'%audio.split('/')[-1])
    print('... transform sr %d to 44100 ...'%sox.file_info.sample_rate(audio))
    audio_n = audio[:-4] + '_new.wav'
    tfm.build_file(audio, audio_n)
    # update audio file
    audio = audio_n


    # check whether segment or not
    audio_length = sox.file_info.duration(audio)
        # if audio length < 5s, pad 
    if audio_length < 10:
        print('... SKIPPED, less than %d seconds ...\n'%seg_len)
        clips_num -= 1
        continue
    # fix the win_shift param
    print('... %s: %.2f ...'%(audio.split('/')[-1], audio_length))
    win_shift = audio_length//30 + 1
    print('... trimming (win shift:%d) ...'%win_shift)
    if audio_length > seg_len:
        audio_out = '/'.join(audio.split('/')[:-1]) + '/'+ 'splited/'
        if not os.path.isdir(audio_out):
            os.mkdir(audio_out)
        trimmer(audio, 44100, seg_len, audio_out, win_shift)
    print('... trimming done! %d segments in total ...'%len(os.listdir(audio_out)))





    all_segs = [audio_out + x for x in os.listdir(audio_out)]
    print('... detecting noises ...')
    for seg in all_segs:
        try:
            segSpec = wav2spec(seg, sampleRate=44100)
        except:
            os.remove(seg)
            continue
        segSpec = segSpec.reshape(1, segSpec.shape[0], segSpec.shape[1], 1)
        predProb = modelTest.predict(segSpec)[0][0]
        if predProb < noise_thres: # consider as noise
            os.remove(seg)
    all_segs = [audio_out + x for x in os.listdir(audio_out)]
    if len(all_segs) == 0:
        print('... SKIPPED, all noises ...\n')
        clips_num -= 1
        continue
    print('... detecting done, obtained %d segments ...'%len(all_segs))


    # extract h5 files


    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 256,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 11000,
        "preemph": 0.97
    }



    h5_out = '/'.join(audio.split('/')[:-1]) + '/'+ 'splited_h5/'
    if not os.path.isdir(h5_out):
        os.mkdir(h5_out)
    print('... extracting features ...')
    for song in all_segs:
        song_name = song.split('/')[-1]
        h5_out_path = h5_out + song_name + '.h5'
        featLogfbank = featExtractWriter(song)

        hf = h5py.File(h5_out_path,'w')
        hf.create_dataset('logfbank', data=featLogfbank)
    print('... extraction feats done ...')

    # extract index
    index_dir = args.index
    utt2wav_text = ""
    utt2label_text = ""
    all_h5_utt2wav = [x[:-7] + ' ' + h5_out+x for x in os.listdir(h5_out)]
    all_h5_utt2label = [x[:-7] + ' ' + 'cuculus_canorus' for x in os.listdir(h5_out)]
    utt2wav_text = '\n'.join(all_h5_utt2wav)
    utt2label_text = '\n'.join(all_h5_utt2label)
    with open(index_dir+'demo_utt2wav','w') as f:
        f.write(utt2wav_text)
    with open(index_dir+'demo_utt2label','w') as f:
        f.write(utt2label_text)



    # g-vector extractor
    config = {
        "n_gpu": 1,
        "dataset": {
            "type": "LogfbankDataset",
            "args": {
                "wav_scp": "/DATA1/ziang/index/large/demo_utt2wav",
                "spk2int": '/DATA1/ziang/index/large/label2int.json',
                "logfbank_kwargs":{
                    "winlen": 0.025,
                    "winstep": 0.01,
                    "nfilt": 256,
                    "nfft": 1024,
                    "lowfreq": 500,
                    "highfreq": None,
                    "preemph": 0.97
                },
                "padding": "wrap",
                "cmn": True
            }
        },
        "dataloader": {
            "type": "SimpleDataLoader",
            "args": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 8,
                "drop_last": True
            }
        },
        "model": {
            "type": "Gvector",
            "args": {
                "channels": 1,
                "block": "BasicBlock", 
                "num_blocks": [
                    3,
                    4,
                    6,
                    3
                ],
                "embd_dim": 1024,
                "drop": 0.5, 
                "n_class": 80
            }
        }
    }


    model = get_instance(module_model, config['model'])
    chkpt = torch.load(model_clf,map_location=torch.device('cpu'))

    try:
        model.load_state_dict(chkpt['model'])
    except:
        model.load_state_dict(chkpt)
    model = model.to('cpu')

    testset = get_instance(dataset, config['dataset'])
    testloader = get_instance(dataloader, config['dataloader'], testset)

    model.eval()
    with open('/DATA1/ziang/index/large/label2int_inv.json') as f:
        identifier = json.load(f)
    # print(identifier)
    voters = {i:0 for i in range(len(identifier))}
    print('... identifying ...')
    for i, (utt, data) in enumerate(testloader):
        utt = utt[0]
        # For utterances longer than 30s, we only truncate the first 30s.
        if data.shape[1] > 3000:
            data = data[:,:3000,:]
        data = data.float().to('cpu')
        with torch.no_grad():
            embd = model.forward(data)
            pred = torch.argmax(embd, dim=1)
            voters[pred.item()] += 1

    # print('voters:', voters)
    voters = dict(sorted(voters.items(), key=lambda item: item[1], reverse=True))
#     print(identifier[str(list(voters.keys())[0])])
#     print([identifier[str(s)] for s in list(voters.keys())[:topk]])
    if identifier[str(list(voters.keys())[0])] == dir_label:
        top1_count+=1
    if dir_label in [identifier[str(s)] for s in list(voters.keys())[:topk]]:
        top3_count+=1
    result = "*** top %d identifications: ***\n"%topk
    for i in range(topk):
        result += "*** %s: %.4f\n"%(identifier[str(list(voters.keys())[i])],list(voters.values())[i]/len(all_segs))
    #     if i==0:
    #         print('Prediction: %s'%identifier[str(list(voters.keys())[0])])
    #     else:
    #         print('top %d predictions: %s'%(topk,identifier[str(list(voters.keys())[i])]))
    print(result)
    shutil.rmtree(h5_out, ignore_errors=True)
    shutil.rmtree(audio_out, ignore_errors=True)
    # vote_result = identifier[str(max(voters, key=voters.get))]
    # print('result is %s'%vote_result)

# print results
print("######## %s seconds ########" % (time.time() - start_time))
print('######## accuracy of %s: ########\n... top1 accuracy: %.4f\n... top3 accuracy: %.4f'%(dir_label, top1_count/clips_num, top3_count/clips_num))
    
# remove files

os.remove(index_dir+'demo_utt2wav')
os.remove(index_dir+'demo_utt2label')
# os.system('rm -rf %ssplited_h5'%audio_dir)
# os.system('rm -r %ssplited'%audio_dir)
os.system('rm %s*_new.wav'%audio_dir)