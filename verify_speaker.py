#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import librosa
import numpy as np
import argparse
import pickle

import random
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

from hparam import hparam as hp


embedder_net = SpeechEmbedder()
embedder_net.load_state_dict(torch.load('../speaker_recognition/model.model')) # file path relative the the called of the class, which is receiver_app/app.py
embedder_net.eval()


utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
utter_num = hp.test.M
utter_start = 0
shuffle = True




def speaker_enroll(wav_file_path):
    utterances_spec = []

    #utter_path = wav_file_path #os.path.join(hp.integration.enroll_upload_folder, 'audio', wav_file_path)         # path of each utterance
    utter, sr = librosa.core.load(wav_file_path, hp.data.sr)        # load utterance audio
    # utter, sr = librosa.core.load(wav_file_path, sr=None)
    # utter, sr = librosa.core.load(wav_file_path)
    # print(sr)
    #intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
    intervals = librosa.effects.split(utter)
    # print("---------------------------------")
    # print(intervals)
    # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
    # for vctk dataset use top_db=100
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                    win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            # mel_basis = librosa.filters.mel(22050, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
            utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
            utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance
    
    
    utterances_spec = np.array(utterances_spec)
    #speaker_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # using date and time as a unique identifier
    speaker_id = os.path.splitext(os.path.basename(wav_file_path))[0]
    speaker_id = speaker_id.split('_')
    speaker_id = "_".join(speaker_id[0:4])
    np.save(os.path.join(hp.integration.enroll_preprocessed_audio, speaker_id + "_audio.npy"), utterances_spec)
    # returns the speaker_id as a string
    return speaker_id  



def speaker_verify(npy_file, wav_file_path):

    utterances_spec = []
    #utter_path = wav_file_path #os.path.join(hp.integration.verify_upload_folder, wav_file_path)         # path of each utterance
    utter, sr = librosa.core.load(wav_file_path, hp.data.sr)        # load utterance audio
    # utter, sr = librosa.core.load(wav_file_path, sr=None)
    # utter, sr = librosa.core.load(wav_file_path)
    #intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
    intervals = librosa.effects.split(utter)
    # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
    # for vctk dataset use top_db=100
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                    win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            # mel_basis = librosa.filters.mel(22050, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
            utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
            utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance
    
    if len(utterances_spec)==0:   # no qualified interval found in this audio
        return -1

    wav_file_npy = np.array(utterances_spec)
    #print("\n############ "+npy_file)
    npy_file = np.load(npy_file)
    
    if shuffle:
        utter_index = np.random.randint(0, wav_file_npy.shape[0], utter_num)   # select M utterances per speaker
        wav_file_npy = wav_file_npy[utter_index]       
        
        utter_index = np.random.randint(0, npy_file.shape[0], utter_num)   # select M utterances per speaker
        npy_file = npy_file[utter_index]       
    else:
        wav_file_npy = wav_file_npy[utter_start: utter_start+ utter_num] # utterances of a speaker [batch(M), n_mels, frames]
        npy_file = npy_file[utter_start: utter_start+ utter_num] # utterances of a speaker [batch(M), n_mels, frames]

    wav_file_npy = wav_file_npy[:,:,:160]               # TODO implement variable length batch size
    wav_file_npy = torch.tensor(np.transpose(wav_file_npy, axes=(0,2,1)))     # transpose [batch, frames, n_mels]

    npy_file = npy_file[:,:,:160]               # TODO implement variable length batch size
    npy_file = torch.tensor(np.transpose(npy_file, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
            

    npy_file = torch.reshape(npy_file, (hp.test.N*hp.test.M//2, npy_file.size(1), npy_file.size(2)))
    wav_file_npy = torch.reshape(wav_file_npy, (hp.test.N*hp.test.M//2, wav_file_npy.size(1), wav_file_npy.size(2)))
    
    perm = random.sample(range(0,wav_file_npy.size(0)), wav_file_npy.size(0))
    unperm = list(perm)
    for i,j in enumerate(perm):
        unperm[j] = i
        
    wav_file_npy = wav_file_npy[perm]
    enrollment_embeddings = embedder_net(npy_file)
    verification_embeddings = embedder_net(wav_file_npy)
    verification_embeddings = verification_embeddings[unperm]
    
    enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
    verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))


    enrollment_centroids = get_centroids(enrollment_embeddings) 
    
    sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)


    # calculating EER
    diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
    
    for thres in [0.01*i+0.5 for i in range(50)]:
        sim_matrix_thresh = sim_matrix>thres
        
        FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
        /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)

        FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
        /(float(hp.test.M/2))/hp.test.N)
        
        # Save threshold when FAR = FRR (=EER)
        if diff> abs(FAR-FRR):
            diff = abs(FAR-FRR)
            EER = (FAR+FRR)/2
            EER_thresh = thres
            EER_FAR = FAR
            EER_FRR = FRR
    # print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))

    sim_matrix_pos = torch.abs(sim_matrix)
    avg = sim_matrix_pos[0,0,1] + sim_matrix_pos[0,1,1] + sim_matrix_pos[0,2,1] + sim_matrix_pos[1,0,0] + sim_matrix_pos[1,1,0] + sim_matrix_pos[1,2,0]
    avg /= 6

    # print(sim_matrix)
    # print(sim_matrix_pos)

    avg = round(avg.item(), 2)
    return avg

# Convert the string argument passsed value to boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Verify or enroll")
    parser.add_argument('--test_wav_file')
    parser.add_argument('--best_identified_speakers')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # speaker_to_test_path = "20200519_191124"   # female
    # speaker_to_test_path = "20200519_191125"   # male

    args = parse_args()
    if(args.verify):
        speaker_to_test_path = args.test_wav_file
        
        #print("\n          Verify the file : " +    os.path.splitext(os.path.basename(speaker_to_test_path))[0])
        #npy_preprocessed = hp.integration.enroll_preprocessed_audio
        
        accuracy_list = []
        
        for npy_file in os.listdir(hp.integration.enroll_preprocessed_audio):
            #print("\n")
            #print("File : " + npy_file)
            accuracy = speaker_verify(hp.integration.enroll_preprocessed_audio + npy_file, speaker_to_test_path)
            #speaker_name = ' '.join(os.path.splitext(npy_file)[0].split('_')[2:])
            id = '_'.join(npy_file.split('_')[:4])
            accuracy_list.append((id, accuracy))
        
        #args.best_identified_speakers = 'speaker_result.txt'
        accuracy_list.sort(key=lambda tup: tup[1], reverse=True)  
        #accuracy_list = accuracy_list[ :hp.integration.restriction_cutoff]

        with open(args.best_identified_speakers + 'speaker_result.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(accuracy_list, filehandle)
        '''with open(args.best_identified_speakers + 'speaker_result.txt', 'w') as fp:
            fp.write([speaker_name, accuracy_max])'''
        
        
    else :
        #enroll_audio_path = os.path.join(hp.integration.enroll_upload_folder, 'audio/')
        speaker_enroll(args.test_wav_file)
        

