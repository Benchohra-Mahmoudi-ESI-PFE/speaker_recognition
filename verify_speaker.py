#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
from hparam import hparam as hp

import random
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim


embedder_net = SpeechEmbedder()
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.eval()

test_wav_path = './test_wav'
enrollment_wav_path = './enrollment_wav'
enrollment_preproccessed_path = './enrollment_preproccessed'

utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
utter_num = hp.test.M
utter_start = 0
shuffle = True




def speaker_enroll(wav_file):
    utterances_spec = []

    utter_path = os.path.join(enrollment_wav_path, wav_file)         # path of each utterance
    utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
    intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
    # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
    # for vctk dataset use top_db=100
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                    win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
            utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
            utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

    utterances_spec = np.array(utterances_spec)
    speaker_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # using date and time as a unique identifier
    np.save(os.path.join(enrollment_preproccessed_path, "speaker_" + speaker_id + ".npy"), utterances_spec)

    # returns the speaker_id as a string
    return speaker_id  



def speaker_verify(wav_file, speaker_id):

    utterances_spec = []
    utter_path = os.path.join(test_wav_path, wav_file)         # path of each utterance
    utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
    intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
    # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
    # for vctk dataset use top_db=100
    for interval in intervals:
        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                    win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
            utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
            utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance
    
    if len(utterances_spec)==0:   # no qualified interval found in this audio
        return -1

    verification_batch = np.array(utterances_spec)
    enrollment_batch = np.load(os.path.join(enrollment_preproccessed_path, "speaker_" + speaker_id + ".npy"))
    
    if shuffle:
        utter_index = np.random.randint(0, verification_batch.shape[0], utter_num)   # select M utterances per speaker
        verification_batch = verification_batch[utter_index]       
        
        utter_index = np.random.randint(0, enrollment_batch.shape[0], utter_num)   # select M utterances per speaker
        enrollment_batch = enrollment_batch[utter_index]       
    else:
        verification_batch = verification_batch[utter_start: utter_start+ utter_num] # utterances of a speaker [batch(M), n_mels, frames]
        enrollment_batch = enrollment_batch[utter_start: utter_start+ utter_num] # utterances of a speaker [batch(M), n_mels, frames]

    verification_batch = verification_batch[:,:,:160]               # TODO implement variable length batch size
    verification_batch = torch.tensor(np.transpose(verification_batch, axes=(0,2,1)))     # transpose [batch, frames, n_mels]

    enrollment_batch = enrollment_batch[:,:,:160]               # TODO implement variable length batch size
    enrollment_batch = torch.tensor(np.transpose(enrollment_batch, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
            

    enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(1), enrollment_batch.size(2)))
    verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(1), verification_batch.size(2)))
    
    perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
    unperm = list(perm)
    for i,j in enumerate(perm):
        unperm[j] = i
        
    verification_batch = verification_batch[perm]
    enrollment_embeddings = embedder_net(enrollment_batch)
    verification_embeddings = embedder_net(verification_batch)
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


if __name__ == "__main__":


    '''for wav_file in os.listdir(enrollment_wav_path):
        print(wav_file)
        speaker_enroll(wav_file)
        time.sleep(1.2)'''

    # speaker_to_test = "20200824_175606"   # sid ali
    speaker_to_test = "20200824_175607"   # benchohra

    print("          Testing speaker : Benchohra ") # + speaker_to_test)

    for wav_file in os.listdir(test_wav_path):
        #print("\n")
        print("File : " + wav_file)
        print("- Precision : " + str(speaker_verify(wav_file, speaker_to_test)))

    #print("\n")
