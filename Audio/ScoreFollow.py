# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:34:42 2017

@author: Niki
"""
import numpy as np
import librosa
import pretty_midi
import sounddevice as sd
import time
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.spatial.distance import cdist
import tables as tb
import os
import pickle

class ScoreFollow(object):
    
    def __init__(self):
        self.samplerate = 11025
        self.hop_size = 1024
        self.windowsize = [-5,5]  # in seconds
        self.livewindow = 5 # in seconds
        self.buffersize = 10 # in seconds   
        self.framesinbuffer = self.samplerate*self.buffersize
        self.featuresinbuffer = int(self.samplerate*self.buffersize/self.hop_size)
        
        self.micwav = np.zeros((0,1))
        self.micfeatures = np.zeros((0,1))
       
        self.reading_prob = [0.]
        self.reading_speed = [0.]
        self.reading_pos = [0.]
        self.reading_delta = [0.]
        self.volume = [0.]
        self.speed = [5.]
        self.position = [0.]
        
        self.featuremode = 'cqt' #'cqt' #'stft'
        self.C = []
        
        self.channels = 1
        self.instrument = 0 # Grand Piano
        self.soundfont = br"C:\Users\Niki\Datasets\Soundfonts\Weeds\WeedsGM3.sf2"
        
        self.songid = 0
        h5file = tb.open_file(os.getcwd()+ r'\MusicData\songbook.h5', "a")
        self.songbook = h5file.root.songbook.songbook
        
        self.mic = sd.InputStream(channels=self.channels, 
                                  callback=self.callback,
                                  samplerate=self.samplerate)
        
    def callback(self, indata, frames, time, status):  
        self.micwav = np.append(self.micwav, indata)
        if self.micwav.shape[0]>self.framesinbuffer:
            self.micwav = self.micwav[-self.framesinbuffer:]
            
    def load_midi(self, midifile=None):
        if midifile is None:
            midifile = os.getcwd()+ '/MusicData/' + self.songbook.row.table[self.songid]['midi_file'].decode('utf-8')
        midi_obj = pretty_midi.PrettyMIDI(midifile)
        for instrument in midi_obj.instruments:        
            instrument.program = self.instrument
        return midi_obj
    
    
    def load_wav(self, wfile=None):
        if wfile is None:
            wfile = os.getcwd()+ '/MusicData/' + self.songbook.row.table[self.songid]['youtube_file'].decode('utf-8')
        wf = read(wfile)
        return wf[1]
    
    def midi2wav(self, midi_obj):
        wav_obj = midi_obj.fluidsynth(fs=self.samplerate, sf2_path = self.soundfont)
        return wav_obj
    
    def play(self, wavObj):
        sd.play(wavObj, self.samplerate)

    def stop(self):
        sd.stop()

    def wav2features(self, wavObj):
        mode = self.featuremode
        if mode =='cqt':
            features = librosa.feature.chroma_cqt(y=wavObj,sr=self.samplerate, hop_length=self.hop_size, norm = 2, threshold=0, n_chroma=8*12, n_octaves=8, fmin = 16.5)
        elif mode == 'cens':
            features = librosa.feature.chroma_cens(y=wavObj,sr=self.samplerate, hop_length=self.hop_size, norm = 2, n_chroma=7*12, n_octaves=7)
        elif mode == 'stft':
            features = librosa.feature.chroma_stft(y=wavObj,sr=self.samplerate, hop_length=self.hop_size, norm = 2, n_chroma=8*12)
        elif mode == 'mel':
            features = librosa.feature.melspectrogram(y=wavObj,sr=self.samplerate, hop_length=self.hop_size)
        elif mode == 'mfcc':
            features = librosa.feature.mfcc(y=wavObj,sr=self.samplerate, hop_length=self.hop_size, n_mfcc=8*12)
        elif mode == 'poly':
            features = librosa.feature.poly_features(y=wavObj,sr=self.samplerate, hop_length=self.hop_size)
        return features
    
    def feature2distance_old(self, livefeature, features):
        if livefeature.ndim==1:
            livefeature.resize(len(livefeature),1)
        dist = np.zeros([features.shape[1], livefeature.shape[1]])
        for i in range(livefeature.shape[1]):
            dist[:, [i]] = np.mean((features - livefeature[:,[i]])**2, axis=0, keepdims=True).T / np.mean(features**2, axis=0, keepdims=True).T
        return dist
    
    def features2distance(self, livefeatures, features):
        dist = cdist(livefeatures.T, features.T)
        return dist.T
      
    def rebinx(self, a, size):
        indices = np.linspace(0, a.shape[1]-1, size)
        return a[:,tuple(indices.astype(int))]    
    
    def prob_model2(self, livefeatures, features, zeroindex, biasindex):
        ntrue = features.shape[1]
        nf, nlive = livefeatures.shape
        six = int(0.5*nlive)
        eix = int(1.5*nlive)
        dmin = 1
        nmin = -1
        jmin = -1
        biasindex = min(biasindex, ntrue)
        biasmult = np.append(np.linspace(1,0.9,biasindex), np.linspace(0.9,1,ntrue-biasindex))

        for n in range(six, eix, 2):
            rebinlive = self.rebinx(livefeatures, n)
            for j in range(n, ntrue):
                d = np.linalg.norm(rebinlive-features[:,j-n:j])/np.linalg.norm(rebinlive+features[:,j-n:j])
                d *= biasmult[j]
                if d<dmin:
                    dmin = d
                    jmin = j
                    nmin = n
        
        pos =  jmin
        speed = nmin/nlive
        prob = dmin
        return pos, speed, prob
    
    def prob_model(self, dist, featuretime, speed, zeroindex, biasindex):
        [ntrue, nlive] = dist.shape
        biasindex = min(biasindex, ntrue)
        #q = np.maximum(0, 1-dist.min(axis=0))
        r = np.linspace(max(1,speed-20), min(60,speed+20), 20)
        offset = featuretime*r
        n = np.linspace(nlive-1, 1, nlive)
        N = offset.reshape([-1,1])*n.reshape([1,-1])
        N = N.astype(int)
        biasmult = np.append(np.linspace(1,1,biasindex), np.linspace(1,1,ntrue-biasindex))
        for i in range(0, N.shape[0]):
            d = np.ones(dist.shape)
            for j in range(0, N.shape[1]):
                if N[i,j]>0:
                    d[N[i,j]:,[j]] = dist[0:-N[i,j],[j]]
            #if any(q):
                #p = biasmult*np.average(d, axis=1, weights = q)
                #p = biasmult*np.average(d, axis=1)
            #else:
                #p = biasmult         
            p = biasmult*np.average(d, axis=1)  
            if 'probs' in locals():
                probs = np.append(probs,p)
            else:
                probs = p
        
        order = probs.argsort()
        #val = probs[order[0:3]]
        #spidx = int(np.sum(val*np.floor(order[0:3]/ntrue))/np.sum(val))
        #deltaidx = np.sum(val*np.mod(order[0:3], ntrue))/np.sum(val)
        val = probs[order[0]]
        spidx = int(order[0]/ntrue)
        deltaidx = np.mod(order[0], ntrue)

        
        posdelta = deltaidx-zeroindex
        speed = r[spidx]
        return posdelta, speed, np.mean(val), probs

        
    def online_follow(self, features, midi_obj):    
        self.mic.start()
        # One feature frame corresponds to 1024/44100 seconds
        featuretime = self.hop_size/self.samplerate
        defaultspeed = 1/featuretime
        history = -int(self.samplerate*self.windowsize[0]/self.hop_size)
        future = int(self.samplerate*self.windowsize[1]/self.hop_size)
        true_num = features.shape[1]
        dt=0
        print("Starting score following. Exit with 'Ctrl+C'.")
        
        piano_roll = midi_obj.get_piano_roll(15)
        #%matplotlib qt
        #%matplotlib inline

        plt.ion()
        f = plt.figure()
        h1 = plt.imshow(piano_roll[::-1, :])
        h1.axes.set_xlim(left = 0, right = 200)
        h2 = plt.plot([0,0],[0,128])
        try:
            while True:
                plt.pause(0.01)
                if len(self.micwav)<self.livewindow*self.samplerate:
                    livewav = self.micwav.squeeze()
                else:
                    livewav = self.micwav[-self.livewindow*self.samplerate:].squeeze()
                self.volume.append(0.6*self.volume[-1]+0.4*np.sqrt(np.mean(livewav**2)))
                livefeatures = self.wav2features(livewav)
                
                if self.volume[-1]>0.02:
                    
                    j = int(self.reading_pos[-1])
                    if j>=history and j<true_num-future:
                        scoreindex = slice(j-history,j+future)
                        zeroindex = history
                    elif j>=true_num-future:
                        scoreindex = slice(j-history,true_num)
                        zeroindex = history
                    elif j<history:
                        scoreindex = slice(0,10+livefeatures.shape[1])
                        zeroindex = j

                    bias = 5
                    
                    #dist = self.feature2distance(livefeatures, features[:, scoreindex])
                    #posdelta, speed, d, probs = self.prob_model(dist, featuretime, defaultspeed, zeroindex, zeroindex+bias)
                    
                    pos, speed, prob = self.prob_model2(livefeatures, features[:, scoreindex], zeroindex, zeroindex+bias)
                    if prob<0.9:
                        posdelta = pos-zeroindex
                        
                        posdelta = max(-4, min(4*bias, posdelta))
                        if dt==0:
                            dt=0.2
                        else:
                            dt = time.clock() - t0
                        t0 = time.clock()    
                        
                        self.reading_pos.append(self.reading_pos[-1]+posdelta)
                        self.reading_speed.append(speed)
                        self.reading_delta.append(posdelta)
                        self.reading_prob.append(prob)
                        self.speed.append(np.median(self.reading_delta[-5:]))
                        self.position.append(0.6*self.position[-1]+ 0.6*self.speed[-1] + 0.4*self.reading_pos[-1])
            
                        # Convert feature position to midi position
                        midi_pos = (self.position[-1]+bias)/features.shape[1]*piano_roll.shape[1]
                        
                        # Update Piano Roll
                        h1.axes.set_xlim(left = max(0, midi_pos-100), right = min(piano_roll.shape[1], max(200,midi_pos+100)))
                        h2[0].set_xdata([midi_pos,midi_pos])
                        f.canvas.draw()
                        
                        # Log
                        print('Current Frame:', self.reading_pos[-1], 'Distance: ', self.reading_prob[-1], 'Speed:', self.speed[-1], 'dt:', dt)
                    
                
        except KeyboardInterrupt:
            self.mic.stop()
            sd.stop()

    def online_follow_dtw(self, features, midi_obj):    
        self.mic.start()
        time.sleep(0.1)
        t0 = time.clock()   
        # One feature frame corresponds to 1024/44100 seconds
        featuretime = self.hop_size/self.samplerate
        true_num = features.shape[1]
        C = np.zeros((true_num, 10*true_num))
        liveix = 0
        
        piano_roll = midi_obj.get_piano_roll(15)
        #%matplotlib qt
        #%matplotlib inline

        plt.ion()
        f = plt.figure()
        h1 = plt.imshow(piano_roll[::-1, :])
        h1.axes.set_xlim(left = 0, right = 200)
        h2 = plt.plot([0,0],[0,128])
        
        memcoef = 0.999
        maxfwd = 5
        maxbkw = -5
                            
        print("Starting score following. Exit with 'Ctrl+C'.")
        try:
            while True:
                plt.pause(0.1)
                dt = time.clock() - t0
                t0 = time.clock()   
                
                livewav = self.micwav.squeeze()
                self.micwav = self.micwav[0:0]
              
                self.volume.append(0.6*self.volume[-1]+0.4*np.sqrt(np.mean(livewav**2)))
                #print(self.volume[-1])
                if self.volume[-1]>0.02:
                    live_features = self.wav2features(livewav)
                    
                    D = cdist(features.T, live_features.T, metric='euclidean')

                    for j in range(D.shape[1]):
                        for i in range(D.shape[0]):
                            thisd = D[i,j]*(1-memcoef)*(1+j/true_num)
                            if i==0 and liveix+j==0:
                                C[i,liveix+j] = thisd
                            elif i==0 and liveix+j>0:
                                C[i,liveix+j] = thisd+memcoef*C[i,liveix+j-1]
                            elif i>0 and liveix+j==0:
                                C[i,liveix+j] = thisd+memcoef*C[i-1,liveix+j]
                            else:
                                C[i,liveix+j] = thisd+memcoef*min([C[i-1,liveix+j], C[i,liveix+j-1], C[i-1,liveix+j-1]])
                    
                    self.C = C
                    liveix += D.shape[1]
                    posdelta = C[:,liveix-1].argmin()-self.reading_pos[-1]
                    posdelta = max(maxbkw, min(maxfwd, posdelta))
                    self.reading_delta.append(posdelta)
                    self.reading_pos.append(self.reading_pos[-1]+posdelta)
                    self.reading_prob.append(C[:,-1].min())
                    self.position.append(0.6*self.position[-1]+ 0.6*self.reading_delta[-1] + 0.4*self.reading_pos[-1])
        
                    # Convert feature position to midi position
                    midi_pos = (self.position[-1]+posdelta)/features.shape[1]*piano_roll.shape[1]
                    
                    # Update Piano Roll
                    h1.axes.set_xlim(left = max(0, midi_pos-100), right = min(piano_roll.shape[1], max(200,midi_pos+100)))
                    h2[0].set_xdata([midi_pos,midi_pos])
                    f.canvas.draw()

                     
                    # Log
                    print('Current Frame:', self.reading_pos[-1], 'Distance: ', self.reading_prob[-1], 'dt:', dt)

                
        except KeyboardInterrupt:
            self.mic.stop()
            sd.stop()
