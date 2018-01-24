if False:
    #%% Save Files
    
    import ScoreFollow
    import pickle
    
    sf = ScoreFollow.ScoreFollow()
    
    sf.midifile = r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major.mid'
    mid_obj   = sf.load_midi()
    wav_obj  = sf.midi2wav(mid_obj)
    features = sf.wav2features(wav_obj)
    ywav = sf.youtube2wav().astype(dtype='float32')
    yfeatures = sf.wav2features(ywav)    
    
    pickle.dump( features, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_features_11025_1024.p', "wb" ) )
    pickle.dump( ywav, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_wav_11025_1024.p', "wb" ))
    
    pickle.dump( yfeatures, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_yfeatures_11025_1024.p', "wb" ) )
    pickle.dump( wav_obj, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_ywav_11025_1024.p', "wb" ))
    
    sf.buffersize = 180
    sf.framesinbuffer = sf.buffersize*sf.samplerate
    sf.mic.start()
    sf.mic.stop()
    
    live_wav = sf.frames
    live_features = sf.wav2features(live_wav.squeeze())
    pickle.dump( live_features, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa_live_features_11025.p', "wb" ) )
    pickle.dump( live_wav, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa_live_wav_11025.p', "wb" ))
  
    distfull = sf.feature2distance(live_features, features)
    pickle.dump( distfull, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_distance_matrix_11025_1024.p', "wb" ))    

    ydistfull = sf.feature2distance(yfeatures, features)
    pickle.dump( ydistfull, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_ydistance_matrix_11025_1024.p', "wb" ))    
    
    yldistfull = sf.feature2distance(live_features, yfeatures)
    pickle.dump( yldistfull, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_yldistance_matrix_11025_1024.p', "wb" ))    

    
    #%% Load Files
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import ScoreFollow
    import time
    import copy
    from scipy.io.wavfile import read

    #live_features = pickle.load(open(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_live_features_11025_1024.p','rb')  )  
    #features = pickle.load(open(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_features_11025_1024.p','rb')  )  
    #ydistfull = pickle.load(open(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_ydistance_matrix_11025_1024.p','rb')  )  
        
    from importlib import reload
    reload(ScoreFollow)

    sf = ScoreFollow.ScoreFollow()

    sf.midifile = r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa.mid'
    mid_obj   = sf.load_midi()
    wav_obj  = sf.midi2wav(mid_obj)
    features = sf.wav2features(wav_obj)
    ywav = read(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa.wav')
    ywav = ywav[1]
    yfeatures = sf.wav2features(ywav)    
    ydistfull = sf.feature2distance_old(yfeatures, features)
    live_features = yfeatures
    
    #%% Full Offline
        
    #distfull = sf.feature2distance(live_features, features)
    #pickle.dump( distfull, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa_distance_matrix.p', "wb" ))
    plt.imshow(distfull, cmap="bwr")
      
    
    #%% Core probability model
    t0 = time.clock()
    [ntrue, nlive] = dist.shape
    q = np.maximum(0, 1-dist.min(axis=0))
    r = np.linspace(max(10,speed-30), min(70,speed+30), 10)
    offset = featuretime*r
    n = np.linspace(nlive-1, 1, nlive)
    N = offset.reshape([-1,1])*n.reshape([1,-1])
    N = N.astype(int)
    biasmult = np.append(np.linspace(1,0.8,biasindex), np.linspace(0.8,1,ntrue-biasindex))
    for i in range(0, N.shape[0]):
        d = np.ones(dist.shape)
        for j in range(0, N.shape[1]):
            if N[i,j]>0:
                d[N[i,j]:,[j]] = dist[0:-N[i,j],[j]]
        if any(q):
            p = biasmult*np.average(d, axis=1, weights = q)
        else:
            p = biasmult
    
        if 'probs' in locals():
            probs = np.append(probs,p)
        else:
            probs = p
            
    print(time.clock()-t0)

    #%% Core probability model - old and slow
    t0 = time.clock()
    [ntrue, nlive] = dist.shape
    r = np.linspace(max(10,speed-30), min(70,speed+30), 10)
    for j in r:
        offset = featuretime*j
        d1 = copy.deepcopy(dist)
        for i in range(nlive-1):
            n = int((nlive-i)*offset)
            if n>0:
                d1[n:,[i]] = d1[0:-n,[i]]
                d1[0:n,[i]] = 1
            #d1 = d1[-truesize:,:]
            q1 = np.maximum(0, 1-d1.min(axis=0))
            if any(q1):
                p = np.average(d1, axis=1, weights = q1)
            else:
                p = 1+0*np.average(d1, axis=1)
        if 'probs' in locals():
            probs = np.concatenate([probs,p])
        else:
            probs = p
    print(time.clock()-t0)
#%%
    #sf = ScoreFollow.ScoreFollow()
    #sf.midifile = r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\fur_elisa.mid'
    #mid_obj   = sf.load_midi()
    piano_roll = midi_obj.get_piano_roll(20)
    
    plt.ion()
    f = plt.figure()
    h1 = plt.imshow(piano_roll[::-1, :])
    h1.axes.set_xlim(left = 0, right = 200)
    h2 = plt.plot([100,100],[0,128])
    
    for i in range(0,100,1):
        h1.axes.set_xlim(left = i, right = i+200)
        h2[0].set_xdata([150,150])
        f.canvas.draw()
        plt.pause(0.01)

#%% Distnace measures

D = cdist(yfeatures.T,features.T, 'cosine') # euclidean minkowski cityblock seuclidean sqeuclidean cosine correlation chebyshev canberra braycurtis
plt.imshow(D)
plt.figure()
plt.plot(D[:, 1000])

ydistfull = np.nan_to_num(D.T,1)

#%% Global align
thislive = yfeatures[:, 1000:1050]
thistrue = features[:, 900:1100]

nf, ntrue = thistrue.shape
nlive = thislive.shape[1]

six = int(0.5*nlive)
eix = int(1.5*nlive)
dmin = 1
nmin = -1
jmin = -1
d = []
for n in range(six, eix):
    nlive = rebinx(thislive, n)
    for j in range(n, ntrue):
        d.append(np.linalg.norm(nlive-thistrue[:,j-n:j])/np.linalg.norm(nlive+thistrue[:,j-n:j]))
        if d[-1]<dmin:
            dmin = d[-1]
            jmin = j
            nmin = n   

#%% Offline simulation of online following
from importlib import reload
reload(ScoreFollow)
self = ScoreFollow.ScoreFollow()
self.songid = 4

midi_obj  = self.load_midi()
wav_obj   = self.midi2wav(midi_obj)
features = self.wav2features(wav_obj)
ywav = self.load_wav()
live_features = self.wav2features(ywav.astype(dtype='float32'))

featuretime = self.hop_size/self.samplerate
defaultspeed = 1/featuretime
history = -int(self.samplerate*self.windowsize[0]/self.hop_size)
future = int(self.samplerate*self.windowsize[1]/self.hop_size)

#thisdistance = ydistfull
thisdistance = self.feature2distance_old(live_features, features)

true_num = features.shape[1]
live_num = live_features.shape[1]
eix = live_num
step = 5

live_chunk = int(self.samplerate*self.livewindow/self.hop_size)
defaultspeed = 1/featuretime
dt=0
try:
    for i in range(20, eix, step): #live_num
        if dt==0:
            dt=0.22
        else:
            dt = time.clock() - t0
        t0 = time.clock()            
        print(i, self.reading_pos[-1], self.reading_speed[-1], self.reading_prob[-1], dt)
        j = int(self.reading_pos[-1])
        if i>=live_chunk:
            liveindex = slice(i-live_chunk,i)
        else:
            liveindex = slice(0,i)
        if j>=history and j<true_num-future:
            scoreindex = slice(j-history,j+future)
            zeroindex = history
        elif j>=true_num-future:
            scoreindex = slice(j-history,true_num)
            zeroindex = history
        elif j<history:
            scoreindex = slice(0,j+future)
            zeroindex = j
       
        if len(self.reading_delta)<20:
            bias=step
        else:
            bias = max(3,int(self.speed[-1]))

        #dist = thisdistance[scoreindex, liveindex]
        #posdelta, speed, d, probs = self.prob_model(dist, featuretime, defaultspeed, zeroindex, zeroindex+bias)
    
        pos, speed, probs, d = self.prob_model2(live_features[:,liveindex], features[:, scoreindex],zeroindex, zeroindex+bias)
        posdelta = pos-zeroindex
                    
        posdelta = max(-4, min(2*step, posdelta))
        self.reading_pos.append(self.reading_pos[-1]+posdelta)
        self.reading_speed.append(speed)
        self.reading_delta.append(posdelta)
        self.reading_prob.append(d)
        self.speed.append(np.median(self.reading_delta[-10:]))
        self.position.append(0.7*self.position[-1]+ 0.7*self.speed[-1] + 0.3*self.reading_pos[-1])

            
except KeyboardInterrupt:
    plt.figure()
    plt.plot(self.reading_pos)
    plt.figure()
    plt.plot(self.reading_speed)

plt.figure(figsize=(20, 20))
plt.imshow(thisdistance, cmap="bwr")
plt.plot(range(20, eix, step), self.reading_pos[1:])
plt.plot(range(20, eix, step), self.position[1:])

plt.figure(figsize=(10, 5))
plt.plot(self.reading_prob)