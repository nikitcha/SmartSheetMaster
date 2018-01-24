import os
cdir = r'C:\Users\Niki\Source\SmartSheetMusic'
os.chdir(cdir)
  
import ScoreFollow
from importlib import reload

reload(ScoreFollow)
sf = ScoreFollow.ScoreFollow()

#sf.songid = 4
#midi_obj  = sf.load_midi()
midi_obj  = sf.load_midi(cdir+r'\MusicData\Chopin - Nocturne op9 No2 2.mid') #Chopin - Nocturne op9 No2 2.mid #Beethoven - Fur Elise.mid
wav_obj   = sf.midi2wav(midi_obj)
features = sf.wav2features(wav_obj)

sf.online_follow(features, midi_obj)

#if False:   

    #%%
    ywav = read(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin - Nocturne op9 No2.wav')
    ywav = ywav[1]
    livefeatures = sf.wav2features(ywav.astype(dtype='float32'))    

    
    thisfeatures = features[:, 0:60]
    thislive = livefeatures[:, 20:60]
    biasindex = 0
    
    ntrue = thisfeatures.shape[1]
    nf, nlive = thislive.shape
    six = int(0.5*nlive)
    eix = int(1.5*nlive)
    dmin = 1
    nmin = -1
    jmin = -1
    biasindex = min(biasindex, ntrue)
    biasmult = np.append(np.linspace(1,0.9,biasindex), np.linspace(0.9,1,ntrue-biasindex))

    for n in range(six, eix, 2):
        rebinlive = self.rebinx(thislive, n)
        for j in range(n, ntrue):
            d = np.linalg.norm(rebinlive-thisfeatures[:,j-n:j])/np.linalg.norm(rebinlive+thisfeatures[:,j-n:j])
            d *= biasmult[j]
            if d<dmin:
                dmin = d
                jmin = j
                nmin = n
    
    pos =  jmin
    speed = nmin/nlive
    prob = dmin
        
    #%%
    import ScoreFollow
    from importlib import reload
    from scipy.io.wavfile import read
    
    reload(ScoreFollow)
    sf = ScoreFollow.ScoreFollow()
    
    midi_obj  = sf.load_midi()
    wav_obj   = sf.midi2wav(midi_obj)
    features = sf.wav2features(wav_obj)
    ywav = sf.youtube2wav()
    
    #ywav = read(r'C:\Users\Niki\Source\SmartSheetMusic\Chopin - Nocturne op9 No2.wav')
    #ywav = ywav[1]
    yfeatures = sf.wav2features(ywav.astype(dtype='float32'), 'stft')
    
    
    #ydistfull = sf.feature2distance(features, yfeatures)
    #pickle.dump( ydistfull, open( r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\Chopin_Nocturne_Op_9_No_2_E_Flat_Major_yldistance_matrix_11025_1024_chroma_stft.p', "wb" ))    
    
    #sf.start_following(features, midi_obj)

#%%
    import ScoreFollow
    from importlib import reload
    reload(ScoreFollow)
    sf = ScoreFollow.ScoreFollow()
    
    cqtfeatures = sf.wav2features(wav_obj, 'cqt')
    censfeatures = sf.wav2features(wav_obj, 'cens')
    melfeatures = sf.wav2features(wav_obj, 'mel')
    stftfeatures = sf.wav2features(wav_obj, 'stft')
    mfccfeatures = sf.wav2features(wav_obj, 'mfcc')
    polyfeatures = sf.wav2features(wav_obj, 'poly')

#%%
    import pytube
    yt = pytube.YouTube(sf.youtube)
    stream = yt.streams.filter(only_audio=True).order_by('bitrate').asc().first()
    stream.download()
    
#%%
    import os
    os.chdir(r'C:\Users\Niki\Source\SmartSheetMusic')
    
    import ScoreFollow
    from importlib import reload
    reload(ScoreFollow)
    sf = ScoreFollow.ScoreFollow()
    
    for i in range(16):
        sf.offline_follow(i)
        
#%%
    #gswin64 -dNOPAUSE -dBATCH -r300 -sDEVICE=pnggray -sOutputFile="C:\Users\Niki\Source\SmartSheetMusic\MusicData\cover.png" C:\Users\Niki\Source\SmartSheetMusic\MusicData\cover.pdf
    
#%% Check Log and compare to Offline DTW
    import pickle
    import matplotlib.pyplot as plt
    import librosa.core as lb
    song_id = 4

    songbook = sf.songbook
    sf.songid = song_id

    name = sf.songbook.row.table[song_id]['composer'].decode('utf-8') + sf.songbook.row.table[song_id]['name'].decode('utf-8')
    res = pickle.load(open(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\log_'+name+'.p','rb')  )  
    
    
    midi_obj  = sf.load_midi()
    wav_obj   = sf.midi2wav(midi_obj)
    features = sf.wav2features(wav_obj)
    ywav = sf.load_wav()
    live_features = sf.wav2features(ywav.astype(dtype='float32'))

    d,wp = lb.dtw(features, live_features, backtrack = True)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(d)
    plt.plot(wp[:,1],wp[:,0])
    plt.plot(res['x'], res['reading_pos'][1:])
    
    plt.figure(figsize=(20, 20))    
    plt.imshow(res['thisdistance'])
    plt.plot(wp[:,1],wp[:,0])
    plt.plot(res['x'], res['reading_pos'][1:])    

    #plt.figure(figsize=(20, 5))
    #plt.plot(res['reading_prob'])

#%% Online DTW
    import numpy as np
    from scipy.spatial.distance import cdist
    song_id = 4

    songbook = sf.songbook
    sf.songid = song_id

    name = sf.songbook.row.table[song_id]['composer'].decode('utf-8') + sf.songbook.row.table[song_id]['name'].decode('utf-8')
    res = pickle.load(open(r'C:\Users\Niki\Source\SmartSheetMusic\MusicData\log_'+name+'.p','rb')  )  
    
    midi_obj  = sf.load_midi()
    wav_obj   = sf.midi2wav(midi_obj)
    features = sf.wav2features(wav_obj)
    ywav = sf.load_wav()
    live_features = sf.wav2features(ywav.astype(dtype='float32'))
    
    d,wp = lb.dtw(features, live_features, backtrack = True)
        
    
    '''
    for i in range(50,features.shape[1],50):
        d,wp = lb.dtw(features[:,0:i], live_features[:,0:i], backtrack = True)
        plt.plot(wp[:,1],wp[:,0])
    '''
#%% FULL
    D = cdist(features.T, live_features.T, metric='euclidean')
    C = np.zeros(D.shape)
    pos = [0]
    memcoef = 0.999
    maxfwd = 200
    maxbkw = -100
    v = np.arange(D.shape[0])
    for j in range(D.shape[1]):
        for i in range(D.shape[0]):
            thisd = D[i,j]*(1-memcoef) 
            if i==0 and j==0:
                C[i,j] = thisd
            elif i==0 and j>0:
                C[i,j] = thisd+memcoef*C[i,j-1]
            elif i>0 and j==0:
                C[i,j] = thisd+memcoef*C[i-1,j]
            else:
                C[i,j] = thisd+memcoef*min([C[i-1,j], C[i,j-1], C[i-1,j-1]])
        dpos = (C[:,j]+0.05*abs(v-pos[-1]-1)/D.shape[0]).argmin()-pos[-1]
        dpos = max(maxbkw, min(maxfwd, dpos))
        pos.append(pos[-1]+dpos)

    plt.imshow(C)
    plt.plot(pos, 'r')
    
    