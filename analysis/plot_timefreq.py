#!/usr/bin/env python3
"""
Time-frequency features vs ERP window means for 4-class decoding.
5 frequency band powers within each ERP window, per channel.
"""
import numpy as np
from scipy import signal as sg
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os, warnings
warnings.filterwarnings('ignore')

for f in ['Microsoft YaHei', 'SimHei']:
    try:
        fp = fm.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['axes.unicode_minus'] = False
        break
    except:
        pass

FS=500.0; GAIN=6.0; SCALE=4.5/(2**23-1)/GAIN*1e6
OUT_DIR=r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
DATA_PATH=r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'
DIR_NAMES=['Up','Down','Left','Right']
DIR_VALS=[2.0001,2.0002,2.0003,2.0004]
ERP_WINDOWS=[('P1',0.080,0.130),('N1',0.140,0.200),('P2',0.200,0.300),('P3',0.300,0.500)]
W_SHORT=['P1','N1','P2','P3']
FREQ_BANDS=[('Delta',1,4),('Theta',4,8),('Alpha',8,13),('Beta',13,30),('Gamma',30,45)]
BAND_COLORS=['#3498db','#2ecc71','#f1c40f','#e67e22','#e74c3c']
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys()); HW_NAMES=[CH_MAP[c] for c in HW_CHS]
REGIONS={'Frontal':['F3','Fz','F4'],'Central':['C3','Cz','C4'],
         'Parietal':['P3','Pz','P4'],'Occipital':['O1','Oz','O2']}

def map_direction(val):
    for di,k in enumerate(DIR_VALS):
        if abs(val-k)<5e-5: return di
    return -1

print('Loading...')
with open(DATA_PATH) as f: lines=f.readlines()
data,markers=[],[]
for line in lines[5:]:
    p=line.strip().split(',')
    if len(p)>33:
        try: data.append([float(p[i]) for i in range(1,17)]); markers.append(float(p[32]))
        except: pass
d=np.array(data,dtype=np.float64).T; m=np.array(markers)
sel=np.array([d[c-1] for c in HW_CHS])
bp=sg.butter(4,[1/250,45/250],btype='band',output='sos')
notch=sg.iirnotch(50/250,30)
filt=np.zeros_like(sel)
for ch in range(12):
    dm=sel[ch]-sel[ch].mean(); ts=sg.sosfiltfilt(bp,dm); filt[ch]=sg.filtfilt(*notch,ts)
filt_uv=filt*SCALE
ons=sorted([i for k in DIR_VALS for i in np.where(np.abs(m-k)<5e-5)[0]])
t_start,t_end=-0.2,0.8; n_pre,n_post=int(abs(t_start)*FS),int(t_end*FS); n_total=n_pre+n_post
t=np.linspace(t_start,t_end,n_total)
epochs_data,epochs_label=[],[]
for idx in ons:
    s,e=idx-n_pre,idx+n_post
    if s<0 or e>=filt_uv.shape[1]: continue
    ep=filt_uv[:,s:e].copy(); ep-=ep[:,:n_pre].mean(axis=1,keepdims=True)
    if np.max(np.abs(ep))>100: continue
    epochs_data.append(ep); epochs_label.append(map_direction(m[idx]))
epochs_data=np.array(epochs_data); epochs_label=np.array(epochs_label)
print(f'Trials: {len(epochs_data)}')

# ====== Feature sets ======
# Set A: 4 ERP window means per channel (baseline)
print('Extracting window means...')
win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# Set B: Band power within full post-stimulus (0-800ms, 400 samples)
print('Computing band power (0-800ms FFT)...')
n_fft=400
freqs=fftfreq(n_fft,1/FS)[:n_fft//2]
band_power_full=np.zeros((len(epochs_data),12,len(FREQ_BANDS)))
for ti in range(len(epochs_data)):
    sig=epochs_data[ti,:,n_pre:]  # 12 x 400 (0-800ms)
    for ci in range(12):
        psd=np.abs(fft(sig[ci]))**2/n_fft
        for bi,(_,fl,fh) in enumerate(FREQ_BANDS):
            msk=(freqs>=fl)&(freqs<fh)
            band_power_full[ti,ci,bi]=psd[:n_fft//2][msk].sum()
# Log transform (power is log-normal)
band_power_full=np.log1p(band_power_full)

# Set C: Band power within each ERP window
print('Computing per-window band power...')
band_power_win=np.zeros((len(epochs_data),12,4,len(FREQ_BANDS)))
for ti in range(len(epochs_data)):
    for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
        msk=(t>=ws)&(t<=we)
        seg=epochs_data[ti,:,msk]  # 12 x n_samples
        n_seg=seg.shape[1]
        freqs_seg=fftfreq(n_seg,1/FS)[:n_seg//2]
        for ci in range(12):
            psd=np.abs(fft(seg[ci]))**2/n_seg
            for bi,(_,fl,fh) in enumerate(FREQ_BANDS):
                fm=(freqs_seg>=fl)&(freqs_seg<fh)
                band_power_win[ti,ci,wi,bi]=psd[:n_seg//2][fm].sum()
band_power_win=np.log1p(band_power_win)

# ====== Decoding comparison ======
TEST_COMBOS=[
    ('Frontal',[HW_NAMES.index(c) for c in REGIONS['Frontal']],'#e74c3c'),
    ('Central',[HW_NAMES.index(c) for c in REGIONS['Central']],'#3498db'),
    ('Parietal',[HW_NAMES.index(c) for c in REGIONS['Parietal']],'#2ecc71'),
    ('Occipital',[HW_NAMES.index(c) for c in REGIONS['Occipital']],'#f39c12'),
    ('F+P',[HW_NAMES.index(c) for c in REGIONS['Frontal']+REGIONS['Parietal']],'#8e44ad'),
    ('All 12ch',list(range(12)),'#333'),
]

def decode_4(X,y):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]
        scl=StandardScaler().fit(Xtr)
        clf=LDA().fit(scl.transform(Xtr),y[tr])
        accs.append(accuracy_score(y[te],clf.predict(scl.transform(Xte))))
    return np.mean(accs),np.std(accs)

print(f'\n{"="*70}')
print(f'  COMPARISON: Window Means vs Time-Frequency Features')
print(f'{"="*70}')
results=[]
for rname,ch_idx,rc in TEST_COMBOS:
    # A: Window means (baseline)  — n_ch*4 features
    Xa=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    a,_=decode_4(Xa,epochs_label)

    # B: Full epoch band power — n_ch*5 features
    Xb=band_power_full[:,ch_idx,:].reshape(len(epochs_data),-1)
    b,_=decode_4(Xb,epochs_label)

    # C: Per-window band power — n_ch*4*5 features → PCA to match
    Xc=band_power_win[:,ch_idx,:,:].reshape(len(epochs_data),-1)
    from sklearn.decomposition import PCA
    # PCA keeping 95% variance
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(Xc,epochs_label):
        Xtr,Xte=Xc[tr],Xc[te]
        scl=StandardScaler().fit(Xtr); Xtr_s=scl.transform(Xtr); Xte_s=scl.transform(Xte)
        pca=PCA(n_components=0.95).fit(Xtr_s)
        clf=LDA().fit(pca.transform(Xtr_s),epochs_label[tr])
        accs.append(accuracy_score(epochs_label[te],clf.predict(pca.transform(Xte_s))))
    c=np.mean(accs)

    results.append((rname,a,b,c,rc))
    print(f'  {rname:<10}:  winMean={a:.1%}  bandFull={b:.1%}  bandWinPCA={c:.1%}')

# ====== Figure ======
fig,ax=plt.subplots(figsize=(14,5))
fig.patch.set_facecolor('white')
x=np.arange(len(TEST_COMBOS));w=0.25
names=[r[0] for r in results]
va=[r[1] for r in results]
vb=[r[2] for r in results]
vc=[r[3] for r in results]
cols=[r[4] for r in results]

ax.bar(x-w,va,w,color='#bdc3c7',edgecolor='#333',lw=0.4,label='4 Window Means (baseline)')
ax.bar(x,vb,w,color='#3498db',edgecolor='#333',lw=0.4,alpha=0.8,label='5 Band Powers (0-800ms)')
ax.bar(x+w,vc,w,color='#e74c3c',edgecolor='#333',lw=0.4,alpha=0.8,label='5 Bands x 4 Windows + PCA')
ax.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')

for i in range(len(results)):
    db=vb[i]-va[i]; dc=vc[i]-va[i]
    ax.text(x[i],vb[i]+0.012,f'{db:+.1%}',ha='center',fontsize=7,fontweight='bold',color='#2980b9',rotation=90)
    ax.text(x[i]+w,vc[i]+0.012,f'{dc:+.1%}',ha='center',fontsize=7,fontweight='bold',color='#c0392b',rotation=90)

ax.set_xticks(x);ax.set_xticklabels(names,fontsize=10)
ax.set_ylabel('4-Class Accuracy',fontsize=12)
ax.set_title('Time-Frequency Features vs ERP Window Means',fontsize=13,fontweight='bold')
ax.legend(fontsize=8,ncol=2);ax.grid(True,axis='y',ls=':',alpha=0.3)
ax.set_ylim(0,max(max(va),max(vb),max(vc))*1.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'96_timefreq_comparison.png'),dpi=200,bbox_inches='tight')
plt.close(fig)
print('Saved: 96_timefreq_comparison.png')

# ====== Best band analysis ======
print(f'\n{"="*70}')
print(f'  BEST BAND ANALYSIS (full epoch, Frontal)')
print(f'{"="*70}')
ch_idx=[HW_NAMES.index(c) for c in REGIONS['Frontal']]
for bi,(bn,_,_) in enumerate(FREQ_BANDS):
    X=band_power_full[:,ch_idx,bi].reshape(len(epochs_data),-1)
    a,_=decode_4(X,epochs_label)
    print(f'  {bn:8s} alone: {a:.1%}')

# Combined
Xall=band_power_full[:,ch_idx,:].reshape(len(epochs_data),-1)
aa,_=decode_4(Xall,epochs_label)
print(f'  All bands: {aa:.1%}')

sp=os.path.join(OUT_DIR,'97_timefreq_summary.txt')
with open(sp,'w') as sf:
    sf.write('Time-Frequency Feature Summary\n')
    for rn,va,vb,vc,_ in results:
        sf.write(f'{rn}: winMean={va:.4f} bandFull={vb:.4f} bandWinPCA={vc:.4f}\n')
print(f'Saved: {sp}')
print('Done!')
