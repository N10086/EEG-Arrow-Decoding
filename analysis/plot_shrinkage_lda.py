#!/usr/bin/env python3
"""
Try shrinkage LDA on full time course — handles high-dim features directly,
no PCA needed. Compare with standard LDA on 4 window means.
"""
import numpy as np
from scipy import signal as sg
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
WINDOW_SHORT=['P1','N1','P2','P3']
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
print(f'Trials: {len(epochs_data)}, Time points: {n_total}')

# Window means
win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# ====== Compare methods ======
# Methods:
# M1: 4 window means + standard LDA   (baseline)
# M2: Full time course + shrinkage LDA  (solver='lsqr', shrinkage='auto')
# M3: Full time course + PCA + standard LDA

def decode(X,y,clf_type='standard'):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr); Xtr_s=scl.transform(Xtr); Xte_s=scl.transform(Xte)
        if clf_type=='standard':
            clf=LDA().fit(Xtr_s,ytr)
        elif clf_type=='shrinkage':
            clf=LDA(solver='lsqr',shrinkage='auto').fit(Xtr_s,ytr)
        accs.append(accuracy_score(yte,clf.predict(Xte_s)))
    return np.mean(accs),np.std(accs)

print('\nComparing methods...')
results={}

# Test combinations
TEST_COMBOS = [
    ('F (F3,Fz,F4)', [HW_NAMES.index(c) for c in REGIONS['Frontal']]),
    ('C (C3,Cz,C4)', [HW_NAMES.index(c) for c in REGIONS['Central']]),
    ('F+P', [HW_NAMES.index(c) for c in REGIONS['Frontal']+REGIONS['Central']]),
    ('All 12ch', list(range(12))),
]

for cname, ch_idx in TEST_COMBOS:
    # M1: 4 window means + standard LDA
    Xm=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    a1,_=decode(Xm,epochs_label,'standard')
    # M2: Full time course + shrinkage LDA
    Xf=epochs_data[:,ch_idx,:].reshape(len(epochs_data),-1)
    a2,_=decode(Xf,epochs_label,'shrinkage')
    # M3: Full time course + PCA95 + standard LDA
    from sklearn.decomposition import PCA
    skf3=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs3=[]
    for tr,te in skf3.split(Xf,epochs_label):
        Xtr3,Xte3=Xf[tr],Xf[te]
        scl3=StandardScaler().fit(Xtr3); Xtr3_s=scl3.transform(Xtr3); Xte3_s=scl3.transform(Xte3)
        pca3=PCA(n_components=0.95).fit(Xtr3_s)
        clf3=LDA().fit(pca3.transform(Xtr3_s),epochs_label[tr])
        accs3.append(accuracy_score(epochs_label[te],clf3.predict(pca3.transform(Xte3_s))))
    a3=np.mean(accs3)

    results[cname]={'win4':a1,'full_shrink':a2,'full_pca':a3}
    print(f'  {cname:12s}:  win4={a1:.1%}  full+shrink={a2:.1%}  full+PCA={a3:.1%}')

sp=os.path.join(OUT_DIR,'84_shrinkage_lda_summary.txt')
with open(sp,'w') as sf:
    sf.write('Shrinkage LDA Comparison\n')
    for cn in results:
        r=results[cn]
        sf.write(f'{cn}: win4={r["win4"]:.4f} shrink={r["full_shrink"]:.4f}\n')
print(f'Saved: {sp}')
