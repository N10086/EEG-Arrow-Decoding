#!/usr/bin/env python3
"""
Binary classification — frontal/prefrontal region (F3, Fz, F4) only.
Single channel, pairs, all 3 combined.
"""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys()); HW_NAMES=[CH_MAP[c] for c in HW_CHS]

# Frontal channels
FRONTAL_CHS=['F3','Fz','F4']
FRONTAL_IDX=[HW_NAMES.index(c) for c in FRONTAL_CHS]

FRONTAL_COMBOS=[
    ('F3',[FRONTAL_IDX[0]]),
    ('Fz',[FRONTAL_IDX[1]]),
    ('F4',[FRONTAL_IDX[2]]),
    ('F3+Fz',[FRONTAL_IDX[0],FRONTAL_IDX[1]]),
    ('F3+F4',[FRONTAL_IDX[0],FRONTAL_IDX[2]]),
    ('Fz+F4',[FRONTAL_IDX[1],FRONTAL_IDX[2]]),
    ('F3+Fz+F4',FRONTAL_IDX),
]

BINARY_PAIRS=[
    ('Up vs Down',[0,1]),
    ('Left vs Right',[2,3]),
    ('Up vs Left',[0,2]),
    ('Up vs Right',[0,3]),
    ('Down vs Left',[1,2]),
    ('Down vs Right',[1,3]),
    ('Axis (Vert vs Horz)',[[0,1],[2,3]]),
]

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

win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

def decode_binary(X,y,pair_info):
    if isinstance(pair_info[0],list):
        cls_a,cls_b=pair_info
        mask=np.isin(y,cls_a+cls_b)
        y_bin=np.isin(y[mask],cls_b).astype(int)
    else:
        cls_a,cls_b=pair_info
        mask=(y==cls_a)|(y==cls_b)
        y_bin=(y[mask]==cls_b).astype(int)
    Xb=X[mask]
    if len(Xb)<20: return 0,0
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(Xb,y_bin):
        scl=StandardScaler().fit(Xb[tr])
        clf=LinearDiscriminantAnalysis().fit(scl.transform(Xb[tr]),y_bin[tr])
        accs.append(accuracy_score(y_bin[te],clf.predict(scl.transform(Xb[te]))))
    return np.mean(accs),np.std(accs)

def decode_4class(X,y):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(X,y):
        scl=StandardScaler().fit(X[tr])
        clf=LinearDiscriminantAnalysis().fit(scl.transform(X[tr]),y[tr])
        accs.append(accuracy_score(y[te],clf.predict(scl.transform(X[te]))))
    return np.mean(accs),np.std(accs)

# ====== Run ======
print('\n前额区(F3,Fz,F4) 二分类结果')
print('='*65)

for cname, ch_idx in FRONTAL_COMBOS:
    X=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    a4,_=decode_4class(X,epochs_label)
    print(f'\n  {cname}  (n_ch={len(ch_idx)}, {X.shape[1]} features)')
    print(f'  {"-"*45}')
    print(f'  {"4-class":20s}: {a4:.1%}')
    for pname,pcls in BINARY_PAIRS:
        acc,std=decode_binary(X,epochs_label,pcls)
        bar='#'*int(acc*30)
        above=' ↑' if acc>0.5 else ' ↓'
        print(f'  {pname:20s}: {acc:.1%}+-{std:.1%} {bar} {above}')

# ====== Summary table ======
print('\n\n前额区二分类汇总表')
print('='*65)
print(f'{"组合":<10s}', end='')
for pn,_ in BINARY_PAIRS:
    short=pn.split('(')[0].strip()
    if len(short)>8: short=short[:8]
    print(f'{short:>10s}', end='')
print(f'{"4-class":>10s}')
print('-'*80)

for cname, ch_idx in FRONTAL_COMBOS:
    X=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    a4,_=decode_4class(X,epochs_label)
    print(f'{cname:<10s}', end='')
    for pname,pcls in BINARY_PAIRS:
        acc,_=decode_binary(X,epochs_label,pcls)
        print(f'{acc:>10.1%}', end='')
    print(f'{a4:>10.1%}')

# ====== Normalized confusion: what does Frontal confuse ======
print('\n\n前额区(F3+Fz+F4) 混淆分析')
print('='*65)
Xf=win_mean[:,FRONTAL_IDX,:].reshape(len(epochs_data),-1)
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
from sklearn.metrics import confusion_matrix
cm_sum=np.zeros((4,4))
for tr,te in skf.split(Xf,epochs_label):
    scl=StandardScaler().fit(Xf[tr])
    clf=LinearDiscriminantAnalysis().fit(scl.transform(Xf[tr]),epochs_label[tr])
    cm_sum+=confusion_matrix(epochs_label[te],clf.predict(scl.transform(Xf[te])))
cm_norm=cm_sum/cm_sum.sum(axis=1,keepdims=True)
print(f'         Pred:   Up     Down   Left   Right')
print(f'{"True":>8s}',end='')
for i,d in enumerate(DIR_NAMES):
    print(f'  {d:>6s}',end='')
print()
for i,d in enumerate(DIR_NAMES):
    print(f'{d:>8s}',end='')
    for j in range(4):
        print(f'  {cm_norm[i,j]:.1%}',end='')
    print()

# What % of errors are axis errors vs within-axis errors?
print(f'\n  错误类型分析:')
total_errors=cm_sum.sum()-np.trace(cm_sum)
# Axis errors: Up misclassified as Down or vice versa (0<->1), Left<->Right (2<->3)
within_axis_errors=cm_sum[0,1]+cm_sum[1,0]+cm_sum[2,3]+cm_sum[3,2]
cross_axis_errors=total_errors-within_axis_errors
print(f'    轴内错误(±同一轴): {within_axis_errors}/{total_errors} = {within_axis_errors/total_errors:.1%}')
print(f'    跨轴错误(垂直vs水平混淆): {cross_axis_errors}/{total_errors} = {cross_axis_errors/total_errors:.1%}')

sp=os.path.join(OUT_DIR,'100_frontal_binary_summary.txt')
with open(sp,'w') as sf:
    sf.write('Frontal Binary Classification Summary\n')
    for cname,ch_idx in FRONTAL_COMBOS:
        X=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
        a4,_=decode_4class(X,epochs_label)
        sf.write(f'\n{cname}: 4-class={a4:.4f}\n')
        for pname,pcls in BINARY_PAIRS:
            acc,_=decode_binary(X,epochs_label,pcls)
            sf.write(f'  {pname}: {acc:.4f}\n')
print(f'\nSaved: {sp}')
print('Done!')
