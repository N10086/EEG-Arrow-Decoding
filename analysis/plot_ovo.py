#!/usr/bin/env python3
"""
One-vs-One voting vs direct multi-class LDA for 4-class decoding.
Train 6 binary classifiers, let them vote, compare.
"""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
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

win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# ====== Compare: multi-class LDA vs OvO LDA ======
TEST_COMBOS = [
    ('Frontal', [HW_NAMES.index(c) for c in REGIONS['Frontal']], '#e74c3c'),
    ('Central', [HW_NAMES.index(c) for c in REGIONS['Central']], '#3498db'),
    ('Parietal', [HW_NAMES.index(c) for c in REGIONS['Parietal']], '#2ecc71'),
    ('Occipital', [HW_NAMES.index(c) for c in REGIONS['Occipital']], '#f39c12'),
    ('F+P', [HW_NAMES.index(c) for c in REGIONS['Frontal']+REGIONS['Parietal']], '#8e44ad'),
    ('All 12ch', list(range(12)), '#333'),
]

print('\nMulti-class LDA vs One-vs-One LDA voting...')
print(f'  {"Region":<10} {"MultiLDA":<10} {"OvO LDA":<10} {"BestOf":<10}')
print(f'  {"-"*44}')
results = []
for rname, ch_idx, rc in TEST_COMBOS:
    X = win_mean[:, ch_idx, :].reshape(len(epochs_data), -1)
    y = epochs_label

    # Multi-class LDA (standard)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_multi = []
    acc_ovo = []
    for tr, te in skf.split(X, y):
        scl = StandardScaler().fit(X[tr])
        Xtr_s = scl.transform(X[tr]); Xte_s = scl.transform(X[te])

        # Standard multi-class LDA
        clf = LDA().fit(Xtr_s, y[tr])
        acc_multi.append(accuracy_score(y[te], clf.predict(Xte_s)))

        # One-vs-One LDA
        ovo = OneVsOneClassifier(LDA()).fit(Xtr_s, y[tr])
        acc_ovo.append(accuracy_score(y[te], ovo.predict(Xte_s)))

    m = np.mean(acc_multi)
    o = np.mean(acc_ovo)
    results.append((rname, m, o, rc))
    best = max(m, o)
    print(f'  {rname:<10} {m:.1%}     {o:.1%}     {best:.1%}')

# ====== Figure ======
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')
x = np.arange(len(TEST_COMBOS)); w = 0.35
names = [r[0] for r in results]
multi_vals = [r[1] for r in results]
ovo_vals = [r[2] for r in results]
colors = [r[3] for r in results]

ax.bar(x-w/2, multi_vals, w, color='#bdc3c7', edgecolor='#333', lw=0.4, label='Standard Multi-Class LDA')
bars = ax.bar(x+w/2, ovo_vals, w, color=colors, edgecolor='#333', lw=0.4, label='One-vs-One LDA Voting')
ax.axhline(0.25, color='#888', ls='--', lw=1.5, label='Chance (25%)')

for i in range(len(results)):
    d = ovo_vals[i] - multi_vals[i]
    c = '#27ae60' if d > 0.005 else '#c0392b' if d < -0.005 else '#888'
    ax.text(i+w/2, ovo_vals[i]+0.012, f'{d:+.1%}', ha='center', fontsize=9, fontweight='bold', color=c)

ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('4-Class Accuracy', fontsize=12)
ax.set_title('Multi-Class LDA vs One-vs-One Voting LDA', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, axis='y', ls=':', alpha=0.3)
ax.set_ylim(0, max(max(multi_vals), max(ovo_vals))*1.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '95_ovo_comparison.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: 95_ovo_comparison.png')

# ====== Summary ======
print(f'\n{"="*70}')
print(f'  OVO VOTING SUMMARY')
print(f'{"="*70}')
for rn, m, o, _ in results:
    print(f'  {rn:<10}: multi={m:.1%}  ovo={o:.1%}  delta={o-m:+.1%}')
avg_m = np.mean([r[1] for r in results])
avg_o = np.mean([r[2] for r in results])
print(f'\n  Average: multi={avg_m:.1%}  ovo={avg_o:.1%}  delta={avg_o-avg_m:+.1%}')
print('Done!')
