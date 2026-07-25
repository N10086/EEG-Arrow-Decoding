#!/usr/bin/env python3
"""
Binary classification — all direction pairs & axis, all key region combos.
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
WINDOW_SHORT=['P1','N1','P2','P3']
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys()); HW_NAMES=[CH_MAP[c] for c in HW_CHS]
REGIONS={'Frontal':['F3','Fz','F4'],'Central':['C3','Cz','C4'],
         'Parietal':['P3','Pz','P4'],'Occipital':['O1','Oz','O2']}
REG_COLORS=['#e74c3c','#3498db','#2ecc71','#f39c12']

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

# Window means: n_trials x 12 x 4
win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# ====== Binary classification ======
BINARY_PAIRS = [
    ('Up vs Down', [0, 1]),
    ('Left vs Right', [2, 3]),
    ('Up vs Left', [0, 2]),
    ('Up vs Right', [0, 3]),
    ('Down vs Left', [1, 2]),
    ('Down vs Right', [1, 3]),
    ('Axis (Vert vs Horz)', [[0,1],[2,3]]),  # special: group classes
]
PAIR_COLORS = ['#4A72C4','#E8833A','#5CB85C','#9B59B6','#D94F70','#34495e','#7f8c8d']

REGION_COMBOS = [
    ('Frontal', [HW_NAMES.index(c) for c in REGIONS['Frontal']], '#e74c3c'),
    ('Central', [HW_NAMES.index(c) for c in REGIONS['Central']], '#3498db'),
    ('Parietal', [HW_NAMES.index(c) for c in REGIONS['Parietal']], '#2ecc71'),
    ('Occipital', [HW_NAMES.index(c) for c in REGIONS['Occipital']], '#f39c12'),
    ('F+P', [HW_NAMES.index(c) for c in REGIONS['Frontal']+REGIONS['Parietal']], '#8e44ad'),
    ('F+P+O', [HW_NAMES.index(c) for c in REGIONS['Frontal']+REGIONS['Parietal']+REGIONS['Occipital']], '#16a085'),
    ('All 12ch', list(range(12)), '#333333'),
]

def decode_binary(X, y, pair_info):
    """Binary LDA decoding with 5-fold CV."""
    if isinstance(pair_info[0], list):  # axis: group classes
        cls_a, cls_b = pair_info
        mask = np.isin(y, cls_a+cls_b)
        y_bin = np.isin(y[mask], cls_b).astype(int)
    else:
        cls_a, cls_b = pair_info
        mask = (y == cls_a) | (y == cls_b)
        y_bin = (y[mask] == cls_b).astype(int)

    Xb = X[mask]
    if len(Xb) < 20:
        return 0, 0, 0  # too few trials

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(Xb, y_bin):
        scl = StandardScaler().fit(Xb[tr])
        clf = LinearDiscriminantAnalysis().fit(scl.transform(Xb[tr]), y_bin[tr])
        accs.append(accuracy_score(y_bin[te], clf.predict(scl.transform(Xb[te]))))
    return np.mean(accs), np.std(accs), len(Xb)

# ====== Run all binary × region ======
print('\nBinary classification (all region combos × all pairs)...')
print(f'  {"Region":<10} {"Pair":<22} {"Acc":<10} {"Trials":<8}')
print(f'  {"-"*54}')
results = {}
for rname, ch_idx, rcolor in REGION_COMBOS:
    X = win_mean[:, ch_idx, :].reshape(len(epochs_data), -1)
    for pname, pcls in BINARY_PAIRS:
        acc, std, n = decode_binary(X, epochs_label, pcls)
        key = (rname, pname)
        results[key] = {'acc': acc, 'std': std, 'n': n, 'pair': pname, 'region': rname}
        print(f'  {rname:<10} {pname:<22} {acc:.1%}+-{std:.1%}  n={n}')

# ====== FIGURE 1: Heatmap ======
fig1, ax1 = plt.subplots(figsize=(14, 7))
fig1.patch.set_facecolor('white')
rnames_order = [r[0] for r in REGION_COMBOS]
pnames_order = [p[0] for p in BINARY_PAIRS]
hm_data = np.zeros((len(rnames_order), len(pnames_order)))
for ri, rn in enumerate(rnames_order):
    for pi, pn in enumerate(pnames_order):
        hm_data[ri, pi] = results[(rn, pn)]['acc']

im1 = ax1.imshow(hm_data, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
ax1.set_xticks(range(len(pnames_order))); ax1.set_xticklabels(pnames_order, fontsize=9, rotation=30, ha='right')
ax1.set_yticks(range(len(rnames_order))); ax1.set_yticklabels(rnames_order, fontsize=10)
ax1.set_title('Binary Classification Accuracy — Region × Direction Pair', fontsize=13, fontweight='bold')
for ri in range(len(rnames_order)):
    for pi in range(len(pnames_order)):
        v = hm_data[ri, pi]
        ax1.text(pi, ri, f'{v:.1%}', ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white' if v > 0.7 else 'black')
fig1.colorbar(im1, ax=ax1, shrink=0.8, label='Accuracy')
# Add 4-class reference lines
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, '91_binary_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close(fig1)
print('Saved: 91_binary_heatmap.png')

# ====== FIGURE 2: Best binary per region ======
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('white')
x2 = np.arange(len(rnames_order)); w = 0.13
# For each region, show all 7 binary accuracies
for pi, (pn, _) in enumerate(BINARY_PAIRS):
    vals = [results[(rn, pn)]['acc'] for rn in rnames_order]
    ax2.bar(x2 + pi*w - w*3, vals, w, color=PAIR_COLORS[pi], alpha=0.85,
            edgecolor='#333', lw=0.3, label=pn)
ax2.axhline(0.5, color='#888', ls='--', lw=1.5, label='Chance (50%)')
ax2.set_xticks(x2); ax2.set_xticklabels(rnames_order, fontsize=10)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Binary Classification by Region — All Direction Pairs', fontsize=12, fontweight='bold')
ax2.legend(fontsize=7, ncol=4); ax2.grid(True, axis='y', ls=':', alpha=0.3)
ax2.set_ylim(0.35, 1.0)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '92_binary_by_region.png'), dpi=200, bbox_inches='tight')
plt.close(fig2)
print('Saved: 92_binary_by_region.png')

# ====== FIGURE 3: 4-class vs best binary scatter ======
# Compare 4-class accuracy with the best binary accuracy for each region
from sklearn.metrics import accuracy_score as acc_score
fig3, ax3 = plt.subplots(figsize=(8, 8))
fig3.patch.set_facecolor('white')
ax3.plot([0.2, 0.8], [0.2, 0.8], '--', color='#888', lw=1, label='Equal')
for rname, ch_idx, rcolor in REGION_COMBOS:
    X = win_mean[:, ch_idx, :].reshape(len(epochs_data), -1)
    # 4-class
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs4 = []
    for tr, te in skf.split(X, epochs_label):
        scl = StandardScaler().fit(X[tr])
        clf = LinearDiscriminantAnalysis().fit(scl.transform(X[tr]), epochs_label[tr])
        accs4.append(acc_score(epochs_label[te], clf.predict(scl.transform(X[te]))))
    acc4 = np.mean(accs4)
    # Best binary
    best_bin = max(results[(rname, pn)]['acc'] for pn in [p[0] for p in BINARY_PAIRS])
    ax3.scatter(acc4, best_bin, s=150, c=rcolor, edgecolors='#333', lw=1, zorder=5)
    ax3.annotate(rname, (acc4, best_bin), textcoords='offset points',
                xytext=(8, 5), fontsize=10, fontweight='bold')
ax3.set_xlabel('4-Class Accuracy', fontsize=12)
ax3.set_ylabel('Best Binary Accuracy', fontsize=12)
ax3.set_title('4-Class vs Best Binary Classification', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10); ax3.grid(True, ls=':', alpha=0.3)
ax3.set_xlim(0.3, 0.8); ax3.set_ylim(0.3, 1.0)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, '93_4class_vs_binary.png'), dpi=200, bbox_inches='tight')
plt.close(fig3)
print('Saved: 93_4class_vs_binary.png')

# ====== Summary ======
print(f'\n{"="*80}')
print(f'  BINARY CLASSIFICATION SUMMARY')
print(f'{"="*80}')

for rname, ch_idx, rcolor in REGION_COMBOS:
    X4 = win_mean[:, ch_idx, :].reshape(len(epochs_data), -1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs4 = []
    for tr, te in skf.split(X4, epochs_label):
        scl = StandardScaler().fit(X4[tr])
        clf = LinearDiscriminantAnalysis().fit(scl.transform(X4[tr]), epochs_label[tr])
        accs4.append(acc_score(epochs_label[te], clf.predict(scl.transform(X4[te]))))
    acc4 = np.mean(accs4)
    print(f'  {rname}: 4-class={acc4:.1%}')

    for pname, _ in BINARY_PAIRS:
        r = results[(rname, pname)]
        bar = '#' * int(r['acc'] * 30)
        print(f'    {pname:<22} {r["acc"]:.1%}  ({r["n"]} trials)')

sp = os.path.join(OUT_DIR, '94_binary_summary.txt')
with open(sp, 'w') as sf:
    sf.write('Binary Classification Summary\n')
    for rname, ch_idx, _ in REGION_COMBOS:
        X4 = win_mean[:, ch_idx, :].reshape(len(epochs_data), -1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs4 = []
        for tr, te in skf.split(X4, epochs_label):
            scl = StandardScaler().fit(X4[tr])
            clf = LinearDiscriminantAnalysis().fit(scl.transform(X4[tr]), epochs_label[tr])
            accs4.append(acc_score(epochs_label[te], clf.predict(scl.transform(X4[te]))))
        sf.write(f'\n{rname} (4-class={np.mean(accs4):.4f}):\n')
        for pname, _ in BINARY_PAIRS:
            r = results[(rname, pname)]
            sf.write(f'  {pname}: {r["acc"]:.4f}\n')

print(f'\nSaved: {sp}')
print('Done!')
