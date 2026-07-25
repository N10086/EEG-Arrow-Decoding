#!/usr/bin/env python3
"""
Feature importance analysis — which channel-window features drive
cross-session decoding. Uses LDA coefficients as importance scores.
"""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
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
BASE=r'E:\deskbook\OpenBCI_GUI\stimulus_logs'
OUT_DIR=r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
DIR_NAMES=['Up','Down','Left','Right']
DIR_VALS=[2.0001,2.0002,2.0003,2.0004]
ERP_WINDOWS=[('P1',0.080,0.130),('N1',0.140,0.200),('P2',0.200,0.300),('P3',0.300,0.500)]
WINDOW_SHORT=['P1','N1','P2','P3']
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys())
HW_NAMES=[CH_MAP[c] for c in HW_CHS]
REGIONS={'Frontal':[7,4,14],'Central':[2,6,5],'Parietal':[9,10,12],'Occipital':[15,1,8]}
REG_NAMES=['Frontal','Central','Parietal','Occipital']
SESSION_PATHS=[
    r'2026-07-07_10-35-51\OpenBCI-RAW-2026-07-07_10-35-51.txt',
    r'2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt',
    r'2026-07-08_08-54-11\OpenBCI-RAW-2026-07-08_08-54-11.txt',
    r'2026-07-08_09-04-45\OpenBCI-RAW-2026-07-08_09-04-45.txt',
    r'2026-07-09_09-02-54\OpenBCI-RAW-2026-07-09_09-02-54.txt',
]
TRAIN_IDX=[0,1,2,3]; TEST_IDX=4

def map_direction(val):
    for di,k in enumerate(DIR_VALS):
        if abs(val-k)<5e-5: return di
    return -1

def load_and_process(path):
    with open(path) as f: lines=f.readlines()
    data,markers=[],[]
    for line in lines[5:]:
        p=line.strip().split(',')
        if len(p)>33:
            try:
                data.append([float(p[i]) for i in range(1,17)])
                markers.append(float(p[32]))
            except: pass
    d=np.array(data,dtype=np.float64).T; m=np.array(markers)
    sel=np.array([d[c-1] for c in HW_CHS])
    bp=sg.butter(4,[1/250,45/250],btype='band',output='sos')
    notch=sg.iirnotch(50/250,30)
    f=np.zeros_like(sel)
    for ch in range(12):
        dm=sel[ch]-sel[ch].mean()
        ts=sg.sosfiltfilt(bp,dm)
        f[ch]=sg.filtfilt(*notch,ts)
    filt_uv=f*SCALE
    ons=sorted([i for k in DIR_VALS for i in np.where(np.abs(m-k)<5e-5)[0]])
    t_start,t_end=-0.2,0.8; n_pre,n_post=int(abs(t_start)*FS),int(t_end*FS); n_total=n_pre+n_post
    epochs,labels=[],[]
    for idx in ons:
        s,e=idx-n_pre,idx+n_post
        if s<0 or e>=filt_uv.shape[1]: continue
        ep=filt_uv[:,s:e].copy()
        ep-=ep[:,:n_pre].mean(axis=1,keepdims=True)
        if np.max(np.abs(ep))>100: continue
        epochs.append(ep); labels.append(map_direction(m[idx]))
    return np.array(epochs),np.array(labels),np.linspace(t_start,t_end,n_total)

# Load
print('Loading 5 sessions...')
all_ep,all_lb=[],[]
for sp in SESSION_PATHS:
    ep,lb,t=load_and_process(os.path.join(BASE,sp))
    all_ep.append(ep); all_lb.append(lb)

train_ep=np.concatenate([all_ep[i] for i in TRAIN_IDX],axis=0)
train_lb=np.concatenate([all_lb[i] for i in TRAIN_IDX],axis=0)
test_ep=all_ep[TEST_IDX]; test_lb=all_lb[TEST_IDX]

# Window means
train_win=np.zeros((len(train_ep),12,4))
test_win=np.zeros((len(test_ep),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    train_win[:,:,wi]=train_ep[:,:,msk].mean(axis=2)
    test_win[:,:,wi]=test_ep[:,:,msk].mean(axis=2)

# ====== Feature importance analysis ======
# Key combinations to analyze
KEY_COMBOS = {
    'Central (C3,Cz,C4)': {'regions':['Central'], 'channels':[2,6,5]},
    'Frontal (F3,Fz,F4)': {'regions':['Frontal'], 'channels':[7,4,14]},
    'Occipital (O1,Oz,O2)': {'regions':['Occipital'], 'channels':[15,1,8]},
    'Parietal (P3,Pz,P4)': {'regions':['Parietal'], 'channels':[9,10,12]},
}

all_ch_importance = np.zeros((12, 4))  # 12 channels x 4 windows

print('\nComputing feature importance...')
fig_imp, axes_imp = plt.subplots(2, 2, figsize=(14, 10))
fig_imp.patch.set_facecolor('white')

for rik, (cname, cinfo) in enumerate(KEY_COMBOS.items()):
    ax = axes_imp[rik//2, rik%2]

    ch_idx = [HW_CHS.index(c) for c in cinfo['channels']]
    ch_names = [HW_NAMES[i] for i in ch_idx]

    Xtr = train_win[:, ch_idx, :].reshape(len(train_ep), -1)
    Xte = test_win[:, ch_idx, :].reshape(len(test_ep), -1)

    scl = StandardScaler().fit(Xtr)
    clf = LinearDiscriminantAnalysis().fit(scl.transform(Xtr), train_lb)
    y_pred = clf.predict(scl.transform(Xte))
    acc = accuracy_score(test_lb, y_pred)

    # Feature importance: sum of abs(coef) across all 4 class discriminants
    # clf.coef_ shape: (4, n_features) where n_features = n_channels * 4
    importance = np.sum(np.abs(clf.coef_), axis=0)  # (n_channels*4,)
    # Normalize to 0-1
    if importance.max() > 0:
        importance = importance / importance.max()

    # Reshape to (n_channels, 4)
    imp_mat = importance.reshape(len(ch_idx), 4)

    # Store for all-channels analysis
    for ci, ci_global in enumerate(ch_idx):
        all_ch_importance[ci_global, :] = imp_mat[ci, :]

    # Heatmap
    im = ax.imshow(imp_mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(WINDOW_SHORT, fontsize=9)
    ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=9)
    ax.set_xlabel('ERP Window', fontsize=10)
    ax.set_ylabel('Channel', fontsize=10)
    ax.set_title(f'{cname}\nCross-Session Acc: {acc:.1%}', fontsize=10, fontweight='bold')
    for ci in range(len(ch_idx)):
        for wj in range(4):
            v = imp_mat[ci, wj]
            ax.text(wj, ci, f'{v:.2f}', ha='center', va='center',
                   fontsize=7, fontweight='bold',
                   color='white' if v > 0.5 else 'black')
    fig_imp.colorbar(im, ax=ax, shrink=0.85)

    # Print interpretation
    max_ch_win = np.unravel_index(imp_mat.argmax(), imp_mat.shape)
    print(f'  {cname}: acc={acc:.1%}')
    print(f'    Top feature: {ch_names[max_ch_win[0]]} x {WINDOW_SHORT[max_ch_win[1]]} (imp={imp_mat[max_ch_win]:.3f})')
    # Window ranking
    win_imp = imp_mat.mean(axis=0)
    window_order = np.argsort(win_imp)[::-1]
    for wi in window_order:
        print(f'    {WINDOW_SHORT[wi]} window: {win_imp[wi]:.3f}')
    # Channel ranking
    ch_imp = imp_mat.mean(axis=1)
    ch_order = np.argsort(ch_imp)[::-1]
    for ci in ch_order:
        print(f'    {ch_names[ci]} channel: {ch_imp[ci]:.3f}')

fig_imp.tight_layout()
fig_imp.savefig(os.path.join(OUT_DIR, '61_feature_importance_regions.png'), dpi=200, bbox_inches='tight')
plt.close(fig_imp)
print('  Saved: 61_feature_importance_regions.png')

# ====== All 12 channels importance ======
print('\nAll 12 channels feature importance...')
Xtr_all = train_win.reshape(len(train_ep), -1)  # 48 features (12ch x 4w)
Xte_all = test_win.reshape(len(test_ep), -1)

scl = StandardScaler().fit(Xtr_all)
clf = LinearDiscriminantAnalysis().fit(scl.transform(Xtr_all), train_lb)
y_pred = clf.predict(scl.transform(Xte_all))
acc_all = accuracy_score(test_lb, y_pred)

importance_all = np.sum(np.abs(clf.coef_), axis=0)
if importance_all.max() > 0:
    importance_all = importance_all / importance_all.max()

imp_all_mat = importance_all.reshape(12, 4)

# Sort channels by overall importance (mean across windows)
ch_imp_all = imp_all_mat.mean(axis=1)
ch_order_all = np.argsort(ch_imp_all)[::-1]

# FIGURE: All channels heatmap
fig_all, axes_all = plt.subplots(1, 2, figsize=(18, 8))
fig_all.patch.set_facecolor('white')

# Heatmap (sorted)
sorted_idx = np.argsort(-imp_all_mat.mean(axis=1))
sorted_names = [HW_NAMES[i] for i in sorted_idx]
sorted_mat = imp_all_mat[sorted_idx, :]

im_all = axes_all[0].imshow(sorted_mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
axes_all[0].set_xticks(range(4)); axes_all[0].set_xticklabels(WINDOW_SHORT, fontsize=9)
axes_all[0].set_yticks(range(12)); axes_all[0].set_yticklabels(sorted_names, fontsize=8)
axes_all[0].set_xlabel('ERP Window', fontsize=11); axes_all[0].set_ylabel('Channel (sorted by importance)', fontsize=11)
axes_all[0].set_title(f'All 12 Channels — Feature Importance (Cross-Session Acc: {acc_all:.1%})',
                     fontsize=11, fontweight='bold')
for ci in range(12):
    for wj in range(4):
        v = sorted_mat[ci, wj]
        axes_all[0].text(wj, ci, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, fontweight='bold',
                        color='white' if v > 0.5 else 'black')
fig_all.colorbar(im_all, ax=axes_all[0], shrink=0.85)

# Bar chart: channel-level importance
ch_imp_sorted = ch_imp_all[sorted_idx]
colors_ch = []
for ci in sorted_idx:
    if ci in [HW_CHS.index(c) for c in REGIONS['Frontal']]:
        colors_ch.append('#e74c3c')
    elif ci in [HW_CHS.index(c) for c in REGIONS['Central']]:
        colors_ch.append('#3498db')
    elif ci in [HW_CHS.index(c) for c in REGIONS['Parietal']]:
        colors_ch.append('#2ecc71')
    else:
        colors_ch.append('#f39c12')

x_ch = np.arange(12)
bars = axes_all[1].barh(x_ch, ch_imp_sorted, color=colors_ch, edgecolor='#333', lw=0.4, height=0.7)
axes_all[1].set_yticks(x_ch); axes_all[1].set_yticklabels(sorted_names, fontsize=8)
axes_all[1].set_xlabel('Mean Importance (across 4 windows)', fontsize=11)
axes_all[1].set_title('Channel-Level Importance (aggregated across windows)', fontsize=11, fontweight='bold')
axes_all[1].axvline(0, color='#888', lw=0.5)
# Add region labels
region_boundaries = {}
for ci, ci_global in enumerate(sorted_idx):
    for rn, chs in REGIONS.items():
        if HW_CHS[ci_global] in chs:
            region_boundaries.setdefault(rn, []).append(ci)
for rn, rn_ci in region_boundaries.items():
    if rn_ci:
        y0 = min(rn_ci); y1 = max(rn_ci)
        color_map = {'Frontal':'#e74c3c','Central':'#3498db','Parietal':'#2ecc71','Occipital':'#f39c12'}
        axes_all[1].axhspan(y0-0.5, y1+0.5, alpha=0.06, color=color_map.get(rn, '#888'))
        axes_all[1].text(ch_imp_sorted.max()*1.05, (y0+y1)/2, rn, fontsize=8, color=color_map.get(rn, '#888'),
                        fontweight='bold', va='center')

fig_all.tight_layout()
fig_all.savefig(os.path.join(OUT_DIR, '62_feature_importance_all_channels.png'), dpi=200, bbox_inches='tight')
plt.close(fig_all)
print('  Saved: 62_feature_importance_all_channels.png')

# ====== Window importance ======
win_imp_all = imp_all_mat.mean(axis=0)
print(f'\nWindow importance across all channels:')
for wi in np.argsort(win_imp_all)[::-1]:
    print(f'  {WINDOW_SHORT[wi]}: {win_imp_all[wi]:.3f}')

# ====== Direction-specific feature profiles ======
# Show the mean feature profile (standardized) for each direction
print('\nDirection-specific feature profiles (Central region, all 5 sessions)...')
all_sessions_ep = np.concatenate(all_ep, axis=0)
all_sessions_lb = np.concatenate(all_lb, axis=0)

all_win = np.zeros((len(all_sessions_ep), 12, 4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    all_win[:,:,wi]=all_sessions_ep[:,:,msk].mean(axis=2)

# Central region
ch_idx_c = [HW_CHS.index(c) for c in REGIONS['Central']]
ch_names_c = [HW_NAMES[i] for i in ch_idx_c]
X_c = all_win[:, ch_idx_c, :].reshape(len(all_sessions_ep), -1)
scl_c = StandardScaler().fit(X_c)
X_c_norm = scl_c.transform(X_c)

# Mean profile per direction
dir_profiles = np.zeros((4, 12))  # 4 directions x 12 features (3ch x 4w)
for di in range(4):
    mask = all_sessions_lb == di
    dir_profiles[di] = X_c_norm[mask].mean(axis=0)

fig_prof, ax_prof = plt.subplots(figsize=(12, 5))
fig_prof.patch.set_facecolor('white')
x_prof = np.arange(12)
labels_prof = []
for cn in ch_names_c:
    for wn in WINDOW_SHORT:
        labels_prof.append(f'{cn}-{wn}')
dir_colors = ['#4A72C4','#E8833A','#5CB85C','#9B59B6']
for di in range(4):
    ax_prof.plot(x_prof, dir_profiles[di], 'o-', color=dir_colors[di], lw=2, ms=6, label=DIR_NAMES[di])
ax_prof.axhline(0, color='#888', ls='--', lw=0.8)
ax_prof.set_xticks(x_prof); ax_prof.set_xticklabels(labels_prof, fontsize=8, rotation=45, ha='right')
ax_prof.set_ylabel('Standardized Amplitude (z-score)', fontsize=11)
ax_prof.set_title('Central Region: Direction-Specific Feature Profile (all 5 sessions)', fontsize=12, fontweight='bold')
ax_prof.legend(fontsize=10); ax_prof.grid(True, axis='y', ls=':', alpha=0.3)
fig_prof.tight_layout()
fig_prof.savefig(os.path.join(OUT_DIR, '63_direction_feature_profiles.png'), dpi=200, bbox_inches='tight')
plt.close(fig_prof)
print('  Saved: 63_direction_feature_profiles.png')

# ====== Summary ======
print(f'\n{"="*80}')
print(f'  FEATURE IMPORTANCE SUMMARY')
print(f'{"="*80}')

print(f'\n  Window importance ranking (all 12 channels, cross-session):')
for wi in np.argsort(win_imp_all)[::-1]:
    print(f'    {WINDOW_SHORT[wi]:4s} ({(ERP_WINDOWS[wi][1]):.0f}-{(ERP_WINDOWS[wi][2]*1000):.0f}ms): importance={win_imp_all[wi]:.3f}')

print(f'\n  Channel importance ranking (mean across windows):')
for ci in ch_order_all:
    region = ''
    for rn, chs in REGIONS.items():
        if HW_CHS[ci] in chs:
            region = f' ({rn})'
            break
    print(f'    {HW_NAMES[ci]:4s}{region}: importance={ch_imp_all[ci]:.3f}')

print(f'\n  Top 5 specific features (channel x window):')
flat_imp = [(ci, wj, imp_all_mat[ci, wj]) for ci in range(12) for wj in range(4)]
flat_imp.sort(key=lambda x: x[2], reverse=True)
for ci, wj, imp in flat_imp[:5]:
    region = ''
    for rn, chs in REGIONS.items():
        if HW_CHS[ci] in chs:
            region = f'({rn})'
            break
    print(f'    {HW_NAMES[ci]:4s} x {WINDOW_SHORT[wj]:4s} {region}: importance={imp:.3f}')

sp = os.path.join(OUT_DIR, '64_feature_importance_summary.txt')
with open(sp, 'w') as sf:
    sf.write(f'Feature Importance Summary\n{"="*60}\n')
    sf.write(f'Cross-session LDA coefficient analysis\n\n')
    sf.write(f'Window ranking:\n')
    for wi in np.argsort(win_imp_all)[::-1]:
        sf.write(f'  {WINDOW_SHORT[wi]}: {win_imp_all[wi]:.4f}\n')
    sf.write(f'\nChannel ranking:\n')
    for ci in ch_order_all:
        sf.write(f'  {HW_NAMES[ci]}: {ch_imp_all[ci]:.4f}\n')

print(f'\nSaved: {sp}')
print('\nFigures: 61-63')
print('Done!')
