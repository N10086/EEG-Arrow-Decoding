#!/usr/bin/env python3
"""Compare frontal source activity: first 10 trials vs entire session."""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
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

FS = 500.0; GAIN = 6.0; SCALE = 4.5/(2**23-1)/GAIN*1e6
SUBJECTS_DIR = r'C:\Users\Zibo\mne_data\MNE-fsaverage-data'
SUBJECT = 'fsaverage'
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'

# Use Session 2 (best quality, 200 trials, 0 bad)
DATA_PATH = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'

ERP_WINDOWS = [('P1',0.080,0.130,'#27ae60'),('N1',0.140,0.200,'#7f8c8d'),
               ('P2',0.200,0.300,'#e67e22'),('P3',0.300,0.500,'#9B59B6')]
ROI_ORDER = ['Frontal','Central','Parietal','Occipital']
CH_MAP = {1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',
          8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS = sorted(CH_MAP.keys()); MNE_NAMES = [CH_MAP[c] for c in HW_CHS]

DK_ROIS = {
    'Frontal': ['superiorfrontal','rostralmiddlefrontal','caudalmiddlefrontal',
                'lateralorbitofrontal','medialorbitofrontal','frontalpole',
                'parsopercularis','parsorbitalis','parstriangularis'],
    'Central': ['precentral','postcentral'],
    'Parietal': ['superiorparietal','inferiorparietal','precuneus',
                 'supramarginal','isthmuscingulate'],
    'Occipital': ['lateraloccipital','cuneus','pericalcarine','lingual'],
}
HEMIS = ['lh','rh']

# ====== Build forward model once ======
print('Setting up fsaverage...')
src = mne.setup_source_space(SUBJECT, spacing='ico3', subjects_dir=SUBJECTS_DIR, add_dist=False)
model = mne.make_bem_model(SUBJECT, ico=4, conductivity=[0.3,0.006,0.3], subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)
labels = mne.read_labels_from_annot(SUBJECT, 'aparc', subjects_dir=SUBJECTS_DIR, verbose=False)
roi_labels = {}
for rn, dks in DK_ROIS.items():
    sel = []
    for dk in dks:
        for hemi in HEMIS:
            l = [x for x in labels if x.name == f'{dk}-{hemi}']
            sel.extend(l)
    roi_labels[rn] = sel

# ====== Load Session 2 ======
print('Loading Session 2...')
with open(DATA_PATH) as f:
    lines = f.readlines()
data, markers = [], []
for line in lines[5:]:
    p = line.strip().split(',')
    if len(p) > 33:
        try:
            data.append([float(p[i]) for i in range(1,17)])
            markers.append(float(p[32]))
        except: pass
d = np.array(data, dtype=np.float64).T; m = np.array(markers)
sel = np.array([d[c-1] for c in HW_CHS])
bp = sg.butter(4, [1/250,45/250], btype='band', output='sos')
notch = sg.iirnotch(50/250,30)
filt = np.zeros_like(sel)
for ch in range(12):
    dm = sel[ch]-sel[ch].mean()
    ts = sg.sosfiltfilt(bp, dm)
    filt[ch] = sg.filtfilt(*notch, ts)
filt_uv = filt * SCALE

ons = sorted([i for k in [2.0001,2.0002,2.0003,2.0004]
              for i in np.where(np.abs(m-k)<5e-5)[0]])

# ====== MNE processing ======
info = mne.create_info(ch_names=MNE_NAMES, sfreq=FS, ch_types=['eeg']*12)
raw = mne.io.RawArray(filt_uv/1e6, info)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage); raw.set_eeg_reference(projection=True)

ev_list = []
for k,v in {2.0001:1,2.0002:2,2.0003:3,2.0004:4}.items():
    for idx in ons:
        if abs(m[idx]-k)<5e-5:
            ev_list.append([idx,0,v])
events = np.array(ev_list, dtype=int)

# ====== Epoch all trials ======
epochs_all = mne.Epochs(raw, events, {'Up':1,'Down':2,'Left':3,'Right':4},
                        -0.2, 0.8, baseline=(-0.2,0.0), preload=True,
                        reject={'eeg':100e-6}, proj=True, verbose=False)
print(f'Total good epochs: {len(epochs_all)}')

t = epochs_all.times

# ====== Forward & inverse from ALL trials ======
fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem,
                                 meg=False, eeg=True, verbose=False)
cov = mne.compute_covariance(epochs_all, tmax=0.0, method='empirical', verbose=False)
inv = make_inverse_operator(epochs_all.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)
print('Inverse operator computed from all 200 trials.')

# ====== Define trial subsets ======
n_all = len(epochs_all)
subsets = {
    'First 10 trials': epochs_all[:10],
    'First 30 trials': epochs_all[:30],
    'All 200 trials': epochs_all[:],
}
# Also add last 10/30
subsets['Last 10 trials'] = epochs_all[-10:]
subsets['Last 30 trials'] = epochs_all[-30:]

# ====== Apply inverse to each subset ======
subset_stc = {}
subset_roi = {}
for sname, ep_sub in subsets.items():
    if len(ep_sub) < 5:
        print(f'  {sname}: too few trials ({len(ep_sub)}), skipping')
        continue
    ev = ep_sub.average()
    stc = apply_inverse(ev, inv, lambda2=1/9.0, method='sLORETA',
                        pick_ori=None, verbose=False)

    # Extract ROI
    roi = {}
    for rn in ROI_ORDER:
        sr = stc.extract_label_time_course(roi_labels[rn], src, mode='mean')
        if isinstance(sr, np.ndarray) and sr.ndim == 1: sr = [sr]
        roi[rn] = np.mean(sr, axis=0) if (sr is not None and len(sr)>0) else np.zeros_like(t)
    subset_stc[sname] = stc
    subset_roi[sname] = roi
    print(f'  {sname}: {len(ep_sub)} trials -> ROI source extracted')

# ====== FIGURE 1: Frontal only, compare subsets ======
colors_list = {k:'#4A72C4'}
colors_list['First 10 trials'] = '#E74C3C'
colors_list['First 30 trials'] = '#E8833A'
colors_list['All 200 trials'] = '#2c3e50'
colors_list['Last 10 trials'] = '#27AE60'
colors_list['Last 30 trials'] = '#2ECC71'
ls_list = {'First 10 trials':'-','First 30 trials':'--','All 200 trials':'-',
           'Last 10 trials':'-','Last 30 trials':'--'}

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
fig.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes[ri]; ax.set_facecolor('#FAFAFA')

    for sname in ['First 10 trials', 'First 30 trials', 'All 200 trials',
                  'Last 30 trials', 'Last 10 trials']:
        if sname not in subset_roi:
            continue
        n_tr = len(subsets[sname])
        if rn in subset_roi[sname]:
            ax.plot(t, subset_roi[sname][rn], color=colors_list[sname],
                    lw=2.0 if sname in ['First 10 trials','All 200 trials'] else 1.0,
                    ls=ls_list[sname],
                    label=f'{sname} (n={n_tr})', alpha=0.85)

    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
        ax.text((t1+t2)/2, ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.12,
                cn, fontsize=7, ha='center', color=cc, fontweight='bold', alpha=0.5)

    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_ylabel(f'{rn}\n(nAm)', fontsize=10, fontweight='bold', color='#444')
    ax.tick_params(labelsize=7, colors='#888')
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.92, rn, transform=ax.transAxes, fontsize=12,
            fontweight='bold', color='#222', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.85))
    if ri == 0:
        ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.85, edgecolor='#ddd')

axes[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
fig.suptitle('Frontal Source Activity: First 10/30 trials vs Last 10/30 vs All 200 (Session 2)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '11_frontal_first10_vs_all.png'), dpi=200,
            facecolor='white', bbox_inches='tight')
plt.close(fig)
print('Saved: 11_frontal_first10_vs_all.png')

# ====== FIGURE 2: Frontal focus with ACC sub-region ======
# Also extract ACC separately for comparison
acc_labels = [l for l in labels if 'caudalanteriorcingulate' in l.name or
              'rostralanteriorcingulate' in l.name]
print(f'ACC labels found: {len(acc_labels)}')

acc_roi = {}
for sname, ep_sub in subsets.items():
    if len(ep_sub) < 5: continue
    ev = ep_sub.average()
    stc = subset_stc[sname]
    sr = stc.extract_label_time_course(acc_labels, src, mode='mean')
    if isinstance(sr, np.ndarray) and sr.ndim == 1: sr = [sr]
    acc_roi[sname] = np.mean(sr, axis=0) if (sr is not None and len(sr)>0) else np.zeros_like(t)

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig2.patch.set_facecolor('white')

# Top: Full Frontal ROI
ax = axes2[0]; ax.set_facecolor('#FAFAFA')
ax.set_title('Frontal ROI (18 labels, full frontal lobe)', fontsize=12, fontweight='bold')
for sname in ['First 10 trials','All 200 trials','Last 10 trials']:
    if sname not in subset_roi: continue
    n_tr = len(subsets[sname])
    ax.plot(t, subset_roi[sname]['Frontal'], color=colors_list[sname],
            lw=2.0, label=f'{sname} (n={n_tr})')
# Add baseline from all 200 trials
ax.axhline(0, color='#999', lw=0.5)
ax.axvline(0, color='#333', ls='--', lw=0.8)
for cn, t1, t2, cc in ERP_WINDOWS:
    ax.axvspan(t1, t2, alpha=0.06, color=cc)
ax.set_ylabel('nAm', fontsize=10)
ax.tick_params(labelsize=8)
ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.legend(fontsize=9, loc='upper right')

# Bottom: ACC only
ax = axes2[1]; ax.set_facecolor('#FAFAFA')
ax.set_title('ACC only (anterior cingulate, 4 labels)', fontsize=12, fontweight='bold', color='#E74C3C')
for sname in ['First 10 trials','All 200 trials','Last 10 trials']:
    if sname not in acc_roi: continue
    n_tr = len(subsets[sname])
    ax.plot(t, acc_roi[sname], color=colors_list[sname],
            lw=2.0, label=f'{sname} (n={n_tr})')
ax.axhline(0, color='#999', lw=0.5)
ax.axvline(0, color='#333', ls='--', lw=0.8)
for cn, t1, t2, cc in ERP_WINDOWS:
    ax.axvspan(t1, t2, alpha=0.06, color=cc)
ax.set_ylabel('nAm', fontsize=10)
ax.set_xlabel('Time (s)', fontsize=10)
ax.tick_params(labelsize=8)
ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.legend(fontsize=9, loc='upper right')

fig2.suptitle('Frontal Sub-region Comparison: Full Frontal ROI vs ACC (Session 2)',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '12_frontal_vs_acc.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig2)
print('Saved: 12_frontal_vs_acc.png')

# ====== Numerical comparison for Frontal ======
print(f'\n{"="*80}')
print(f'  Frontal ROI source activity (nAm) -- first 10 vs all 200 vs last 10')
print(f'{"="*80}')
hdr = f'{"Window":<8} {"First 10":>10} {"All 200":>10} {"Last 10":>10} {"First-All":>10}'
print(f'  {hdr}')
print(f'  {"-"*42}')
for cn, t1, t2, cc in ERP_WINDOWS:
    mask = (t >= t1) & (t <= t2)
    f10 = subset_roi['First 10 trials']['Frontal'][mask].mean()
    all_ = subset_roi['All 200 trials']['Frontal'][mask].mean()
    l10 = subset_roi['Last 10 trials']['Frontal'][mask].mean()
    diff = f10 - all_
    print(f'  {cn:<8} {f10:+10.3f} {all_:+10.3f} {l10:+10.3f} {diff:+10.3f}')

# Also ACC
print(f'\n  ACC source activity (nAm):')
print(f'  {hdr}')
print(f'  {"-"*42}')
for cn, t1, t2, cc in ERP_WINDOWS:
    mask = (t >= t1) & (t <= t2)
    f10 = acc_roi['First 10 trials'][mask].mean()
    all_ = acc_roi['All 200 trials'][mask].mean()
    l10 = acc_roi['Last 10 trials'][mask].mean()
    diff = f10 - all_
    print(f'  {cn:<8} {f10:+10.3f} {all_:+10.3f} {l10:+10.3f} {diff:+10.3f}')

print(f'\n  First 10 vs All 200: positive diff = frontal activity decreases with practice')
print(f'  (supports "cognitive economization" hypothesis)')
print('\nDone!')
