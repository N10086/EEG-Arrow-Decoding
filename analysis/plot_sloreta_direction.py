#!/usr/bin/env python3
"""
Per-direction sLORETA source time courses — 4 brain regions × 4 arrow directions.
"""
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
DATA_PATH = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'

DIRECTIONS = ['Up', 'Down', 'Left', 'Right']
DIR_MAP = {2.0001: 0, 2.0002: 1, 2.0003: 2, 2.0004: 3}
DIR_COLORS = ['#4A72C4', '#E8833A', '#5CB85C', '#9B59B6']
DIR_LS = ['-', '--', '-.', ':']

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

# ====== Forward model (shared) ======
print('Setting up fsaverage forward model...')
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

# ====== Load & filter ======
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

# ====== MNE Raw ======
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
t = epochs_all.times
print(f'Total: {len(epochs_all)} epochs')

# ====== Forward & inverse from ALL trials ======
fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem,
                                 meg=False, eeg=True, verbose=False)
cov = mne.compute_covariance(epochs_all, tmax=0.0, method='empirical', verbose=False)
inv = make_inverse_operator(epochs_all.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)

# ====== Per-direction epoch groups ======
epochs_dir = {}
for di, direction in enumerate(DIRECTIONS):
    idx = epochs_all.events[:,-1] == di+1
    epochs_dir[direction] = epochs_all[idx]
    print(f'  {direction}: {np.sum(idx)} trials')

# ====== Apply inverse per direction ======
roi_dir = {}   # roi_name -> direction_name -> time course
dir_n = {}     # direction_name -> trial count
for direction in DIRECTIONS:
    ep = epochs_dir[direction]
    if len(ep) < 10:
        print(f'  WARNING: {direction} too few trials ({len(ep)})')
        continue
    dir_n[direction] = len(ep)
    evoked = ep.average()
    stc = apply_inverse(evoked, inv, lambda2=1/9.0, method='sLORETA',
                         pick_ori=None, verbose=False)
    for rn in ROI_ORDER:
        sr = stc.extract_label_time_course(roi_labels[rn], src, mode='mean')
        if isinstance(sr, np.ndarray) and sr.ndim == 1: sr = [sr]
        val = np.mean(sr, axis=0) if (sr is not None and len(sr)>0) else np.zeros_like(t)
        if rn not in roi_dir:
            roi_dir[rn] = {}
        roi_dir[rn][direction] = val

# ====== FIGURE 1: Source space — 4 ROIs × 4 directions ======
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes[ri]; ax.set_facecolor('#FAFAFA')

    for di, direction in enumerate(DIRECTIONS):
        if direction in roi_dir.get(rn, {}):
            n = dir_n.get(direction, 0)
            ax.plot(t, roi_dir[rn][direction], color=DIR_COLORS[di],
                    lw=2.0, ls=DIR_LS[di], label=f'{direction} (n={n})')

    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
        ax.text((t1+t2)/2, ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.14,
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
        ax.legend(fontsize=8, loc='upper right', ncol=4, framealpha=0.85, edgecolor='#ddd')

axes[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
fig.suptitle('sLORETA Source Activity by Direction — 4 Brain Regions (Session 2)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '13_sloreta_direction_sources.png'), dpi=200,
            facecolor='white', bbox_inches='tight')
plt.close(fig)
print('Saved: 13_sloreta_direction_sources.png')

# ====== Scalp ERP by direction (for comparison) ======
scalp_dir = {}
for direction in DIRECTIONS:
    ep = epochs_dir[direction]
    if len(ep) < 10: continue
    scalp_dir[direction] = ep.get_data().mean(axis=0)

SCALP_ROIS = {
    'Frontal': [7,4,14], 'Central': [2,6,5],
    'Parietal': [9,10,12], 'Occipital': [15,1,8],
}

# ====== FIGURE 2: Scalp ERP by direction ======
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig2.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes2[ri]; ax.set_facecolor('#FAFAFA')
    chs = SCALP_ROIS[rn]
    ch_idx = [HW_CHS.index(c) for c in chs]

    for di, direction in enumerate(DIRECTIONS):
        if direction in scalp_dir:
            n = dir_n.get(direction, 0)
            erp = scalp_dir[direction][ch_idx].mean(axis=0)
            ax.plot(t, erp, color=DIR_COLORS[di], lw=2.0, ls=DIR_LS[di],
                    label=f'{direction} (n={n})')

    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
        ax.text((t1+t2)/2, ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.14,
                cn, fontsize=7, ha='center', color=cc, fontweight='bold', alpha=0.5)

    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_ylabel(f'{rn}\n(µV)', fontsize=10, fontweight='bold', color='#444')
    ax.tick_params(labelsize=7, colors='#888')
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.92, rn, transform=ax.transAxes, fontsize=12,
            fontweight='bold', color='#222', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.85))

    if ri == 0:
        ax.legend(fontsize=8, loc='upper right', ncol=4, framealpha=0.85, edgecolor='#ddd')

axes2[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
fig2.suptitle('Scalp ERP by Direction — 4 Brain Regions (Session 2)',
              fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '14_scalp_direction_erp.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig2)
print('Saved: 14_scalp_direction_erp.png')

# ====== FIGURE 3: Focus on P3 — bar comparison ======
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 5))
fig3.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes3[ri]; ax.set_facecolor('#FAFAFA')
    vals = []
    for direction in DIRECTIONS:
        if direction in roi_dir.get(rn, {}):
            mask = (t >= 0.300) & (t <= 0.500)
            vals.append(roi_dir[rn][direction][mask].mean())
        else:
            vals.append(0)
    bars = ax.bar(DIRECTIONS, vals, color=DIR_COLORS, edgecolor='#333', lw=0.5, alpha=0.85)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_title(f'{rn}\nP3 (300-500ms)', fontsize=11, fontweight='bold', color='#222')
    ax.set_ylabel('nAm', fontsize=9, color='#555')
    ax.tick_params(labelsize=8, colors='#888')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

fig3.suptitle('P3 Source Amplitude by Direction (sLORETA, Session 2)',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, '15_p3_direction_bar.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig3)
print('Saved: 15_p3_direction_bar.png')

# ====== FIGURE 4: Topographic map of P3 scalp by direction ======
# Use MNE's built-in topomap
from mne.viz import plot_evoked_topo
try:
    evokeds = []
    for direction in DIRECTIONS:
        ep = epochs_dir[direction]
        if len(ep) < 10: continue
        ev = ep.average()
        ev.comment = direction
        evokeds.append(ev)
    fig4 = plot_evoked_topo(evokeds, layout=None, fig_background='white',
                            title='P3 (300-500ms) — Scalp Topography by Direction')
    fig4.savefig(os.path.join(OUT_DIR, '16_p3_topo_by_direction.png'), dpi=200,
                 facecolor='white', bbox_inches='tight')
    plt.close(fig4)
    print('Saved: 16_p3_topo_by_direction.png')
except Exception as e:
    print(f'  Skipping topomap: {e}')

# ====== Numerical summary ======
print(f'\n{"="*80}')
print(f'  sLORETA source activity by direction (nAm) — P3 window (300-500ms)')
print(f'{"="*80}')
hdr = f'{"Region":<12} {"Up":>10} {"Down":>10} {"Left":>10} {"Right":>10} {"MaxDiff":>10}'
print(f'  {hdr}')
print(f'  {"-"*56}')
for rn in ROI_ORDER:
    vals = []
    for direction in DIRECTIONS:
        if direction in roi_dir.get(rn, {}):
            mask = (t >= 0.300) & (t <= 0.500)
            vals.append(roi_dir[rn][direction][mask].mean())
        else:
            vals.append(np.nan)
    max_diff = np.nanmax(vals) - np.nanmin(vals)
    print(f'  {rn:<12} {vals[0]:+10.3f} {vals[1]:+10.3f} {vals[2]:+10.3f} {vals[3]:+10.3f} {max_diff:+10.3f}')

# For P2 also
print(f'\n{"="*80}')
print(f'  sLORETA source activity by direction (nAm) — P2 window (200-300ms)')
print(f'{"="*80}')
print(f'  {hdr}')
print(f'  {"-"*56}')
for rn in ROI_ORDER:
    vals = []
    for direction in DIRECTIONS:
        if direction in roi_dir.get(rn, {}):
            mask = (t >= 0.200) & (t <= 0.300)
            vals.append(roi_dir[rn][direction][mask].mean())
        else:
            vals.append(np.nan)
    max_diff = np.nanmax(vals) - np.nanmin(vals)
    print(f'  {rn:<12} {vals[0]:+10.3f} {vals[1]:+10.3f} {vals[2]:+10.3f} {vals[3]:+10.3f} {max_diff:+10.3f}')

print(f'\nDone! Files in {OUT_DIR}')
