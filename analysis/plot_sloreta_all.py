#!/usr/bin/env python3
"""sLORETA on all 5 sessions — source-level ROI time courses."""
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

FS = 500.0; GAIN = 6.0
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6
SUBJECTS_DIR = r'C:\Users\Zibo\mne_data\MNE-fsaverage-data'
SUBJECT = 'fsaverage'
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'

PATHS = [
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-35-51\OpenBCI-RAW-2026-07-07_10-35-51.txt',
     'S1 (07-07 10h35)', '#4A72C4'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt',
     'S2 (07-07 10h47)', '#E8833A'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_08-54-11\OpenBCI-RAW-2026-07-08_08-54-11.txt',
     'S3 (07-08 08h54)', '#5CB85C'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_09-04-45\OpenBCI-RAW-2026-07-08_09-04-45.txt',
     'S4 (07-08 09h04)', '#9B59B6'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-09_09-02-54\OpenBCI-RAW-2026-07-09_09-02-54.txt',
     'S5 (07-09 09h02)', '#D94F70'),
]

ERP_WINDOWS = [
    ('P1', 0.080, 0.130, '#27ae60'),
    ('N1', 0.140, 0.200, '#7f8c8d'),
    ('P2', 0.200, 0.300, '#e67e22'),
    ('P3', 0.300, 0.500, '#9B59B6'),
]

CH_MAP = {1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',
          8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS = sorted(CH_MAP.keys())
MNE_NAMES = [CH_MAP[c] for c in HW_CHS]

DK_ROIS = {
    'Frontal': ['superiorfrontal','rostralmiddlefrontal','caudalmiddlefrontal',
                'lateralorbitofrontal','medialorbitofrontal','frontalpole',
                'parsopercularis','parsorbitalis','parstriangularis'],
    'Central': ['precentral','postcentral'],
    'Parietal': ['superiorparietal','inferiorparietal','precuneus',
                 'supramarginal','isthmuscingulate'],
    'Occipital': ['lateraloccipital','cuneus','pericalcarine','lingual'],
}
ROI_ORDER = ['Frontal','Central','Parietal','Occipital']
HEMIS = ['lh','rh']

# Precompute common forward model
print('Setting up fsaverage forward model...')
src = mne.setup_source_space(SUBJECT, spacing='ico3',
                             subjects_dir=SUBJECTS_DIR, add_dist=False)
model = mne.make_bem_model(SUBJECT, ico=4, conductivity=[0.3,0.006,0.3],
                            subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)
labels = mne.read_labels_from_annot(SUBJECT, 'aparc', subjects_dir=SUBJECTS_DIR, verbose=False)

# Build ROI label list
roi_labels = {}
for rn, dk_names in DK_ROIS.items():
    sel = []
    for dk in dk_names:
        for hemi in HEMIS:
            lbl = [l for l in labels if l.name == f'{dk}-{hemi}']
            sel.extend(lbl)
    roi_labels[rn] = sel

def process_session(path):
    """Returns (roi_src_dict, t_array, n_trials)"""
    with open(path) as f:
        lines = f.readlines()
    data, markers = [], []
    for line in lines[5:]:
        p = line.strip().split(',')
        if len(p) > 33:
            try:
                data.append([float(p[i]) for i in range(1, 17)])
                markers.append(float(p[32]))
            except:
                pass
    d = np.array(data, dtype=np.float64).T
    m = np.array(markers)
    sel = np.array([d[c-1] for c in HW_CHS])

    bp = sg.butter(4, [1/250, 45/250], btype='band', output='sos')
    notch = sg.iirnotch(50/250, 30)
    filt = np.zeros_like(sel)
    for ch in range(12):
        dm = sel[ch] - sel[ch].mean()
        ts = sg.sosfiltfilt(bp, dm)
        filt[ch] = sg.filtfilt(*notch, ts)
    filt_uv = filt * SCALE

    ons = sorted([i for k in [2.0001,2.0002,2.0003,2.0004]
                  for i in np.where(np.abs(m - k) < 5e-5)[0]])

    info = mne.create_info(ch_names=MNE_NAMES, sfreq=FS, ch_types=['eeg']*12)
    raw = mne.io.RawArray(filt_uv / 1e6, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)

    ev_list = []
    for k, v in {2.0001:1,2.0002:2,2.0003:3,2.0004:4}.items():
        for idx in ons:
            if abs(m[idx]-k) < 5e-5:
                ev_list.append([idx,0,v])
    events = np.array(ev_list, dtype=int)
    event_id = {d:i+1 for i,d in enumerate(['Up','Down','Left','Right'])}

    epochs = mne.Epochs(raw, events, event_id, -0.2, 0.8,
                        baseline=(-0.2,0.0), preload=True,
                        reject={'eeg':100e-6}, proj=True, verbose=False)
    n = len(epochs)
    if n == 0:
        return None, None, 0

    evoked = epochs.average()
    t = evoked.times

    fwd = mne.make_forward_solution(raw.info, trans='fsaverage',
                                     src=src, bem=bem,
                                     meg=False, eeg=True, verbose=False)
    cov = mne.compute_covariance(epochs, tmax=0.0, method='empirical', verbose=False)
    inv = make_inverse_operator(evoked.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)
    stc = apply_inverse(evoked, inv, lambda2=1/9.0, method='sLORETA',
                        pick_ori=None, verbose=False)

    roi_src = {}
    for rn in ROI_ORDER:
        sr = stc.extract_label_time_course(roi_labels[rn], src, mode='mean')
        if isinstance(sr, np.ndarray) and sr.ndim == 1:
            sr = [sr]
        if isinstance(sr, np.ndarray) and sr.ndim == 1:
            sr = [sr]
        roi_src[rn] = np.mean(sr, axis=0) if (sr is not None and len(sr) > 0) else np.zeros_like(t)

    return roi_src, t, n

# ========== Process ==========
results = []  # (label, n, color, roi_src, t)
for path, label, color in PATHS:
    roi_src, t, n = process_session(path)
    if roi_src is not None:
        results.append((label, n, color, roi_src, t))
        print(f'{label}: {n} trials')

if not results:
    print('No data!'); exit()

t_ref = results[0][4]

# ========== Fig 1: All sessions ==========
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
fig.patch.set_facecolor('white')
for ri, rn in enumerate(ROI_ORDER):
    ax = axes[ri]; ax.set_facecolor('#FAFAFA')
    for label, n, color, roi_src, t in results:
        ax.plot(t, roi_src[rn], color=color, lw=1.5, label=f'{label} (n={n})')
    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
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
fig.suptitle('sLORETA Source Time Courses -- All 5 Sessions x 4 ROIs',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '10_sloreta_all_sessions.png'), dpi=200,
            facecolor='white', bbox_inches='tight')
plt.close(fig)
print('Saved: 10_sloreta_all_sessions.png')

# ========== Fig 2: Grand average ==========
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
fig2.patch.set_facecolor('white')
for ri, rn in enumerate(ROI_ORDER):
    ax = axes2[ri]; ax.set_facecolor('#FAFAFA')
    all_tc = []
    for label, n, color, roi_src, t in results:
        tc = roi_src[rn]
        all_tc.append(tc)
        ax.plot(t, tc, color=color, lw=0.6, alpha=0.4)
    if all_tc:
        grand = np.mean(all_tc, axis=0)
        se = np.std(all_tc, axis=0) / np.sqrt(len(all_tc))
        ax.plot(t_ref, grand, color='#111', lw=3.0, label='Grand avg')
        ax.fill_between(t_ref, grand-2*se, grand+2*se, color='#111', alpha=0.1)
    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
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
        ax.legend(fontsize=9, loc='upper right')
axes2[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
fig2.suptitle('Grand Average sLORETA Source (5 sessions) -- Error bar = +/-2 SE',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '10_sloreta_grand_avg.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig2)
print('Saved: 10_sloreta_grand_avg.png')

# ========== Table ==========
print('\n' + '='*80)
print('  sLORETA source amplitude (nAm) -- mean within each ERP window')
print('='*80)
hdr = f'{"Window":<8} {"Frontal":>8} {"Central":>8} {"Parietal":>8} {"Occipital":>8}'

# Per session
for label, n, color, roi_src, t in results:
    print(f'\n  {label} (n={n}):')
    print(f'  {hdr}')
    print('  ' + '-'*42)
    for cn, t1, t2, cc in ERP_WINDOWS:
        mask = (t >= t1) & (t <= t2)
        vals = [f'{roi_src[rn][mask].mean():+7.3f}' for rn in ROI_ORDER]
        print(f'  {cn:<8} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}')

# Grand average
print(f'\n  GRAND AVERAGE (5 sessions):')
print(f'  {hdr}')
print('  ' + '-'*42)
for cn, t1, t2, cc in ERP_WINDOWS:
    mask = (t_ref >= t1) & (t_ref <= t2)
    gvals = []
    for rn in ROI_ORDER:
        sv = [roi_src[rn][mask].mean() for _, _, _, roi_src, t in results]
        gvals.append(f'{np.mean(sv):+7.3f}+-{np.std(sv):.3f}')
    print(f'  {cn:<8} {gvals[0]:>14} {gvals[1]:>14} {gvals[2]:>14} {gvals[3]:>14}')

print(f'\nAll figures in {OUT_DIR}')
print('Done!')
