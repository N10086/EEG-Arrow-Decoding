#!/usr/bin/env python3
"""
sLORETA source localization - Session 2 EEG data.
12 channels -> forward model (fsaverage) -> inverse -> 4 ROI source time courses
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

# Chinese font
for f in ['Microsoft YaHei', 'SimHei']:
    try:
        fp = fm.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['axes.unicode_minus'] = False
        break
    except:
        pass

# ===== Config =====
FS = 500.0
GAIN = 6.0
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6

DATA_PATH = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
SUBJECTS_DIR = r'C:\Users\Zibo\mne_data\MNE-fsaverage-data'
SUBJECT = 'fsaverage'
os.makedirs(OUT_DIR, exist_ok=True)

ERP_WINDOWS = [
    ('P1', 0.080, 0.130, '#27ae60'),
    ('N1', 0.140, 0.200, '#7f8c8d'),
    ('P2', 0.200, 0.300, '#e67e22'),
    ('P3', 0.300, 0.500, '#9B59B6'),
]

CH_MAP = {
    1: 'Oz', 2: 'C3', 4: 'Fz', 5: 'C4', 6: 'Cz', 7: 'F3',
    8: 'O2', 9: 'P3', 10: 'Pz', 12: 'P4', 14: 'F4', 15: 'O1'
}
HW_CHS = sorted(CH_MAP.keys())
MNE_CH_NAMES = [CH_MAP[c] for c in HW_CHS]

SCALP_ROIS = {
    'Frontal': [7, 4, 14],
    'Central': [2, 6, 5],
    'Parietal': [9, 10, 12],
    'Occipital': [15, 1, 8],
}
ROI_ORDER = ['Frontal', 'Central', 'Parietal', 'Occipital']

DK_ROIS = {
    'Frontal': ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal',
                'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole',
                'parsopercularis', 'parsorbitalis', 'parstriangularis'],
    'Central': ['precentral', 'postcentral'],
    'Parietal': ['superiorparietal', 'inferiorparietal', 'precuneus',
                 'supramarginal', 'isthmuscingulate'],
    'Occipital': ['lateraloccipital', 'cuneus', 'pericalcarine', 'lingual'],
}
HEMIS = ['lh', 'rh']

# ========== 1. Load & filter ==========
print('Loading data...')
with open(DATA_PATH) as f:
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

ons = sorted([i for k in [2.0001, 2.0002, 2.0003, 2.0004]
              for i in np.where(np.abs(m - k) < 5e-5)[0]])
print(f'Found {len(ons)} events')

# ========== 2. MNE Raw ==========
print('Creating MNE Raw...')
info = mne.create_info(ch_names=MNE_CH_NAMES, sfreq=FS, ch_types=['eeg']*12)
raw = mne.io.RawArray(filt_uv / 1e6, info)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)

# ========== 3. Events ==========
event_map = {2.0001: 1, 2.0002: 2, 2.0003: 3, 2.0004: 4}
events_list = []
for idx in ons:
    for k, v in event_map.items():
        if abs(m[idx] - k) < 5e-5:
            events_list.append([idx, 0, v])
            break
events = np.array(events_list, dtype=int)
event_id = {d: i+1 for i, d in enumerate(['Up', 'Down', 'Left', 'Right'])}
print(f'Events: {events.shape}')

# ========== 4. Epochs ==========
print('Epoching...')
tmin, tmax = -0.2, 0.8
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(-0.2, 0.0), preload=True,
                    reject={'eeg': 100e-6},
                    proj=True, verbose=False)
print(f'Kept: {len(epochs)}/{len(events)} epochs')

# ========== 5. Evoked ==========
evoked = epochs.average()
t = evoked.times * 1.0
print(f'Evoked: {evoked.data.shape}')

# ========== 6. Forward model ==========
print('Forward model...')
src = mne.setup_source_space(SUBJECT, spacing='ico3',
                             subjects_dir=SUBJECTS_DIR, add_dist=False)
print(f'  Sources: {sum(s["nuse"] for s in src)}')

model = mne.make_bem_model(SUBJECT, ico=4,
                           conductivity=[0.3, 0.006, 0.3],
                           subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(raw.info, trans='fsaverage',
                                 src=src, bem=bem,
                                 meg=False, eeg=True,
                                 verbose=False)
print(f'  Forward: {fwd["nchan"]} sensors -> {fwd["nsource"]} sources')

# ========== 7. Noise covariance ==========
print('Noise covariance...')
cov = mne.compute_covariance(epochs, tmax=0.0, method='empirical', verbose=False)
print(f'  Cov: {cov.data.shape}')

# ========== 8. sLORETA inverse ==========
print('sLORETA inverse...')
inv = make_inverse_operator(evoked.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)
print(f'  Inverse operator: {inv}')

stc = apply_inverse(evoked, inv, lambda2=1/9.0, method='sLORETA',
                    pick_ori=None, verbose=False)
print(f'  STC: {stc.shape}')

# ========== 9. Extract ROI source time courses ==========
print('Extracting ROI source time courses...')
labels = mne.read_labels_from_annot(SUBJECT, 'aparc',
                                     subjects_dir=SUBJECTS_DIR,
                                     verbose=False)
print(f'  Loaded {len(labels)} labels')

roi_source = {}
for roi_name, dk_names in DK_ROIS.items():
    sel_labels = []
    for dk in dk_names:
        for hemi in HEMIS:
            full_name = f'{dk}-{hemi}'
            lbl = [l for l in labels if l.name == full_name]
            sel_labels.extend(lbl)

    if not sel_labels:
        print(f'  WARNING: no labels for {roi_name}')
        continue
    src_result = stc.extract_label_time_course(sel_labels, src, mode='mean')
    if isinstance(src_result, np.ndarray) and src_result.ndim == 1:
        src_result = [src_result]
    if src_result is not None and len(src_result) > 0:
        mean_tc = np.mean(src_result, axis=0)
        roi_source[roi_name] = mean_tc
        print(f'  {roi_name}: {len(sel_labels)} labels -> {mean_tc.shape}')

# ========== 10. Scalp ROI ERPs for comparison ==========
scalp_erp = {}
for rn, chs in SCALP_ROIS.items():
    idx = [HW_CHS.index(c) for c in chs]
    scalp_erp[rn] = epochs.get_data()[:, idx, :].mean(axis=(0, 1))

# ========== 11. FIGURE 1: Scalp vs Source side-by-side ==========
print('Plotting comparison...')
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    # Left: scalp
    ax = axes[ri, 0]
    ax.set_facecolor('#FAFAFA')
    ax.plot(t, scalp_erp[rn], '#2c3e50', lw=2.0)
    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_title(f'Scalp ERP: {rn}', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('uV', fontsize=9, color='#555')
    ax.tick_params(labelsize=7, colors='#888')
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # Right: source
    ax = axes[ri, 1]
    ax.set_facecolor('#FAFAFA')
    if rn in roi_source:
        ax.plot(t, roi_source[rn], '#E74C3C', lw=2.0)
        ax.set_title(f'Source: {rn} (sLORETA)', fontsize=11, fontweight='bold', color='#E74C3C')
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='#999')
    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_ylabel('nAm', fontsize=9, color='#555')
    ax.tick_params(labelsize=7, colors='#888')
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

for ax in axes[-1, :]:
    ax.set_xlabel('Time (s)', fontsize=10, color='#555')

fig.suptitle('Scalp ERP (uV) vs sLORETA Source Activity (nAm) - Session 2',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, '09_sloreta_vs_scalp.png')
fig.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ========== 12. FIGURE 2: Source activity alone ==========
fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig2.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes2[ri]
    ax.set_facecolor('#FAFAFA')

    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.07, color=cc)
        ax.text((t1+t2)/2, ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.12,
                cn, fontsize=7.5, ha='center', color=cc, fontweight='bold', alpha=0.5)

    if rn in roi_source:
        ax.plot(t, roi_source[rn], '#E74C3C', lw=2.5)
        ax.fill_between(t, roi_source[rn]*0, roi_source[rn],
                        where=roi_source[rn] > 0, color='#E74C3C', alpha=0.08)
        ax.fill_between(t, roi_source[rn]*0, roi_source[rn],
                        where=roi_source[rn] < 0, color='#2980B9', alpha=0.08)
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                transform=ax.transAxes)

    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_ylabel(f'{rn}\n(nAm)', fontsize=9, fontweight='bold', color='#444')
    ax.tick_params(labelsize=7, colors='#888')
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    ax.text(0.02, 0.92, f'{rn}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='#333', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='#ddd', alpha=0.85))

axes2[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
fig2.suptitle('sLORETA Source Time Courses - 4 Brain Regions (Session 2)',
              fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, '09_sloreta_source_tc.png')
fig2.savefig(out2, dpi=200, facecolor='white', bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {out2}')

# ========== 13. FIGURE 3: Normalized overlay ==========
fig3, axes3 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig3.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes3[ri]
    ax.set_facecolor('#FAFAFA')

    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)

    if rn in roi_source:
        s = roi_source[rn]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10) * 2 - 1
        ax.plot(t, s_norm, '#E74C3C', lw=2.0, alpha=0.9, label='sLORETA source')

    s2 = scalp_erp[rn]
    s2_norm = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-10) * 2 - 1
    ax.plot(t, s2_norm, '#2c3e50', lw=2.0, alpha=0.9, ls='--', label='Scalp ERP')

    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_ylabel(f'{rn}\n(norm.)', fontsize=9, fontweight='bold', color='#444')
    ax.tick_params(labelsize=7, colors='#888')
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    if ri == 0:
        ax.legend(fontsize=9, loc='upper right')

axes3[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
fig3.suptitle('Normalized Overlay: Scalp ERP vs sLORETA Source (Session 2)',
              fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, '09_sloreta_overlay.png')
fig3.savefig(out3, dpi=200, facecolor='white', bbox_inches='tight')
plt.close(fig3)
print(f'Saved: {out3}')

# ========== 14. Numerical comparison ==========
print('\n' + '='*70)
print('   Numerical comparison: Scalp ERP vs sLORETA source')
print('='*70)
for rn in ROI_ORDER:
    print(f'\n  {rn}:')
    if rn not in roi_source:
        print('    Not available')
        continue
    for cn, t1, t2, _ in ERP_WINDOWS:
        mask = (t >= t1) & (t <= t2)
        scalp_mean = scalp_erp[rn][mask].mean()
        src_mean = roi_source[rn][mask].mean()
        print(f'    {cn:5s} ({t1*1000:.0f}-{t2*1000:.0f}ms): '
              f'Scalp={scalp_mean:+.4f} uV  Source={src_mean:+.4f} nAm')

print('\nDone!')
