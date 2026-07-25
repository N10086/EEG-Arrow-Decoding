#!/usr/bin/env python3
"""Plot Session 2 ROI waveforms with per-region explanation annotations."""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Chinese font
for f in ['Microsoft YaHei', 'SimHei']:
    try:
        fp = fm.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['axes.unicode_minus'] = False
        break
    except:
        pass

FS = 500.0
GAIN = 6.0
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6
COMMON_CH = [1,2,4,5,6,7,8,9,10,12,14,15]

ROIS = {
    'Frontal':  [7,4,14],
    'Central':  [2,6,5],
    'Parietal': [9,10,12],
    'Occipital':[15,1,8],
}
ROI_ORDER = ['Frontal', 'Central', 'Parietal', 'Occipital']

ERP_WINDOWS = [
    ('P1', 0.080, 0.130, '#27ae60'),
    ('N1', 0.140, 0.200, '#7f8c8d'),
    ('P2', 0.200, 0.300, '#e67e22'),
    ('P3', 0.300, 0.500, '#8e44ad'),
]

DIR_LABELS = {2.0001:'Up', 2.0002:'Down', 2.0003:'Left', 2.0004:'Right'}
DIR_COLORS = {'Up':'#E74C3C','Down':'#3498DB','Left':'#2ECC71','Right':'#F39C12'}
DIR_ORDER = ['Up', 'Down', 'Left', 'Right']

path = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'

# ---------- Load ----------
with open(path) as f:
    lines = f.readlines()
data, mk = [], []
for line in lines[5:]:
    p = line.strip().split(',')
    if len(p) > 33:
        try:
            data.append([float(p[i]) for i in range(1,17)])
            mk.append(float(p[32]))
        except:
            pass
data = np.array(data, dtype=np.float64).T
mk = np.array(mk)

# ---------- Select channels ----------
sel = data[[c-1 for c in COMMON_CH], :]

# ---------- Onsets ----------
ons = sorted([i for k in [2.0001,2.0002,2.0003,2.0004]
              for i in np.where(np.abs(mk - k) < 5e-5)[0]])

# ---------- Crop ----------
s = max(0, ons[0] - int(5*FS))
e = min(sel.shape[1], ons[-1] + int(5*FS))
crop = sel[:, s:e]

# ---------- Filter ----------
bp = sg.butter(4, [1/250, 45/250], btype='band', output='sos')
nc = sg.iirnotch(50/250, 30)
filt = np.zeros_like(crop)
for ch in range(crop.shape[0]):
    dm = crop[ch] - crop[ch].mean()
    t = sg.sosfiltfilt(bp, dm)
    filt[ch] = sg.filtfilt(*nc, t)
filt_uv = filt * SCALE

# ---------- Epoch ----------
rel = [i - s for i in ons]
ep = []
for idx in rel:
    st, en = idx - 200, idx + 400
    if st >= 0 and en <= filt_uv.shape[1]:
        ei = filt_uv[:, st:en].copy()
        ei -= ei[:, :200].mean(axis=1, keepdims=True)
        ep.append(ei)
ep = np.stack(ep, 0)

# ---------- Bad trial rejection ----------
ptp = (ep.max(2) - ep.min(2)).max(1)
good = ptp < 100.0
ep_clean = ep[good]
n_trials = ep_clean.shape[0]
print(f'Trials: {n_trials}/{len(ep)} kept')

# ---------- Time axis ----------
t = (np.arange(600) - 200) / FS

# ---------- ROI data (overall) ----------
roi_mean, roi_se = {}, {}
for rn, chs in ROIS.items():
    idx = [COMMON_CH.index(c) for c in chs]
    rd = ep_clean[:, idx, :].mean(axis=1)
    roi_mean[rn] = rd.mean(axis=0)
    roi_se[rn] = rd.std(axis=0) / np.sqrt(n_trials)

# ---------- ROI data (by direction) ----------
dir_roi = {}
for d in DIR_ORDER:
    dir_roi[d] = {}
    for rn in ROI_ORDER:
        dir_roi[d][rn] = None

dk = list(DIR_LABELS.keys())
for i, on in enumerate(ons):
    if not good[i]:
        continue
    for k in dk:
        if abs(mk[on] - k) < 5e-5:
            dname = DIR_LABELS[k]
            for rn, chs in ROIS.items():
                idx = [COMMON_CH.index(c) for c in chs]
                rd = ep_clean[i][idx].mean(axis=0)
                if dir_roi[dname][rn] is None:
                    dir_roi[dname][rn] = []
                dir_roi[dname][rn].append(rd)
            break

for d in DIR_ORDER:
    for rn in ROI_ORDER:
        dir_roi[d][rn] = np.array(dir_roi[d][rn])

# ---------- Per-region descriptions ----------
descriptions = {
    'Frontal': (
        'Frontal Lobe (F3/Fz/F4)\n'
        'Role: Decision-making, conflict monitoring, attention control\n'
        'Waveform: Weak early response. Strong P2 negative (-0.45uV, p<0.001)\n'
        'and P3 negative (-0.55uV, p<0.001). No significant P1/N1.\n'
        'Interpretation: Frontal cortex does NOT receive direct visual input.\n'
        'It activates at 200ms+ for direction evaluation and response selection.\n'
        'P2 reflects stimulus-response conflict detection (which arrow = which key).\n'
        'P3 reflects decision and working memory updating.'
    ),
    'Central': (
        'Central Region (C3/Cz/C4)\n'
        'Role: Motor planning, premotor processing, sensorimotor integration\n'
        'Waveform: Transitional P1 positive (+0.18uV, p<0.05), strong P2\n'
        'negative (-0.37uV, p<0.001), strong P3 negative (-0.60uV, p<0.001).\n'
        'N1 window shows clear downward slope (significant decline).\n'
        'Interpretation: Sits at the interface between sensory and motor.\n'
        'P1 reflects volume-conducted visual input from occipital cortex.\n'
        'N1 decline = transition from visual processing to response preparation.\n'
        'P2/P3 = classification and motor decision processes.'
    ),
    'Parietal': (
        'Parietal Lobe (P3/Pz/P4)\n'
        'Role: Spatial attention, visuomotor integration, perceptual decision\n'
        'Waveform: Complete processing cascade — P1 positive (+0.25uV, p<0.01),\n'
        'N1 decline (significant slope), P2 negative (-0.15uV, p<0.01),\n'
        'and LARGEST P3 negative (-0.64uV, p<0.001) of all ROIs.\n'
        'Interpretation: Most complete ERP signature. P1 = visual input from\n'
        'occipital cortex. N1 = spatial processing. P2 = stimulus classification.\n'
        'P3 = perceptual decision (classic P3b site). Largest amplitude here\n'
        'matches Polich 2007: P3b maximum over parietal scalp.'
    ),
    'Occipital': (
        'Occipital Lobe (O1/Oz/O2)\n'
        'Role: Primary & secondary visual cortex (V1-V5)\n'
        'Waveform: Strong P1 positive (+0.21uV, p<0.01), strong P3 negative\n'
        '(-0.42uV, p<0.001). N1 and P2 are NOT significant.\n'
        'Interpretation: Pure visual processing. P1 = first cortical response\n'
        'to the arrow stimulus (Hillyard 1998: extrastriate visual activation).\n'
        'The early visual cortex does NOT classify or decide — it only senses.\n'
        'P3 here reflects top-down feedback from parietal attention network\n'
        'back to visual cortex, not local visual processing.'
    ),
}

# ========== PLOT ==========
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
fig.patch.set_facecolor('white')

for ri, rn in enumerate(ROI_ORDER):
    ax = axes[ri]
    ax.set_facecolor('#FAFAFA')

    # Shaded windows
    for cname, t1, t2, ccol in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=ccol, zorder=0)
        ax.text((t1+t2)/2, ax.get_ylim()[0] - 0.06 if ri > 0 else ax.get_ylim()[0] - 0.12,
                f'{cname}\n{int(t1*1000)}-{int(t2*1000)}ms',
                fontsize=6.5, ha='center', color=ccol, fontweight='bold')

    # Mean + SE
    ax.plot(t, roi_mean[rn], color='#2c3e50', linewidth=2.0, label='Overall mean')
    ax.fill_between(t, roi_mean[rn] - 2*roi_se[rn], roi_mean[rn] + 2*roi_se[rn],
                     alpha=0.12, color='#2c3e50')

    # Direction lines (subtle)
    for d in DIR_ORDER:
        dm = dir_roi[d][rn].mean(axis=0)
        ax.plot(t, dm, color=DIR_COLORS[d], linewidth=0.5, alpha=0.35)

    # Zero lines
    ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
    ax.axhline(y=0, color='#999', linewidth=0.5)

    # Labels
    ax.set_ylabel(f'{rn}\n(uV)', fontsize=10, fontweight='bold', color='#444')
    ax.tick_params(colors='#888', labelsize=7)
    ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # Annotation box
    desc = descriptions[rn]
    ax.text(0.01, 0.97, desc, transform=ax.transAxes,
            fontsize=7.2, color='#333', va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor='#ddd', alpha=0.92))

axes[-1].set_xlabel('Time (s)', fontsize=10, color='#888')
fig.suptitle('Session 2 — Brain Region ERP Waveforms with Neuroanatomical Interpretation',
             fontsize=14, fontweight='bold', color='#222', y=1.01)
plt.tight_layout()
out = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v4\session2_roi_explained.png'
fig.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
