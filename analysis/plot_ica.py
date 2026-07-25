#!/usr/bin/env python3
"""
ICA decomposition on all 5 sessions.
Infomax ICA -> 12 components/session -> plot topography + time course -> assign to ROIs
"""
import numpy as np
from scipy import signal as sg
from sklearn.decomposition import FastICA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
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

FS = 500.0
GAIN = 6.0
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6

PATHS = [
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-35-51\OpenBCI-RAW-2026-07-07_10-35-51.txt',
     'S1-07-07_10h35'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt',
     'S2-07-07_10h47'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_08-54-11\OpenBCI-RAW-2026-07-08_08-54-11.txt',
     'S3-07-08_08h54'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_09-04-45\OpenBCI-RAW-2026-07-08_09-04-45.txt',
     'S4-07-08_09h04'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-09_09-02-54\OpenBCI-RAW-2026-07-09_09-02-54.txt',
     'S5-07-09_09h02'),
]
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5\ica'
os.makedirs(OUT_DIR, exist_ok=True)

CH_NAMES = {1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',
            8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
COMMON_CH = [1,2,4,5,6,7,8,9,10,12,14,15]
MNE_NAMES = [CH_NAMES[c] for c in COMMON_CH]

ERP_WINDOWS = [
    ('P1', 0.080, 0.130, '#27ae60'),
    ('N1', 0.140, 0.200, '#7f8c8d'),
    ('P2', 0.200, 0.300, '#e67e22'),
    ('P3', 0.300, 0.500, '#9B59B6'),
]

# Electrode positions for topography plotting (2D)
# Approximate 10-20 positions in 2D (normalized)
ELEC_POS = {
    'F3': (-0.4, 0.7), 'Fz': (0.0, 0.75), 'F4': (0.4, 0.7),
    'C3': (-0.45, 0.35), 'Cz': (0.0, 0.35), 'C4': (0.45, 0.35),
    'P3': (-0.4, 0.0), 'Pz': (0.0, 0.0), 'P4': (0.4, 0.0),
    'O1': (-0.2, -0.35), 'Oz': (0.0, -0.4), 'O2': (0.2, -0.35),
}


def load_and_process(path):
    """Load one session, filter, epoch, return epoch data in uV"""
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

    sel = np.array([d[c-1] for c in COMMON_CH])

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

    s = max(0, ons[0] - int(5*FS))
    e = min(filt_uv.shape[1], ons[-1] + int(5*FS))
    crop = filt_uv[:, s:e]

    rel = [i - s for i in ons]
    epochs = []
    for idx in rel:
        st, en = idx - 200, idx + 400
        if st >= 0 and en <= crop.shape[1]:
            ep = crop[:, st:en].copy()
            ep -= ep[:, :200].mean(axis=1, keepdims=True)
            epochs.append(ep)
    epochs = np.stack(epochs, 0)

    ptp = (epochs.max(2) - epochs.min(2)).max(1)
    good = ptp < 100.0
    epochs_clean = epochs[good]
    return epochs_clean, n_bad, good, m, ons


def load_and_process(path):
    """Load one session, filter, epoch, return epoch data in uV"""
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

    sel = np.array([d[c-1] for c in COMMON_CH])

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

    s = max(0, ons[0] - int(5*FS))
    e = min(filt_uv.shape[1], ons[-1] + int(5*FS))
    crop = filt_uv[:, s:e]

    rel = [i - s for i in ons]
    epochs = []
    for idx in rel:
        st, en = idx - 200, idx + 400
        if st >= 0 and en <= crop.shape[1]:
            ep = crop[:, st:en].copy()
            ep -= ep[:, :200].mean(axis=1, keepdims=True)
            epochs.append(ep)
    epochs = np.stack(epochs, 0)

    ptp = (epochs.max(2) - epochs.min(2)).max(1)
    good = ptp < 100.0
    epochs_clean = epochs[good]
    n_bad = epochs.shape[0] - epochs_clean.shape[0]
    return epochs_clean, n_bad, good, m, ons


def plot_topography(weights, ch_names, ax, title=''):
    """Plot 2D topography from ICA weights (12 channels)"""
    from scipy.interpolate import griddata
    pos = np.array([ELEC_POS[n] for n in ch_names])
    xi, yi = np.meshgrid(np.linspace(-0.7, 0.7, 100),
                          np.linspace(-0.6, 0.9, 100))
    zi = griddata(pos, weights, (xi, yi), method='cubic', fill_value=0)
    ax.contourf(xi, yi, zi, levels=30, cmap='RdBu_r', vmin=-abs(weights).max(), vmax=abs(weights).max())
    ax.scatter(pos[:, 0], pos[:, 1], c='k', s=30, zorder=5)
    for n, p in zip(ch_names, pos):
        ax.text(p[0], p[1]+0.06, n, ha='center', fontsize=6, color='k')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.6, 0.9)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')


# ========== Main ==========
for si, (path, label) in enumerate(PATHS):
    print(f'\n{"="*60}')
    print(f'  ICA: {label}')
    print(f'{"="*60}')

    epochs_uv, n_bad, good, m, ons = load_and_process(path)
    n_trials = epochs_uv.shape[0]
    print(f'  Trials: {n_trials} good, {n_bad} bad')

    # Reshape to 2D: channels x (trials*time) for ICA
    n_ch, n_t, n_pts = epochs_uv.shape[1], epochs_uv.shape[0], epochs_uv.shape[2]
    X = epochs_uv.transpose(1, 0, 2).reshape(n_ch, -1).T  # (trials*time, channels)

    # Infomax ICA via FastICA with logcosh (approximates Infomax)
    ica = FastICA(n_components=12, algorithm='deflation', fun='logcosh',
                  max_iter=1000, random_state=42)
    S = ica.fit_transform(X)  # (trials*time, 12)
    A = ica.mixing_  # (12, 12) mixing matrix

    # Reshape sources back to (trials, time, components)
    S_ep = S.reshape(n_trials, n_pts, 12).transpose(2, 0, 1)  # (12, trials, time)
    # Average across trials -> component ERP
    comp_erp = S_ep.mean(axis=1)  # (12, time)

    # Sort components by variance explained
    var_explained = np.var(S, axis=0)
    order = np.argsort(var_explained)[::-1]

    print(f'  Component variance (%):')
    for rank, idx in enumerate(order):
        print(f'    IC{idx+1:2d}: {var_explained[idx]/var_explained.sum()*100:5.1f}%')

    # ====== FIGURE 1: Component grid ======
    t = (np.arange(600) - 200) / FS
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'ICA Components (Infomax) — {label}  (n={n_trials} trials)',
                 fontsize=14, fontweight='bold', y=1.01, color='#222')

    for rank, idx in enumerate(order):
        ri, ci = divmod(rank, 3)
        ax = axes[ri, ci]
        ax.set_facecolor('#FAFAFA')

        # Topography
        ax_topo = ax.inset_axes([0.03, 0.58, 0.4, 0.4])
        plot_topography(A[idx, :], MNE_NAMES, ax_topo,
                        f'IC{idx+1} ({var_explained[idx]/var_explained.sum()*100:.0f}%)')

        # ERP
        ax.plot(t, comp_erp[idx], '#2c3e50', lw=1.5)
        for cn, t1, t2, cc in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.05, color=cc)
        ax.axvline(0, color='#999', ls='--', lw=0.5)
        ax.axhline(0, color='#999', lw=0.5)
        ax.set_xlim(-0.2, 0.8)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.2)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        # Baseline topography label
        ax.set_ylabel('a.u.', fontsize=7, color='#888')

        if ri == 3:
            ax.set_xlabel('Time (s)', fontsize=7, color='#888')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT_DIR, f'{label}_ica_components.png')
    fig.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')

    # ====== FIGURE 2: Map ICs to ROIs ======
    # Use topography to assign each IC to the ROI where it has max weight
    roi_channels = {
        'Frontal': ['F3', 'Fz', 'F4'],
        'Central': ['C3', 'Cz', 'C4'],
        'Parietal': ['P3', 'Pz', 'P4'],
        'Occipital': ['O1', 'Oz', 'O2'],
    }
    ic_to_roi = {}
    for idx in range(12):
        w = A[idx, :]
        roi_energy = {}
        for rn, chs in roi_channels.items():
            ch_idx = [MNE_NAMES.index(c) for c in chs]
            roi_energy[rn] = np.mean(w[ch_idx]**2)
        best_roi = max(roi_energy, key=roi_energy.get)
        ic_to_roi[idx] = best_roi

    # Group ICs by ROI and plot
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig2.patch.set_facecolor('white')
    fig2.suptitle(f'ICA Components grouped by ROI — {label}',
                  fontsize=14, fontweight='bold', y=1.01, color='#222')

    roi_colors = {'Frontal': '#E74C3C', 'Central': '#3498DB',
                  'Parietal': '#9B59B6', 'Occipital': '#27AE60'}
    ls_list = ['-', '--', '-.', ':', (0, (3,1,1,1))]

    for ri, rn in enumerate(['Frontal', 'Central', 'Parietal', 'Occipital']):
        ax = axes2[ri]
        ax.set_facecolor('#FAFAFA')

        ics_in_roi = [idx for idx, r in ic_to_roi.items() if r == rn]
        li = 0
        for idx in ics_in_roi:
            ax.plot(t, comp_erp[idx], color=roi_colors[rn],
                    lw=1.5, alpha=0.8, ls=ls_list[li % len(ls_list)],
                    label=f'IC{idx+1} ({var_explained[idx]/var_explained.sum()*100:.0f}%)')
            li += 1

        if not ics_in_roi:
            ax.text(0.5, 0.5, 'No IC assigned', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#999')

        for cn, t1, t2, cc in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.05, color=cc)
            ax.text((t1+t2)/2, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.92,
                    cn, fontsize=6.5, ha='center', color=cc, fontweight='bold', alpha=0.4)

        ax.axvline(0, color='#333', ls='--', lw=0.8)
        ax.axhline(0, color='#999', lw=0.5)
        ax.set_ylabel(f'{rn}\n(a.u.)', fontsize=10, fontweight='bold', color=roi_colors[rn])
        ax.tick_params(labelsize=7, colors='#888')
        ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        if ics_in_roi:
            ax.legend(fontsize=7, loc='upper right', ncol=min(len(ics_in_roi), 3))

    axes2[-1].set_xlabel('Time (s)', fontsize=10, color='#555')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path2 = os.path.join(OUT_DIR, f'{label}_ica_by_roi.png')
    fig2.savefig(out_path2, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved: {out_path2}')

    # ====== Print assignment ======
    print(f'\n  IC -> ROI assignment:')
    for idx in range(12):
        print(f'    IC{idx+1:2d} -> {ic_to_roi[idx]:10s}  '
              f'(var={var_explained[idx]/var_explained.sum()*100:.1f}%)')

    # ====== FIGURE 3: Reconstruct ROI time courses from ICA ======
    # For each ROI, take the IC with highest variance that maps to it
    # Back-project: reconstruct the ROI channel group from ICA
    # Take the top IC per ROI -> use its time course as "ROI source"
    selected_ics = {}
    for rn in ['Frontal', 'Central', 'Parietal', 'Occipital']:
        candidates = [(idx, var_explained[idx]) for idx, r in ic_to_roi.items() if r == rn]
        if candidates:
            best_idx = max(candidates, key=lambda x: x[1])[0]
            selected_ics[rn] = best_idx
            print(f'    Selected IC{best_idx+1} for {rn}')

    fig3, axes3 = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig3.patch.set_facecolor('white')
    fig3.suptitle(f'ICA-derived ROI source signals — {label}',
                  fontsize=14, fontweight='bold', y=1.01, color='#222')

    for ri, rn in enumerate(['Frontal', 'Central', 'Parietal', 'Occipital']):
        ax = axes3[ri]
        ax.set_facecolor('#FAFAFA')

        if rn in selected_ics:
            idx = selected_ics[rn]
            ax.plot(t, comp_erp[idx], color=roi_colors[rn], lw=2.5)
            ax.set_ylabel(f'{rn}\n(a.u.)', fontsize=10, fontweight='bold', color=roi_colors[rn])
        else:
            ax.text(0.5, 0.5, 'No IC assigned', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#999')

        for cn, t1, t2, cc in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.06, color=cc)
        ax.axvline(0, color='#333', ls='--', lw=0.8)
        ax.axhline(0, color='#999', lw=0.5)
        ax.tick_params(labelsize=7, colors='#888')
        ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    axes3[-1].set_xlabel('Time (s)', fontsize=10, color='#555')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path3 = os.path.join(OUT_DIR, f'{label}_ica_roi_sources.png')
    fig3.savefig(out_path3, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig3)
    print(f'  Saved: {out_path3}')

    # Save ICA results
    np.savez(os.path.join(OUT_DIR, f'{label}_ica.npz'),
             mixing=A, sources=S, comp_erp=comp_erp,
             var=var_explained, order=order,
             ic_to_roi=str(ic_to_roi))

print(f'\n{"="*60}')
print(f'  All ICA done! -> {OUT_DIR}')
print(f'{"="*60}')
