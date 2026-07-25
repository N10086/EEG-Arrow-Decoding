#!/usr/bin/env python3
"""Plot all 4 sessions — 4 brain region ERP waveforms in a 2×2 grid + grand average."""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import warnings, os
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
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6  # 0.08941 uV/count

PATHS = [
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-35-51\OpenBCI-RAW-2026-07-07_10-35-51.txt',
     'Session 1\n07-07 10h35'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt',
     'Session 2\n07-07 10h47'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_08-54-11\OpenBCI-RAW-2026-07-08_08-54-11.txt',
     'Session 3\n07-08 08h54'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_09-04-45\OpenBCI-RAW-2026-07-08_09-04-45.txt',
     'Session 4\n07-08 09h04'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-09_09-02-54\OpenBCI-RAW-2026-07-09_09-02-54.txt',
     'Session 5\n07-09 09h02'),
]
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'

SESSION_COLORS = ['#4A72C4', '#E8833A', '#5CB85C', '#9B59B6', '#D94F70']
SESSION_LS = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

COMMON_CH = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]

ROIS = {
    'Frontal\n(F3/Fz/F4)':  [7, 4, 14],
    'Central\n(C3/Cz/C4)':  [2, 6, 5],
    'Parietal\n(P3/Pz/P4)': [9, 10, 12],
    'Occipital\n(O1/Oz/O2)': [15, 1, 8],
}
ROI_ORDER = list(ROIS.keys())

ERP_WINDOWS = [
    ('P1', 0.080, 0.130, '#27ae60'),
    ('N1', 0.140, 0.200, '#7f8c8d'),
    ('P2', 0.200, 0.300, '#e67e22'),
    ('P3', 0.300, 0.500, '#8e44ad'),
]

N_BOOT = 5000

# ===== Functions =====
def load_session(path):
    with open(path) as f:
        lines = f.readlines()
    data, markers = [], []
    for line in lines[5:]:
        parts = line.strip().split(',')
        if len(parts) > 33:
            try:
                data.append([float(parts[i]) for i in range(1, 17)])
                markers.append(float(parts[32]))
            except:
                pass
    return np.array(data, dtype=np.float64).T, np.array(markers)


def process(path):
    data, markers = load_session(path)
    exg = data[[c-1 for c in COMMON_CH], :]

    # Onsets
    ons = sorted([i for k in [2.0001,2.0002,2.0003,2.0004]
                  for i in np.where(np.abs(markers - k) < 5e-5)[0]])

    # Crop
    s = max(0, ons[0] - int(5*FS))
    e = min(exg.shape[1], ons[-1] + int(5*FS))
    crop = exg[:, s:e]

    # Filter
    bp = sg.butter(4, [1/250, 45/250], btype='band', output='sos')
    nc = sg.iirnotch(50/250, 30)
    filt = np.zeros_like(crop)
    for ch in range(crop.shape[0]):
        dm = crop[ch] - crop[ch].mean()
        t = sg.sosfiltfilt(bp, dm)
        filt[ch] = sg.filtfilt(*nc, t)
    filt_uv = filt * SCALE

    # Epoch
    rel = [i - s for i in ons]
    ep = []
    for idx in rel:
        st, en = idx - 200, idx + 400
        if st >= 0 and en <= filt_uv.shape[1]:
            ei = filt_uv[:, st:en].copy()
            ei -= ei[:, :200].mean(axis=1, keepdims=True)
            ep.append(ei)
    ep = np.stack(ep, 0)

    # Reject
    ptp = (ep.max(2) - ep.min(2)).max(1)
    good = ptp < 100.0
    ep_clean = ep[good]
    kept = ep_clean.shape[0]

    # ROI
    roi_data = {}
    for rn, chs in ROIS.items():
        idx = [COMMON_CH.index(c) for c in chs]
        rd = ep_clean[:, idx, :].mean(axis=1)
        roi_data[rn] = rd
    return roi_data, kept, ep_clean.shape[0] - kept


# ===== Main =====
def main():
    t = (np.arange(600) - 200) / FS

    # Process all sessions
    all_roi = []
    n_kept = []
    n_bad = []
    for path, label in PATHS:
        rd, kept, bad = process(path)
        all_roi.append(rd)
        n_kept.append(kept)
        n_bad.append(bad)
        print(f'{label.split(chr(10))[-1].strip()}: {kept} good, {bad} bad')

    # ====== FIGURE 1: 2×2 grid ======
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    axes_flat = axes.flatten()

    for ri, rn in enumerate(ROI_ORDER):
        ax = axes_flat[ri]
        ax.set_facecolor('#FAFAFA')

        # ERP windows
        for cname, t1, t2, ccol in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.07, color=ccol, zorder=0)
            # window labels in top margin done below for 2x2
            # Place window labels at top of each panel
            ax.text((t1+t2)/2, ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.15 if ri > 0 else ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.12,
                    cname, fontsize=7, ha='center', color=ccol, fontweight='bold',
                    alpha=0.5)

        # Session lines
        for si in range(len(PATHS)):
            mu = all_roi[si][rn].mean(axis=0)
            ax.plot(t, mu, color=SESSION_COLORS[si], linewidth=1.3,
                    linestyle=SESSION_LS[si],
                    label=f'{PATHS[si][1].replace(chr(10), " ")} (n={n_kept[si]})')

        # Grand average
        grand = np.mean([all_roi[si][rn].mean(axis=0) for si in range(len(PATHS))], axis=0)
        ax.plot(t, grand, color='#111', linewidth=2.5, alpha=0.6, label='Grand avg')

        # Zero lines
        ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.5)

        # Labels
        ax.set_title(rn, fontsize=13, fontweight='bold', color='#222', pad=10)
        ax.set_ylabel('µV', fontsize=10, color='#555')
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        # Legend on first panel only
        if ri == 0:
            ax.legend(fontsize=7, loc='upper right', ncol=2,
                     framealpha=0.85, edgecolor='#ddd')

    for ax in axes_flat[2:]:
        ax.set_xlabel('Time (s)', fontsize=10, color='#555')

    fig.suptitle('5 Sessions — Brain Region ERP Waveforms (Arrow Stimulus, All Directions)',
                 fontsize=15, fontweight='bold', color='#111', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '08_all_sessions_roi_2x2.png')
    fig.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')

    # ====== FIGURE 2: Grand average only ======
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig2.patch.set_facecolor('white')

    for ri, rn in enumerate(ROI_ORDER):
        ax = axes2[ri]
        ax.set_facecolor('#FAFAFA')

        # ERP windows
        for cname, t1, t2, ccol in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.06, color=ccol, zorder=0)

        # Window labels at top
        y_top = ax.get_ylim()[1] if ri == 0 else ax.get_ylim()[1]
        for cname, t1, t2, ccol in ERP_WINDOWS:
            ax.text((t1+t2)/2, y_top - 0.08, cname, fontsize=7, ha='center',
                    color=ccol, fontweight='bold', alpha=0.5)

        # Individual sessions (thin)
        for si in range(len(PATHS)):
            mu = all_roi[si][rn].mean(axis=0)
            se = all_roi[si][rn].std(axis=0) / np.sqrt(n_kept[si])
            ax.plot(t, mu, color=SESSION_COLORS[si], linewidth=0.6,
                    linestyle=SESSION_LS[si], alpha=0.4)
            ax.fill_between(t, mu - 2*se, mu + 2*se,
                           color=SESSION_COLORS[si], alpha=0.04)

        # Grand average (thick)
        grand_se = np.std([all_roi[si][rn].mean(axis=0) for si in range(len(PATHS))], axis=0) / np.sqrt(len(PATHS))
        ax.plot(t, grand, color='#111', linewidth=2.8, label='Grand average (n=4 sessions)')
        ax.fill_between(t, grand - 2*grand_se, grand + 2*grand_se,
                       color='#111', alpha=0.1)

        # Zero lines
        ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.5)

        # Stats annotation
        ax.text(0.97, 0.05, f'{rn}', transform=ax.transAxes,
               fontsize=11, fontweight='bold', color='#333', ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.85))

        ax.set_ylabel('µV', fontsize=10, color='#555')
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    axes2[0].legend(fontsize=8, loc='upper right', framealpha=0.85, edgecolor='#ddd')
    axes2[-1].set_xlabel('Time (s)', fontsize=11, color='#555')
    fig2.suptitle('Grand Average — 5 Sessions × 4 Brain Regions',
                  fontsize=15, fontweight='bold', color='#111', y=1.01)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, '08_grand_average_roi.png')
    fig2.savefig(out2, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out2}')

    # ====== FIGURE 3: Per-session individual figures ======
    for si in range(len(PATHS)):
        fig3, axes3 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        fig3.patch.set_facecolor('white')
        label_short = PATHS[si][1].split('\n')[1].strip()

        for ri, rn in enumerate(ROI_ORDER):
            ax = axes3[ri]
            ax.set_facecolor('#FAFAFA')

            for cname, t1, t2, ccol in ERP_WINDOWS:
                ax.axvspan(t1, t2, alpha=0.06, color=ccol, zorder=0)
                ax.text((t1+t2)/2, 0.55, cname, fontsize=6.5, ha='center',
                       color=ccol, fontweight='bold', alpha=0.5)

            mu = all_roi[si][rn].mean(axis=0)
            se = all_roi[si][rn].std(axis=0) / np.sqrt(n_kept[si])
            ax.plot(t, mu, color=SESSION_COLORS[si], linewidth=2.0)
            ax.fill_between(t, mu - 2*se, mu + 2*se,
                           color=SESSION_COLORS[si], alpha=0.15)

            ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
            ax.axhline(y=0, color='#999', linewidth=0.5)
            ax.set_ylabel('µV', fontsize=9, color='#555')
            ax.tick_params(colors='#888', labelsize=7)
            ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)

            # Region label
            ax.text(0.98, 0.93, rn.replace('\n', ' '),
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   color='#333', ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#ddd', alpha=0.85))

        axes3[-1].set_xlabel('Time (s)', fontsize=10, color='#555')
        fig3.suptitle(f'{PATHS[si][1].replace(chr(10), " — ")}  (n={n_kept[si]} trials)',
                     fontsize=13, fontweight='bold', color='#111', y=1.01)
        plt.tight_layout()
        out3 = os.path.join(OUT_DIR, f'session{si+1}_roi_waveforms.png')
        fig3.savefig(out3, dpi=200, facecolor='white', bbox_inches='tight')
        plt.close(fig3)
        print(f'Saved: {out3}')

    print('\nDone — 6 PNG files generated.')


if __name__ == '__main__':
    main()
