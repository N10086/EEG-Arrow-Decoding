#!/usr/bin/env python3
"""
5-session analysis — Arrow stimulus ERP + ROI activation
================================================================
Two new sessions (10:35 & 10:47), 50 trials/direction, all 16ch active
"""
import numpy as np
from scipy import signal as sg
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import warnings, json, os
warnings.filterwarnings('ignore')

# ============ Chinese font setup ============
_CHINESE_FONTS = [
    'Microsoft YaHei', 'SimHei', 'Source Han Sans CN',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
]
for fname in _CHINESE_FONTS:
    try:
        fp = fm.findfont(fname, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [fname] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

# ============ Config ============
PATHS = [
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-35-51\OpenBCI-RAW-2026-07-07_10-35-51.txt',
     'Session 1 (07-07 10h35)'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt',
     'Session 2 (07-07 10h47)'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_08-54-11\OpenBCI-RAW-2026-07-08_08-54-11.txt',
     'Session 3 (07-08 08h54)'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-08_09-04-45\OpenBCI-RAW-2026-07-08_09-04-45.txt',
     'Session 4 (07-08 09h04)'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-09_09-02-54\OpenBCI-RAW-2026-07-09_09-02-54.txt',
     'Session 5 (07-09 09h02)'),
]
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
SESSION_COLORS = ['#4A72C4', '#E8833A', '#5CB85C', '#9B59B6', '#D94F70']
SESSION_LS = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

FS = 500.0
GAIN = 6.0
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6  # 0.08941 uV/count

# ============ Channel config (all 16ch) ============
CH_MAP = {
    1:'Oz', 2:'C3', 3:'unused', 4:'Fz', 5:'C4', 6:'Cz', 7:'F3',
    8:'O2', 9:'P3', 10:'Pz', 11:'unused', 12:'P4', 13:'unused', 14:'F4', 15:'O1'
}
COMMON_CH = [1,2,4,5,6,7,8,9,10,12,14,15]  # 12 channels

ROIS = {
    'Frontal (F)':  [7,4,14],    # F3, Fz, F4
    'Central (C)':  [2,6,5],     # C3, Cz, C4
    'Parietal (P)': [9,10,12],   # P3, Pz, P4
    'Occipital (O)':[15,1,8],    # O1, Oz, O2
}
ROI_ORDER = ['Frontal (F)', 'Central (C)', 'Parietal (P)', 'Occipital (O)']

ERP_WINDOWS = [
    ('P1', 0.080, 0.130, 'Early visual\n80-130ms'),
    ('N1', 0.140, 0.200, 'Spatial attention\n140-200ms'),
    ('P2', 0.200, 0.300, 'Feature detection\n200-300ms'),
    ('P3', 0.300, 0.500, 'Cognitive evaluation\n300-500ms'),
]

DIR_LABELS = {2.0001: 'Up', 2.0002: 'Down', 2.0003: 'Left', 2.0004: 'Right'}
DIR_COLORS = {'Up': '#E74C3C', 'Down': '#3498DB', 'Left': '#2ECC71', 'Right': '#F39C12'}
DIR_ORDER = ['Up', 'Down', 'Left', 'Right']

N_BOOT = 5000

# ============ Data I/O ============
def load_session(path):
    with open(path) as f:
        lines = f.readlines()
    data, markers, timestamps = [], [], []
    for line in lines[5:]:
        parts = line.strip().split(',')
        if len(parts) > 33:
            try:
                data.append([float(parts[i].strip()) for i in range(1, 17)])
                markers.append(float(parts[32].strip()))
                timestamps.append(float(parts[30].strip()))
            except:
                pass
    data = np.array(data, dtype=np.float64).T
    markers = np.array(markers, dtype=np.float64)
    timestamps = np.array(timestamps)
    return data, markers, timestamps


def filter_data(data, fs=500):
    """1-45 Hz bandpass + 50 Hz notch"""
    sos_bp = sg.butter(4, [1/(fs/2), 45/(fs/2)], btype='band', output='sos')
    sos_notch = sg.iirnotch(50/(fs/2), 30)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        dm = data[ch] - data[ch].mean()
        temp = sg.sosfiltfilt(sos_bp, dm)
        filtered[ch] = sg.filtfilt(*sos_notch, temp)
    return filtered


def epoch_data(filtered, onsets, before, after):
    """Segment + baseline correction"""
    epochs = []
    for idx in onsets:
        start, end = idx - before, idx + after
        if start >= 0 and end <= filtered.shape[1]:
            ep = filtered[:, start:end].copy()
            ep -= ep[:, :before].mean(axis=1, keepdims=True)
            epochs.append(ep)
    return np.stack(epochs, axis=0)


def reject_bad_trials(epochs_uv, threshold=100.0):
    """Peak-to-peak > 100 uV -> bad trial"""
    ptp = epochs_uv.max(axis=2) - epochs_uv.min(axis=2)
    max_ptp = ptp.max(axis=1)
    good = max_ptp < threshold
    return good, max_ptp


def bootstrap_vs_baseline(data, n_iter=5000):
    """Bootstrap test H0: mean = 0"""
    obs_mean = data.mean()
    n = len(data)
    boot_means = np.zeros(n_iter)
    for i in range(n_iter):
        idx = np.random.randint(0, n, size=n)
        boot_means[i] = data[idx].mean()
    ci = np.percentile(boot_means, [2.5, 97.5])
    p = 2 * min(np.mean(boot_means >= 0), np.mean(boot_means <= 0))
    return obs_mean, ci[0], ci[1], p


# ============ Main processing ============
def process_session(path, label):
    out_dir_s = os.path.join(OUT_DIR, label.replace(' ','_'))
    os.makedirs(out_dir_s, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'  Processing: {label}')
    print(f'{"="*70}')

    # 1. Load
    data, markers, timestamps = load_session(path)
    exg_idx = [ch - 1 for ch in COMMON_CH]
    sel = data[exg_idx, :]

    # 2. Segment crop
    all_onsets = sorted([i for k in [2.0001,2.0002,2.0003,2.0004]
                          for i in np.where(np.abs(markers - k) < 0.00005)[0]])
    seg_start = max(0, all_onsets[0] - int(5 * FS))
    seg_end = min(sel.shape[1], all_onsets[-1] + int(5 * FS))
    crop = sel[:, seg_start:seg_end]
    print(f'  Total frames: {sel.shape[1]}, cropped: {crop.shape[1]} ({crop.shape[1]/FS:.0f}s)')

    # 3. Filter
    filt = filter_data(crop, FS)
    filt_uv = filt * SCALE

    # 4. Epoch
    rel_onsets = [i - seg_start for i in all_onsets]
    t_before, t_after = 200, 400
    epochs_uv = epoch_data(filt_uv, rel_onsets, t_before, t_after)
    print(f'  Raw epochs: {epochs_uv.shape[0]} trials')

    # 5. Bad trial rejection
    good_ptp, ptp_vals = reject_bad_trials(epochs_uv, 100.0)
    n_bad = epochs_uv.shape[0] - good_ptp.sum()
    print(f'  Bad trials: {n_bad}/{epochs_uv.shape[0]} rejected')
    epochs_clean = epochs_uv[good_ptp]
    print(f'  Kept: {epochs_clean.shape[0]} trials')

    # 6. Split by direction
    dir_epochs = {}
    for k in [2.0001, 2.0002, 2.0003, 2.0004]:
        dir_name = DIR_LABELS[k]
        idx_in_onsets = [i for i, idx in enumerate(all_onsets)
                         if np.abs(markers[idx] - k) < 0.00005]
        good_rel_idx = [i for i, orig_i in enumerate(idx_in_onsets) if good_ptp[orig_i]]
        dir_data = epochs_clean[good_rel_idx]
        dir_epochs[dir_name] = dir_data
        print(f'  {dir_name}: {dir_data.shape[0]} trials')

    # 7. ROI merge
    roi_epochs = {}
    for roi_name, hw_chs in ROIS.items():
        idx_in_common = [COMMON_CH.index(ch) for ch in hw_chs]
        roi_data = epochs_clean[:, idx_in_common, :].mean(axis=1)
        roi_epochs[roi_name] = roi_data

    # Direction x ROI
    dir_roi_epochs = {}
    for d in DIR_ORDER:
        dir_roi_epochs[d] = {}
        for roi_name in ROI_ORDER:
            hw_chs = ROIS[roi_name]
            idx_in_common = [COMMON_CH.index(ch) for ch in hw_chs]
            dir_roi_epochs[d][roi_name] = dir_epochs[d][:, idx_in_common, :].mean(axis=1)

    return {
        'label': label,
        'n_trials_total': epochs_clean.shape[0],
        'n_bad': n_bad,
        'n_per_direction': {d: int(dir_epochs[d].shape[0]) for d in DIR_ORDER},
        'epochs': epochs_clean,
        'roi_epochs': roi_epochs,
        'dir_epochs': dir_epochs,
        'dir_roi_epochs': dir_roi_epochs,
        't_before': t_before,
        't_after': t_after,
    }


# ============ Figure 1: ERP by direction ============
def plot_direction_erp(res, out_dir_s):
    t = (np.arange(600) - 200) / FS
    fig, axes = plt.subplots(len(ROI_ORDER), 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('#FAFAFA')

    for ri, roi_name in enumerate(ROI_ORDER):
        ax = axes[ri]
        ax.set_facecolor('#FAFAFA')

        for comp_name, t1, t2, label in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.06, color='#1f77b4', zorder=0)
            ax.text((t1+t2)/2, ax.get_ylim()[0] if ri==0 else ax.get_ylim()[0],
                    label, fontsize=5.5, ha='center', color='#1f77b4', alpha=0.6)

        for d in DIR_ORDER:
            erp = res['dir_roi_epochs'][d][roi_name].mean(axis=0)
            ax.plot(t, erp, color=DIR_COLORS[d], linewidth=1.0, alpha=0.85,
                    label=f"{d} (n={res['dir_epochs'][d].shape[0]})")

        ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.4)
        ax.set_title(roi_name, fontsize=11, color='#444', fontweight='bold')
        ax.tick_params(colors='#888', labelsize=7)
        ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    ax.legend(fontsize=6.5, ncol=4, loc='lower left')
    axes[-1].set_xlabel('Time (s)', fontsize=9, color='#888')
    fig.suptitle(f'{res["label"]} — ERP across 4 directions x 4 ROIs', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir_s, '01_direction_erp.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA')
    plt.close(fig)
    print(f'  [Fig] Direction ERP -> {path}')


# ============ Figure 2: ROI activation (all directions) ============
def plot_roi_erp(res_list, out_dir):
    fig, axes = plt.subplots(len(ROI_ORDER), 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('#FAFAFA')
    t = (np.arange(600) - 200) / FS

    for ri, roi_name in enumerate(ROI_ORDER):
        ax = axes[ri]
        ax.set_facecolor('#FAFAFA')
        for comp_name, t1, t2, label in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.06, color='#1f77b4', zorder=0)
        for si, res in enumerate(res_list):
            erp = res['roi_epochs'][roi_name].mean(axis=0)
            ax.plot(t, erp, color=SESSION_COLORS[si], linewidth=1.2,
                    label=f"{res['label']} (n={res['n_trials_total']})")
        ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.4)
        ax.set_title(roi_name, fontsize=11, color='#444', fontweight='bold')
        ax.tick_params(colors='#888', labelsize=7)
        ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        if ri == 0:
            ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('Time (s)', fontsize=9, color='#888')
    fig.suptitle('Arrow-stimulus evoked ROI activity (all directions)', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, '02_roi_activation_erp.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA')
    plt.close(fig)
    print(f'  [Fig] ROI activation ERP -> {path}')


# ============ Figure 3: Activation heatmap ============
def plot_heatmap(res_list, out_dir):
    fig, axes = plt.subplots(1, len(res_list), figsize=(10, 4), sharey=True)
    fig.patch.set_facecolor('#FAFAFA')
    if len(res_list) == 1:
        axes = [axes]
    comp_names = [e[0] for e in ERP_WINDOWS]

    for si, (res, ax) in enumerate(zip(res_list, axes)):
        ax.set_facecolor('#FAFAFA')
        matrix = np.zeros((len(ROI_ORDER), len(comp_names)))
        sig = np.zeros_like(matrix, dtype=bool)

        for ri, roi_name in enumerate(ROI_ORDER):
            for ci, (comp_name, t1, t2, _) in enumerate(ERP_WINDOWS):
                roi_data = res['roi_epochs'][roi_name]
                mask = ((np.arange(600) - 200)/FS >= t1) & ((np.arange(600) - 200)/FS <= t2)
                trial_means = roi_data[:, mask].mean(axis=1)
                mean_val, ci_l, ci_h, p = bootstrap_vs_baseline(trial_means, N_BOOT)
                matrix[ri, ci] = mean_val
                sig[ri, ci] = p < 0.05

        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.15)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')

        for ri in range(len(ROI_ORDER)):
            for ci in range(len(comp_names)):
                val = matrix[ri, ci]
                text_color = 'white' if abs(val) > vmax*0.5 else '#444'
                ax.text(ci, ri, f'{val:+.3f}', ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold')
                if sig[ri, ci]:
                    ax.text(ci, ri-0.3, '●', ha='center', fontsize=8, color='#FF3333')

        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels(comp_names, fontsize=9)
        ax.set_yticks(range(len(ROI_ORDER)))
        ax.set_yticklabels([r.split(' ')[1].strip('()') for r in ROI_ORDER], fontsize=9)
        ax.set_title(res['label'], fontsize=10, color='#444')
        for sp in ax.spines.values():
            sp.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('uV', fontsize=7, color='#888')
        cbar.ax.tick_params(colors='#888', labelsize=6)

    fig.suptitle('Activation amplitude (red dot = significant p<0.05)', fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, '03_activation_heatmap.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA', bbox_inches='tight')
    plt.close(fig)
    print(f'  [Fig] Heatmap -> {path}')


# ============ Figure 4: P1/N1/P2/P3 bar charts ============
def plot_comp_bars(res_list, out_dir):
    for comp_name in ['P1', 'N1', 'P2', 'P3']:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FAFAFA')

        x = np.arange(len(ROI_ORDER))
        w = 0.30

        for si, res in enumerate(res_list):
            vals, cis_l, cis_h = [], [], []
            for ri, roi_name in enumerate(ROI_ORDER):
                for cn, t1, t2, _ in ERP_WINDOWS:
                    if cn != comp_name:
                        continue
                    roi_data = res['roi_epochs'][roi_name]
                    mask = ((np.arange(600) - 200)/FS >= t1) & ((np.arange(600) - 200)/FS <= t2)
                    trial_means = roi_data[:, mask].mean(axis=1)
                    m, cl, ch, _ = bootstrap_vs_baseline(trial_means, N_BOOT)
                    vals.append(m)
                    cis_l.append(cl)
                    cis_h.append(ch)

            color = SESSION_COLORS[si]
            bars = ax.bar(x + (si-0.5)*w, vals, w,
                         color=color, alpha=0.85, edgecolor='white', linewidth=0.5,
                         label=res['label'])
            yerr_low = np.array(vals) - np.array(cis_l)
            yerr_high = np.array(cis_h) - np.array(vals)
            ax.errorbar(x + (si-0.5)*w, vals,
                       yerr=[yerr_low, yerr_high],
                       fmt='none', color='#333', capsize=2, capthick=1, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([r.split(' (')[0] for r in ROI_ORDER], fontsize=10)
        ax.set_ylabel(f'{comp_name} amplitude (uV)', fontsize=9, color='#888')
        ax.axhline(y=0, color='#999', linewidth=0.5)
        ax.legend(fontsize=8)
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.set_title(f'{comp_name} — all sessions (error bar = 95% CI)', fontsize=11, color='#444')
        plt.tight_layout()
        path = os.path.join(out_dir, f'04_{comp_name}_bar.png')
        fig.savefig(path, dpi=150, facecolor='#FAFAFA')
        plt.close(fig)
        print(f'  [Fig] {comp_name} bar -> {path}')


# ============ Figure 5: Activation timing ============
def plot_timing(res_list, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')
    t = (np.arange(600) - 200) / FS

    for si, res in enumerate(res_list):
        for ri, roi_name in enumerate(ROI_ORDER):
            erp = res['roi_epochs'][roi_name].mean(axis=0)
            ax.plot(t, erp, color=SESSION_COLORS[si], linewidth=1.0, linestyle=SESSION_LS[si],
                   alpha=0.8, label=f"{res['label']} — {roi_name.split(' ')[0]}")

    ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
    ax.axhline(y=0, color='#999', linewidth=0.4)
    ax.set_xlabel('Time (s)', fontsize=9, color='#888')
    ax.set_ylabel('uV', fontsize=9, color='#888')
    ax.tick_params(colors='#888', labelsize=8)
    ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=5.5, ncol=2, loc='upper right')
    ax.set_title('All sessions — ROI activation time course', fontsize=12, color='#444')
    plt.tight_layout()
    path = os.path.join(out_dir, '05_activation_timing.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA')
    plt.close(fig)
    print(f'  [Fig] Timing -> {path}')


# ============ Figure 6: Direction comparison (P3 window) ============
def plot_direction_p3(res_list, out_dir):
    n_sessions = len(res_list)
    if n_sessions <= 2:
        fig, axes = plt.subplots(1, n_sessions, figsize=(10, 4), sharey=True)
        if n_sessions == 1:
            axes = [axes]
    else:
        n_cols = 2
        n_rows = (n_sessions + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3*n_rows), sharey=True)
        axes = axes.flatten()
        for i in range(n_sessions, n_rows*n_cols):
            axes[i].set_visible(False)
    fig.patch.set_facecolor('#FAFAFA')

    for si, (res, ax) in enumerate(zip(res_list, axes)):
        ax.set_facecolor('#FAFAFA')
        x = np.arange(len(ROI_ORDER))
        w = 0.15

        for di, d in enumerate(DIR_ORDER):
            vals = []
            for roi_name in ROI_ORDER:
                roi_data = res['dir_roi_epochs'][d][roi_name]
                mask = ((np.arange(600) - 200)/FS >= 0.300) & ((np.arange(600) - 200)/FS <= 0.500)
                vals.append(roi_data[:, mask].mean(axis=1).mean())
            ax.bar(x + (di-1.5)*w, vals, w,
                  color=DIR_COLORS[d], alpha=0.8, label=d)

        ax.set_xticks(x)
        ax.set_xticklabels([r.split(' ')[1].strip('()') for r in ROI_ORDER], fontsize=9)
        ax.axhline(y=0, color='#999', linewidth=0.5)
        ax.set_title(res['label'], fontsize=10)
        ax.tick_params(colors='#888', labelsize=8)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    axes[0].legend(fontsize=7)
    fig.suptitle('Direction comparison — P3 (300-500ms) mean amplitude', fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, '06_direction_p3_compare.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA', bbox_inches='tight')
    plt.close(fig)
    print(f'  [Fig] Direction P3 compare -> {path}')


# ============ Figure 7: Channel ERP grid ============
def plot_topo_overview(res_list, out_dir):
    REAL_CH = {1:'Oz', 2:'C3', 4:'Fz', 5:'C4', 6:'Cz', 7:'F3',
               8:'O2', 9:'P3', 10:'Pz', 12:'P4', 14:'F4', 15:'O1'}
    FIG_CH_ORDER = [7,4,14, 2,6,5, 9,10,12, 15,1,8]
    FIG_ROW_NAMES = ['Frontal', 'Central', 'Parietal', 'Occipital']

    fig, axes = plt.subplots(4, 3, figsize=(8, 7))
    fig.patch.set_facecolor('#FAFAFA')
    t = (np.arange(600) - 200) / FS

    for pi, hwch in enumerate(FIG_CH_ORDER):
        ri, ci = divmod(pi, 3)
        ax = axes[ri, ci]
        ax.set_facecolor('#FAFAFA')

        ch_idx = COMMON_CH.index(hwch)
        for si, res in enumerate(res_list):
            erp = res['epochs'][:, ch_idx, :].mean(axis=0)
            ax.plot(t, erp, color=SESSION_COLORS[si], linewidth=1.0,
                   alpha=0.85, label=res['label'] if ci == 2 else '')

        ax.axvline(x=0, color='#333', linewidth=0.5, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.3)
        ax.set_title(f'{REAL_CH[hwch]} (ch{hwch})', fontsize=8, color='#444')
        ax.tick_params(colors='#888', labelsize=6)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        if ci == 0:
            ax.set_ylabel(FIG_ROW_NAMES[ri], fontsize=8, color='#888')
        if ri == 3:
            ax.set_xlabel('Time (s)', fontsize=7, color='#888')

    axes[0, 2].legend(fontsize=6, loc='upper right')
    fig.suptitle('Channel ERP — all sessions', fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, '07_channel_erp_grid.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA', bbox_inches='tight')
    plt.close(fig)
    print(f'  [Fig] Channel ERP grid -> {path}')


# ============ Stats output ============
def print_stats(res_list, out_dir):
    print(f'\n{"="*80}')
    print(f'  All-session — ROI activation statistics (bootstrap H0: mean=0)')
    print(f'{"="*80}')

    for si, res in enumerate(res_list):
        print(f'\n  --- {res["label"]} ---')
        print(f'  Valid trials: {res["n_trials_total"]}, rejected: {res["n_bad"]} bad trials')
        print(f'  Per direction: {res["n_per_direction"]}')
        print(f'  {"ROI":<20} {"Comp":<8} {"Window":<10} {"Mean(uV)":<12} {"95% CI":<22} {"p":<10} {"Sig."}')
        print(f'  {"-"*80}')

        stat_rows = []
        for roi_name in ROI_ORDER:
            for comp_name, t1, t2, _ in ERP_WINDOWS:
                roi_data = res['roi_epochs'][roi_name]
                mask = ((np.arange(600) - 200)/FS >= t1) & ((np.arange(600) - 200)/FS <= t2)
                trial_means = roi_data[:, mask].mean(axis=1)
                mean_val, ci_l, ci_h, p = bootstrap_vs_baseline(trial_means, N_BOOT)
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                print(f'  {roi_name:<20} {comp_name:<8} {t1*1000:.0f}-{t2*1000:.0f}ms    '
                      f'{mean_val:<+10.4f}  [{ci_l:<+7.4f}, {ci_h:<+7.4f}]  '
                      f'{p:<10.4f} {sig}')
                stat_rows.append({
                    'roi': roi_name,
                    'component': comp_name,
                    'window_ms': f'{t1*1000:.0f}-{t2*1000:.0f}',
                    'mean_uV': float(round(mean_val, 5)),
                    'ci_low': float(round(ci_l, 5)),
                    'ci_high': float(round(ci_h, 5)),
                    'p': float(round(p, 5)),
                    'sig': sig,
                })
        res['stats'] = stat_rows

    all_data = [{
        'label': res['label'],
        'n_trials': int(res['n_trials_total']),
        'n_bad': int(res['n_bad']),
        'n_per_direction': {k: int(v) for k, v in res['n_per_direction'].items()},
        'results': res['stats'],
    } for res in res_list]

    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f'\n  [JSON] -> {os.path.join(out_dir, "stats.json")}')


# ============ Main ============
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    res_list = []
    for path, label in PATHS:
        res = process_session(path, label)
        if res:
            res_list.append(res)
            out_dir_s = os.path.join(OUT_DIR, label.replace(' ','_'))
            plot_direction_erp(res, out_dir_s)

    if len(res_list) == 0:
        print('No data to analyze.')
        return

    plot_roi_erp(res_list, OUT_DIR)
    plot_heatmap(res_list, OUT_DIR)
    plot_comp_bars(res_list, OUT_DIR)
    plot_timing(res_list, OUT_DIR)
    plot_direction_p3(res_list, OUT_DIR)
    plot_topo_overview(res_list, OUT_DIR)
    print_stats(res_list, OUT_DIR)

    print(f'\n  Analysis complete! All figures -> {OUT_DIR}')


if __name__ == '__main__':
    main()
