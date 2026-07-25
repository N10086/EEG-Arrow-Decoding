#!/usr/bin/env python3
"""
OpenBCI 箭头刺激 ERP 分析
==========================
对两批 EEG 数据 (CytonWifiDaisy, 16ch, 500 Hz) 做：
  1. 读取 + 通道选择 + 去除 DC + 1–45 Hz 带通
  2. 按刺激标记分段 (epoch) 并基线校正
  3. 坏 trial 剔除 (±100 µV)
  4. ERP 叠加平均 (每方向 × 每通道)
  5. ROI 合并 (额/中央/顶/枕)
  6. 提取 P1(80–120), N1(140–200), P3(300–500) 指标
  7. 两次 bootstrap 差异检验
  8. 全部可视化

作者: Claude
日期: 2026-07-06
"""

import os, sys, json, warnings
from collections import Counter, OrderedDict
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)

# ======================================================================
#  配置参数  (按需修改)
# ======================================================================
PATHS = [
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-06_14-39-25\OpenBCI-RAW-2026-07-06_14-39-25.txt',
     'Session 1 (14:39)'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-06_14-16-16\OpenBCI-RAW-2026-07-06_14-16-16.txt',
     'Session 2 (14:16)'),
]
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis'

# 硬件通道 → 电极名  (用户提供)
CH_MAP = {
    1: 'Oz', 2: 'C3', 4: 'Fz', 5: 'C4', 6: 'Cz', 7: 'F3',
    8: 'O2', 9: 'P3', 10: 'Pz', 12: 'P4', 14: 'F4', 15: 'O1',
}
# 共同通道 (两次都有, 且 Oz 不贴头皮 → 排除)
COMMON_CH = [2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]  # 硬件通道号
COMMON_CH_LABELS = [CH_MAP[ch] for ch in COMMON_CH]

# ROI 定义  (用硬件通道号)
ROIS = OrderedDict([
    ('Frontal (F)',  [7, 4, 14]),   # F3, Fz, F4
    ('Central (C)',  [2, 6, 5]),    # C3, Cz, C4
    ('Parietal (P)', [9, 10, 12]),  # P3, Pz, P4
    ('Occipital (O)', [15, 8]),     # O1, O2
])
ROI_LABELS = list(ROIS.keys())
ROI_CHS = [ROIS[k] for k in ROI_LABELS]

# 刺激标记编码
DIR_MAP = {
    2.0001: ('A ↑', 'Up'),
    2.0002: ('B ↓', 'Down'),
    2.0003: ('C ←', 'Left'),
    2.0004: ('D →', 'Right'),
}
DIR_KEYS = sorted(DIR_MAP.keys())

# 信号参数
FS = 500.0                     # 采样率 Hz
VREF = 4.5                     # ADS1299 Vref
GAIN = 6.0                     # ADS1299 增益 (×6 — 用户手动设置)
SCALE_UV = VREF / (2**23 - 1) / GAIN * 1e6   # 0.08941 µV/count (×6)

# 滤波参数
BP_LOW, BP_HIGH = 1.0, 45.0   # 带通 (Hz)
NOTCH = 50.0                   # 陷波频率 (Hz)
FILTER_ORDER = 4

# Epoch 参数
T_BEFORE = 0.2                 # 刺激前 (s)
T_AFTER  = 0.8                 # 刺激后 (s)
N_BEFORE = int(T_BEFORE * FS)  # 100 采样点
N_AFTER  = int(T_AFTER * FS)   # 400 采样点
EPOCH_LEN = N_BEFORE + N_AFTER  # 500 采样点

# 坏 trial 阈值 (µV, peak-to-peak)
REJECT_THRESH_UV = 100.0

# ERP 指标时间窗
ERP_WINDOWS = OrderedDict([
    ('P1', (0.080, 0.120)),   # [80, 120] ms
    ('N1', (0.140, 0.200)),   # [140, 200] ms
    ('P3', (0.300, 0.500)),   # [300, 500] ms
])

# Bootstrap
N_BOOT = 2000
ALPHA = 0.05

# 绘图颜色
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ROI_COLORS = ['#E3242B', '#FD5E34', '#E1B530', '#317159']
BG_COLOR = '#FAFAFA'
GRID_COLOR = '#E0E0E0'
AXIS_COLOR = '#888888'
MARKER_COLOR = '#FF3333'

# ======================================================================
#  工具函数
# ======================================================================
def design_filters(fs, bp_low=1.0, bp_high=45.0, notch=50.0, order=4):
    """设计 1–45 Hz 带通 + 50 Hz 陷波 (级联)"""
    sos_bp = sg.butter(order, [bp_low/(fs/2), bp_high/(fs/2)],
                       btype='band', output='sos')
    b_notch, a_notch = sg.iirnotch(notch, 30, fs)
    sos_notch = sg.tf2sos(b_notch, a_notch)
    return np.vstack([sos_bp, sos_notch])


def read_openbcitxt(path):
    """读取 OpenBCI TXT，返回 data (n_channels × n_samples), marker 列, sr"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析列名
    cols = lines[4].strip().split(',')
    cols = [c.strip() for c in cols]

    # 找到 EXG 通道列和标记列
    exg_cols = [i for i, c in enumerate(cols) if c.startswith('EXG Channel')]
    stim_col = cols.index('Stimulus Marker')

    # 解析数据
    data_rows = []
    markers = []
    for line in lines[5:]:
        parts = line.strip().split(',')
        if len(parts) < max(exg_cols + [stim_col]) + 1:
            continue
        try:
            row = [float(parts[ci].strip()) for ci in exg_cols]
            m = float(parts[stim_col].strip())
        except (ValueError, IndexError):
            continue
        data_rows.append(row)
        markers.append(m)

    data = np.array(data_rows, dtype=np.float64).T  # (nch, nsamples)
    markers = np.array(markers, dtype=np.float64)
    return data, markers, len(exg_cols)


def find_stimulus_onsets(markers):
    """从标记序列中找到 2.0001–2.0004 的刺激起始下标"""
    onsets = {}
    for i, m in enumerate(markers):
        for key in DIR_MAP:
            if abs(m - key) < 0.00005:   # 浮点容差
                onsets.setdefault(key, []).append(i)
                break
    return onsets


def epoch_data(data, onsets_idx, n_before, n_after):
    """分段: 返回 (n_trials, n_channels, epoch_len) 和每 trial 的标记类型"""
    n_ch = data.shape[0]
    epochs_list = []
    labels_list = []
    for key, idx_list in onsets_idx.items():
        for idx in idx_list:
            start = idx - n_before
            end   = idx + n_after
            if start < 0 or end > data.shape[1]:
                continue
            epoch = data[:, start:end]
            epochs_list.append(epoch)
            labels_list.append(key)
    epochs = np.stack(epochs_list, axis=0)  # (n, nch, T)
    return epochs, np.array(labels_list)


def apply_filters(data, sos):
    """零相位滤波 (sosfiltfilt)"""
    # 处理边缘: 前后填充
    pad_len = data.shape[-1]
    padded = np.pad(data, ((0,0), (pad_len, pad_len)), mode='reflect')
    filtered = sg.sosfiltfilt(sos, padded, axis=1)
    return filtered[:, pad_len:-pad_len]


def baseline_correct(epochs, n_before):
    """减去基线 (-200~0 ms) 均值"""
    baseline = epochs[:, :, :n_before].mean(axis=2, keepdims=True)
    return epochs - baseline


def reject_bad_trials(epochs_uv, threshold=100.0):
    """幅度阈值剔除, 返回保留的 trial 索引"""
    peak_to_peak = epochs_uv.max(axis=2) - epochs_uv.min(axis=2)  # (n, nch)
    max_ptp = peak_to_peak.max(axis=1)  # (n,)  — 所有通道中最大的峰峰值
    good = max_ptp < threshold
    return good


def erp_waveform(epochs):
    """叠加平均: (n_trials, n_ch, T) → (n_ch, T)"""
    return epochs.mean(axis=0)


def extract_metrics(erp, fs, windows):
    """从 ERP 波形提取各窗口指标"""
    metrics = {}
    t = np.arange(erp.shape[1]) / fs - T_BEFORE
    for name, (t1, t2) in windows.items():
        mask = (t >= t1) & (t <= t2)
        if mask.sum() == 0:
            continue
        win_data = erp[:, mask]
        mean_amp = win_data.mean(axis=1)
        # 峰值和潜伏期 (找窗口内最大绝对值)
        peak_idx = np.argmax(np.abs(win_data), axis=1)
        peak_amp = win_data[np.arange(len(win_data)), peak_idx]
        peak_lat = t1 + peak_idx / fs
        metrics[name] = {
            'mean_amp': mean_amp,
            'peak_amp': peak_amp,
            'peak_lat': peak_lat,
        }
    return metrics


def bootstrap_diff(a, b, n_iter=N_BOOT):
    """bootstrap 检验两组均值的差异, 返回 (diff_mean, ci_low, ci_high, p)"""
    # a, b 是 (n_trials,) 的数值
    all_vals = np.concatenate([a, b])
    n_a, n_b = len(a), len(b)
    obs_diff = a.mean() - b.mean()

    boot_diffs = []
    for _ in range(n_iter):
        idx = np.random.randint(0, len(all_vals), size=n_a + n_b)
        a_boot = all_vals[idx[:n_a]]
        b_boot = all_vals[idx[n_a:]]
        boot_diffs.append(a_boot.mean() - b_boot.mean())
    boot_diffs = np.array(boot_diffs)
    ci = np.percentile(boot_diffs, [100*ALPHA/2, 100*(1-ALPHA/2)])

    # 双尾 p 值 (基于重采样分布)
    p_val = np.mean(np.abs(boot_diffs) >= abs(obs_diff))
    return obs_diff, ci[0], ci[1], p_val


# ======================================================================
#  主分析函数
# ======================================================================
def analyze_session(path, label, sos, out_dir):
    """分析单批数据, 返回结果字典"""
    print(f'\n{"="*60}')
    print(f'  分析: {label}')
    print(f'  文件: {path}')
    print(f'{"="*60}')

    data_raw, markers, n_ch_raw = read_openbcitxt(path)
    print(f'  原始数据: {n_ch_raw} 通道 × {data_raw.shape[1]} 采样点 ({data_raw.shape[1]/FS:.1f}s)')

    # 选择共同通道: 硬件通道 N → EXG 索引 N-1
    exg_idx = [ch - 1 for ch in COMMON_CH]
    sel_data = data_raw[exg_idx, :]  # (11, n)
    print(f'  选 {len(COMMON_CH)} 个共同通道: {", ".join(COMMON_CH_LABELS)}')

    # 0) 找到标记
    onsets = find_stimulus_onsets(markers)
    total_trials = sum(len(v) for v in onsets.values())
    print(f'  刺激标记: ', end='')
    for k in DIR_KEYS:
        print(f'{DIR_MAP[k][0]}={len(onsets.get(k,[]))}', end='  ')
    print(f'  → 合计 {total_trials} 个 trial')

    # 去掉开头/结尾不稳定段: 从第一个刺激前 5s 到最后一个刺激后 5s
    all_onsets = sorted([idx for lst in onsets.values() for idx in lst])
    seg_start = max(0, all_onsets[0] - int(5 * FS))
    seg_end   = min(sel_data.shape[1], all_onsets[-1] + int(5 * FS))
    print(f'  稳定段: {seg_start/FS:.1f}s – {seg_end/FS:.1f}s '
          f'({(seg_end-seg_start)/FS:.1f}s)')

    crop_data = sel_data[:, seg_start:seg_end]

    # 1) 去除 DC (每通道减均值)
    raw_demean = crop_data - crop_data.mean(axis=1, keepdims=True)

    # 2) 滤波
    filt_data = apply_filters(raw_demean, sos)

    # 3) 转换为 µV
    data_uv = filt_data * SCALE_UV

    # 4) 分段
    # 调整 onsets index 到 crop 后的坐标
    adjusted_onsets = {}
    for k, idx_list in onsets.items():
        adj = [idx - seg_start for idx in idx_list
               if seg_start <= idx < seg_end]
        adjusted_onsets[k] = adj

    epochs, epoch_labels = epoch_data(data_uv, adjusted_onsets,
                                      N_BEFORE, N_AFTER)
    print(f'  分段后: {epochs.shape[0]} trial × {epochs.shape[1]} 通道 × {epochs.shape[2]} 采样点')

    # 5) 基线校正
    epochs_bc = baseline_correct(epochs, N_BEFORE)

    # 6) 坏 trial 剔除
    good = reject_bad_trials(epochs_bc, REJECT_THRESH_UV)
    print(f'  剔除: {good.sum()}/{len(good)} trial 保留 (丢 {good.sum()==0} 个)')
    print(f'        丢弃: {np.where(~good)[0].tolist() if (~good).sum() <= 10 else f"{ (~good).sum()} 个"}'  )

    epochs_clean = epochs_bc[good]
    labels_clean = epoch_labels[good]

    # 7) 按方向叠加
    erp_by_dir = {}
    trial_counts = {}
    for k in DIR_KEYS:
        mask = labels_clean == k
        n_trials = mask.sum()
        trial_counts[k] = n_trials
        if n_trials >= 3:
            ep = epochs_clean[mask]
            erp = ep.mean(axis=0)
        else:
            erp = np.zeros((len(COMMON_CH), EPOCH_LEN)) * np.nan
        erp_by_dir[k] = erp
        print(f'  {DIR_MAP[k][0]}: {n_trials} trial → ERP')

    # 8) ROI 平均
    roi_erp_by_dir = {}
    for k in DIR_KEYS:
        erp = erp_by_dir[k]
        roi_list = []
        for chs in ROI_CHS:
            roi_idx = [COMMON_CH.index(ch) for ch in chs]  # 在 COMMON_CH 中的位置
            roi_erp = erp[roi_idx, :].mean(axis=0)          # 通道平均
            roi_list.append(roi_erp)
        roi_erp_by_dir[k] = np.array(roi_list)  # (4, T)

    return {
        'label': label,
        'data_uv': data_uv,
        'epochs_clean': epochs_clean,
        'labels_clean': labels_clean,
        'erp_by_dir': erp_by_dir,
        'roi_erp_by_dir': roi_erp_by_dir,
        'trial_counts': trial_counts,
        'all_trial_counts': {k: len(onsets.get(k, [])) for k in DIR_KEYS},
    }


# ======================================================================
#  可视化函数
# ======================================================================
def plot_raw_trace(data_uv, fs, title, save_path):
    """画出原始 (滤波后) 波形, 每个通道一条线"""
    n_ch = data_uv.shape[0]
    dur = data_uv.shape[1] / fs
    t = np.arange(data_uv.shape[1]) / fs

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 0.8*n_ch+1),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'{title} — 滤波后波形 (1–45 Hz)', color='#444', fontsize=12, y=0.98)

    for i in range(n_ch):
        ax = axes[i][0]
        ax.set_facecolor(BG_COLOR)
        ax.plot(t, data_uv[i], color=COLORS[i % 4], linewidth=0.3)
        ax.set_ylim(-50, 50)
        ax.set_ylabel(COMMON_CH_LABELS[i], fontsize=7, color=AXIS_COLOR)
        ax.axhline(y=0, color='#DDD', linewidth=0.3)
        ax.tick_params(colors=AXIS_COLOR, labelsize=6)
        ax.grid(True, axis='y', linestyle=':', color=GRID_COLOR, linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    axes[-1][0].set_xlabel('Time (s)', fontsize=8, color=AXIS_COLOR)
    axes[-1][0].spines['bottom'].set_visible(True)
    axes[-1][0].spines['bottom'].set_color(GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  [图] 滤波波形 → {save_path}')


def plot_erp_directions(erp_dict, trial_counts, fs, title, save_path):
    """画四个方向的 ERP 叠加图, 每个通道一张子图"""
    n_ch = len(COMMON_CH)
    t = np.arange(EPOCH_LEN) / fs - T_BEFORE
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 0.7*n_ch+1),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'{title} — 各方向 ERP (按通道)', color='#444', fontsize=11, y=0.98)

    for i in range(n_ch):
        ax = axes[i][0]
        ax.set_facecolor(BG_COLOR)
        for j, k in enumerate(DIR_KEYS):
            erp = erp_dict[k]
            if not np.any(np.isnan(erp)):
                ax.plot(t, erp[i], color=COLORS[j], linewidth=0.9,
                        label=f'{DIR_MAP[k][0]} (n={trial_counts[k]})')
        ax.axvline(x=0, color='#CCC', linewidth=0.5, linestyle='--')
        ax.axhline(y=0, color='#DDD', linewidth=0.3)
        ax.set_ylabel(COMMON_CH_LABELS[i], fontsize=7, color=AXIS_COLOR)
        ax.tick_params(colors=AXIS_COLOR, labelsize=6)
        ax.grid(True, axis='y', linestyle=':', color=GRID_COLOR, linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right',
                      facecolor=BG_COLOR, edgecolor=GRID_COLOR)

    axes[-1][0].set_xlabel('Time (s)', fontsize=8, color=AXIS_COLOR)
    axes[-1][0].spines['bottom'].set_visible(True)
    axes[-1][0].spines['bottom'].set_color(GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  [图] 各方向 ERP → {save_path}')


def plot_roi_erp(roi_dict, trial_counts, fs, title, save_path):
    """画四个 ROI 的 ERP (同一图上每个方向一条线)"""
    n_roi = len(ROI_LABELS)
    t = np.arange(EPOCH_LEN) / fs - T_BEFORE
    fig, axes = plt.subplots(n_roi, 1, figsize=(10, 2.5*n_roi+0.5),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'{title} — ROI ERP', color='#444', fontsize=12, y=0.98)

    for ri in range(n_roi):
        ax = axes[ri][0]
        ax.set_facecolor(BG_COLOR)
        for j, k in enumerate(DIR_KEYS):
            erp = roi_dict[k][ri]
            if not np.any(np.isnan(erp)):
                ax.plot(t, erp, color=COLORS[j], linewidth=1.0,
                        label=f'{DIR_MAP[k][0]} (n={trial_counts[k]})')
        # 标注 P1/N1/P3 窗口
        for w_name, (t1, t2) in ERP_WINDOWS.items():
            ax.axvspan(t1, t2, alpha=0.06, color='#1f77b4')
            ax.text((t1+t2)/2, ax.get_ylim()[1]*0.9, w_name,
                    fontsize=6, ha='center', color='#1f77b4', alpha=0.7)
        ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.4)
        ax.set_title(ROI_LABELS[ri], fontsize=10, color='#444', pad=1)
        ax.tick_params(colors=AXIS_COLOR, labelsize=7)
        ax.grid(True, axis='y', linestyle=':', color=GRID_COLOR, linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        if ri == 0:
            ax.legend(fontsize=7, loc='upper right',
                      facecolor=BG_COLOR, edgecolor=GRID_COLOR, ncol=2)

    axes[-1][0].set_xlabel('Time (s)', fontsize=9, color=AXIS_COLOR)
    axes[-1][0].spines['bottom'].set_visible(True)
    axes[-1][0].spines['bottom'].set_color(GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  [图] ROI ERP → {save_path}')


def plot_comparison_bar(s1_roi_dir, s2_roi_dir,
                        s1_counts, s2_counts, save_path):
    """P3 幅度柱状图 (两次对比)"""
    n_roi = len(ROI_LABELS)
    n_dir = len(DIR_KEYS)
    dir_short = [DIR_MAP[k][0] for k in DIR_KEYS]

    # 提取 P3 平均幅度
    def get_p3(roi_dict, trial_counts):
        """返回 (n_roi, n_dir) P3 均值"""
        result = np.full((n_roi, n_dir), np.nan)
        for di, k in enumerate(DIR_KEYS):
            roi_erp = roi_dict[k]
            if np.any(np.isnan(roi_erp)):
                continue
            for ri in range(n_roi):
                t = np.arange(EPOCH_LEN) / FS - T_BEFORE
                mask = (t >= 0.300) & (t <= 0.500)
                result[ri, di] = roi_erp[ri, mask].mean()
        return result

    p3_s1 = get_p3(s1_roi_dir, s1_counts)
    p3_s2 = get_p3(s2_roi_dir, s2_counts)

    fig, axes = plt.subplots(1, n_roi, figsize=(5*n_roi, 4.5),
                             sharey=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('P3 幅度 (300–500 ms) — 两次对比', color='#444',
                 fontsize=13, y=0.98)

    if n_roi == 1:
        axes = [axes]

    for ri in range(n_roi):
        ax = axes[ri]
        ax.set_facecolor(BG_COLOR)

        x = np.arange(n_dir)
        w = 0.30

        s1_vals = p3_s1[ri]
        s2_vals = p3_s2[ri]

        bars1 = ax.bar(x - w/2, s1_vals, w, color='#4A72C4', alpha=0.85,
                       label='Session 1', edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + w/2, s2_vals, w, color='#E8833A', alpha=0.85,
                       label='Session 2', edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(dir_short, fontsize=9)
        ax.set_title(ROI_LABELS[ri], fontsize=11, color='#444')
        ax.axhline(y=0, color='#999', linewidth=0.5)
        ax.tick_params(colors=AXIS_COLOR, labelsize=8)
        ax.grid(True, axis='y', linestyle=':', color=GRID_COLOR, linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        # 加标注: trial count
        for bi, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            k = DIR_KEYS[bi]
            ax.text(bar1.get_x() + bar1.get_width()/2,
                    bar1.get_height() + 0.2,
                    f'n={s1_counts[k]}', fontsize=6, ha='center',
                    color='#4A72C4')
            ax.text(bar2.get_x() + bar2.get_width()/2,
                    bar2.get_height() + 0.2,
                    f'n={s2_counts[k]}', fontsize=6, ha='center',
                    color='#E8833A')

        if ri == 0:
            ax.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  [图] P3 对比 → {save_path}')

    return p3_s1, p3_s2


def plot_difference_waves(s1_roi, s2_roi, fs, save_path):
    """差异波 (Session 2 - Session 1)"""
    t = np.arange(EPOCH_LEN) / fs - T_BEFORE
    fig, axes = plt.subplots(1, len(ROI_LABELS), figsize=(5*len(ROI_LABELS), 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('差异波 (Session 2 − Session 1)', color='#444',
                 fontsize=12, y=0.98)

    if len(ROI_LABELS) == 1:
        axes = [axes]

    for ri in range(len(ROI_LABELS)):
        ax = axes[ri]
        ax.set_facecolor(BG_COLOR)

        for j, k in enumerate(DIR_KEYS):
            if np.any(np.isnan(s1_roi[k][ri])) or np.any(np.isnan(s2_roi[k][ri])):
                continue
            diff = s2_roi[k][ri] - s1_roi[k][ri]
            ax.plot(t, diff, color=COLORS[j], linewidth=0.9,
                    label=f'{DIR_MAP[k][0]}')

        ax.axvline(x=0, color='#333', linewidth=0.6, linestyle='--')
        ax.axhline(y=0, color='#999', linewidth=0.4)
        ax.set_title(ROI_LABELS[ri], fontsize=11, color='#444')
        ax.tick_params(colors=AXIS_COLOR, labelsize=7)
        ax.grid(True, axis='y', linestyle=':', color=GRID_COLOR, linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    axes[0].legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    axes[-1].set_xlabel('Time (s)', fontsize=9, color=AXIS_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  [图] 差异波 → {save_path}')


def plot_bootstrap_heatmap(s1_epochs, s1_labels, s2_epochs, s2_labels,
                           fs, save_path):
    """Bootstrap 差异检验热力图: 横轴时间, 纵轴通道"""
    n_ch = len(COMMON_CH)
    t = np.arange(EPOCH_LEN) / fs - T_BEFORE

    # 对每个时间点 x 每个通道做 bootstrap
    p_matrix = np.ones((n_ch, EPOCH_LEN))
    diff_matrix = np.zeros((n_ch, EPOCH_LEN))

    for ch in range(n_ch):
        for tp in range(EPOCH_LEN):
            a = s1_epochs[:, ch, tp]
            b = s2_epochs[:, ch, tp]
            diff, _, _, p = bootstrap_diff(a, b)
            diff_matrix[ch, tp] = diff
            p_matrix[ch, tp] = p

    # FDR 校正
    n_tests = n_ch * EPOCH_LEN
    p_flat = p_matrix.flatten()
    ranked = np.argsort(p_flat)
    # Benjamini-Hochberg
    thresholds = np.arange(1, n_tests+1) / n_tests * ALPHA
    significant = p_flat[ranked] <= thresholds
    max_sig_idx = np.where(significant)[0]
    if len(max_sig_idx) > 0:
        p_thresh = p_flat[ranked[max_sig_idx[-1]]]
    else:
        p_thresh = 0

    sig_mask = p_matrix <= p_thresh  # True = 显著

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    im = ax.imshow(diff_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[t[0], t[-1], n_ch-0.5, -0.5])
    # 叠加显著性打点
    for ch in range(n_ch):
        for tp in range(EPOCH_LEN):
            if sig_mask[ch, tp]:
                ax.plot(t[tp], ch, '.', color='#333', markersize=0.4, alpha=0.4)

    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(COMMON_CH_LABELS, fontsize=7, color=AXIS_COLOR)
    ax.set_xlabel('Time (s)', fontsize=9, color=AXIS_COLOR)
    ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
    ax.tick_params(colors=AXIS_COLOR, labelsize=7)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('µV (S2 − S1)', fontsize=8, color=AXIS_COLOR)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=6)

    ax.set_title(f'Bootstrap 差异热力图 (FDR α={ALPHA}, 点=显著)',
                 fontsize=10, color='#444')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  [图] Bootstrap 热力图 → {save_path}')

    return diff_matrix, sig_mask


def print_stats_table(metrics_s1, metrics_s2, trial_counts_s1, trial_counts_s2):
    """打印统计汇总表"""
    print(f'\n{"="*80}')
    print(f'  统计汇总')
    print(f'{"="*80}')

    for cmp_name in ['P3']:
        print(f'\n--- {cmp_name} 平均幅度 (µV) ---')
        print(f'{"ROI":<20} {"方向":<6}', end='')
        for k in DIR_KEYS:
            print(f' {DIR_MAP[k][0]:>12}', end='')
        print()

        for ri, name in enumerate(ROI_LABELS):
            print(f'{name:<20} {"S1":<6}', end='')
            for k in DIR_KEYS:
                v = metrics_s1[k][cmp_name]['mean_amp'][ri]
                print(f' {v:>10.2f}  ', end='')
            print()
            print(f'{"":<20} {"S2":<6}', end='')
            for k in DIR_KEYS:
                v = metrics_s2[k][cmp_name]['mean_amp'][ri]
                print(f' {v:>10.2f}  ', end='')
            print()
            print(f'{"":<20} {"n":<6}', end='')
            for k in DIR_KEYS:
                print(f' {f"({trial_counts_s1[k]}/{trial_counts_s2[k]})":>12}', end='')
            print()


# ======================================================================
#  主流程
# ======================================================================
def main():
    print('='*60)
    print('  OpenBCI 箭头刺激 ERP 分析')
    print('='*60)
    print(f'  采样率: {FS} Hz')
    print(f'  uV 系数: {SCALE_UV:.6f} uV/count (Vref={VREF}V, gain=x{GAIN})')
    print(f'  带通滤波: {BP_LOW}–{BP_HIGH} Hz')
    print(f'  陷波: {NOTCH} Hz')
    print(f'  Epoch: [{T_BEFORE*1000:.0f}, +{T_AFTER*1000:.0f}] ms')
    print(f'  坏 trial 阈值: ±{REJECT_THRESH_UV} µV')
    print(f'  Bootstrap: {N_BOOT} 次, α={ALPHA}')

    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'session1'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'session2'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'comparison'), exist_ok=True)

    # 设计滤波器
    sos = design_filters(FS, BP_LOW, BP_HIGH, NOTCH, FILTER_ORDER)
    print(f'  滤波器级数: {sos.shape[0]}')

    # 分析每个 session
    results = []
    for i, (path, label) in enumerate(PATHS):
        prefix = f'session{i+1}'
        out_sub = os.path.join(OUT_DIR, prefix)

        res = analyze_session(path, label, sos, out_sub)
        results.append(res)

        # 画出滤波波形
        plot_raw_trace(res['data_uv'], FS, label,
                       os.path.join(out_sub, '01_filtered_raw.png'))

        # 画出各方向 ERP
        plot_erp_directions(res['erp_by_dir'], res['trial_counts'], FS, label,
                           os.path.join(out_sub, '02_erp_directions.png'))

        # 画出 ROI ERP
        plot_roi_erp(res['roi_erp_by_dir'], res['trial_counts'], FS, label,
                    os.path.join(out_sub, '03_erp_roi.png'))

        # 提取指标
        metrics = {}
        for k in DIR_KEYS:
            erp = res['erp_by_dir'][k]
            if not np.any(np.isnan(erp)):
                metrics[k] = extract_metrics(erp, FS, ERP_WINDOWS)
            else:
                metrics[k] = None
        res['metrics'] = metrics

    s1, s2 = results

    # ---- 对比 ----
    comp_dir = os.path.join(OUT_DIR, 'comparison')

    # 1) P3 柱状图
    p3_s1, p3_s2 = plot_comparison_bar(
        s1['roi_erp_by_dir'], s2['roi_erp_by_dir'],
        s1['trial_counts'], s2['trial_counts'],
        os.path.join(comp_dir, '01_p3_comparison.png'))

    # 2) 差异波
    plot_difference_waves(s1['roi_erp_by_dir'], s2['roi_erp_by_dir'], FS,
                         os.path.join(comp_dir, '02_difference_waves.png'))

    # 3) Bootstrap 热力图 (在所有 trial 上进行)
    print(f'\n  [运行 Bootstrap 差异检验...]')
    plot_bootstrap_heatmap(
        s1['epochs_clean'], s1['labels_clean'],
        s2['epochs_clean'], s2['labels_clean'],
        FS, os.path.join(comp_dir, '03_bootstrap_heatmap.png'))

    # 4) 统计表格
    print_stats_table(s1['metrics'], s2['metrics'],
                      s1['trial_counts'], s2['trial_counts'])

    # ---- 保存汇总 JSON ----
    summary = {
        'session1_trials': {f'{DIR_MAP[k][0]}': int(s1['trial_counts'][k])
                           for k in DIR_KEYS},
        'session2_trials': {f'{DIR_MAP[k][0]}': int(s2['trial_counts'][k])
                           for k in DIR_KEYS},
        'p3_values_s1': p3_s1.tolist(),
        'p3_values_s2': p3_s2.tolist(),
        'roi_labels': ROI_LABELS,
        'direction_labels': [DIR_MAP[k][0] for k in DIR_KEYS],
    }
    with open(os.path.join(comp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  [JSON] 汇总 → {os.path.join(comp_dir, "summary.json")}')

    print(f'\n{"="*60}')
    print(f'  分析完成! 全部输出在: {OUT_DIR}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
