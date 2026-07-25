#!/usr/bin/env python3
"""
脑区激活分析 — 箭头刺激诱发的 ROI 对比
========================================
对两批 EEG 数据，合并所有方向，比较四个脑区 (F/C/P/O) 的 ERP 响应，
找出哪些脑区被显著激活及其时序。
"""

import numpy as np
from scipy import signal as sg
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, json, os
warnings.filterwarnings('ignore')

# ============ 配置 ============
PATHS = [
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-06_14-39-25\OpenBCI-RAW-2026-07-06_14-39-25.txt',
     'Session 1 (14:39)'),
    (r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-06_14-16-16\OpenBCI-RAW-2026-07-06_14-16-16.txt',
     'Session 2 (14:16)'),
]
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis'

FS = 500.0
GAIN = 6.0
SCALE = 4.5 / (2**23 - 1) / GAIN * 1e6  # 0.08941

# 通道映射 (硬件通道 -> EXG索引)
CH_MAP = {
    1:'Oz',2:'C3',3:'unused',4:'Fz',5:'C4',6:'Cz',7:'F3',
    8:'O2',9:'P3',10:'Pz',11:'unused',12:'P4',13:'unused',14:'F4',15:'O1'
}
COMMON_CH = [2,4,5,6,7,8,9,10,12,14,15]  # 硬件通道

ROIS = {
    'Frontal (F)':  [7,4,14],   # F3,Fz,F4
    'Central (C)':  [2,6,5],    # C3,Cz,C4
    'Parietal (P)': [9,10,12],  # P3,Pz,P4
    'Occipital (O)':[15,8],     # O1,O2
}
ROI_ORDER = ['Frontal (F)', 'Central (C)', 'Parietal (P)', 'Occipital (O)']

ERP_WINDOWS = [
    ('P1', 0.080, 0.120, '早期视觉\n80-120ms'),
    ('N1', 0.140, 0.200, '空间注意\n140-200ms'),
    ('P3', 0.300, 0.500, '认知处理\n300-500ms'),
]

N_BOOT = 5000

# ============ 数据处理 ============
def process_session(path, label):
    """读取、滤波、分段、合并所有方向"""
    with open(path) as f:
        lines = f.readlines()

    data = []
    markers = []
    for line in lines[5:]:
        parts = line.strip().split(',')
        if len(parts) > 32:
            try:
                data.append([float(parts[i].strip()) for i in range(1, 17)])
                markers.append(float(parts[32].strip()))
            except:
                pass

    data = np.array(data, dtype=np.float64).T  # (16, n)
    markers = np.array(markers, dtype=np.float64)

    # 选共同通道
    exg_idx = [ch - 1 for ch in COMMON_CH]
    sel = data[exg_idx, :]  # (11, n)

    # 找到稳定段
    all_onsets = [i for k in [2.0001,2.0002,2.0003,2.0004]
                   for i in np.where(np.abs(markers - k) < 0.00005)[0]]
    all_onsets = sorted(all_onsets)

    if len(all_onsets) < 5:
        print(f'{label}: 刺激太少 ({len(all_onsets)}), 跳过')
        return None

    # 滤波
    sos = sg.butter(4, [1/(FS/2), 45/(FS/2)], btype='band', output='sos')
    sel_filt = np.zeros_like(sel)
    for ch in range(sel.shape[0]):
        dm = sel[ch] - sel[ch].mean()
        sel_filt[ch] = sg.sosfiltfilt(sos, dm) * SCALE

    # 分段 (合并所有方向)
    t_before, t_after = 200, 400
    epochs = []
    for idx in all_onsets:
        start, end = idx - t_before, idx + t_after
        if start >= 0 and end <= sel_filt.shape[1]:
            ep = sel_filt[:, start:end]
            ep -= ep[:, :t_before].mean(axis=1, keepdims=True)
            epochs.append(ep)

    epochs = np.stack(epochs, axis=0)  # (n_trials, 11, 600)
    print(f'{label}: {epochs.shape[0]} trials, {epochs.shape[1]} channels')

    # ROI 合并
    roi_epochs = {}
    for roi_name, hw_chs in ROIS.items():
        idx_in_common = [COMMON_CH.index(ch) for ch in hw_chs]
        # 取平均: (n_trials, 600)
        roi_data = epochs[:, idx_in_common, :].mean(axis=1)
        roi_epochs[roi_name] = roi_data  # (n_trials, 600)

    return {
        'label': label,
        'n_trials': epochs.shape[0],
        'epochs': epochs,           # (n, 11, 600)
        'roi_epochs': roi_epochs,   # dict of (n, 600)
    }


def bootstrap_vs_baseline(data, n_iter=5000):
    """bootstrap 检验: data 是否显著 ≠ 0 (双尾)
    data: (n_trials,) — 某个时间窗口内均值的数组
    返回: (mean, ci_low, ci_high, p)
    """
    obs_mean = data.mean()
    n = len(data)
    boot_means = np.zeros(n_iter)
    for i in range(n_iter):
        idx = np.random.randint(0, n, size=n)
        boot_means[i] = data[idx].mean()
    ci = np.percentile(boot_means, [2.5, 97.5])
    # 双尾 p: H0 = 均值=0
    p = 2 * min(np.mean(boot_means >= 0), np.mean(boot_means <= 0))
    return obs_mean, ci[0], ci[1], p


def analyze_roi_activation(res):
    """对每个 ROI 计算 P1/N1/P3 幅值并进行 bootstrap 显著性检验"""
    t = (np.arange(600) - 200) / FS  # -200 to 800 ms
    results = []

    for roi_name in ROI_ORDER:
        roi_data = res['roi_epochs'][roi_name]  # (n_trials, 500)

        for comp_name, t1, t2, _ in ERP_WINDOWS:
            mask = (t >= t1) & (t <= t2)
            if mask.sum() == 0:
                continue
            # 每个 trial 在该窗口的均值
            trial_means = roi_data[:, mask].mean(axis=1)  # (n_trials,)

            mean_val, ci_low, ci_high, p = bootstrap_vs_baseline(trial_means, N_BOOT)
            sig_stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

            results.append({
                'roi': roi_name,
                'component': comp_name,
                'window_ms': f'{t1*1000:.0f}-{t2*1000:.0f}',
                'mean_uV': mean_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p': p,
                'sig': sig_stars,
            })

    return results


# ============ 可视化 ============
def plot_roi_comparison(res_list, out_dir):
    """图1: 四个 ROI 的 ERP (合并方向), 标注显著窗口"""
    fig, axes = plt.subplots(len(ROI_ORDER), 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle('箭头刺激诱发的脑区激活 (所有方向合并)', fontsize=13, y=0.98)

    colors_sessions = ['#4A72C4', '#E8833A']
    t = (np.arange(600) - 200) / FS

    # 对每个 ROI 画图
    for ri, roi_name in enumerate(ROI_ORDER):
        ax = axes[ri]
        ax.set_facecolor('#FAFAFA')

        # 标注 ERP 窗口背景
        window_ymax = -1  # track for window label placement
        for comp_name, t1, t2, label in ERP_WINDOWS:
            ax.axvspan(t1, t2, alpha=0.06, color='#1f77b4', zorder=0)
            ax.text((t1+t2)/2, ax.get_ylim()[0] if ri==0 else ax.get_ylim()[0],
                    label, fontsize=5.5, ha='center', color='#1f77b4', alpha=0.7)

        # 画两条曲线
        for si, res in enumerate(res_list):
            roi_erp = res['roi_epochs'][roi_name].mean(axis=0)  # (500,)
            ax.plot(t, roi_erp, color=colors_sessions[si], linewidth=1.2,
                    label=f"{res['label']} (n={res['n_trials']})")

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
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].spines['bottom'].set_color('#E0E0E0')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, 'comparison', '04_roi_activation_erp.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA')
    plt.close(fig)
    print(f'  [图] 脑区激活 ERP → {path}')


def plot_activation_heatmap(res_list, out_dir):
    """图2: ROI × 成分热力图"""
    fig, axes = plt.subplots(1, len(res_list), figsize=(10, 4), sharey=True)
    fig.patch.set_facecolor('#FAFAFA')

    if len(res_list) == 1:
        axes = [axes]

    for si, (res, ax) in enumerate(zip(res_list, axes)):
        results = analyze_roi_activation(res)
        ax.set_facecolor('#FAFAFA')

        # 构建矩阵: ROI rows × component cols
        comps = [e[0] for e in ERP_WINDOWS]
        matrix = np.zeros((len(ROI_ORDER), len(comps)))
        sig_matrix = np.zeros_like(matrix, dtype=bool)

        for r in results:
            ri = ROI_ORDER.index(r['roi'])
            ci = comps.index(r['component'])
            matrix[ri, ci] = r['mean_uV']
            sig_matrix[ri, ci] = r['p'] < 0.05

        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.15)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='equal')

        # 标注数值和显著性
        for ri in range(len(ROI_ORDER)):
            for ci in range(len(comps)):
                val = matrix[ri, ci]
                sig = sig_matrix[ri, ci]
                text_color = 'white' if abs(val) > vmax*0.5 else '#444'
                ax.text(ci, ri, f'{val:+.3f}', ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold')
                if sig:
                    ax.text(ci, ri-0.3, '●', ha='center', fontsize=8, color='#FF3333')

        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(comps, fontsize=9)
        ax.set_yticks(range(len(ROI_ORDER)))
        rois_short = [r.split(' ')[1].strip('()') for r in ROI_ORDER]
        ax.set_yticklabels(rois_short, fontsize=9)
        ax.set_title(res['label'], fontsize=10, color='#444')
        for sp in ax.spines.values():
            sp.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('µV', fontsize=7, color='#888')
        cbar.ax.tick_params(colors='#888', labelsize=6)

    fig.suptitle('箭头刺激诱发的脑区激活幅度 (红色=显著, p<0.05)',
                fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison', '05_roi_activation_heatmap.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA', bbox_inches='tight')
    plt.close(fig)
    print(f'  [图] 激活热力图 → {path}')


def plot_activation_bar_summary(all_results, out_dir):
    """图3: 顶叶/枕叶 P3 柱状图对比"""
    for comp_name in ['P3', 'P1', 'N1']:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FAFAFA')

        x = np.arange(len(ROI_ORDER))
        w = 0.30

        for si, res in enumerate(all_results):
            results = analyze_roi_activation(res)
            vals = [r['mean_uV'] for r in results if r['component'] == comp_name]
            cis_low = [r['ci_low'] for r in results if r['component'] == comp_name]
            cis_high = [r['ci_high'] for r in results if r['component'] == comp_name]

            color = ['#4A72C4', '#E8833A'][si]
            bars = ax.bar(x + (si-0.5)*w, vals, w,
                         color=color, alpha=0.85, edgecolor='white', linewidth=0.5,
                         label=res['label'])
            # error bars
            yerr_low = np.array(vals) - np.array(cis_low)
            yerr_high = np.array(cis_high) - np.array(vals)
            ax.errorbar(x + (si-0.5)*w, vals,
                       yerr=[yerr_low, yerr_high],
                       fmt='none', color='#333', capsize=2, capthick=1, linewidth=1)

        ax.set_xticks(x)
        rois_short = [r.split(' (')[0] for r in ROI_ORDER]
        ax.set_xticklabels(rois_short, fontsize=10)
        ax.set_ylabel(f'{comp_name} 幅度 (µV)', fontsize=9, color='#888')
        ax.axhline(y=0, color='#999', linewidth=0.5)
        ax.legend(fontsize=8)
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        ax.set_title(f'{comp_name} 幅度 — 各脑区对比 (误差线=95% CI)', fontsize=11, color='#444')
        plt.tight_layout()
        path = os.path.join(out_dir, 'comparison', f'06_{comp_name}_bar.png')
        fig.savefig(path, dpi=150, facecolor='#FAFAFA')
        plt.close(fig)
        print(f'  [图] {comp_name} 柱状图 → {path}')


def plot_activation_sequential(all_results, out_dir):
    """图4: 各脑区激活的时间进程 — 反应顺序"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    t = (np.arange(600) - 200) / FS

    for si, res in enumerate(all_results):
        linestyle = '-' if si == 0 else '--'
        for ri, roi_name in enumerate(ROI_ORDER):
            roi_erp = res['roi_epochs'][roi_name].mean(axis=0)
            # 只画 P3 时段
            ax.plot(t, roi_erp, color=plt.cm.Set2(ri/4), linewidth=1.0,
                   linestyle=linestyle,
                   label=f"{roi_name.split(' ')[0]} ({res['label'][:9]})")

    ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='--')
    ax.axhline(y=0, color='#999', linewidth=0.4)
    ax.set_xlabel('Time (s)', fontsize=9, color='#888')
    ax.set_ylabel('µV', fontsize=9, color='#888')
    ax.tick_params(colors='#888', labelsize=8)
    ax.grid(True, axis='y', linestyle=':', color='#E0E0E0', linewidth=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6, ncol=2, loc='upper right')
    ax.set_title('四个脑区激活的时间进程', fontsize=12, color='#444')
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison', '07_activation_timing.png')
    fig.savefig(path, dpi=150, facecolor='#FAFAFA')
    plt.close(fig)
    print(f'  [图] 激活时间进程 → {path}')


# ============ 主入口 ============
def main():
    os.makedirs(os.path.join(OUT_DIR, 'comparison'), exist_ok=True)

    # 处理两批数据
    res_list = []
    for path, label in PATHS:
        res = process_session(path, label)
        if res:
            res_list.append(res)

    # 画图
    plot_roi_comparison(res_list, OUT_DIR)
    plot_activation_heatmap(res_list, OUT_DIR)
    plot_activation_bar_summary(res_list, OUT_DIR)
    plot_activation_sequential(res_list, OUT_DIR)

    # 打印统计表
    print(f'\n{"="*80}')
    print(f'  脑区激活统计 — bootstrap 检验 H0: 均值=0')
    print(f'  (显著 = 该脑区-成分组合被箭头刺激显著激活)')
    print(f'{"="*80}')

    for si, res in enumerate(res_list):
        print(f'\n  --- {res["label"]} ---')
        print(f'  {"ROI":<20} {"成分":<8} {"窗口":<10} {"均值(µV)":<12} {"95% CI":<20} {"p":<10} {"显著性"}')
        results = analyze_roi_activation(res)
        for r in results:
            print(f'  {r["roi"]:<20} {r["component"]:<8} {r["window_ms"]:<10} '
                  f'{r["mean_uV"]:<+10.4f}  '
                  f'[{r["ci_low"]:<+7.4f}, {r["ci_high"]:<+7.4f}]  '
                  f'{r["p"]:<10.4f} {r["sig"]}')

    # 保存 JSON
    all_data = []
    for si, res in enumerate(res_list):
        results = analyze_roi_activation(res)
        all_data.append({
            'label': res['label'],
            'n_trials': res['n_trials'],
            'results': results,
        })
    with open(os.path.join(OUT_DIR, 'comparison', 'activation_stats.json'), 'w') as f:
        # convert to serializable
        clean = []
        for d in all_data:
            clean.append({
                'label': d['label'],
                'n_trials': d['n_trials'],
                'results': [{k: float(v) if isinstance(v, (np.floating,)) else v
                            for k, v in r.items()} for r in d['results']]
            })
        json.dump(clean, f, indent=2)
    print(f'\n  [JSON] → {os.path.join(OUT_DIR, "comparison", "activation_stats.json")}')
    print(f'\n  分析完成!')


if __name__ == '__main__':
    main()
