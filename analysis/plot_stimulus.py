#!/usr/bin/env python3
"""
OpenBCI ODF TXT → W_TimeSeries 风格可视化
===========================================
完全模拟 OpenBCI GUI 的实时绘图管线:
  ODF 文件(µV) → 去 DC → BP 5-50 Hz → 50 Hz notch → ±200 µV y 轴 → 5 秒滚动窗口

用法:
    python plot_stimulus.py <文件路径>
    python plot_stimulus.py "C:/path/to/OpenBCI-RAW-*.txt"
"""

import sys, os, numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#818181','#7C4B8D','#36579E','#317159','#E1B530','#FD5E34','#E3242B','#A25231']
MARKER_COLOR = '#FF3333'
BG_COLOR = '#FFFFFF'
GRID_COLOR = '#E8E8E8'
AXIS_COLOR = '#BBBBBB'
FIG_W_MM = 297
FIG_H_PER_CH = 28
DPI = 200
Y_RANGE = 200.0
WIN_SEC = 5  # 显示窗口宽度（秒）


def parse_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sr = 500
    col_names = []
    data_line = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('%Sample Rate'):
            sr = float(s.split('=')[1].strip().split()[0])
        elif s.startswith('Sample'):
            col_names = [c.strip() for c in s.split(',')]
            data_line = i + 1
            break
    if not col_names:
        raise ValueError('找不到列名行')

    stim_col = next((i for i, n in enumerate(col_names) if n == 'Stimulus Marker'), -1)
    exg_cols = [i for i, n in enumerate(col_names) if n.startswith('EXG Channel')]
    if not exg_cols:
        exg_cols = [i for i, n in enumerate(col_names)
                    if i != stim_col and n not in (
                        'Timestamp (Formatted)', 'Timestamp', 'Sample Index',
                        'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
                        'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2')]

    # 读取所有数据行
    raw_lines = []
    stim_events = []
    for line in lines[data_line:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        nc = len(parts) - 1  # 跳过 Timestamp (Formatted)
        if nc < 3:
            continue
        try:
            row = [float(parts[j].strip()) for j in range(nc)]
        except (ValueError, IndexError):
            continue
        raw_lines.append(row)
        si = len(raw_lines) - 1
        if stim_col >= 0 and stim_col < nc:
            mv = row[stim_col]
            if mv >= 2.0:
                mn = int(round((mv - 2.0) * 10000))
                if mn < 1: mn = int(mv)
                stim_events.append((si / sr, mn))

    data = np.array(raw_lines, dtype=np.float64).T  # (ncols, nsamples)
    n = data.shape[1]
    dur = n / sr
    return {'sr': sr, 'data': data, 'exg_cols': exg_cols,
            'stim': stim_events, 'duration': dur,
            'file_name': os.path.splitext(os.path.basename(path))[0]}


def plot(info, out_path):
    data = info['data']
    exg = info['exg_cols']
    nchan = len(exg)
    sr = info['sr']
    stim = info['stim']
    dur = info['duration']
    mn = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    t = np.arange(data.shape[1]) / sr

    # ── 设计滤波器（同 GUI: BP 5-50 Hz + 50 Hz notch）──
    sos_bp = sg.butter(4, [5/(sr/2), 50/(sr/2)], btype='band', output='sos')
    sos_notch = sg.tf2sos(*sg.iirnotch(50, 30, sr))
    sos_all = np.vstack([sos_bp, sos_notch])
    print(f'  Filter: BP 5-50 Hz + 50 Hz notch')

    # ── 逐通道处理 ──
    ch_data = []
    for col in exg:
        # ODF 数据已是 µV，只需去 DC
        x = data[col] - np.mean(data[col])
        x = sg.sosfiltfilt(sos_all, x)
        # 去边缘暂态
        edge = int(sr)
        if len(x) > 2 * edge:
            x[:edge] = 0; x[-edge:] = 0
        ch_data.append(x)

    # ── 只显示最后 5 秒 ──
    t0 = max(0, dur - WIN_SEC)
    mask = t >= t0
    stim_win = [(t_, n_) for t_, n_ in stim if t0 <= t_ <= dur]
    print(f'  Window: {t0:.1f}s – {dur:.1f}s  |  Y: +/-{Y_RANGE:.0f} uV')

    fig_h_mm = max(50, nchan * FIG_H_PER_CH + 25)
    fig, axes = plt.subplots(nchan, 1,
                             figsize=(FIG_W_MM/25.4, fig_h_mm/25.4),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f'EEG (BP 5-50Hz + notch) -- {info["file_name"]}',
                 color='#444', fontsize=11, fontweight='normal', y=0.98)

    for ci in range(nchan):
        ax = axes[ci][0]
        ax.set_facecolor(BG_COLOR)
        c = COLORS[ci % len(COLORS)]
        ax.plot(t, ch_data[ci], color=c, linewidth=0.35, antialiased=True)
        ax.set_xlim(t0, dur)
        ax.set_ylim(-Y_RANGE, Y_RANGE)
        ax.axhline(y=0, color='#DDD', linewidth=0.3, zorder=0)

        for t_, n_ in stim_win:
            ax.axvline(x=t_, color=MARKER_COLOR, linewidth=0.4, alpha=0.6, zorder=2)
            if ci == 0:
                ax.text(t_, Y_RANGE*0.85, mn.get(n_, f'M#{n_}'),
                        fontsize=5.5, color=MARKER_COLOR,
                        ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.08',
                                  facecolor='#FFF', edgecolor=MARKER_COLOR, alpha=0.7))

        # y 轴
        ticks = np.linspace(-Y_RANGE, Y_RANGE, 3).astype(int)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks], fontsize=5.5, color=AXIS_COLOR)
        ax.tick_params(axis='y', length=2, pad=1)
        ax.set_ylabel('uV', fontsize=5.5, color=AXIS_COLOR, labelpad=0)
        ax.grid(True, axis='y', linestyle=':', color=GRID_COLOR, linewidth=0.3)
        ax.tick_params(colors=AXIS_COLOR, labelsize=5.5)
        for sp in ['top', 'right', 'bottom']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID_COLOR)
        ax.text(0.005, 0.04, str(ci+1), transform=ax.transAxes,
                fontsize=7.5, fontweight='bold', color=c, va='bottom', ha='left')

        # 通道编号内的 y 值标签（标注当前通道的 DC 值？不需要，去 DC 后自然居零）

    axes[-1][0].set_xlabel('Time (sec)', fontsize=7, color=AXIS_COLOR)
    axes[-1][0].tick_params(colors=AXIS_COLOR, labelsize=6)
    axes[-1][0].spines['bottom'].set_visible(True)
    axes[-1][0].spines['bottom'].set_color(GRID_COLOR)

    if stim_win:
        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Line2D([0],[0],color=COLORS[0],linewidth=1,label=f'EEG {nchan}ch'),
            Line2D([0],[0],color=MARKER_COLOR,linewidth=1.5,label=f'Stimulus x{len(stim_win)}'),
        ], loc='upper right', fontsize=6, labelcolor=AXIS_COLOR,
           facecolor=BG_COLOR, edgecolor=GRID_COLOR, framealpha=0.9)

    fig.text(0.5, 0.005,
             f'Sample Rate: {sr:.0f} Hz | Y: +/-{Y_RANGE:.0f} uV | Window: {WIN_SEC}s',
             ha='center', fontsize=5.5, color=AXIS_COLOR)

    plt.tight_layout(rect=[0, 0.015, 1, 0.97])
    fig.savefig(out_path, dpi=DPI, facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_stimulus.py <ODF文件路径>')
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print('ERROR: File not found:', path)
        sys.exit(1)
    print('Reading:', path)
    info = parse_txt(path)
    out = os.path.join(os.path.dirname(path) or '.',
                       info['file_name'] + '_stimulus.png')
    plot(info, out)
