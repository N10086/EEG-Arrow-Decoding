#!/usr/bin/env python3
"""
sLORETA fitting residual analysis — how well do source activities reconstruct
the observed scalp ERP?

Pipeline:  actual scalp φ → sLORETA → source ĵ → forward φ̂ = K·ĵ → compare φ vs φ̂
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

ERP_WINDOWS = [('P1',0.080,0.130,'#27ae60'),('N1',0.140,0.200,'#7f8c8d'),
               ('P2',0.200,0.300,'#e67e22'),('P3',0.300,0.500,'#9B59B6')]
ROI_ORDER = ['Frontal','Central','Parietal','Occipital']
CH_MAP = {1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',
          8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS = sorted(CH_MAP.keys()); MNE_NAMES = [CH_MAP[c] for c in HW_CHS]

SCALP_ROIS = {
    'Frontal': [7,4,14], 'Central': [2,6,5],
    'Parietal': [9,10,12], 'Occipital': [15,1,8],
}

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

# ====== Forward model ======
print('Setting up fsaverage...')
src = mne.setup_source_space(SUBJECT, spacing='ico3', subjects_dir=SUBJECTS_DIR, add_dist=False)
model = mne.make_bem_model(SUBJECT, ico=4, conductivity=[0.3,0.006,0.3], subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)
labels = mne.read_labels_from_annot(SUBJECT, 'aparc', subjects_dir=SUBJECTS_DIR, verbose=False)

# ====== Load ======
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

epochs = mne.Epochs(raw, events, {'Up':1,'Down':2,'Left':3,'Right':4},
                     -0.2, 0.8, baseline=(-0.2,0.0), preload=True,
                     reject={'eeg':100e-6}, proj=True, verbose=False)
evoked = epochs.average()
t = evoked.times
print(f'Good epochs: {len(epochs)}')

# ====== sLORETA (for source display) + MNE (for residual, has physical units) ======
fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem,
                                 meg=False, eeg=True, verbose=False)
cov = mne.compute_covariance(epochs, tmax=0.0, method='empirical', verbose=False)
inv = make_inverse_operator(evoked.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)

# sLORETA for source display (statistical map, no physical units)
stc_sloreta = apply_inverse(evoked, inv, lambda2=1/9.0, method='sLORETA',
                              pick_ori=None, verbose=False)
# MNE for residual computation: use 'normal' orientation to preserve sign.
# pick_ori=None returns the L2 norm (unsigned), which cannot be forward-projected.
# pick_ori='normal' returns signed normal component → correct polarity.
stc_mne = apply_inverse(evoked, inv, lambda2=1/9.0, method='MNE',
                         pick_ori='normal', verbose=False)
print(f'sLORETA: {stc_sloreta.shape}, MNE (normal): {stc_mne.shape}')

# ====== Forward project source → scalp ======
print('Forward-projecting sources back to scalp...')

# stc_mne is in Am (signed normal dipole moment); forward maps Am → V
pred_evoked = mne.apply_forward(fwd, stc_mne, evoked.info, verbose=False)

# pred_evoked is an EvokedArray — scalp potentials in V
phi_actual = evoked.data          # (12, 501) in V
phi_pred   = pred_evoked.data    # (12, 501) in V

# Debug: check relative scaling
rms_actual = np.sqrt(np.mean(phi_actual**2))
rms_pred   = np.sqrt(np.mean(phi_pred**2))
print(f'  Actual RMS: {rms_actual:.6e} V')
print(f'  Pred RMS:   {rms_pred:.6e} V')
print(f'  Ratio:      {rms_actual/rms_pred:.2f} (MNE shrinks amplitude due to regularization)')

# ====== Optimal scaling per channel ======
# MNE regularization shrinks amplitude; find alpha that minimizes ||phi - alpha*phi_hat||^2
n_channels = 12
opt_scale = np.zeros(n_channels)
for ch in range(n_channels):
    num = np.dot(phi_actual[ch], phi_pred[ch])
    den = np.dot(phi_pred[ch], phi_pred[ch])
    opt_scale[ch] = num / den if den > 0 else 1.0
# Apply optimal scaling
phi_pred_scaled = phi_pred * opt_scale[:, np.newaxis]

# ====== Per-time-point residual (with optimal scaling) ======
# Residual variance: 1 - R^2 after optimal linear scaling
residual_var = np.zeros(len(t))
residual_var_raw = np.zeros(len(t))  # without optimal scaling
for i in range(len(t)):
    # Raw (as-is) residual
    rn = np.sum((phi_actual[:, i] - phi_pred[:, i])**2)
    rd = np.sum(phi_actual[:, i]**2)
    residual_var_raw[i] = rn / rd if rd > 0 else 1.0
    # Scaled residual (amplitude-invariant shape mismatch)
    sn = np.sum((phi_actual[:, i] - phi_pred_scaled[:, i])**2)
    sd = np.sum(phi_actual[:, i]**2)
    residual_var[i] = sn / sd if sd > 0 else 1.0

# GFP: Global Field Power
gfp_actual = np.std(phi_actual, axis=0)
gfp_pred   = np.std(phi_pred, axis=0)
gfp_pred_scaled = np.std(phi_pred_scaled, axis=0)

# Per-channel correlation (against optimally scaled pred)
ch_corr = []
for ch in range(n_channels):
    r = np.corrcoef(phi_actual[ch], phi_pred_scaled[ch])[0, 1]
    ch_corr.append(r)

# ROI residual (using scaled predictions)
roi_residual = {}  # residual variance (shape-only, after optimal scaling)
roi_residual_raw = {}  # raw residual (includes amplitude shrinkage)
roi_actual = {}
roi_pred_scaled = {}
roi_pred_raw = {}
for rn, chs in SCALP_ROIS.items():
    ch_idx = [HW_CHS.index(c) for c in chs]
    a = phi_actual[ch_idx].mean(axis=0)
    p_raw = phi_pred[ch_idx].mean(axis=0)
    p_scl = phi_pred_scaled[ch_idx].mean(axis=0)
    roi_actual[rn] = a
    roi_pred_scaled[rn] = p_scl
    roi_pred_raw[rn] = p_raw
    # Scaled residual per time point
    rr = np.zeros(len(t))
    rr_raw = np.zeros(len(t))
    for i in range(len(t)):
        sn = np.sum((a[i] - p_scl[i])**2)
        sd = np.sum(a[i]**2)
        rr[i] = sn / sd if sd > 0 else 1.0
        rn_raw = np.sum((a[i] - p_raw[i])**2)
        rd_raw = np.sum(a[i]**2)
        rr_raw[i] = rn_raw / rd_raw if rd_raw > 0 else 1.0
    roi_residual[rn] = rr
    roi_residual_raw[rn] = rr_raw

# ====== FIGURE 1: Channel-wise actual vs predicted ======
fig1, axes1 = plt.subplots(4, 3, figsize=(16, 12))
fig1.patch.set_facecolor('white')
ch_colors = {0:'#4A72C4',1:'#E8833A',2:'#5CB85C',3:'#9B59B6',4:'#D94F70',
             5:'#34495E',6:'#1ABC9C',7:'#E74C3C',8:'#3498DB',9:'#F39C12',
             10:'#2ECC71',11:'#8E44AD'}

for chi, ch_name in enumerate(MNE_NAMES):
    ax = axes1[chi//3, chi%3]
    ax.set_facecolor('#FAFAFA')
    # Scale to µV for display
    ax.plot(t, phi_actual[chi]*1e6, color='#2c3e50', lw=2.0, label='Actual')
    ax.plot(t, phi_pred[chi]*1e6, color='#E74C3C', lw=1.0, ls=':', alpha=0.5, label=f'Raw (α={opt_scale[chi]:.2f})')
    ax.plot(t, phi_pred_scaled[chi]*1e6, color='#E74C3C', lw=2.0, ls='--', label='Scaled')
    ax.set_title(f'{ch_name} (r={ch_corr[chi]:.3f})', fontsize=9, fontweight='bold')
    ax.axvline(0, color='#333', ls='--', lw=0.5)
    ax.axhline(0, color='#999', lw=0.3)
    ax.tick_params(labelsize=7)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    if chi == 0: ax.legend(fontsize=7)

fig1.suptitle('Scalp ERP: Actual vs sLORETA Reconstructed (Session 2, 200 trials)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, '17_residual_channel_fit.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig1)
print('Saved: 17_residual_channel_fit.png')

# ====== FIGURE 2: Residual variance time course ======
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
fig2.patch.set_facecolor('white')

ax = axes2[0]; ax.set_facecolor('#FAFAFA')
ax.fill_between(t, residual_var_raw*100, 0, color='#999', alpha=0.2, label='Raw (incl. amplitude shrinkage)')
ax.plot(t, residual_var_raw*100, color='#999', lw=1.5, ls=':')
ax.fill_between(t, residual_var*100, 0, color='#E74C3C', alpha=0.3, label='Scaled (shape mismatch only)')
ax.plot(t, residual_var*100, color='#E74C3C', lw=2.0)
for cn, t1, t2, cc in ERP_WINDOWS:
    ax.axvspan(t1, t2, alpha=0.06, color=cc)
    mean_r = residual_var[(t>=t1)&(t<=t2)].mean() * 100
    ax.text((t1+t2)/2, 85, f'{cn}\n{mean_r:.1f}%', ha='center', fontsize=8,
            color=cc, fontweight='bold')
ax.axvline(0, color='#333', ls='--', lw=0.8)
ax.set_ylabel('Residual Variance (%)', fontsize=11, fontweight='bold')
ax.set_title('Global Fitting Residual (all 12 channels)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax = axes2[1]; ax.set_facecolor('#FAFAFA')
colors_roi = ['#4A72C4', '#E8833A', '#5CB85C', '#9B59B6']
for ri, rn in enumerate(ROI_ORDER):
    ax.plot(t, roi_residual_raw[rn]*100, color=colors_roi[ri], lw=1.0, ls=':', alpha=0.4)
    ax.plot(t, roi_residual[rn]*100, color=colors_roi[ri], lw=2.0, label=f'{rn} (scaled)')
for cn, t1, t2, cc in ERP_WINDOWS:
    ax.axvspan(t1, t2, alpha=0.06, color=cc)
ax.axvline(0, color='#333', ls='--', lw=0.8)
ax.set_ylabel('Residual Variance (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_title('ROI-wise Residual Variance', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

fig2.suptitle('sLORETA Model Fit — Residual Variance (Actual vs Reconstructed)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '18_residual_time_course.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig2)
print('Saved: 18_residual_time_course.png')

# ====== FIGURE 3: ROI-level comparison ======
fig3, axes3 = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
fig3.patch.set_facecolor('white')
for ri, rn in enumerate(ROI_ORDER):
    ax = axes3[ri]; ax.set_facecolor('#FAFAFA')
    ax.plot(t, roi_actual[rn]*1e6, color='#2c3e50', lw=3.0, label='Actual (scalp)')
    ch_idx_opt = HW_CHS.index(SCALP_ROIS[rn][0])
    ax.plot(t, roi_pred_raw[rn]*1e6, color='#E74C3C', lw=1.0, ls=':', alpha=0.4, label=f'Raw (α={opt_scale[ch_idx_opt]:.2f})')
    ax.plot(t, roi_pred_scaled[rn]*1e6, color='#E74C3C', lw=2.5, ls='--', label='Scaled')
    for cn, t1, t2, cc in ERP_WINDOWS:
        ax.axvspan(t1, t2, alpha=0.06, color=cc)
    ax.axvline(0, color='#333', ls='--', lw=0.8)
    ax.axhline(0, color='#999', lw=0.5)
    ax.set_ylabel(f'{rn}\n(µV)', fontsize=10, fontweight='bold', color='#444')
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    # ROI residual annotation
    mean_r = roi_residual[rn].mean() * 100
    ax.text(0.98, 0.92, f'{rn}\nResid: {mean_r:.1f}%', transform=ax.transAxes,
            fontsize=10, fontweight='bold', color='#222', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.85))
    if ri == 0: ax.legend(fontsize=9)

axes3[-1].set_xlabel('Time (s)', fontsize=11)
fig3.suptitle('ROI Scalp ERP: Actual vs sLORETA Reconstruction (Session 2, 200 trials)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, '19_residual_roi_comparison.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig3)
print('Saved: 19_residual_roi_comparison.png')

# ====== FIGURE 4: GFP comparison ======
fig4, ax4 = plt.subplots(figsize=(12, 5))
fig4.patch.set_facecolor('white')
ax4.set_facecolor('#FAFAFA')
ax4.plot(t, gfp_actual*1e6, color='#2c3e50', lw=2.5, label='Actual GFP')
ax4.plot(t, gfp_pred*1e6, color='#E74C3C', lw=1.5, ls=':', alpha=0.5, label=f'Raw GFP (α=mean({np.mean(opt_scale):.2f}))')
ax4.plot(t, gfp_pred_scaled*1e6, color='#E74C3C', lw=2.5, ls='--', label='Scaled GFP')
for cn, t1, t2, cc in ERP_WINDOWS:
    ax4.axvspan(t1, t2, alpha=0.06, color=cc)
    mid = (t1+t2)/2
    gfp_a = np.mean(gfp_actual[(t>=t1)&(t<=t2)])*1e6
    gfp_p = np.mean(gfp_pred[(t>=t1)&(t<=t2)])*1e6
    gfp_s = np.mean(gfp_pred_scaled[(t>=t1)&(t<=t2)])*1e6
    expl = (1 - gfp_s/gfp_a) * 100 if gfp_a > 0 else 0
    ax4.text(mid, ax4.get_ylim()[1]*0.82, f'{cn}\nShape R^2={expl:.0f}%',
             ha='center', fontsize=8, color=cc, fontweight='bold')
ax4.axvline(0, color='#333', ls='--', lw=0.8)
ax4.set_ylabel('Global Field Power (µV)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_title('Global Field Power: Actual vs Reconstructed', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.tick_params(labelsize=8)
ax4.grid(True, axis='y', ls=':', color='#E0E0E0', lw=0.3)
for sp in ['top','right']: ax4.spines[sp].set_visible(False)
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, '20_residual_gfp.png'), dpi=200,
             facecolor='white', bbox_inches='tight')
plt.close(fig4)
print('Saved: 20_residual_gfp.png')

# ====== Numerical summary ======
print(f'\n{"="*80}')
print(f'  sLORETA Model Fit Summary')
print(f'{"="*80}')

# Overall
print(f'\n  Overall fitting residual (full time window -200 to 800ms):')
print(f'    Raw residual (incl. amplitude shrinkage):  {residual_var_raw.mean()*100:.2f}%')
print(f'    Scaled residual (shape mismatch only):     {residual_var.mean()*100:.2f}%')
print(f'    Shape R^2 (1 - scaled residual):            {(1-residual_var.mean())*100:.2f}%')
print(f'    Raw GFP correlation: r = {np.corrcoef(gfp_actual, gfp_pred)[0,1]:.4f}')
print(f'    Scaled GFP corr:     r = {np.corrcoef(gfp_actual, gfp_pred_scaled)[0,1]:.4f}')

# Optimal scaling factors
print('\n  Optimal channel scaling factors (alpha, pred x alpha ~ actual):')
for chi, ch_name in enumerate(MNE_NAMES):
    print(f'    {ch_name:4s}: alpha = {opt_scale[chi]:.3f}')
print(f'    Mean alpha = {np.mean(opt_scale):.3f}  (MNE shrinkage = {1/np.mean(opt_scale):.2f}x)')

# Per channel
print('\n  Per-channel r (actual vs scaled-reconstructed):')
for chi, ch_name in enumerate(MNE_NAMES):
    print(f'    {ch_name:4s}: r = {ch_corr[chi]:.4f}')
print(f'    Mean channel r:  {np.mean(ch_corr):.4f}')

# Per ERP window
print('\n  Scaled residual variance by ERP window (%):')
hdr = f'{"Window":<10} {"Global":>10} {"Frontal":>10} {"Central":>10} {"Parietal":>10} {"Occipital":>10}'
print(f'  {hdr}')
print(f'  {"-"*64}')
for cn, t1, t2, cc in ERP_WINDOWS:
    mask = (t >= t1) & (t <= t2)
    g = residual_var[mask].mean() * 100
    vals = [roi_residual[rn][mask].mean()*100 for rn in ROI_ORDER]
    print(f'  {cn:<10} {g:>10.2f} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f} {vals[3]:>10.2f}')

# Baseline
bl = (t >= -0.2) & (t < 0)
bl_r = residual_var[bl].mean() * 100
print(f'\n  Baseline scaled residual (-200 to 0ms): {bl_r:.2f}%')

# Shape R^2
print('\n  Shape R^2 (1 - scaled residual) by window:')
for cn, t1, t2, cc in ERP_WINDOWS:
    mask = (t >= t1) & (t <= t2)
    g = (1 - residual_var[mask].mean()) * 100
    print(f'    {cn:<8}: R^2 = {g:.1f}%')

# ROI correlation
print('\n  ROI-level r (actual vs scaled-reconstructed):')
for rn in ROI_ORDER:
    r = np.corrcoef(roi_actual[rn], roi_pred_scaled[rn])[0, 1]
    print(f'    {rn:<10}: r = {r:.4f}')

# Summary block
print('\n  Summary:')
print(f'    Mean channel r:      {np.mean(ch_corr):>10.4f}')
print(f'    Mean ROI r:          {np.mean([np.corrcoef(roi_actual[rn], roi_pred_scaled[rn])[0,1] for rn in ROI_ORDER]):>10.4f}')
print(f'    Global scaled resid: {residual_var.mean()*100:>10.2f}%')
print(f'    Global Shape R^2:     {(1-residual_var.mean())*100:>10.2f}%')
print(f'    MNE shrink factor:   {1/np.mean(opt_scale):>10.2f}x')
print(f'\nDone! All figures in {OUT_DIR}')
