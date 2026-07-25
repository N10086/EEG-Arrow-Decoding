#!/usr/bin/env python3
"""
Hierarchical binary decoding: axis (Vert vs Horz) -> within-axis direction.
15 region combos x (axis, UvsD, LvsR) + time-frequency comparison.
"""
import numpy as np
from scipy import signal as sg
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

FS=500.0; GAIN=6.0; SCALE=4.5/(2**23-1)/GAIN*1e6
OUT_DIR=r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
DATA_PATH=r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'
DIR_NAMES=['Up','Down','Left','Right']
DIR_VALS=[2.0001,2.0002,2.0003,2.0004]
ERP_WINDOWS=[('P1',0.080,0.130),('N1',0.140,0.200),('P2',0.200,0.300),('P3',0.300,0.500)]
W_SHORT=['P1','N1','P2','P3']
FREQ_BANDS=[('Delta',1,4),('Theta',4,8),('Alpha',8,13),('Beta',13,30),('Gamma',30,45)]
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys()); HW_NAMES=[CH_MAP[c] for c in HW_CHS]
REGIONS={'Frontal':['F3','Fz','F4'],'Central':['C3','Cz','C4'],
         'Parietal':['P3','Pz','P4'],'Occipital':['O1','Oz','O2']}

def map_direction(val):
    for di,k in enumerate(DIR_VALS):
        if abs(val-k)<5e-5: return di
    return -1

print('Loading...')
with open(DATA_PATH) as f: lines=f.readlines()
data,markers=[],[]
for line in lines[5:]:
    p=line.strip().split(',')
    if len(p)>33:
        try: data.append([float(p[i]) for i in range(1,17)]); markers.append(float(p[32]))
        except: pass
d=np.array(data,dtype=np.float64).T; m=np.array(markers)
sel=np.array([d[c-1] for c in HW_CHS])
bp=sg.butter(4,[1/250,45/250],btype='band',output='sos')
notch=sg.iirnotch(50/250,30)
filt=np.zeros_like(sel)
for ch in range(12):
    dm=sel[ch]-sel[ch].mean(); ts=sg.sosfiltfilt(bp,dm); filt[ch]=sg.filtfilt(*notch,ts)
filt_uv=filt*SCALE
ons=sorted([i for k in DIR_VALS for i in np.where(np.abs(m-k)<5e-5)[0]])
t_start,t_end=-0.2,0.8; n_pre,n_post=int(abs(t_start)*FS),int(t_end*FS); n_total=n_pre+n_post
t=np.linspace(t_start,t_end,n_total)
epochs_data,epochs_label=[],[]
for idx in ons:
    s,e=idx-n_pre,idx+n_post
    if s<0 or e>=filt_uv.shape[1]: continue
    ep=filt_uv[:,s:e].copy(); ep-=ep[:,:n_pre].mean(axis=1,keepdims=True)
    if np.max(np.abs(ep))>100: continue
    epochs_data.append(ep); epochs_label.append(map_direction(m[idx]))
epochs_data=np.array(epochs_data); epochs_label=np.array(epochs_label)
print(f'Trials: {len(epochs_data)}')

# Window means
win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# All 15 region combos
SINGLE=[('Frontal',[HW_NAMES.index(c) for c in REGIONS['Frontal']],'#e74c3c'),
        ('Central',[HW_NAMES.index(c) for c in REGIONS['Central']],'#3498db'),
        ('Parietal',[HW_NAMES.index(c) for c in REGIONS['Parietal']],'#2ecc71'),
        ('Occipital',[HW_NAMES.index(c) for c in REGIONS['Occipital']],'#f39c12')]
PAIRS=[(f'{SINGLE[i][0]}+{SINGLE[j][0]}',sorted(set(SINGLE[i][1]+SINGLE[j][1])),'#8e44ad')
       for i in range(4) for j in range(i+1,4)]
TRIPLES=[(f'{SINGLE[i][0]}+{SINGLE[j][0]}+{SINGLE[k][0]}',sorted(set(SINGLE[i][1]+SINGLE[j][1]+SINGLE[k][1])),'#16a085')
         for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]
ALL=[('F+P+O+C',list(range(12)),'#333333')]
REGION_COMBOS=SINGLE+PAIRS+TRIPLES+ALL  # 4+6+4+1=15

def decode_binary(X,y,cls_a,cls_b):
    mask=(y==cls_a)|(y==cls_b)
    y_bin=(y[mask]==cls_b).astype(int); Xb=X[mask]
    if len(Xb)<20: return 0.0
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(Xb,y_bin):
        scl=StandardScaler().fit(Xb[tr])
        clf=LinearDiscriminantAnalysis().fit(scl.transform(Xb[tr]),y_bin[tr])
        accs.append(accuracy_score(y_bin[te],clf.predict(scl.transform(Xb[te]))))
    return np.mean(accs)

def decode_cv(X,y):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(X,y):
        scl=StandardScaler().fit(X[tr])
        clf=LinearDiscriminantAnalysis().fit(scl.transform(X[tr]),y[tr])
        accs.append(accuracy_score(y[te],clf.predict(scl.transform(X[te]))))
    return np.mean(accs)

# ====== Hierarchical decoding ======
print('\n'+'='*70)
print('  HIERARCHICAL DECODING: Axis (Vert/Horz) -> Within-Axis Direction')
print('='*70)
print(f'  {"Region":<14s} {"Axis(V/H)":<10s} {"UvsD":<10s} {"LvsR":<10s} {"4-class":<10s} {"hier":<10s} {"bottleneck":<12s}')
print(f'  {"-"*68}')

results=[]
for rname,ch_idx,rcolor in REGION_COMBOS:
    X=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    y=epochs_label
    y_axis=(y>=2).astype(int)
    axis_acc=decode_cv(X,y_axis)
    vert_acc=decode_binary(X,y,0,1)
    horz_acc=decode_binary(X,y,2,3)
    acc4=decode_cv(X,y)
    # Hierarchical: axis must be correct first, THEN within-axis direction correct
    # If axis wrong, direction is guaranteed wrong in 4-class sense
    hier_acc=axis_acc*((vert_acc+horz_acc)/2)
    # Identify bottleneck: lowest among axis, UvsD, LvsR
    bottleneck=min(axis_acc,vert_acc,horz_acc)
    bn_name='axis' if axis_acc==bottleneck else ('UvsD' if vert_acc==bottleneck else 'LvsR')
    results.append((rname,axis_acc,vert_acc,horz_acc,acc4,hier_acc,bottleneck,bn_name,rcolor))
    print(f'  {rname:<14s} {axis_acc:.1%}    {vert_acc:.1%}   {horz_acc:.1%}   {acc4:.1%}    {hier_acc:.1%}   {bn_name}={bottleneck:.1%}')

# ====== Bottleneck analysis: why is axis hard? ======
# For the best axis region, compare ERP waveforms between V and H
print('\n'+'='*70)
print('  BOTTLENECK ANALYSIS: Axis discrimination by channel')
print('='*70)

# Take best region for axis
best_axis_region=sorted(results,key=lambda r:r[1],reverse=True)[0][0]
best_axis_idx=[i for i,(rn,_,_,_,_,_,_,_,_) in enumerate(results) if rn==best_axis_region][0]
_,ch_idx,_=REGION_COMBOS[best_axis_idx]
print(f'  Best axis region: {best_axis_region} ({len(ch_idx)} channels)')

# Single-channel axis accuracy
print(f'\n  {"Channel":<6s} {"Axis(V/H)":<10s} {"UvsD":<10s} {"LvsR":<10s}')
print(f'  {"-"*34}')
for ci in ch_idx:
    X1=win_mean[:,[ci],:].reshape(len(epochs_data),-1)
    y_axis=(epochs_label>=2).astype(int)
    aa=decode_cv(X1,y_axis)
    vu=decode_binary(X1,epochs_label,0,1)
    lr=decode_binary(X1,epochs_label,2,3)
    print(f'  {HW_NAMES[ci]:<6s} {aa:.1%}       {vu:.1%}      {lr:.1%}')

# ====== Figure 1: All 15 regions hierarchical breakdown ======
fig1,ax1=plt.subplots(figsize=(16,6))
fig1.patch.set_facecolor('white')
x=np.arange(15); w=0.18
names=[r[0] for r in results]
ax1.bar(x-1.5*w,[r[1] for r in results],w,color='#3498db',ec='#333',lw=0.3,label='Axis (Vert vs Horz)')
ax1.bar(x-0.5*w,[r[2] for r in results],w,color='#2ecc71',ec='#333',lw=0.3,label='U vs D')
ax1.bar(x+0.5*w,[r[3] for r in results],w,color='#e67e22',ec='#333',lw=0.3,label='L vs R')
ax1.bar(x+1.5*w,[r[4] for r in results],w,color='#e74c3c',ec='#333',lw=0.3,label='4-Class (direct)')
ax1.axhline(0.25,color='#888',ls='--',lw=1.5,alpha=0.7)
ax1.axhline(0.5,color='#aaa',ls=':',lw=1,alpha=0.5)
ax1.set_xticks(x);ax1.set_xticklabels(names,fontsize=6.5,rotation=60,ha='right')
ax1.set_ylabel('Accuracy',fontsize=11)
ax1.set_title('All 15 Region Combinations: Hierarchical Decoding Breakdown',fontsize=12,fontweight='bold')
ax1.legend(fontsize=8,ncol=4);ax1.grid(True,axis='y',ls=':',alpha=0.3)
ax1.set_ylim(0,max(max(r[1:5]) for r in results)*1.2)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'98a_hierarchical_breakdown.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('Saved: 98a_hierarchical_breakdown.png')

# ====== Figure 2: Scatter hierarchical vs direct ======
fig2,(ax2a,ax2b)=plt.subplots(1,2,figsize=(14,6))
fig2.patch.set_facecolor('white')

# Left: Hier vs Direct
ax2a.plot([0.2,0.7],[0.2,0.7],'--',color='#888',lw=1,label='Equal')
for rn,aa,vu,lr,a4,ha,_,_,rc in results:
    ax2a.scatter(a4,ha,s=120,c=rc,ec='#333',lw=0.6,zorder=5)
    ax2a.annotate(rn,(a4,ha),textcoords='offset points',xytext=(4,3),fontsize=6.5)
ax2a.set_xlabel('Direct 4-Class',fontsize=11);ax2a.set_ylabel('Hierarchical (Axis x Within)',fontsize=11)
ax2a.set_title('Hierarchical vs Direct 4-Class',fontsize=12,fontweight='bold')
ax2a.grid(True,ls=':',alpha=0.3);ax2a.legend(fontsize=9)
ax2a.set_xlim(0.2,0.7);ax2a.set_ylim(0.2,0.65)

# Right: Bottleneck (minimum of axis/UvsD/LvsR) vs 4-class
ax2b.plot([0.2,0.7],[0.2,0.7],'--',color='#888',lw=1,label='Equal')
for rn,aa,vu,lr,a4,ha,bn,bnn,rc in results:
    ax2b.scatter(a4,bn,s=120,c=rc,ec='#333',lw=0.6,zorder=5)
    ax2b.annotate(rn,(a4,bn),textcoords='offset points',xytext=(4,3),fontsize=6.5)
ax2b.set_xlabel('Direct 4-Class',fontsize=11);ax2b.set_ylabel('Bottleneck (min component)',fontsize=11)
ax2b.set_title('4-Class Accuracy vs Worst Sub-Problem',fontsize=12,fontweight='bold')
ax2b.grid(True,ls=':',alpha=0.3);ax2b.legend(fontsize=9)
ax2b.set_xlim(0.2,0.7);ax2b.set_ylim(0.2,0.8)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'98b_hierarchical_scatter.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('Saved: 98b_hierarchical_scatter.png')

# ====== Does time-frequency help the axis bottleneck? ======
print('\n'+'='*70)
print('  CAN TIME-FREQUENCY FIX THE AXIS BOTTLENECK?')
print('='*70)

n_fft=400; freqs=fftfreq(n_fft,1/FS)[:n_fft//2]
band_power=np.zeros((len(epochs_data),12,len(FREQ_BANDS)))
for ti in range(len(epochs_data)):
    sig=epochs_data[ti,:,n_pre:]
    for ci in range(12):
        psd=np.abs(fft(sig[ci]))**2/n_fft
        for bi,(_,fl,fh) in enumerate(FREQ_BANDS):
            band_power[ti,ci,bi]=psd[:n_fft//2][(freqs>=fl)&(freqs<fh)].sum()
band_power=np.log1p(band_power)

# Test axis discrimination: win_means vs band_power
print(f'  {"Region":<14s} {"Axis(win)":<10s} {"Axis(band)":<10s} {"diff":<8s} {"UvsD(win)":<10s} {"UvsD(band)":<10s} {"LvsR(win)":<10s} {"LvsR(band)":<10s}')
print(f'  {"-"*82}')
for rname,ch_idx,rcolor in REGION_COMBOS:
    Xw=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    Xb=band_power[:,ch_idx,:].reshape(len(epochs_data),-1)
    y=epochs_label; y_axis=(y>=2).astype(int)
    aw=decode_cv(Xw,y_axis); ab=decode_cv(Xb,y_axis)
    uw=decode_binary(Xw,y,0,1); ub=decode_binary(Xb,y,0,1)
    lw=decode_binary(Xw,y,2,3); lb=decode_binary(Xb,y,2,3)
    print(f'  {rname:<14s} {aw:.1%}      {ab:.1%}      {ab-aw:+.1%}    {uw:.1%}       {ub:.1%}       {lw:.1%}       {lb:.1%}')

# ====== Summary ======
print('\n'+'='*70)
print('  HIERARCHICAL BINARY SUMMARY')
print('='*70)
ranked=sorted(results,key=lambda r:r[5],reverse=True)
print(f'  {"Rank":<5s} {"Region":<14s} {"Axis":<8s} {"UvsD":<8s} {"LvsR":<8s} {"4class":<8s} {"Hier":<8s} {"BN":<10s}')
print(f'  {"-"*63}')
for i,(rn,aa,vu,lr,a4,ha,bn,bnn,_) in enumerate(ranked):
    print(f'  #{i+1:<2d}  {rn:<14s} {aa:.1%}   {vu:.1%}  {lr:.1%}  {a4:.1%}  {ha:.1%}  {bnn}={bn:.1%}')

print(f'\n  Key insight:')
bottleneck_counts={}
for _,_,_,_,_,_,_,bnn,_ in results:
    bottleneck_counts[bnn]=bottleneck_counts.get(bnn,0)+1
total_bn=sum(bottleneck_counts.values())
for k,v in sorted(bottleneck_counts.items()):
    print(f'    {k}: {v}/{total_bn} regions ({v/total_bn:.0%}) bottleneck')

sp=os.path.join(OUT_DIR,'99_hierarchical_summary.txt')
with open(sp,'w') as sf:
    sf.write('Hierarchical Binary Decoding Summary\n')
    sf.write(f'{"="*60}\n\n')
    for rn,aa,vu,lr,a4,ha,_,_,_ in ranked:
        sf.write(f'{rn}: axis={aa:.4f} UvsD={vu:.4f} LvsR={lr:.4f} 4class={a4:.4f} hier={ha:.4f}\n')
    sf.write(f'\nBottleneck counts: {bottleneck_counts}\n')
print(f'Saved: {sp}')
print('Done!')
