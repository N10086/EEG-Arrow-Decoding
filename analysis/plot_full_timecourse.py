#!/usr/bin/env python3
"""
Full time-course decoding vs 4-window means.
Use all 500 time points per channel + PCA, compare with 4-window mean approach.
"""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
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
WINDOW_SHORT=['P1','N1','P2','P3']
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys()); HW_NAMES=[CH_MAP[c] for c in HW_CHS]
REGIONS={'Frontal':['F3','Fz','F4'],'Central':['C3','Cz','C4'],
         'Parietal':['P3','Pz','P4'],'Occipital':['O1','Oz','O2']}
REG_COLORS=['#e74c3c','#3498db','#2ecc71','#f39c12']

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
print(f'Trials: {len(epochs_data)}, Time points: {n_total}')

# Pre-compute window means
win_mean=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_mean[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# ====== Decoding function ======
def decode_cv(X,y,n_folds=5,use_pca=False,pca_var=0.95):
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr)
        Xtr_s=scl.transform(Xtr); Xte_s=scl.transform(Xte)
        if use_pca:
            pca=PCA(n_components=pca_var).fit(Xtr_s)
            Xtr_s=pca.transform(Xtr_s); Xte_s=pca.transform(Xte_s)
        clf=LinearDiscriminantAnalysis().fit(Xtr_s,ytr)
        accs.append(accuracy_score(yte,clf.predict(Xte_s)))
    return np.mean(accs),np.std(accs)

# ====== 1. Single-channel: windows vs full time course + PCA ======
print('\nSingle-channel: 4 windows vs full time course+PCA...')
sc_res={}
for ci in range(12):
    # 4 windows: 4 features
    X4=win_mean[:,ci,:]
    a4,s4=decode_cv(X4,epochs_label,use_pca=False)

    # Full time course: 500 features + PCA
    Xfull=epochs_data[:,ci,:]
    af,sf=decode_cv(Xfull,epochs_label,use_pca=True)
    # Re-run with return_components to get n_components
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    ncs=[]
    for tr,te in skf.split(Xfull,epochs_label):
        Xtr=Xfull[tr]; scl=StandardScaler().fit(Xtr)
        pca=PCA(n_components=0.95).fit(scl.transform(Xtr))
        ncs.append(pca.n_components_)

    sc_res[HW_NAMES[ci]]={'win4':a4,'full':af,'ncomp_mean':np.mean(ncs),'ncomp_std':np.std(ncs)}
    delta=af-a4
    print(f'  {HW_NAMES[ci]:4s}: windows={a4:.1%}  full+PCA={af:.1%} ({np.mean(ncs):.0f} PCs)  delta={delta:+.1%}')

# ====== 2. Key single-region: windows vs full + PCA ======
print('\nSingle-region: 4 windows vs full time course+PCA...')
KEY_REGIONS=[('Frontal','F'),('Central','C'),('Parietal','P'),('Occipital','O')]
rr_res={}
for rn,short in KEY_REGIONS:
    ch_idx=[HW_NAMES.index(c) for c in REGIONS[rn]]
    X4=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    a4,s4=decode_cv(X4,epochs_label)
    Xfull=epochs_data[:,ch_idx,:].reshape(len(epochs_data),-1)
    af,sf=decode_cv(Xfull,epochs_label,use_pca=True)
    # n_components
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    ncs=[]
    for tr,te in skf.split(Xfull,epochs_label):
        Xtr=Xfull[tr]; scl=StandardScaler().fit(Xtr)
        pca=PCA(n_components=0.95).fit(scl.transform(Xtr)); ncs.append(pca.n_components_)
    rr_res[short]={'win4':a4,'full':af,'ncomp_mean':np.mean(ncs)}
    print(f'  {short:4s}: windows={a4:.1%}  full+PCA={af:.1%} ({np.mean(ncs):.0f} PCs)  delta={af-a4:+.1%}')

# ====== 3. All 12 channels: windows vs full + PCA ======
print('\nAll 12 channels: windows vs full + PCA...')
X4_all=win_mean.reshape(len(epochs_data),-1)
a4_all,_=decode_cv(X4_all,epochs_label)
Xfull_all=epochs_data.reshape(len(epochs_data),-1)
af_all,_=decode_cv(Xfull_all,epochs_label,use_pca=True)
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
ncs_all=[]
for tr,te in skf.split(Xfull_all,epochs_label):
    Xtr=Xfull_all[tr]; scl=StandardScaler().fit(Xtr)
    pca=PCA(n_components=0.95).fit(scl.transform(Xtr)); ncs_all.append(pca.n_components_)
print(f'  All:  windows={a4_all:.1%}  full+PCA={af_all:.1%} ({np.mean(ncs_all):.0f} PCs)  delta={af_all-a4_all:+.1%}')

# ====== FIGURE 1: Single-channel comparison ======
fig1,(ax1a,ax1b)=plt.subplots(1,2,figsize=(18,6))
fig1.patch.set_facecolor('white')
order_sc=sorted(sc_res.keys(),key=lambda k:sc_res[k]['win4'],reverse=True)
x1=np.arange(12); w=0.35
vw=[sc_res[k]['win4'] for k in order_sc]
vf=[sc_res[k]['full'] for k in order_sc]
colors_sc=[]
for k in order_sc:
    for ri,(rn,chs) in enumerate(REGIONS.items()):
        if k in chs: colors_sc.append(REG_COLORS[ri]); break

ax1a.bar(x1-w/2,vw,w,color=colors_sc,edgecolor='#333',lw=0.4,alpha=0.6,label='4 Window Means')
bars1=ax1a.bar(x1+w/2,vf,w,color=colors_sc,edgecolor='#333',lw=0.4,alpha=1.0,label=f'Full Time Course + PCA')
ax1a.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for i,k in enumerate(order_sc):
    d=sc_res[k]['full']-sc_res[k]['win4']
    c='#27ae60' if d>0.01 else '#c0392b' if d<-0.01 else '#888'
    ax1a.text(i+w/2,sc_res[k]['full']+0.01,f'{d:+.1%}',ha='center',fontsize=6,fontweight='bold',color=c,rotation=90)
ax1a.set_xticks(x1);ax1a.set_xticklabels(order_sc,fontsize=9)
ax1a.set_ylabel('4-Class Accuracy',fontsize=11)
ax1a.set_title('Single-Channel: 4 Windows vs Full Time Course + PCA',fontsize=11,fontweight='bold')
ax1a.legend(fontsize=8);ax1a.grid(True,axis='y',ls=':',alpha=0.3)
ax1a.set_ylim(0,max(max(vw),max(vf))*1.25)

# PCs bar
npc=[sc_res[k]['ncomp_mean'] for k in order_sc]
ax1b.bar(x1-w/2,[4]*12,w,color='#bdc3c7',edgecolor='#333',lw=0.4,label='4 windows (fixed)')
ax1b.bar(x1+w/2,npc,w,color='#e74c3c',edgecolor='#333',lw=0.4,label=f'Full time course PCs')
for i,k in enumerate(order_sc):
    ax1b.text(i+w/2,npc[i]+0.3,f'{npc[i]:.0f}',ha='center',fontsize=7,fontweight='bold',color='#c0392b')
ax1b.axhline(4,color='#888',ls=':',lw=0.8,alpha=0.5)
ax1b.set_xticks(x1);ax1b.set_xticklabels(order_sc,fontsize=9)
ax1b.set_ylabel('Number of Features',fontsize=11)
ax1b.set_title('Feature Count: 4 Windows vs PCA Components (95% var)',fontsize=11,fontweight='bold')
ax1b.legend(fontsize=8);ax1b.grid(True,axis='y',ls=':',alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'81_full_timecourse_single.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('Saved: 81_full_timecourse_single.png')

# ====== FIGURE 2: Region & All comparison ======
fig2,ax2=plt.subplots(figsize=(10,5))
fig2.patch.set_facecolor('white')
labels2=['F','C','P','O','All']
vals2_4=[rr_res[k]['win4'] for k in ['F','C','P','O']]+[a4_all]
vals2_f=[rr_res[k]['full'] for k in ['F','C','P','O']]+[af_all]
npc2=[rr_res[k]['ncomp_mean'] for k in ['F','C','P','O']]+[np.mean(ncs_all)]
x2=np.arange(5); w=0.35
ax2.bar(x2-w/2,vals2_4,w,color='#bdc3c7',edgecolor='#333',lw=0.4,label='4 Window Means (12 feat)')
bars2=ax2.bar(x2+w/2,vals2_f,w,color='#27ae60',edgecolor='#333',lw=0.4,label='Full Time Course + PCA')
ax2.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for i in range(5):
    d=vals2_f[i]-vals2_4[i]; pcs=npc2[i]
    c='#27ae60' if d>0.01 else '#c0392b'
    ax2.text(i+w/2,vals2_f[i]+0.015,f'{d:+.1%}\n({pcs:.0f} PCs)',ha='center',fontsize=8,fontweight='bold',color=c)
ax2.set_xticks(x2);ax2.set_xticklabels(labels2,fontsize=11)
ax2.set_ylabel('4-Class Accuracy',fontsize=11)
ax2.set_title('Single-Region & All Channels: Windows vs Full Time Course + PCA',fontsize=11,fontweight='bold')
ax2.legend(fontsize=8);ax2.grid(True,axis='y',ls=':',alpha=0.3)
ax2.set_ylim(0,max(max(vals2_4),max(vals2_f))*1.25)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'82_full_timecourse_regions.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('Saved: 82_full_timecourse_regions.png')

# ====== Summary ======
print(f'\n{"="*70}')
print(f'  FULL TIME COURSE + PCA vs 4 WINDOW MEANS')
print(f'{"="*70}')
print(f'\nSingle-channel totals:')
avg_w=np.mean([sc_res[k]['win4'] for k in sc_res])
avg_f=np.mean([sc_res[k]['full'] for k in sc_res])
n_better=sum(1 for k in sc_res if sc_res[k]['full']>sc_res[k]['win4'])
print(f'  Windows: {avg_w:.1%}  Full+PCA: {avg_f:.1%}  delta={avg_f-avg_w:+.1%}')
print(f'  Channels improved: {n_better}/12')
print(f'\nKey results:')
for rn,short in KEY_REGIONS:
    r=rr_res[short]
    print(f'  {short}: {r["win4"]:.1%} -> {r["full"]:.1%} (delta={r["full"]-r["win4"]:+.1%}, {r["ncomp_mean"]:.0f} PCs)')
print(f'  All: {a4_all:.1%} -> {af_all:.1%} (delta={af_all-a4_all:+.1%}, {np.mean(ncs_all):.0f} PCs)')

sp=os.path.join(OUT_DIR,'83_full_timecourse_summary.txt')
with open(sp,'w') as sf:
    sf.write('Full Time Course + PCA vs 4 Window Means\n')
    sf.write(f'{"="*50}\n')
    sf.write(f'\nSingle-channel:\n')
    for k in sorted(sc_res,key=lambda x:sc_res[x]['win4'],reverse=True):
        r=sc_res[k]; d=r['full']-r['win4']
        sf.write(f'  {k}: win4={r["win4"]:.4f} full+PCA={r["full"]:.4f} (nPC={r["ncomp_mean"]:.0f}) delta={d:+.4f}\n')
    sf.write(f'\nRegions:\n')
    for rn,short in KEY_REGIONS:
        r=rr_res[short]
        sf.write(f'  {short}: win4={r["win4"]:.4f} full+PCA={r["full"]:.4f} delta={r["full"]-r["win4"]:+.4f}\n')
    sf.write(f'\nAll: win4={a4_all:.4f} full+PCA={af_all:.4f} delta={af_all-a4_all:+.4f}\n')
print(f'Saved: {sp}')
print('\nDone!')
