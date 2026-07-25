#!/usr/bin/env python3
"""
Add peak-to-peak range per ERP window as additional features.
Compare: 4 means vs 4 means + 4 ranges per channel.
"""
import numpy as np
from scipy import signal as sg
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
WINDOW_SHORT=['P1','N1','P2','P3']
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys()); HW_NAMES=[CH_MAP[c] for c in HW_CHS]
REGIONS={'Frontal':['F3','Fz','F4'],'Central':['C3','Cz','C4'],
         'Parietal':['P3','Pz','P4'],'Occipital':['O1','Oz','O2']}
REG_COLORS=['#e74c3c','#3498db','#2ecc71','#f39c12']
ALL_COMBOS=[
    (['Frontal'],'F','F'),(['Central'],'C','C'),(['Parietal'],'P','P'),(['Occipital'],'O','O'),
    (['Frontal','Central'],'F+C','FC'),(['Frontal','Parietal'],'F+P','FP'),
    (['Frontal','Occipital'],'F+O','FO'),(['Central','Parietal'],'C+P','CP'),
    (['Central','Occipital'],'C+O','CO'),(['Parietal','Occipital'],'P+O','PO'),
    (['Frontal','Central','Parietal'],'F+C+P','FCP'),(['Frontal','Central','Occipital'],'F+C+O','FCO'),
    (['Frontal','Parietal','Occipital'],'F+P+O','FPO'),(['Central','Parietal','Occipital'],'C+P+O','CPO'),
    (['Frontal','Central','Parietal','Occipital'],'All','All'),
]

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

# ====== Dual feature set ======
print('Extracting features (mean + range)...')
win_mean=np.zeros((len(epochs_data),12,4))
win_range=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    d=epochs_data[:,:,msk]
    win_mean[:,:,wi]=d.mean(axis=2)
    win_range[:,:,wi]=d.max(axis=2)-d.min(axis=2)

def decode_4class(X,y,n_folds=5):
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    accs,ap,at=[],[],[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr)
        clf=LinearDiscriminantAnalysis().fit(scl.transform(Xtr),ytr)
        yp=clf.predict(scl.transform(Xte))
        ap.extend(yp);at.extend(yte);accs.append(accuracy_score(yte,yp))
    return np.mean(accs),np.std(accs),np.array(ap),np.array(at)

# ====== Single-channel comparison ======
print('\nSingle-channel: mean-only vs mean+range...')
sc_results={}
for ci in range(12):
    Xm=win_mean[:,ci,:]  # 4 features
    Xmr=np.concatenate([win_mean[:,ci,:],win_range[:,ci,:]],axis=1)  # 8 features
    am,sm,_,_=decode_4class(Xm,epochs_label)
    ar,sr,_,_=decode_4class(Xmr,epochs_label)
    sc_results[HW_NAMES[ci]]={'mean4':am,'mean4_std':sm,'mean8':ar,'mean8_std':sr}
    delta=ar-am
    print(f'  {HW_NAMES[ci]:4s}: mean4={am:.1%}  mean+range={ar:.1%}  delta={delta:+.1%}')

# ====== Region combination comparison ======
print('\nRegion combos: mean-only vs mean+range...')
rc_results={}
for combo,label,short in ALL_COMBOS:
    ch_idx=[]
    for rn in combo:
        for ch_name in REGIONS[rn]:
            ch_idx.append(HW_NAMES.index(ch_name))

    # Mean only: n_ch * 4 features
    Xm=win_mean[:,ch_idx,:].reshape(len(epochs_data),-1)
    # Mean + range: n_ch * 8 features
    Xmr=np.concatenate([win_mean[:,ch_idx,:],win_range[:,ch_idx,:]],axis=1).reshape(len(epochs_data),-1)

    am,sm,_,_=decode_4class(Xm,epochs_label)
    ar,sr,_,_=decode_4class(Xmr,epochs_label)
    rc_results[short]={'label':label,'mean4':am,'mean4_std':sm,'mean8':ar,'mean8_std':sr}
    print(f'  {label:>7s}: mean4={am:.1%}  mean+range={ar:.1%}  delta={ar-am:+.1%}')

# ====== FIGURE 1: Single-channel comparison ======
fig1,(ax1a,ax1b)=plt.subplots(1,2,figsize=(18,6))
fig1.patch.set_facecolor('white')

order_sc=sorted(sc_results.keys(),key=lambda k:sc_results[k]['mean4'],reverse=True)
x1=np.arange(12); w=0.35
vm=[sc_results[k]['mean4'] for k in order_sc]
vr=[sc_results[k]['mean8'] for k in order_sc]
colors_sc=[]
for k in order_sc:
    for ri,(rn,chs) in enumerate(REGIONS.items()):
        if k in chs: colors_sc.append(REG_COLORS[ri]); break
ax1a.bar(x1-w/2,vm,w,color=colors_sc,edgecolor='#333',lw=0.4,alpha=0.7,label='4 Means only')
bars1a=ax1a.bar(x1+w/2,vr,w,color=colors_sc,edgecolor='#333',lw=0.4,alpha=1.0,label='4 Means + 4 Ranges')
ax1a.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for i,k in enumerate(order_sc):
    d=sc_results[k]['mean8']-sc_results[k]['mean4']
    c='#27ae60' if d>0.005 else '#c0392b' if d<-0.005 else '#888'
    ax1a.text(i+w/2,sc_results[k]['mean8']+0.008,f'{d:+.1%}',ha='center',fontsize=6,fontweight='bold',color=c,rotation=90)
ax1a.set_xticks(x1);ax1a.set_xticklabels(order_sc,fontsize=9)
ax1a.set_ylabel('4-Class Accuracy',fontsize=11)
ax1a.set_title('Single-Channel: Mean Only vs Mean + Range',fontsize=12,fontweight='bold')
ax1a.legend(fontsize=8,ncol=2);ax1a.grid(True,axis='y',ls=':',alpha=0.3)
ax1a.set_ylim(0,max(max(vm),max(vr))*1.25)

# Right: improvement vs original accuracy scatter
ch_imps=[sc_results[k]['mean8']-sc_results[k]['mean4'] for k in sc_results]
ch_orig=[sc_results[k]['mean4'] for k in sc_results]
ax1b.scatter(ch_orig,ch_imps,c='#333',s=60,alpha=0.7,zorder=5)
for i,k in enumerate(sc_results):
    ax1b.annotate(k,(ch_orig[i],ch_imps[i]),textcoords='offset points',xytext=(5,5),fontsize=7)
ax1b.axhline(0,color='#888',ls='--',lw=1.2)
ax1b.set_xlabel('Accuracy with 4 Means Only',fontsize=11)
ax1b.set_ylabel('Improvement (Mean+Range minus Mean Only)',fontsize=11)
ax1b.set_title('Does Adding Range Features Help?',fontsize=12,fontweight='bold')
ax1b.grid(True,ls=':',alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'75_single_channel_range_compare.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('Saved: 75_single_channel_range_compare.png')

# ====== FIGURE 2: Region combo comparison ======
fig2,ax2=plt.subplots(figsize=(14,6))
fig2.patch.set_facecolor('white')
order_rc=sorted(rc_results.keys(),key=lambda k:rc_results[k]['mean4'],reverse=True)
x2=np.arange(15)
vm2=[rc_results[k]['mean4'] for k in order_rc]
vr2=[rc_results[k]['mean8'] for k in order_rc]
# Color by #regions
gc2=[]
for k in order_rc:
    n_reg=len(rc_results[k]['label'].split('+'))
    gc2.append(['#3498db','#2ecc71','#e67e22','#e74c3c'][min(3,n_reg-1)])
ax2.bar(x2-w/2,vm2,w,color='#bdc3c7',edgecolor='#333',lw=0.4,label='12 Means only')
bars2=ax2.bar(x2+w/2,vr2,w,color=gc2,edgecolor='#333',lw=0.4,label='12 Means + 12 Ranges')
ax2.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for i,k in enumerate(order_rc):
    d=rc_results[k]['mean8']-rc_results[k]['mean4']
    c='#27ae60' if d>0.005 else '#c0392b' if d<-0.005 else '#888'
    ax2.text(i+w/2,rc_results[k]['mean8']+0.01,f'{d:+.1%}',ha='center',fontsize=6,fontweight='bold',color=c,rotation=90)
ax2.set_xticks(x2);ax2.set_xticklabels([rc_results[k]['label'] for k in order_rc],fontsize=7,rotation=45,ha='right')
ax2.set_ylabel('4-Class Accuracy',fontsize=11)
ax2.set_title('Region Combinations: 4 Means vs 4 Means + 4 Ranges per Channel',fontsize=12,fontweight='bold')
ax2.legend(fontsize=8,ncol=2);ax2.grid(True,axis='y',ls=':',alpha=0.3)
ax2.set_ylim(0,max(max(vm2),max(vr2))*1.25)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'76_region_combo_range_compare.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('Saved: 76_region_combo_range_compare.png')

# ====== Summary ======
avg_m=np.mean([sc_results[k]['mean4'] for k in sc_results])
avg_r=np.mean([sc_results[k]['mean8'] for k in sc_results])
print(f'\n{"="*70}')
print(f'  MEAN vs MEAN+RANGE COMPARISON')
print(f'{"="*70}')
print(f'\nSingle-channel averages: mean4={avg_m:.1%}  mean+range={avg_r:.1%}  delta={avg_r-avg_m:+.1%}')
n_better=sum(1 for k in sc_results if sc_results[k]['mean8']>sc_results[k]['mean4'])
print(f'  Channels improved: {n_better}/12')
print(f'\nRegion combo averages:')
avg_rm=np.mean([rc_results[k]['mean4'] for k in rc_results])
avg_rr=np.mean([rc_results[k]['mean8'] for k in rc_results])
print(f'  mean4={avg_rm:.1%}  mean+range={avg_rr:.1%}  delta={avg_rr-avg_rm:+.1%}')
n_rbetter=sum(1 for k in rc_results if rc_results[k]['mean8']>rc_results[k]['mean4'])
print(f'  Combos improved: {n_rbetter}/15')

sp=os.path.join(OUT_DIR,'77_range_comparison_summary.txt')
with open(sp,'w') as sf:
    sf.write(f'Range Feature Comparison\n{"="*50}\n')
    sf.write(f'Single-channel:\n')
    for k in sorted(sc_results,key=lambda x:sc_results[x]['mean4'],reverse=True):
        r=sc_results[k]; d=r['mean8']-r['mean4']
        sf.write(f'  {k}: mean4={r["mean4"]:.4f} mean8={r["mean8"]:.4f} delta={d:+.4f}\n')
    sf.write(f'\nRegion combos:\n')
    for k in sorted(rc_results,key=lambda x:rc_results[x]['mean4'],reverse=True):
        r=rc_results[k]; d=r['mean8']-r['mean4']
        sf.write(f'  {r["label"]}: mean4={r["mean4"]:.4f} mean8={r["mean8"]:.4f} delta={d:+.4f}\n')
print(f'Saved: {sp}')
print('\nDone!')
