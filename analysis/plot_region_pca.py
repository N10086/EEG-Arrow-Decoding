#!/usr/bin/env python3
"""
PCA dimensionality reduction comparison — all 15 region combinations,
contrasting decoding accuracy with and without PCA (95% variance).
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
from sklearn.metrics import accuracy_score, confusion_matrix
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
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys())
REGIONS={'Frontal':[7,4,14],'Central':[2,6,5],'Parietal':[9,10,12],'Occipital':[15,1,8]}
REG_NAMES=['Frontal','Central','Parietal','Occipital']

def map_direction(val):
    for di,k in enumerate(DIR_VALS):
        if abs(val-k)<5e-5: return di
    return -1

# ====== Load ======
print('Loading...')
with open(DATA_PATH) as f: lines=f.readlines()
data,markers=[],[]
for line in lines[5:]:
    p=line.strip().split(',')
    if len(p)>33:
        try:
            data.append([float(p[i]) for i in range(1,17)])
            markers.append(float(p[32]))
        except: pass
d=np.array(data,dtype=np.float64).T; m=np.array(markers)
sel=np.array([d[c-1] for c in HW_CHS])

print('Filtering...')
bp=sg.butter(4,[1/250,45/250],btype='band',output='sos')
notch=sg.iirnotch(50/250,30)
filt=np.zeros_like(sel)
for ch in range(12):
    dm=sel[ch]-sel[ch].mean()
    ts=sg.sosfiltfilt(bp,dm)
    filt[ch]=sg.filtfilt(*notch,ts)
filt_uv=filt*SCALE

ons=sorted([i for k in DIR_VALS for i in np.where(np.abs(m-k)<5e-5)[0]])
t_start,t_end=-0.2,0.8; n_pre,n_post=int(abs(t_start)*FS),int(t_end*FS); n_total=n_pre+n_post
t=np.linspace(t_start,t_end,n_total)

epochs_data,epochs_label=[],[]
for idx in ons:
    s,e=idx-n_pre,idx+n_post
    if s<0 or e>=filt_uv.shape[1]: continue
    ep=filt_uv[:,s:e].copy()
    ep-=ep[:,:n_pre].mean(axis=1,keepdims=True)
    if np.max(np.abs(ep))>100: continue
    epochs_data.append(ep)
    epochs_label.append(map_direction(m[idx]))
epochs_data=np.array(epochs_data); epochs_label=np.array(epochs_label)
print(f'Trials: {len(epochs_data)}')

# Window means: n_trials x 12 x 4
win_data=np.zeros((len(epochs_data),12,len(ERP_WINDOWS)))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_data[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# ====== Combinations ======
ALL_COMBOS=[
    (['Frontal'],'F','F'),(['Central'],'C','C'),(['Parietal'],'P','P'),(['Occipital'],'O','O'),
    (['Frontal','Central'],'F+C','FC'),(['Frontal','Parietal'],'F+P','FP'),
    (['Frontal','Occipital'],'F+O','FO'),(['Central','Parietal'],'C+P','CP'),
    (['Central','Occipital'],'C+O','CO'),(['Parietal','Occipital'],'P+O','PO'),
    (['Frontal','Central','Parietal'],'F+C+P','FCP'),(['Frontal','Central','Occipital'],'F+C+O','FCO'),
    (['Frontal','Parietal','Occipital'],'F+P+O','FPO'),(['Central','Parietal','Occipital'],'C+P+O','CPO'),
    (['Frontal','Central','Parietal','Occipital'],'All','All'),
]
GROUP_COLORS=['#3498db','#2ecc71','#e67e22','#e74c3c']

# ====== Run all combinations with and without PCA ======
print('\nRunning 15 combinations (no PCA vs PCA)...')
results={}
for combo,label,short in ALL_COMBOS:
    ch_idx=[]
    for rn in combo:
        ch_idx.extend([HW_CHS.index(c) for c in REGIONS[rn]])
    X=win_data[:,ch_idx,:].reshape(len(epochs_data),-1)
    y=epochs_label
    n_feat=X.shape[1]
    n_region=len(combo)

    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    acc_no,std_no=[],[]
    acc_pca,std_pca=[],[]
    ncomp_list=[]

    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr)
        Xtr_s=scl.transform(Xtr); Xte_s=scl.transform(Xte)

        # No PCA
        clf=LinearDiscriminantAnalysis().fit(Xtr_s,ytr)
        acc_no.append(accuracy_score(yte,clf.predict(Xte_s)))

        # PCA 95% variance
        pca=PCA(n_components=0.95).fit(Xtr_s)
        Xtr_p=pca.transform(Xtr_s); Xte_p=pca.transform(Xte_s)
        ncomp_list.append(pca.n_components_)
        clf_p=LinearDiscriminantAnalysis().fit(Xtr_p,ytr)
        acc_pca.append(accuracy_score(yte,clf_p.predict(Xte_p)))

    res={
        'label':label,'short':short,'n_feat':n_feat,'n_region':n_region,
        'no_mean':np.mean(acc_no),'no_std':np.std(acc_no),
        'pca_mean':np.mean(acc_pca),'pca_std':np.std(acc_pca),
        'ncomp_mean':np.mean(ncomp_list),'ncomp_std':np.std(ncomp_list),
    }
    results[short]=res

    imp=res['pca_mean']-res['no_mean']
    print(f'  {label:>7s} ({n_feat:2d}feat->{res["ncomp_mean"]:.0f}PCs) '
          f'noPCA={res["no_mean"]:.1%}  PCA={res["pca_mean"]:.1%}  delta={imp:+.1%}')

# ====== FIGURE 1: Side-by-side bars ======
print('\n[1/3] Comparing no PCA vs PCA...')
order=sorted(results.keys(),key=lambda k:results[k]['no_mean'],reverse=True)
labels=[results[k]['label'] for k in order]
x=np.arange(len(labels)); w=0.35

fig1,(ax1a,ax1b)=plt.subplots(1,2,figsize=(20,6.5))
fig1.patch.set_facecolor('white')

# Subplot 1: bar comparison
vals_no=[results[k]['no_mean'] for k in order]
err_no=[results[k]['no_std'] for k in order]
vals_pca=[results[k]['pca_mean'] for k in order]
err_pca=[results[k]['pca_std'] for k in order]

bars_no=ax1a.bar(x-w/2,vals_no,w,yerr=err_no,color='#7f8c8d',capsize=3,edgecolor='#333',
                 lw=0.4,label=f'No PCA (original {results[order[0]]["n_feat"]} feat)')
bars_pca=ax1a.bar(x+w/2,vals_pca,w,yerr=err_pca,color='#e74c3c',capsize=3,edgecolor='#333',
                  lw=0.4,label=f'PCA (95% var)')
ax1a.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
ax1a.set_xticks(x);ax1a.set_xticklabels(labels,fontsize=7.5,rotation=45,ha='right')
ax1a.set_ylabel('4-Class Accuracy',fontsize=12)
ax1a.set_title('4-Class Accuracy: Original Features vs PCA-Reduced',fontsize=12,fontweight='bold')
ax1a.legend(fontsize=8,ncol=2);ax1a.grid(True,axis='y',ls=':',alpha=0.3)
ax1a.set_ylim(0,max(max(vals_no),max(vals_pca))*1.35)

# Improvement labels
for i,k in enumerate(order):
    imp=results[k]['pca_mean']-results[k]['no_mean']
    c='#27ae60' if imp>0 else '#c0392b' if imp<0 else '#888'
    ax1a.text(x[i]+w/2,results[k]['pca_mean']+0.015,f'{imp:+.1%}',
             ha='center',fontsize=6,fontweight='bold',color=c,rotation=90)

# Subplot 2: components reduction
n_orig=[results[k]['n_feat'] for k in order]
n_pca=[results[k]['ncomp_mean'] for k in order]
bars_o=ax1b.bar(x-w/2,n_orig,w,color='#7f8c8d',edgecolor='#333',lw=0.4,label='Original features')
bars_p=ax1b.bar(x+w/2,n_pca,w,color='#e74c3c',edgecolor='#333',lw=0.4,label='PCA components (95% var)')
# Reduction % labels
for i,k in enumerate(order):
    r=results[k]
    reduction=(r['n_feat']-r['ncomp_mean'])/r['n_feat']*100
    ax1b.text(x[i]+w/2,r['ncomp_mean']+0.5,f'{reduction:.0f}%',
             ha='center',fontsize=6,fontweight='bold',color='#c0392b',rotation=90)
ax1b.set_xticks(x);ax1b.set_xticklabels(labels,fontsize=7.5,rotation=45,ha='right')
ax1b.set_ylabel('Number of Features / Components',fontsize=12)
ax1b.set_title('Dimensionality Reduction: Original vs PCA Components',fontsize=12,fontweight='bold')
ax1b.legend(fontsize=8);ax1b.grid(True,axis='y',ls=':',alpha=0.3)

fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'41_pca_comparison_bars.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('  Saved: 41_pca_comparison_bars.png')

# ====== FIGURE 2: Scatter before vs after ======
print('[2/3] Scatter comparison...')
fig2,ax2=plt.subplots(figsize=(8,8))
fig2.patch.set_facecolor('white')

for gi in range(4):
    g_start=sum([4,6,4,1][:gi]); g_end=g_start+[4,6,4,1][gi]
    ks=list(results.keys())[g_start:g_end]
    xs=[results[k]['no_mean'] for k in ks]
    ys=[results[k]['pca_mean'] for k in ks]
    ax2.scatter(xs,ys,c=GROUP_COLORS[gi],s=100,alpha=0.8,edgecolors='#333',lw=0.5,
               label=['1R','2R','3R','4R'][gi],zorder=5)
    for k in ks:
        ax2.annotate(results[k]['label'],(results[k]['no_mean'],results[k]['pca_mean']),
                    textcoords='offset points',xytext=(5,5),fontsize=8,alpha=0.8)

# Diagonal (no change)
all_vals=[results[k]['no_mean'] for k in results]+[results[k]['pca_mean'] for k in results]
lo,hi=min(all_vals)-0.03,max(all_vals)+0.03
ax2.plot([lo,hi],[lo,hi],'--',color='#888',lw=1,zorder=1,label='No change')
ax2.axhline(0.25,color='#ccc',ls=':',lw=0.8);ax2.axvline(0.25,color='#ccc',ls=':',lw=0.8)

ax2.set_xlabel('Accuracy without PCA',fontsize=12)
ax2.set_ylabel('Accuracy with PCA',fontsize=12)
ax2.set_title('Before vs After PCA — Each Dot = One Region Combination',fontsize=12,fontweight='bold')
ax2.legend(fontsize=9);ax2.grid(True,ls=':',alpha=0.3)
ax2.set_xlim(lo,hi);ax2.set_ylim(lo,hi)
ax2.set_aspect('equal')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'42_pca_comparison_scatter.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('  Saved: 42_pca_comparison_scatter.png')

# ====== FIGURE 3: Improvement by combo ======
print('[3/3] Improvement bar chart...')
order_imp=sorted(results.keys(),key=lambda k:results[k]['pca_mean']-results[k]['no_mean'],reverse=True)

fig3,ax3=plt.subplots(figsize=(14,5))
fig3.patch.set_facecolor('white')

imps=[results[k]['pca_mean']-results[k]['no_mean'] for k in order_imp]
imp_labels=[results[k]['label'] for k in order_imp]
bar_colors=['#27ae60' if i>=0 else '#c0392b' for i in imps]
x3=np.arange(len(imps))
bars=ax3.bar(x3,imps,color=bar_colors,edgecolor='#333',lw=0.4,width=0.7)
ax3.axhline(0,color='#888',ls='--',lw=1)
for b,v in zip(bars,imps):
    ax3.text(b.get_x()+b.get_width()/2,b.get_height()+0.003*(1 if v>=0 else -1),
            f'{v:+.1%}',ha='center',fontsize=7,fontweight='bold')
ax3.set_xticks(x3);ax3.set_xticklabels(imp_labels,fontsize=8,rotation=45,ha='right')
ax3.set_ylabel('Accuracy Change (PCA - No PCA)',fontsize=12)
ax3.set_title('Impact of PCA on Decoding Accuracy — Sorted by Improvement',fontsize=12,fontweight='bold')
ax3.grid(True,axis='y',ls=':',alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR,'43_pca_improvement.png'),dpi=200,bbox_inches='tight')
plt.close(fig3)
print('  Saved: 43_pca_improvement.png')

# ====== Summary ======
print(f'\n{"="*80}')
print(f'  PCA COMPARISON SUMMARY')
print(f'  {len(epochs_data)} trials, 5-fold CV LDA, PCA 95% variance')
print(f'{"="*80}')
print(f'\n{"Combo":<8} {"OrigFeat":<9} {"PCs":<6} {"Reduc%":<8} {"NoPCA":<10} {"PCA":<10} {"Delta":<8}')
print(f'  {"-"*65}')
for k in sorted(results.keys(),key=lambda x:results[x]['no_mean'],reverse=True):
    r=results[k]
    reduc=(r['n_feat']-r['ncomp_mean'])/r['n_feat']*100
    imp=r['pca_mean']-r['no_mean']
    print(f'  {r["label"]:<8} {r["n_feat"]:<9} {r["ncomp_mean"]:<5.0f} {reduc:<7.0f}% '
          f'{r["no_mean"]:.1%}+-{r["no_std"]:.1%}  {r["pca_mean"]:.1%}+-{r["pca_std"]:.1%}  {imp:+.1%}')

# Averages
avg_no=np.mean([results[k]['no_mean'] for k in results])
avg_pca=np.mean([results[k]['pca_mean'] for k in results])
avg_reduc=np.mean([(results[k]['n_feat']-results[k]['ncomp_mean'])/results[k]['n_feat']*100 for k in results])
print(f'\n  Averages: NoPCA={avg_no:.1%}  PCA={avg_pca:.1%}  delta={avg_pca-avg_no:+.1%}')
print(f'  Avg reduction: {avg_reduc:.0f}% (from {np.mean([results[k]["n_feat"] for k in results]):.0f} to {np.mean([results[k]["ncomp_mean"] for k in results]):.0f} components)')

# Save
sp=os.path.join(OUT_DIR,'44_pca_summary.txt')
with open(sp,'w') as sf:
    sf.write(f'PCA Comparison Summary\n{"="*60}\n')
    sf.write(f'Trials: {len(epochs_data)}, 5-fold LDA, PCA 95% var\n\n')
    sf.write(f'{"Combo":<8} {"Orig":<6} {"PCs":<5} {"NoPCA":<10} {"PCA":<10} {"Delta":<8}\n')
    sf.write(f'{"-"*50}\n')
    for k in sorted(results.keys(),key=lambda x:results[x]['no_mean'],reverse=True):
        r=results[k]
        imp=r['pca_mean']-r['no_mean']
        sf.write(f'{r["label"]:<8} {r["n_feat"]:<6} {r["ncomp_mean"]:<5.0f} {r["no_mean"]:.4f} {r["pca_mean"]:.4f} {imp:+.4f}\n')
print(f'\nSaved: {sp}')
print('\nFigures: 41-43')
print('Done!')
