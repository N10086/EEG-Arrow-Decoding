#!/usr/bin/env python3
"""
Single-region direction decoding accuracy -- test how well each brain region
can classify Up/Down/Left/Right from single-trial EEG features.
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
from sklearn.metrics import confusion_matrix, accuracy_score
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
OUT_DIR = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
DATA_PATH = r'E:\deskbook\OpenBCI_GUI\stimulus_logs\2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt'
DIRECTIONS = ['Up', 'Down', 'Left', 'Right']
DIR_VALS = [2.0001, 2.0002, 2.0003, 2.0004]
def map_direction(val):
    for di, k in enumerate(DIR_VALS):
        if abs(val - k) < 5e-5:
            return di
    return -1
DIR_COLORS = ['#4A72C4','#E8833A','#5CB85C','#9B59B6']
ERP_WINDOWS = [('P1',0.080,0.130),('N1',0.140,0.200),('P2',0.200,0.300),('P3',0.300,0.500)]
WCOLORS = ['#27ae60','#7f8c8d','#e67e22','#9B59B6']
CH_MAP = {1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS = sorted(CH_MAP.keys())
SCALP_ROIS = {'Frontal':[7,4,14],'Central':[2,6,5],'Parietal':[9,10,12],'Occipital':[15,1,8]}
ROI_ORDER = ['Frontal','Central','Parietal','Occipital']

# ====== Load & preprocess ======
print('Loading data...')
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
d = np.array(data,dtype=np.float64).T
m = np.array(markers)
sel = np.array([d[c-1] for c in HW_CHS])

print('Filtering...')
bp = sg.butter(4,[1/250,45/250],btype='band',output='sos')
notch = sg.iirnotch(50/250,30)
filt = np.zeros_like(sel)
for ch in range(12):
    dm = sel[ch]-sel[ch].mean()
    ts = sg.sosfiltfilt(bp,dm)
    filt[ch] = sg.filtfilt(*notch,ts)
filt_uv = filt*SCALE

ons = sorted([i for k in DIR_VALS
              for i in np.where(np.abs(m-k)<5e-5)[0]])

t_start,t_end=-0.2,0.8; n_pre,n_post=int(abs(t_start)*FS),int(t_end*FS); n_total=n_pre+n_post
t = np.linspace(t_start,t_end,n_total)

epochs_data,epochs_label=[],[]
for idx in ons:
    s,e = idx-n_pre,idx+n_post
    if s<0 or e>=filt_uv.shape[1]: continue
    ep = filt_uv[:,s:e].copy()
    ep -= ep[:,:n_pre].mean(axis=1,keepdims=True)
    if np.max(np.abs(ep))>100: continue
    epochs_data.append(ep)
    epochs_label.append(map_direction(m[idx]))

epochs_data=np.array(epochs_data); epochs_label=np.array(epochs_label)
print(f'Total: {len(epochs_data)} trials')
for di in range(4):
    print(f'  {DIRECTIONS[di]}: {np.sum(epochs_label==di)}')

# ====== Decoding helper ======
def decode_lda(X,y,n_folds=5,return_all=False):
    skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    accs,ap,at=[],[],[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr)
        clf=LinearDiscriminantAnalysis().fit(scl.transform(Xtr),ytr)
        yp=clf.predict(scl.transform(Xte))
        ap.extend(yp);at.extend(yte);accs.append(accuracy_score(yte,yp))
    r={'acc_mean':np.mean(accs),'acc_std':np.std(accs)}
    if return_all: r['all_pred']=np.array(ap);r['all_true']=np.array(at)
    return r

parietal_chs=SCALP_ROIS['Parietal']
parietal_idx=[HW_CHS.index(c) for c in parietal_chs]
chance=0.25

# ====== FIGURE 1: Time-resolved (Parietal) ======
print('\n[1/4] Time-resolved (Parietal)...')
time_acc=np.zeros(n_total)
for ti in range(n_total):
    X=epochs_data[:,parietal_idx,ti]
    time_acc[ti]=decode_lda(X,epochs_label)['acc_mean']

print('  Permutation test...')
np.random.seed(42); n_perm=500; perm_peaks=[]
for p in range(n_perm):
    ys=np.random.permutation(epochs_label)
    pa=np.zeros(n_total)
    for ti in range(0,n_total,5):
        X=epochs_data[:,parietal_idx,ti]
        pa[ti]=decode_lda(X,ys)['acc_mean']
    msk=np.zeros(n_total,bool);msk[::5]=True
    pa_full=np.interp(np.arange(n_total),np.where(msk)[0],pa[msk])
    perm_peaks.append(np.max(pa_full))
sig_thresh=np.percentile(perm_peaks,95)
print(f'  Chance={chance:.0%}, Sig>{sig_thresh:.1%}, Peak={time_acc.max():.1%}')

fig1,ax1=plt.subplots(figsize=(14,5))
ax1.fill_between(t,chance,time_acc,where=time_acc>chance,color='#4A72C4',alpha=0.2)
ax1.plot(t,time_acc,color='#4A72C4',lw=2.5)
ax1.axhline(chance,color='#888',ls='--',lw=1.5,label=f'Chance ({chance:.0%})')
ax1.axhline(sig_thresh,color='#c0392b',ls=':',lw=1.5,label=f'p<0.05 ({sig_thresh:.1%})')
for cn,ws,we,cc in zip(['P1','N1','P2','P3'],[0.08,0.14,0.2,0.3],[0.13,0.20,0.30,0.50],WCOLORS):
    ax1.axvspan(ws,we,alpha=0.08,color=cc)
    ax1.text((ws+we)/2,ax1.get_ylim()[1]*0.93,cn,ha='center',fontsize=9,fontweight='bold',color=cc,alpha=0.7)
ax1.axvline(0,color='#333',ls='--',lw=0.8)
ax1.set_xlabel('Time (s)',fontsize=12); ax1.set_ylabel('Decoding Accuracy',fontsize=12)
ax1.set_title('Direction Decoding Accuracy over Time -- Parietal Channels (P3, Pz, P4)',fontsize=13,fontweight='bold')
ax1.legend(fontsize=10,loc='lower right'); ax1.grid(True,axis='y',ls=':',alpha=0.3)
ax1.set_ylim(0.15,max(1.0,time_acc.max()*1.3))
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'21_parietal_time_acc.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('  Saved: 21_parietal_time_acc.png')

# ====== FIGURE 2: Window-based bar (Parietal) ======
print('\n[2/4] Window-based (Parietal)...')
wnames,wmeans,wstds=[],[],[]
for wname,wstart,wend in ERP_WINDOWS:
    msk=(t>=wstart)&(t<=wend)
    X=epochs_data[:,parietal_idx,:][:,:,msk].mean(axis=2)
    r=decode_lda(X,epochs_label)
    wnames.append(wname);wmeans.append(r['acc_mean']);wstds.append(r['acc_std'])
# Combined
Xc=np.column_stack([epochs_data[:,parietal_idx,:][:,:,(t>=s)&(t<=e)].mean(axis=2) for _,s,e in ERP_WINDOWS])
rc=decode_lda(Xc,epochs_label)
wnames.append('Combined');wmeans.append(rc['acc_mean']);wstds.append(rc['acc_std'])

fig2,ax2=plt.subplots(figsize=(8,5))
cbar=WCOLORS+['#34495e'];xp=np.arange(len(wnames))
bars=ax2.bar(xp,wmeans,yerr=wstds,color=cbar,capsize=5,edgecolor='#333',lw=0.5,width=0.6)
ax2.axhline(chance,color='#888',ls='--',lw=1.5,label=f'Chance ({chance:.0%})')
for b,m in zip(bars,wmeans):
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{m:.1%}',ha='center',fontsize=10,fontweight='bold')
ax2.set_xticks(xp);ax2.set_xticklabels(wnames,fontsize=11)
ax2.set_ylabel('Decoding Accuracy',fontsize=12)
ax2.set_title('Parietal Region -- Window-based Decoding Accuracy',fontsize=13,fontweight='bold')
ax2.legend(fontsize=10);ax2.set_ylim(0,max(wmeans)*1.25);ax2.grid(True,axis='y',ls=':',alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'22_parietal_window_acc.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('  Saved: 22_parietal_window_acc.png')

# ====== FIGURE 3: Confusion matrix (Parietal, P3) ======
print('\n[3/4] Confusion matrix (Parietal, P3)...')
msk_p3=(t>=0.300)&(t<=0.500)
X_p3=epochs_data[:,parietal_idx,:][:,:,msk_p3].mean(axis=2)
r_p3=decode_lda(X_p3,epochs_label,return_all=True)
cm=confusion_matrix(r_p3['all_true'],r_p3['all_pred'])
cm_norm=cm.astype(float)/cm.sum(axis=1,keepdims=True)*100

fig3,(ax3a,ax3b)=plt.subplots(1,2,figsize=(14,5.5))
im1=ax3a.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max()*1.2)
ax3a.set_xticks(range(4));ax3a.set_xticklabels(DIRECTIONS,fontsize=10)
ax3a.set_yticks(range(4));ax3a.set_yticklabels(DIRECTIONS,fontsize=10)
ax3a.set_xlabel('Predicted',fontsize=11);ax3a.set_ylabel('Actual',fontsize=11)
ax3a.set_title('Confusion Matrix (Counts)',fontsize=12,fontweight='bold')
for i in range(4):
    for j in range(4):
        c='white' if cm[i,j]>cm.max()/2 else 'black'
        ax3a.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11,fontweight='bold',color=c)
fig3.colorbar(im1,ax=ax3a,shrink=0.8)
im2=ax3b.imshow(cm_norm,cmap='YlOrRd',vmin=0,vmax=100)
ax3b.set_xticks(range(4));ax3b.set_xticklabels(DIRECTIONS,fontsize=10)
ax3b.set_yticks(range(4));ax3b.set_yticklabels(DIRECTIONS,fontsize=10)
ax3b.set_xlabel('Predicted',fontsize=11);ax3b.set_ylabel('Actual',fontsize=11)
ax3b.set_title('Confusion Matrix (%)',fontsize=12,fontweight='bold')
for i in range(4):
    for j in range(4):
        c='white' if cm_norm[i,j]>50 else 'black'
        ax3b.text(j,i,f'{cm_norm[i,j]:.0f}%',ha='center',va='center',fontsize=11,fontweight='bold',color=c)
fig3.colorbar(im2,ax=ax3b,shrink=0.8)
fig3.suptitle('Parietal Channels -- P3 Window Direction Confusion',fontsize=13,fontweight='bold',y=1.02)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR,'23_parietal_confusion.png'),dpi=200,bbox_inches='tight')
plt.close(fig3)
print('  Saved: 23_parietal_confusion.png')

# ====== FIGURE 4: Region comparison ======
print('\n[4/4] Region comparison (P3 window)...')
rnames,rmeans,rstds=[],[],[]
for rname,chs in SCALP_ROIS.items():
    ci=[HW_CHS.index(c) for c in chs]
    X=epochs_data[:,ci,:][:,:,msk_p3].mean(axis=2)
    r=decode_lda(X,epochs_label)
    rnames.append(rname);rmeans.append(r['acc_mean']);rstds.append(r['acc_std'])

reg_colors=['#e74c3c','#3498db','#2ecc71','#f39c12']
fig4,ax4=plt.subplots(figsize=(8,5))
bars=ax4.bar(rnames,rmeans,yerr=rstds,color=reg_colors,capsize=5,edgecolor='#333',lw=0.5,width=0.6)
ax4.axhline(chance,color='#888',ls='--',lw=1.5,label=f'Chance ({chance:.0%})')
for b,m in zip(bars,rmeans):
    ax4.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{m:.1%}',ha='center',fontsize=11,fontweight='bold')
ax4.set_xticklabels(rnames,fontsize=11);ax4.set_ylabel('Decoding Accuracy',fontsize=12)
ax4.set_title('Cross-Region Comparison -- P3 Window (300-500ms)',fontsize=13,fontweight='bold')
ax4.legend(fontsize=10);ax4.set_ylim(0,max(rmeans)*1.25);ax4.grid(True,axis='y',ls=':',alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR,'24_region_decoding.png'),dpi=200,bbox_inches='tight')
plt.close(fig4)
print('  Saved: 24_region_decoding.png')

# ====== Pairwise decoding (Parietal, P3) ======
print('\nPairwise (Parietal, P3)...')
pairwise={}
for d1 in range(3):
    for d2 in range(d1+1,4):
        mp=(epochs_label==d1)|(epochs_label==d2)
        Xp,yb=X_p3[mp],(epochs_label[mp]==d2).astype(int)
        rp=decode_lda(Xp,yb)
        pairwise[f'{DIRECTIONS[d1]}-{DIRECTIONS[d2]}']=(rp['acc_mean'],rp['acc_std'])
        print(f'  {DIRECTIONS[d1]} vs {DIRECTIONS[d2]}: {rp["acc_mean"]:.1%} +/- {rp["acc_std"]:.1%}')

# ====== Summary ======
print(f'\n{"="*70}')
print(f'  DECODING SUMMARY -- Parietal Region (Session 2)')
print(f'  Chance: {chance:.0%}, Trials: {len(epochs_data)}')
print(f'{"="*70}')
print(f'  Time-resolved: peak={time_acc.max():.1%} @ {t[np.argmax(time_acc)]:.3f}s')
print(f'  Sig threshold: {sig_thresh:.1%}')
print(f'\n  Window-based:')
for wn,wm,ws in zip(wnames[:-1],wmeans[:-1],wstds[:-1]):
    print(f'    {wn}: {wm:.1%} +/- {ws:.1%}')
print(f'    Combined: {wmeans[-1]:.1%} +/- {wstds[-1]:.1%}')
print(f'\n  Region (P3):')
for rn,rm,rs in zip(rnames,rmeans,rstds):
    print(f'    {rn}: {rm:.1%} +/- {rs:.1%}')

sp=os.path.join(OUT_DIR,'25_decoding_summary.txt')
with open(sp,'w') as sf:
    sf.write(f'Decoding Summary -- Parietal Region\n{"="*60}\n')
    sf.write(f'Session 2, 4-class LDA, 5-fold CV, Chance={chance:.0%}\n')
    sf.write(f'Trials: {len(epochs_data)}\n\n')
    sf.write(f'Time-resolved peak: {time_acc.max():.4f} @ {t[np.argmax(time_acc)]:.3f}s\n')
    sf.write(f'Significance threshold: {sig_thresh:.4f}\n\n')
    sf.write(f'Window-based:\n')
    for wn,wm,ws in zip(wnames,wmeans,wstds):
        sf.write(f'  {wn}: {wm:.4f} +/- {ws:.4f}\n')
    sf.write(f'\nRegion comparison (P3):\n')
    for rn,rm,rs in zip(rnames,rmeans,rstds):
        sf.write(f'  {rn}: {rm:.4f} +/- {rs:.4f}\n')
    sf.write(f'\nPairwise (Parietal, P3):\n')
    for pk,pv in pairwise.items():
        sf.write(f'  {pk}: {pv[0]:.4f} +/- {pv[1]:.4f}\n')
print(f'\nSaved: {sp}')
print('Done!')
