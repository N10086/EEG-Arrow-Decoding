#!/usr/bin/env python3
"""
Comprehensive region-combination decoding — all 15 combinations of
1/2/3/4 brain regions, evaluated on 4-class exact accuracy and
2-class spatial-axis accuracy.
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
DIR_NAMES = ['Up','Down','Left','Right']
DIR_VALS = [2.0001,2.0002,2.0003,2.0004]
ERP_WINDOWS = [('P1',0.080,0.130),('N1',0.140,0.200),('P2',0.200,0.300),('P3',0.300,0.500)]
WCOLORS = ['#27ae60','#7f8c8d','#e67e22','#9B59B6']
CH_MAP = {1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS = sorted(CH_MAP.keys())

REGIONS = {
    'Frontal':[7,4,14],'Central':[2,6,5],
    'Parietal':[9,10,12],'Occipital':[15,1,8],
}
REG_COLORS = {'Frontal':'#e74c3c','Central':'#3498db','Parietal':'#2ecc71','Occipital':'#f39c12'}

def map_direction(val):
    for di,k in enumerate(DIR_VALS):
        if abs(val-k)<5e-5: return di
    return -1

# ====== Load ======
print('Loading...')
with open(DATA_PATH) as f:
    lines=f.readlines()
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
for di in range(4): print(f'  {DIR_NAMES[di]}: {np.sum(epochs_label==di)}')

# Pre-compute window means: n_trials x 12 x 4
print('Extracting features...')
n_win=len(ERP_WINDOWS)
win_data=np.zeros((len(epochs_data),12,n_win))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_data[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# Pre-compute per-region gradient features: n_trials x 4 x 2
#   [:, :, 0] = mean P3 window slope (avg first-difference within P3)
#   [:, :, 1] = P2->P3 transition (P3_mean - P2_mean)
REGION_NAMES = ['Frontal','Central','Parietal','Occipital']
grad_data = np.zeros((len(epochs_data), 4, 2))
p3m = (t >= 0.300) & (t <= 0.500)
p2m = (t >= 0.200) & (t <= 0.300)
for ri, rn in enumerate(REGION_NAMES):
    ci = [HW_CHS.index(c) for c in REGIONS[rn]]
    pd = epochs_data[:, ci, :][:, :, p3m]           # trials x 3 x n_p3
    grad_data[:, ri, 0] = np.diff(pd, axis=2).mean(axis=(1,2))  # mean slope
    p2d = epochs_data[:, ci, :][:, :, p2m].mean(axis=2)         # trials x 3
    p3d = pd.mean(axis=2)                                        # trials x 3
    grad_data[:, ri, 1] = (p3d - p2d).mean(axis=1)              # mean transition

# ====== Combination definitions ======
ALL_COMBOS = [
    (['Frontal'],'F','F'),(['Central'],'C','C'),(['Parietal'],'P','P'),(['Occipital'],'O','O'),
    (['Frontal','Central'],'F+C','FC'),(['Frontal','Parietal'],'F+P','FP'),
    (['Frontal','Occipital'],'F+O','FO'),(['Central','Parietal'],'C+P','CP'),
    (['Central','Occipital'],'C+O','CO'),(['Parietal','Occipital'],'P+O','PO'),
    (['Frontal','Central','Parietal'],'F+C+P','FCP'),(['Frontal','Central','Occipital'],'F+C+O','FCO'),
    (['Frontal','Parietal','Occipital'],'F+P+O','FPO'),(['Central','Parietal','Occipital'],'C+P+O','CPO'),
    (['Frontal','Central','Parietal','Occipital'],'All','All'),
]
N_COMBO_GROUPS = [4,6,4,1]
GROUP_LABELS = ['1 Region','2 Regions','3 Regions','4 Regions']
GROUP_COLORS = ['#3498db','#2ecc71','#e67e22','#e74c3c']

# ====== Decoding function ======
def decode_4class(X,y,n_folds=5,return_all=False):
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    accs,ap,at=[],[],[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr)
        clf=LinearDiscriminantAnalysis().fit(scl.transform(Xtr),ytr)
        yp=clf.predict(scl.transform(Xte))
        ap.extend(yp);at.extend(yte);accs.append(accuracy_score(yte,yp))
    r={'acc4':np.mean(accs),'acc4_std':np.std(accs)}
    if return_all:
        r['pred']=np.array(ap);r['true']=np.array(at)
    return r

# ====== Run all combinations ======
print('\nRunning 15 combinations...')
results={}
for combo,label,short in ALL_COMBOS:
    # Get channel indices
    ch_idx=[]
    for rn in combo:
        ch_idx.extend([HW_CHS.index(c) for c in REGIONS[rn]])
    n_ch=len(ch_idx)

    # Window means: trials x (n_ch * 4)
    X_win=win_data[:,ch_idx,:].reshape(len(epochs_data),-1)
    # Gradient features: trials x (n_regions * 2)
    ri_list=[REGION_NAMES.index(rn) for rn in combo]
    X_grad=grad_data[:,ri_list,:].reshape(len(epochs_data),-1)
    X=np.concatenate([X_win,X_grad],axis=1)
    n_feat=X.shape[1]

    # 4-class
    r=decode_4class(X,epochs_label,return_all=True)

    # 2-class axis (vertical=0,1 vs horizontal=2,3)
    axis_true=(epochs_label>=2).astype(int)
    axis_pred=(r['pred']>=2).astype(int)
    ax_acc=accuracy_score(axis_true,axis_pred)

    # Per-class accuracy from confusion matrix
    cm=confusion_matrix(r['true'],r['pred'])
    per_class=np.diag(cm)/cm.sum(axis=1)*100

    # Within-axis accuracy
    vert_mask=epochs_label<2
    horz_mask=epochs_label>=2
    if vert_mask.sum()>0:
        Xv=X[vert_mask]; yv=epochs_label[vert_mask]
        rv=decode_4class(Xv,yv)
        vert_acc=rv['acc4']
    else:
        vert_acc=0
    if horz_mask.sum()>0:
        Xh=X[horz_mask]; yh=epochs_label[horz_mask]
        rh=decode_4class(Xh,yh)
        horz_acc=rh['acc4']
    else:
        horz_acc=0

    results[short]={
        'combo':combo,'label':label,'short':short,
        'n_ch':n_ch,'n_feat':n_feat,
        'acc4':r['acc4'],'acc4_std':r['acc4_std'],
        'ax_acc':ax_acc,'vert_acc':vert_acc,'horz_acc':horz_acc,
        'per_class':per_class,'cm':cm,
    }

    print(f'  {label:>7s} ({n_ch}ch,{n_feat}feat)'
          f'  4-class={r["acc4"]:.1%}+-{r["acc4_std"]:.1%}'
          f'  axis={ax_acc:.1%}'
          f'  vert={vert_acc:.1%} horz={horz_acc:.1%}')

# ====== FIGURE 1: 4-class accuracy ======
print('\n[1/4] 4-class accuracy comparison...')
order_4=sorted(results.keys(),key=lambda k:results[k]['acc4'],reverse=True)

fig1,(ax1a,ax1b)=plt.subplots(1,2,figsize=(18,6))
fig1.patch.set_facecolor('white')

# Left: all 15 sorted
labels=[results[k]['label'] for k in order_4]
vals=[results[k]['acc4'] for k in order_4]
errs=[results[k]['acc4_std'] for k in order_4]
colors_4=[GROUP_COLORS[min(3,sum(1 for _ in results[k]['combo'])-1)] for k in order_4]

x=np.arange(len(labels))
bars=ax1a.bar(x,vals,yerr=errs,color=colors_4,capsize=3,edgecolor='#333',lw=0.4,width=0.7)
ax1a.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for i,(b,v) in enumerate(zip(bars,vals)):
    ax1a.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{v:.1%}',
              ha='center',fontsize=7,fontweight='bold',rotation=45)
ax1a.set_xticks(x);ax1a.set_xticklabels(labels,fontsize=8,rotation=45,ha='right')
ax1a.set_ylabel('4-Class Accuracy',fontsize=12)
ax1a.set_title('4-Class Exact Accuracy -- All 15 Region Combinations',fontsize=12,fontweight='bold')
ax1a.legend(fontsize=9);ax1a.grid(True,axis='y',ls=':',alpha=0.3)
ax1a.set_ylim(0,max(vals)*1.35)

# Right: grouped by # regions
gp_x=[]; gp_y=[]; gp_err=[]; gp_ticks=[]; gp_colors=[]
offset=0
for gi,grp in enumerate([4,6,4,1]):
    g_start=sum([4,6,4,1][:gi])
    g_end=g_start+grp
    combo_slice=list(results.keys())[g_start:g_end]
    for ci,k in enumerate(combo_slice):
        gp_x.append(offset+ci*0.9)
        gp_y.append(results[k]['acc4'])
        gp_err.append(results[k]['acc4_std'])
        gp_ticks.append(results[k]['label'])
        gp_colors.append(GROUP_COLORS[gi])
    if gi<3: offset+=grp*0.9+1.5
    else: offset+=grp*0.9

bars2=ax1b.bar(gp_x,gp_y,yerr=gp_err,color=gp_colors,capsize=3,edgecolor='#333',lw=0.4,width=0.7)
ax1b.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for b,v in zip(bars2,gp_y):
    ax1b.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{v:.1%}',
              ha='center',fontsize=7,fontweight='bold')

# Add group dividers
cum=[0]+[sum([4,6,4,1][:i+1])*0.9+i*1.5 for i in range(3)]
for cx in cum[1:]:
    ax1b.axvline(cx-0.75,color='#ccc',ls='-',lw=1)
for i,gx in enumerate([0,cum[1]-0.75,(cum[1]+cum[2])/2-0.75,(cum[2]+cum[3])/2-0.75]):
    mid=(cum[i]+(cum[i+1] if i<3 else gp_x[-1]+0.45))/2
    ax1b.text(mid,ax1b.get_ylim()[1]*0.96,GROUP_LABELS[i],ha='center',fontsize=10,fontweight='bold',color=GROUP_COLORS[i])

ax1b.set_xticks(gp_x);ax1b.set_xticklabels(gp_ticks,fontsize=7,rotation=45,ha='right')
ax1b.set_ylabel('4-Class Accuracy',fontsize=12)
ax1b.set_title('Grouped by Number of Regions',fontsize=12,fontweight='bold')
ax1b.legend(fontsize=9);ax1b.grid(True,axis='y',ls=':',alpha=0.3)
ax1b.set_ylim(0,max(gp_y)*1.35)

fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'31_region_4class_accuracy.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('  Saved: 31_region_4class_accuracy.png')

# ====== FIGURE 2: 2-class spatial accuracy ======
print('[2/4] 2-class spatial accuracy...')
order_ax=sorted(results.keys(),key=lambda k:results[k]['ax_acc'],reverse=True)

fig2,ax2=plt.subplots(figsize=(14,6))
fig2.patch.set_facecolor('white')
labels_ax=[results[k]['label'] for k in order_ax]
vals_ax=[results[k]['ax_acc'] for k in order_ax]
color_ax=[GROUP_COLORS[min(3,sum(1 for _ in results[k]['combo'])-1)] for k in order_ax]

x2=np.arange(len(labels_ax))
bars=ax2.bar(x2,vals_ax,color=color_ax,edgecolor='#333',lw=0.4,width=0.7)
ax2.axhline(0.5,color='#888',ls='--',lw=1.5,label='Chance (50%)')
for i,(b,v) in enumerate(zip(bars,vals_ax)):
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.008,f'{v:.1%}',
              ha='center',fontsize=7,fontweight='bold',rotation=90)
ax2.set_xticks(x2);ax2.set_xticklabels(labels_ax,fontsize=8,rotation=45,ha='right')
ax2.set_ylabel('Spatial Axis Accuracy (Vertical vs Horizontal)',fontsize=12)
ax2.set_title('2-Class Spatial Axis Discrimination -- All 15 Region Combinations',fontsize=12,fontweight='bold')
ax2.legend(fontsize=9);ax2.grid(True,axis='y',ls=':',alpha=0.3)
ax2.set_ylim(0.3,max(vals_ax)*1.15)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'32_region_2class_accuracy.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('  Saved: 32_region_2class_accuracy.png')

# ====== FIGURE 3: Per-class & within-axis breakdown for top 10 ======
print('[3/4] Per-class breakdown...')
top10=list(results.keys())[:10]
fig3,axes3=plt.subplots(2,1,figsize=(16,10))
fig3.patch.set_facecolor('white')

# Top: per-class accuracy heatmap
pc_data=np.array([results[k]['per_class'] for k in top10])
im3=axes3[0].imshow(pc_data,cmap='RdYlGn',vmin=0,vmax=100,aspect='auto')
axes3[0].set_xticks(range(4));axes3[0].set_xticklabels(DIR_NAMES,fontsize=10)
axes3[0].set_yticks(range(len(top10)));axes3[0].set_yticklabels([results[k]['label'] for k in top10],fontsize=8)
axes3[0].set_xlabel('True Direction',fontsize=11);axes3[0].set_ylabel('Region Combination',fontsize=11)
axes3[0].set_title('Per-Class Decoding Accuracy (%) -- Top 10 Combinations',fontsize=12,fontweight='bold')
for i in range(len(top10)):
    for j in range(4):
        axes3[0].text(j,i,f'{pc_data[i,j]:.0f}%',ha='center',va='center',fontsize=8,
                     fontweight='bold',color='white' if pc_data[i,j]>50 else 'black')
fig3.colorbar(im3,ax=axes3[0],shrink=0.8)

# Bottom: within-axis vertical & horizontal bars
b_labels=[results[k]['label'] for k in top10]
b_vert=[results[k]['vert_acc'] for k in top10]
b_horz=[results[k]['horz_acc'] for k in top10]
b_axis=[results[k]['ax_acc'] for k in top10]

x3=np.arange(len(b_labels)); w=0.25
axes3[1].bar(x3-w,b_vert,w,label='Vertical (Up vs Down)',color='#8e44ad',alpha=0.8)
axes3[1].bar(x3,b_horz,w,label='Horizontal (Left vs Right)',color='#16a085',alpha=0.8)
axes3[1].bar(x3+w,b_axis,w,label='Axis (Vertical vs Horizontal)',color='#f39c12',alpha=0.8)
axes3[1].axhline(0.5,color='#888',ls='--',lw=1,label='Chance (50%)')
axes3[1].set_xticks(x3);axes3[1].set_xticklabels(b_labels,fontsize=8,rotation=45,ha='right')
axes3[1].set_ylabel('Accuracy',fontsize=11)
axes3[1].set_title('Within-Axis vs Axis-Level Decoding -- Top 10 Combinations',fontsize=12,fontweight='bold')
axes3[1].legend(fontsize=9);axes3[1].grid(True,axis='y',ls=':',alpha=0.3)
axes3[1].set_ylim(0,1.0)

fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR,'33_region_perclass_breakdown.png'),dpi=200,bbox_inches='tight')
plt.close(fig3)
print('  Saved: 33_region_perclass_breakdown.png')

# ====== FIGURE 4: Accuracy vs #features scatter ======
print('[4/4] Accuracy vs features...')
fig4,(ax4a,ax4b)=plt.subplots(1,2,figsize=(16,6))
fig4.patch.set_facecolor('white')

n_feats=[results[k]['n_feat'] for k in results]
acc4s=[results[k]['acc4'] for k in results]
axs=[results[k]['ax_acc'] for k in results]
combo_shorts=list(results.keys())

# Add small jitter for visibility
for ax_target,y_vals,ylabel,title in [(ax4a,acc4s,'4-Class Accuracy','4-Class'),(ax4b,axs,'Axis Accuracy','2-Class Axis')]:
    for gi,(grp_name,grp_n) in enumerate(zip(GROUP_LABELS,[4,6,4,1])):
        g_start=sum([4,6,4,1][:gi]); g_end=g_start+grp_n
        xs=n_feats[g_start:g_end]
        ys=y_vals[g_start:g_end]
        labs=[combo_shorts[i] for i in range(g_start,g_end)]
        ax_target.scatter(xs,ys,c=GROUP_COLORS[gi],s=120,alpha=0.8,edgecolors='#333',lw=0.5,
                         label=grp_name,zorder=5)
        for xi,yi,li in zip(xs,ys,labs):
            ax_target.annotate(results[li]['label'],(xi,yi),textcoords='offset points',
                             xytext=(5,5),fontsize=7,alpha=0.7)
    ax_target.set_xlabel('Number of Features',fontsize=11)
    ax_target.set_ylabel(ylabel,fontsize=11)
    ax_target.set_title(f'{title} -- Accuracy vs Model Complexity',fontsize=12,fontweight='bold')
    ax_target.legend(fontsize=9);ax_target.grid(True,ls=':',alpha=0.3)
    if '4-Class' in ylabel:
        ax_target.axhline(0.25,color='#888',ls='--',lw=1,alpha=0.5)
    else:
        ax_target.axhline(0.5,color='#888',ls='--',lw=1,alpha=0.5)

fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR,'34_region_accuracy_vs_features.png'),dpi=200,bbox_inches='tight')
plt.close(fig4)
print('  Saved: 34_region_accuracy_vs_features.png')

# ====== FIGURE 5: Best combo confusion matrix ======
best_4=max(results.keys(),key=lambda k:results[k]['acc4'])
print(f'[5/5] Best confusion ({best_4})...')
cm_best=results[best_4]['cm']
cm_norm=cm_best.astype(float)/cm_best.sum(axis=1,keepdims=True)*100

fig5,ax5=plt.subplots(figsize=(7,6))
fig5.patch.set_facecolor('white')
im5=ax5.imshow(cm_norm,cmap='YlOrRd',vmin=0,vmax=100)
ax5.set_xticks(range(4));ax5.set_xticklabels(DIR_NAMES,fontsize=11)
ax5.set_yticks(range(4));ax5.set_yticklabels(DIR_NAMES,fontsize=11)
ax5.set_xlabel('Predicted',fontsize=12);ax5.set_ylabel('Actual',fontsize=12)
ax5.set_title(f'Best Combination: {results[best_4]["label"]}'
             f'\n4-Class Accuracy: {results[best_4]["acc4"]:.1%}'
             f'  Axis: {results[best_4]["ax_acc"]:.1%}',
             fontsize=12,fontweight='bold')
for i in range(4):
    for j in range(4):
        c='white' if cm_norm[i,j]>50 else 'black'
        ax5.text(j,i,f'{cm_norm[i,j]:.0f}%\n({cm_best[i,j]})',ha='center',va='center',fontsize=10,fontweight='bold',color=c)
fig5.colorbar(im5,ax=ax5,shrink=0.85)
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR,'35_region_best_confusion.png'),dpi=200,bbox_inches='tight')
plt.close(fig5)
print('  Saved: 35_region_best_confusion.png')

# ====== Numerical summary ======
print(f'\n{"="*80}')
print(f'  REGION COMBINATION DECODING -- COMPLETE RESULTS')
print(f'  {len(epochs_data)} trials, 4-class LDA, 5-fold CV')
print(f'  Features: 4 ERP windows (P1/N1/P2/P3) per channel + 2 gradient features per region')
print(f'  Single region: 3x4 + 2 = 14 features')
print(f'{"="*80}')

print(f'\n{"Combo":<9} {"Ch":<4} {"Feat":<5} {"4-class":<12} {"Axis":<10} {"Vert":<10} {"Horz":<10} {"Improv":<10}')
print(f'  {"-"*64}')
baseline=None
for k in sorted(results.keys(),key=lambda x:results[x]['acc4'],reverse=True):
    r=results[k]
    if r['n_feat']==14 and len(r['combo'])==1 and baseline is None:
        baseline=r['acc4']
    improv=r['acc4']-baseline if baseline is not None else 0
    print(f'  {r["label"]:<9} {r["n_ch"]:<4} {r["n_feat"]:<5} {r["acc4"]:.1%}+-{r["acc4_std"]:.1%}  {r["ax_acc"]:.1%}     {r["vert_acc"]:.1%}     {r["horz_acc"]:.1%}     {improv:+.1%}')

# Summary statistics
print(f'\n  KEY FINDINGS:')
print(f'  Best 4-class: {results[best_4]["label"]} = {results[best_4]["acc4"]:.1%}')
best_single=max([k for k in results if len(results[k]['combo'])==1],key=lambda k:results[k]['acc4'])
print(f'  Best single: {results[best_single]["label"]} = {results[best_single]["acc4"]:.1%}')
print(f'  Best pair: {max([k for k in results if len(results[k]["combo"])==2],key=lambda k:results[k]["acc4"])} = {max(results[k]["acc4"] for k in results if len(results[k]["combo"])==2):.1%}')
print(f'  Best triple: {max([k for k in results if len(results[k]["combo"])==3],key=lambda k:results[k]["acc4"])} = {max(results[k]["acc4"] for k in results if len(results[k]["combo"])==3):.1%}')

print(f'\n  4-class ranking:')
for i,k in enumerate(order_4):
    r=results[k]
    print(f'  {i+1:2d}. {r["label"]:<9} {r["acc4"]:.1%} (axis={r["ax_acc"]:.1%}, vert={r["vert_acc"]:.1%}, horz={r["horz_acc"]:.1%})')

print(f'\n  Axis accuracy ranking:')
for i,k in enumerate(order_ax):
    r=results[k]
    print(f'  {i+1:2d}. {r["label"]:<9} {r["ax_acc"]:.1%} (4-class={r["acc4"]:.1%})')

# Save
sp=os.path.join(OUT_DIR,'36_region_combo_summary.txt')
with open(sp,'w') as sf:
    sf.write(f'Region Combination Decoding Summary\n{"="*60}\n')
    sf.write(f'Trials: {len(epochs_data)}, 4-class LDA, 5-fold CV\n')
    sf.write(f'Features: 4 ERP windows + 2 gradients per region\n\n')
    sf.write(f'{"Combo":<9} {"4-class":<12} {"Axis":<10} {"Vert":<10} {"Horz":<10}\n')
    sf.write(f'{"-"*55}\n')
    for k in sorted(results.keys(),key=lambda x:results[x]['acc4'],reverse=True):
        r=results[k]
        sf.write(f'{r["label"]:<9} {r["acc4"]:.4f}+-{r["acc4_std"]:.4f} {r["ax_acc"]:.4f} {r["vert_acc"]:.4f} {r["horz_acc"]:.4f}\n')
    sf.write(f'\nPer-class accuracy:\n')
    for k in sorted(results.keys(),key=lambda x:results[x]['acc4'],reverse=True):
        r=results[k]
        sf.write(f'{r["label"]:<9} Up={r["per_class"][0]:.1f}% Down={r["per_class"][1]:.1f}% Left={r["per_class"][2]:.1f}% Right={r["per_class"][3]:.1f}%\n')

print(f'\nSaved: {sp}')
print(f'\nFigures: 31-35 in {OUT_DIR}')
print('Done!')
