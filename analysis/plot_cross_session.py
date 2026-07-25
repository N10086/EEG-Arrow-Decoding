#!/usr/bin/env python3
"""
Cross-session validation: train on Sessions 1-4, test on Session 5.
All 15 region combinations, 4 ERP window features per channel (12 per region).
"""
import numpy as np
from scipy import signal as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
BASE=r'E:\deskbook\OpenBCI_GUI\stimulus_logs'
OUT_DIR=r'E:\deskbook\OpenBCI_GUI\stimulus_logs\analysis_v5'
DIR_NAMES=['Up','Down','Left','Right']
DIR_VALS=[2.0001,2.0002,2.0003,2.0004]
ERP_WINDOWS=[('P1',0.080,0.130),('N1',0.140,0.200),('P2',0.200,0.300),('P3',0.300,0.500)]
CH_MAP={1:'Oz',2:'C3',4:'Fz',5:'C4',6:'Cz',7:'F3',8:'O2',9:'P3',10:'Pz',12:'P4',14:'F4',15:'O1'}
HW_CHS=sorted(CH_MAP.keys())
REGIONS={'Frontal':[7,4,14],'Central':[2,6,5],'Parietal':[9,10,12],'Occipital':[15,1,8]}
REG_NAMES=['Frontal','Central','Parietal','Occipital']
SESSION_LABELS=['Session 1 (07/07 10:35)','Session 2 (07/07 10:47)','Session 3 (07/08 08:54)',
                'Session 4 (07/08 09:04)','Session 5 (07/09 09:02)']
SESSION_PATHS=[
    r'2026-07-07_10-35-51\OpenBCI-RAW-2026-07-07_10-35-51.txt',
    r'2026-07-07_10-47-49\OpenBCI-RAW-2026-07-07_10-47-49.txt',
    r'2026-07-08_08-54-11\OpenBCI-RAW-2026-07-08_08-54-11.txt',
    r'2026-07-08_09-04-45\OpenBCI-RAW-2026-07-08_09-04-45.txt',
    r'2026-07-09_09-02-54\OpenBCI-RAW-2026-07-09_09-02-54.txt',
]
SESSION_COLORS=['#4A72C4','#E8833A','#5CB85C','#9B59B6','#D94F70']
TRAIN_IDX=[0,1,2,3]; TEST_IDX=4

def map_direction(val):
    for di,k in enumerate(DIR_VALS):
        if abs(val-k)<5e-5: return di
    return -1

def load_and_process(path):
    with open(path) as f: lines=f.readlines()
    data,markers=[],[]
    for line in lines[5:]:
        p=line.strip().split(',')
        if len(p)>33:
            try:
                data.append([float(p[i]) for i in range(1,17)])
                markers.append(float(p[32]))
            except: pass
    d=np.array(data,dtype=np.float64).T
    m=np.array(markers)
    sel=np.array([d[c-1] for c in HW_CHS])

    bp=sg.butter(4,[1/250,45/250],btype='band',output='sos')
    notch=sg.iirnotch(50/250,30)
    f=np.zeros_like(sel)
    for ch in range(12):
        dm=sel[ch]-sel[ch].mean()
        ts=sg.sosfiltfilt(bp,dm)
        f[ch]=sg.filtfilt(*notch,ts)
    filt_uv=f*SCALE

    ons=sorted([i for k in DIR_VALS for i in np.where(np.abs(m-k)<5e-5)[0]])
    t_start,t_end=-0.2,0.8
    n_pre,n_post=int(abs(t_start)*FS),int(t_end*FS)
    n_total=n_pre+n_post

    epochs,labels=[],[]
    for idx in ons:
        s,e=idx-n_pre,idx+n_post
        if s<0 or e>=filt_uv.shape[1]: continue
        ep=filt_uv[:,s:e].copy()
        ep-=ep[:,:n_pre].mean(axis=1,keepdims=True)
        if np.max(np.abs(ep))>100: continue
        epochs.append(ep)
        labels.append(map_direction(m[idx]))

    return np.array(epochs), np.array(labels), np.linspace(t_start,t_end,n_total)

# ====== Load all sessions ======
print('Loading 5 sessions...')
all_epochs,all_labels=[],[]
for si,sp in enumerate(SESSION_PATHS):
    ep,lb,t=load_and_process(os.path.join(BASE,sp))
    all_epochs.append(ep); all_labels.append(lb)
    print(f'  {SESSION_LABELS[si]}: {len(ep)} trials')

# Combine train/test
train_ep=np.concatenate([all_epochs[i] for i in TRAIN_IDX],axis=0)
train_lb=np.concatenate([all_labels[i] for i in TRAIN_IDX],axis=0)
test_ep=all_epochs[TEST_IDX]
test_lb=all_labels[TEST_IDX]
print(f'\nTrain: {len(train_ep)} trials ({len(TRAIN_IDX)} sessions)')
print(f'Test:  {len(test_ep)} trials (Session 5)')

# Pre-compute window means
n_win=len(ERP_WINDOWS)
train_win=np.zeros((len(train_ep),12,n_win))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    train_win[:,:,wi]=train_ep[:,:,msk].mean(axis=2)
test_win=np.zeros((len(test_ep),12,n_win))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    test_win[:,:,wi]=test_ep[:,:,msk].mean(axis=2)

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
GROUP_LABELS=['1 Region','2 Regions','3 Regions','4 Regions']

# ====== Cross-session decoding ======
print('\nCross-session decoding (train S1-4, test S5)...')
results={}
for combo,label,short in ALL_COMBOS:
    ch_idx=[]
    for rn in combo:
        ch_idx.extend([HW_CHS.index(c) for c in REGIONS[rn]])

    Xtr=train_win[:,ch_idx,:].reshape(len(train_ep),-1)
    Xte=test_win[:,ch_idx,:].reshape(len(test_ep),-1)
    n_feat=Xtr.shape[1]

    # Standardize on training data, transform test
    scl=StandardScaler().fit(Xtr)
    Xtr_s=scl.transform(Xtr); Xte_s=scl.transform(Xte)

    # LDA
    clf=LinearDiscriminantAnalysis().fit(Xtr_s,train_lb)
    y_pred=clf.predict(Xte_s)

    acc4=accuracy_score(test_lb,y_pred)

    # Axis
    axis_true=(test_lb>=2).astype(int)
    axis_pred=(y_pred>=2).astype(int)
    ax_acc=accuracy_score(axis_true,axis_pred)

    # Per-class
    cm=confusion_matrix(test_lb,y_pred)
    per_class=np.diag(cm)/cm.sum(axis=1)*100

    # Within-axis
    vm=test_lb<2; hm=test_lb>=2
    vert_acc=accuracy_score(test_lb[vm],y_pred[vm]) if vm.sum()>0 else 0
    horz_acc=accuracy_score(test_lb[hm],y_pred[hm]) if hm.sum()>0 else 0

    results[short]={
        'label':label,'combo':combo,'n_feat':n_feat,
        'acc4':acc4,'ax_acc':ax_acc,'vert_acc':vert_acc,'horz_acc':horz_acc,
        'per_class':per_class,'cm':cm,
    }
    print(f'  {label:>7s} ({n_feat:2d}feat)  4-class={acc4:.1%}  axis={ax_acc:.1%}  vert={vert_acc:.1%}  horz={horz_acc:.1%}')

# ====== Load within-session results for comparison ======
# We'll reference the known within-session results from Session 2
within_s2={'F':61.5,'C':49.5,'P':48.0,'O':41.0,'FC':60.5,'FP':65.0,'FO':62.0,
           'CP':44.5,'CO':46.0,'PO':52.5,'FCP':57.0,'FCO':59.0,'FPO':60.5,'CPO':48.0,'All':54.0}

# ====== FIGURE 1: Cross-session accuracy ======
print('\n[1/4] Cross-session accuracy...')
order=sorted(results.keys(),key=lambda k:results[k]['acc4'],reverse=True)

fig1,(ax1a,ax1b)=plt.subplots(1,2,figsize=(18,6.5))
fig1.patch.set_facecolor('white')
labels=[results[k]['label'] for k in order]
vals=[results[k]['acc4'] for k in order]
colors=[GROUP_COLORS[min(3,len(results[k]['combo'])-1)] for k in order]
x=np.arange(len(labels))

# Left: cross-session bars
bars=ax1a.bar(x,vals,color=colors,edgecolor='#333',lw=0.4,width=0.7)
ax1a.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for b,v in zip(bars,vals):
    ax1a.text(b.get_x()+b.get_width()/2,b.get_height()+0.008,f'{v:.1%}',
             ha='center',fontsize=7,fontweight='bold',rotation=90)
ax1a.set_xticks(x);ax1a.set_xticklabels(labels,fontsize=7.5,rotation=45,ha='right')
ax1a.set_ylabel('4-Class Accuracy (Cross-Session)',fontsize=12)
ax1a.set_title('Cross-Session Validation: Train S1-4, Test S5',fontsize=12,fontweight='bold')
ax1a.legend(fontsize=9);ax1a.grid(True,axis='y',ls=':',alpha=0.3)
ax1a.set_ylim(0,max(vals)*1.35)

# Right: Cross-session vs within-session comparison
xs=np.arange(len(order)); w=0.35
within_vals=[within_s2[k]/100 for k in order]
cross_vals=[results[k]['acc4'] for k in order]
b1=ax1b.bar(xs-w/2,within_vals,w,color='#3498db',edgecolor='#333',lw=0.4,label='Within-Session (S2)')
b2=ax1b.bar(xs+w/2,cross_vals,w,color='#e74c3c',edgecolor='#333',lw=0.4,label='Cross-Session (S1-4->S5)')
ax1b.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
# Delta labels
for i,k in enumerate(order):
    delta=cross_vals[i]-within_vals[i]
    c='#27ae60' if delta>=-0.02 else '#c0392b'
    ax1b.text(i+w/2,cross_vals[i]+0.01,f'{delta:+.1%}',ha='center',fontsize=6,fontweight='bold',color=c,rotation=90)
ax1b.set_xticks(xs);ax1b.set_xticklabels(labels,fontsize=7.5,rotation=45,ha='right')
ax1b.set_ylabel('4-Class Accuracy',fontsize=12)
ax1b.set_title('Within-Session vs Cross-Session Comparison',fontsize=12,fontweight='bold')
ax1b.legend(fontsize=8);ax1b.grid(True,axis='y',ls=':',alpha=0.3)
ax1b.set_ylim(0,max(max(within_vals),max(cross_vals))*1.35)

fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'51_cross_session_bars.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('  Saved: 51_cross_session_bars.png')

# ====== FIGURE 2: Confusion matrix (best cross-session combo) ======
print('[2/4] Confusion matrix...')
best_cross=max(results.keys(),key=lambda k:results[k]['acc4'])
cm_cross=results[best_cross]['cm']
cm_norm=cm_cross.astype(float)/cm_cross.sum(axis=1,keepdims=True)*100

fig2,ax2=plt.subplots(figsize=(7,6))
fig2.patch.set_facecolor('white')
im2=ax2.imshow(cm_norm,cmap='YlOrRd',vmin=0,vmax=100)
ax2.set_xticks(range(4));ax2.set_xticklabels(DIR_NAMES,fontsize=11)
ax2.set_yticks(range(4));ax2.set_yticklabels(DIR_NAMES,fontsize=11)
ax2.set_xlabel('Predicted',fontsize=12);ax2.set_ylabel('Actual',fontsize=12)
ax2.set_title(f'Cross-Session Best: {results[best_cross]["label"]}'
             f'\n{results[best_cross]["acc4"]:.1%}  Axis={results[best_cross]["ax_acc"]:.1%}',
             fontsize=12,fontweight='bold')
for i in range(4):
    for j in range(4):
        c='white' if cm_norm[i,j]>50 else 'black'
        ax2.text(j,i,f'{cm_norm[i,j]:.0f}%\n({cm_cross[i,j]})',ha='center',va='center',fontsize=10,fontweight='bold',color=c)
fig2.colorbar(im2,ax=ax2,shrink=0.85)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'52_cross_session_confusion.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('  Saved: 52_cross_session_confusion.png')

# ====== FIGURE 3: Cross-session per-class & within-axis ======
print('[3/4] Per-class & within-axis...')
top10=list(results.keys())[:10]
fig3,axes3=plt.subplots(2,1,figsize=(16,10))
fig3.patch.set_facecolor('white')

# Per-class heatmap
pc_data=np.array([results[k]['per_class'] for k in top10])
im3=axes3[0].imshow(pc_data,cmap='RdYlGn',vmin=0,vmax=100,aspect='auto')
axes3[0].set_xticks(range(4));axes3[0].set_xticklabels(DIR_NAMES,fontsize=10)
axes3[0].set_yticks(range(len(top10)));axes3[0].set_yticklabels([results[k]['label'] for k in top10],fontsize=8)
axes3[0].set_xlabel('True Direction',fontsize=11);axes3[0].set_ylabel('Combination',fontsize=11)
axes3[0].set_title('Cross-Session: Per-Class Accuracy (%) -- Top 10',fontsize=12,fontweight='bold')
for i in range(len(top10)):
    for j in range(4):
        axes3[0].text(j,i,f'{pc_data[i,j]:.0f}%',ha='center',va='center',fontsize=8,
                     fontweight='bold',color='white' if pc_data[i,j]>50 else 'black')
fig3.colorbar(im3,ax=axes3[0],shrink=0.8)

# Within-axis bars
b_labels=[results[k]['label'] for k in top10]
b_vert=[results[k]['vert_acc'] for k in top10]
b_horz=[results[k]['horz_acc'] for k in top10]
b_axis=[results[k]['ax_acc'] for k in top10]
x3=np.arange(len(b_labels)); w=0.25
axes3[1].bar(x3-w,b_vert,w,label='Vertical (Up vs Down)',color='#8e44ad',alpha=0.8)
axes3[1].bar(x3,b_horz,w,label='Horizontal (Left vs Right)',color='#16a085',alpha=0.8)
axes3[1].bar(x3+w,b_axis,w,label='Axis (Vert vs Horz)',color='#f39c12',alpha=0.8)
axes3[1].axhline(0.5,color='#888',ls='--',lw=1,label='Chance (50%)')
axes3[1].set_xticks(x3);axes3[1].set_xticklabels(b_labels,fontsize=8,rotation=45,ha='right')
axes3[1].set_ylabel('Accuracy',fontsize=11)
axes3[1].set_title('Cross-Session: Within-Axis vs Axis-Level Decoding',fontsize=12,fontweight='bold')
axes3[1].legend(fontsize=9);axes3[1].grid(True,axis='y',ls=':',alpha=0.3)
axes3[1].set_ylim(0,1.0)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR,'53_cross_session_perclass.png'),dpi=200,bbox_inches='tight')
plt.close(fig3)
print('  Saved: 53_cross_session_perclass.png')

# ====== FIGURE 4: Scatter within vs cross ======
print('[4/4] Scatter comparison...')
fig4,ax4=plt.subplots(figsize=(8,8))
fig4.patch.set_facecolor('white')
for gi in range(4):
    g_start=sum([4,6,4,1][:gi]); g_end=g_start+[4,6,4,1][gi]
    ks=list(results.keys())[g_start:g_end]
    xs=[within_s2.get(k,0)/100 for k in ks]
    ys=[results[k]['acc4'] for k in ks]
    ax4.scatter(xs,ys,c=GROUP_COLORS[gi],s=120,alpha=0.8,edgecolors='#333',lw=0.5,
               label=GROUP_LABELS[gi],zorder=5)
    for k in ks:
        ax4.annotate(results[k]['label'],(within_s2.get(k,0)/100,results[k]['acc4']),
                    textcoords='offset points',xytext=(5,5),fontsize=8)

lo,hi=0.3,0.7
ax4.plot([lo,hi],[lo,hi],'--',color='#888',lw=1,zorder=1,label='Equal')
ax4.axhline(0.25,color='#ccc',ls=':',lw=0.8);ax4.axvline(0.25,color='#ccc',ls=':',lw=0.8)
ax4.set_xlabel('Within-Session Accuracy (Session 2)',fontsize=12)
ax4.set_ylabel('Cross-Session Accuracy (Train S1-4, Test S5)',fontsize=12)
ax4.set_title('Within-Session vs Cross-Session Generalization',fontsize=12,fontweight='bold')
ax4.legend(fontsize=9,loc='lower right');ax4.grid(True,ls=':',alpha=0.3)
ax4.set_xlim(lo,hi);ax4.set_ylim(lo,hi);ax4.set_aspect('equal')
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR,'54_cross_session_scatter.png'),dpi=200,bbox_inches='tight')
plt.close(fig4)
print('  Saved: 54_cross_session_scatter.png')

# ====== Summary ======
print(f'\n{"="*80}')
print(f'  CROSS-SESSION VALIDATION: Train S1-4 ({len(train_ep)} trials) -> Test S5 ({len(test_ep)} trials)')
print(f'{"="*80}')
print(f'\n{"Combo":<8} {"Feat":<6} {"Within":<10} {"Cross":<10} {"Delta":<10} {"Axis":<10}')
print(f'  {"-"*58}')
for k in sorted(results.keys(),key=lambda x:results[x]['acc4'],reverse=True):
    r=results[k]
    w=within_s2.get(k,0)/100
    delta=r['acc4']-w
    print(f'  {r["label"]:<8} {r["n_feat"]:<6} {w:.1%}    {r["acc4"]:.1%}    {delta:+.1%}   {r["ax_acc"]:.1%}')

avg_within=np.mean([within_s2.get(k,0)/100 for k in results])
avg_cross=np.mean([results[k]['acc4'] for k in results])
print(f'\n  Average: Within={avg_within:.1%}  Cross={avg_cross:.1%}  Drop={avg_cross-avg_within:+.1%}')
print(f'  Best cross-session: {results[best_cross]["label"]} = {results[best_cross]["acc4"]:.1%}')

sp=os.path.join(OUT_DIR,'55_cross_session_summary.txt')
with open(sp,'w') as sf:
    sf.write(f'Cross-Session Validation Summary\n{"="*60}\n')
    sf.write(f'Train: Sessions 1-4 ({len(train_ep)} trials)\n')
    sf.write(f'Test:  Session 5 ({len(test_ep)} trials)\n\n')
    sf.write(f'{"Combo":<8} {"Within":<10} {"Cross":<10} {"Delta":<10}\n')
    sf.write(f'{"-"*40}\n')
    for k in sorted(results.keys(),key=lambda x:results[x]['acc4'],reverse=True):
        r=results[k]; w=within_s2.get(k,0)/100
        sf.write(f'{r["label"]:<8} {w:.4f}     {r["acc4"]:.4f}     {r["acc4"]-w:+.4f}\n')

print(f'\nSaved: {sp}')
print('\nFigures: 51-54')
print('Done!')
