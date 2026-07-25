#!/usr/bin/env python3
"""
Single-channel decoding — each of the 12 channels independently.
4 ERP windows per channel, 4-class LDA with 5-fold CV.
Shows the decoding topography across the scalp.
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
HW_CHS=sorted(CH_MAP.keys())
HW_NAMES=[CH_MAP[c] for c in HW_CHS]

# 10-20 positions for topomap
CH_POS = {
    'F3':(-0.3,0.7),'Fz':(0,0.8),'F4':(0.3,0.7),
    'C3':(-0.5,0.35),'Cz':(0,0.4),'C4':(0.5,0.35),
    'P3':(-0.4,0),'Pz':(0,0),'P4':(0.4,0),
    'O1':(-0.25,-0.4),'Oz':(0,-0.5),'O2':(0.25,-0.4),
}

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
    epochs_data.append(ep); epochs_label.append(map_direction(m[idx]))
epochs_data=np.array(epochs_data); epochs_label=np.array(epochs_label)
print(f'Trials: {len(epochs_data)}')

# Pre-compute window means per channel: n_trials x 12 x 4
win_data=np.zeros((len(epochs_data),12,4))
for wi,(_,ws,we) in enumerate(ERP_WINDOWS):
    msk=(t>=ws)&(t<=we)
    win_data[:,:,wi]=epochs_data[:,:,msk].mean(axis=2)

# ====== Single-channel decoding ======
print('\nSingle-channel decoding (4 ERP windows each)...')
ch_results={}
for ci in range(12):
    X=win_data[:,ci,:]  # n_trials x 4
    y=epochs_label
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs,all_pred,all_true=[],[],[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        scl=StandardScaler().fit(Xtr)
        clf=LinearDiscriminantAnalysis().fit(scl.transform(Xtr),ytr)
        yp=clf.predict(scl.transform(Xte))
        all_pred.extend(yp);all_true.extend(yte);accs.append(accuracy_score(yte,yp))
    acc4=np.mean(accs)
    # Axis
    ax_acc=accuracy_score((np.array(all_true)>=2).astype(int),(np.array(all_pred)>=2).astype(int))
    # Per-class
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(all_true,all_pred)
    per_class=np.diag(cm)/cm.sum(axis=1)*100

    ch_results[HW_NAMES[ci]]={'acc4':acc4,'ax_acc':ax_acc,'per_class':per_class}
    print(f'  {HW_NAMES[ci]:4s} (ch{HW_CHS[ci]:2d}): 4-class={acc4:.1%}  axis={ax_acc:.1%}')

# ====== Figure 1: Bar chart ======
fig1,ax1=plt.subplots(figsize=(14,5))
fig1.patch.set_facecolor('white')
order_ch=sorted(ch_results.keys(),key=lambda k:ch_results[k]['acc4'],reverse=True)
x1=np.arange(12)
vals1=[ch_results[k]['acc4'] for k in order_ch]
colors1=[]
for k in order_ch:
    for ri,(rn,chs) in enumerate(REGIONS.items()):
        if k in chs:
            colors1.append(REG_COLORS[ri])
            break
bars=ax1.bar(x1,vals1,color=colors1,edgecolor='#333',lw=0.5,width=0.7)
ax1.axhline(0.25,color='#888',ls='--',lw=1.5,label='Chance (25%)')
for b,v in zip(bars,vals1):
    ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{v:.1%}',
            ha='center',fontsize=9,fontweight='bold')
ax1.set_xticks(x1);ax1.set_xticklabels(order_ch,fontsize=10)
ax1.set_ylabel('4-Class Accuracy',fontsize=12)
ax1.set_title('Single-Channel Decoding Accuracy — Each Channel\'s 4 ERP Windows',fontsize=13,fontweight='bold')
ax1.legend(fontsize=10);ax1.grid(True,axis='y',ls=':',alpha=0.3)
ax1.set_ylim(0,max(vals1)*1.25)
# Add region labels at top
for ri,(rn,chs) in enumerate(REGIONS.items()):
    for k in chs:
        if k in order_ch:
            pass
    r_x=[x1[i] for i,k in enumerate(order_ch) if k in chs]
    if r_x:
        mid=sum(r_x)/len(r_x)
        ax1.text(mid,max(vals1)*1.18,rn,ha='center',fontsize=9,fontweight='bold',color=REG_COLORS[ri])
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR,'71_single_channel_decoding.png'),dpi=200,bbox_inches='tight')
plt.close(fig1)
print('  Saved: 71_single_channel_decoding.png')

# ====== Figure 2: Topomap ======
fig2,ax2=plt.subplots(figsize=(7,6))
fig2.patch.set_facecolor('white')
acc_vals=np.array([ch_results[n]['acc4'] for n in HW_NAMES])
pos_vals=np.array([CH_POS[n] for n in HW_NAMES])
# Simple interpolation for topomap look
from scipy.interpolate import griddata
xi=np.linspace(-0.7,0.7,100); yi=np.linspace(-0.7,0.7,100)
XI,YI=np.meshgrid(xi,yi)
ZI=griddata(pos_vals,acc_vals,(XI,YI),method='cubic',fill_value=0.25)
# Mask outside convex hull
from scipy.spatial import ConvexHull
hull=ConvexHull(pos_vals)
mask=np.zeros(XI.shape,dtype=bool)
for i in range(XI.shape[0]):
    for j in range(XI.shape[1]):
        # Simple distance-based mask
        d=np.min(np.sqrt((pos_vals[:,0]-XI[i,j])**2+(pos_vals[:,1]-YI[i,j])**2))
        if d>0.3: mask[i,j]=True
ZI_m=np.ma.array(ZI,mask=mask)

cont=ax2.contourf(XI,YI,ZI_m,levels=20,cmap='YlOrRd',vmin=0.2,vmax=max(acc_vals)*1.05)
# Plot electrode positions
for ri,(rn,chs) in enumerate(REGIONS.items()):
    for ch in chs:
        xp,yp=CH_POS[ch]
        acc=ch_results[ch]['acc4']
        ax2.scatter(xp,yp,s=200,c=[REG_COLORS[ri]],edgecolors='#333',lw=1.5,zorder=5)
        ax2.text(xp,yp-0.02,f'{ch}\n{acc:.1%}',ha='center',va='top',fontsize=7,fontweight='bold',color='#333')
# Head outline
circle=plt.Circle((0,0),0.6,fill=False,color='#555',lw=2,ls='-')
ax2.add_patch(circle)
# Nose
ax2.plot([-0.08,0,0.08],[0.6,0.72,0.6],color='#555',lw=2)
fig2.colorbar(cont,ax=ax2,shrink=0.8,label='Decoding Accuracy')
ax2.set_xlim(-0.7,0.7);ax2.set_ylim(-0.7,0.75)
ax2.set_aspect('equal');ax2.axis('off')
ax2.set_title('Scalp Topography of Single-Channel Decoding Accuracy\n(Session 2, 4-class LDA)',fontsize=12,fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR,'72_single_channel_topomap.png'),dpi=200,bbox_inches='tight')
plt.close(fig2)
print('  Saved: 72_single_channel_topomap.png')

# ====== Figure 3: Per-class accuracy per channel ======
fig3,ax3=plt.subplots(figsize=(14,6))
fig3.patch.set_facecolor('white')
dir_ch=np.zeros((4,12))
for ci,name in enumerate(HW_NAMES):
    dir_ch[:,ci]=ch_results[name]['per_class']
x3=np.arange(12); w=0.2
dcolors=['#4A72C4','#E8833A','#5CB85C','#9B59B6']
for di in range(4):
    ax3.bar(x3+di*w-w*1.5,dir_ch[di],w,color=dcolors[di],alpha=0.8,
            edgecolor='#333',lw=0.3,label=DIR_NAMES[di])
ax3.axhline(25,color='#888',ls='--',lw=1,label='Chance (25%)')
ax3.set_xticks(x3);ax3.set_xticklabels(HW_NAMES,fontsize=9)
ax3.set_ylabel('Per-Class Accuracy (%)',fontsize=11)
ax3.set_title('Per-Class Accuracy by Channel',fontsize=12,fontweight='bold')
ax3.legend(fontsize=9,ncol=5);ax3.grid(True,axis='y',ls=':',alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR,'73_perclass_by_channel.png'),dpi=200,bbox_inches='tight')
plt.close(fig3)
print('  Saved: 73_perclass_by_channel.png')

# ====== Summary ======
print(f'\n{"="*70}')
print(f'  SINGLE-CHANNEL DECODING RESULTS')
print(f'  {len(epochs_data)} trials, 4 ERP features/channel, 4-class LDA, 5-fold CV')
print(f'{"="*70}')
print(f'\n{"Channel":<6} {"Region":<10} {"4-class":<10} {"Axis":<10} {"BestClass":<12}')
print(f'  {"-"*52}')
for k in order_ch:
    region=''
    for rn,chs in REGIONS.items():
        if k in chs: region=rn; break
    r=ch_results[k]
    best_d=np.argmax(r['per_class'])
    print(f'  {k:<6} {region:<10} {r["acc4"]:.1%}     {r["ax_acc"]:.1%}     {DIR_NAMES[best_d]}={r["per_class"][best_d]:.0f}%')

best_ch=max(ch_results.keys(),key=lambda k:ch_results[k]['acc4'])
print(f'\n  Best channel: {best_ch} = {ch_results[best_ch]["acc4"]:.1%}')
print(f'  Worst channel: {min(ch_results.keys(),key=lambda k:ch_results[k]["acc4"])} = {ch_results[min(ch_results.keys(),key=lambda k:ch_results[k]["acc4"])]["acc4"]:.1%}')
print(f'  Range: {max(r["acc4"] for r in ch_results.values())-min(r["acc4"] for r in ch_results.values()):.1%}')
print(f'  Chance: 25%')

sp=os.path.join(OUT_DIR,'74_single_channel_summary.txt')
with open(sp,'w') as sf:
    sf.write(f'Single-Channel Decoding Summary\n{"="*50}\n')
    for k in order_ch:
        r=ch_results[k]
        sf.write(f'{k}: {r["acc4"]:.4f} (axis={r["ax_acc"]:.4f})\n')
print(f'Saved: {sp}')
print('\nFigures: 71-73')
print('Done!')
