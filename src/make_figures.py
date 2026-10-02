"""Training-curve and confusion-matrix figures for the report.  (Owner: Aanyaa Agarwwal)

Run (after training):  python src/make_figures.py
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","Times New Roman","DejaVu Serif"],
  "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e",
  "axes.labelcolor":"#0b0b0b","xtick.color":"#52514e","ytick.color":"#52514e","axes.grid":True,
  "grid.color":"#e6e5e1","grid.linewidth":0.6})
import os; os.makedirs("figures", exist_ok=True)
R={m:json.load(open(f"results/{m}_results.json")) for m in ["mlp","cnn"]}   # add "bilstm","transformer" once trained
col={"mlp":"#2a78d6","cnn":"#eb6834"}; name={"mlp":"MLP","cnn":"1D CNN"}
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5))
for m,r in R.items():
    h=r["history"]; e=[x["epoch"] for x in h]
    ax[0].plot(e,[x["train_acc"]*100 for x in h],"--",color=col[m],lw=1.5,marker="o",ms=3.5,label=f"{name[m]} train")
    ax[0].plot(e,[x["val_acc"]*100 for x in h],"-",color=col[m],lw=2,marker="o",ms=4,label=f"{name[m]} validation")
    ax[1].plot(e,[x["train_loss"] for x in h],"--",color=col[m],lw=1.5,marker="o",ms=3.5,label=f"{name[m]} train")
    ax[1].plot(e,[x["val_loss"] for x in h],"-",color=col[m],lw=2,marker="o",ms=4,label=f"{name[m]} validation")
    b=r["best_epoch"]; ax[1].plot([b],[h[b-1]["val_loss"]],marker="o",ms=8,mfc="none",mec=col[m],mew=1.2)
ax[0].set(xlabel="Epoch",ylabel="Accuracy (%)",title="(a) Accuracy"); ax[1].set(xlabel="Epoch",ylabel="Cross-entropy loss",title="(b) Loss (circle = selected epoch)")
for a in ax: a.set_xticks(range(1,7)); a.title.set_fontsize(9)
ax[0].legend(fontsize=7,frameon=False,loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig1_curves.png",dpi=220)

C=["World","Sports","Business","Sci/Tech"]
cmap=LinearSegmentedColormap.from_list("b",["#fcfcfb","#cde2fb","#6da7ec","#256abf","#104281"])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.7))
for a,(m,r) in zip(ax,R.items()):
    cm=np.array(r["confusion_matrix"]); a.imshow(cm,cmap=cmap,vmin=0,vmax=1900); a.grid(False)
    for i in range(4):
        for j in range(4):
            a.text(j,i,cm[i,j],ha="center",va="center",fontsize=8,color="white" if cm[i,j]>900 else "#0b0b0b")
    a.set_xticks(range(4),C,fontsize=7.5); a.set_yticks(range(4),C,fontsize=7.5)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(f"{'(a)' if m=='mlp' else '(b)'} {name[m]} — accuracy {r['test_acc']*100:.2f}%",fontsize=9)
    for s in a.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig2_cm.png",dpi=220)
