"""Render the two presentation figures directly from the validated study summary."""
import argparse
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def render(root,out):
    font=Path(os.environ.get("STUDY_CJK_FONT","C:/Windows/Fonts/msyh.ttc"))
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=font).get_name()
    else:
        plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","Microsoft YaHei","DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":11})
    out.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((root/"summary.json").read_text(encoding="utf8"))["tasks"]
    names=["奶酪入碗","开抽屉","牛奶入篮"]
    arms=["base","plain","branch","zero_gap"]
    colors=["#516477","#7394a7","#00796c","#918477"]
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,record,name in zip(axes,tasks,names):
        for y,(arm,color) in enumerate(zip(arms,colors)):
            v=record["errors"][arm]["gap_rmse"]
            ax.scatter(v,y,c=color,s=56,zorder=3)
            ax.annotate(f"{v:.3f}",(v,y),xytext=(7,0),textcoords="offset points",va="center",fontsize=10)
        ax.set(title=name,xscale="log",xlim=(.0025,65),ylim=(3.5,-.5),xticks=[.01,.1,1,10])
        ax.set_xticklabels(["0.01","0.1","1","10"])
        ax.grid(axis="x",alpha=.22)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    axes[0].set_yticks(range(4),["原始 Q","补普通续跑","补同状态分支","恒预测差值为零"])
    fig.suptitle("同预算下，分支训练降低了三个任务的动作价值差误差",fontsize=15)
    fig.supxlabel("动作价值差 RMSE，越小越好（对数刻度）",fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.94])
    fig.savefig(out/"q_gap_rmse.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    keys=["base_QV","plain_QV","branch_QV","N1","N5","N10","N20"]
    vals=np.array([[s["ranking"][k]["accuracy"] for k in keys] for s in tasks],dtype=float)
    fig,ax=plt.subplots(figsize=(10,3.45))
    ax.imshow(vals,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    for i,j in np.ndindex(vals.shape):
        percent=(Decimal(str(vals[i,j]))*100).quantize(Decimal(".1"),rounding=ROUND_HALF_UP)
        ax.text(j,i,f"{percent}%",ha="center",va="center",color="white" if vals[i,j]>.62 else "#19324b",fontsize=11)
    ax.set_xticks(range(7),["原始 Q−V","普通 Q−V","分支 Q−V","单步 TD","5 步","10 步","20 步"])
    ax.set_yticks(range(3),[f"{name}（{s['counts']['eligible_pairs']}/300 对）" for name,s in zip(names,tasks)])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("可区分动作对上的排序准确率",fontsize=15,pad=13)
    fig.text(.5,.025,"按快照和初始状态等权。多步方法读取独立评分续跑的后续步骤。",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.07,1,1])
    fig.savefig(out/"ranking.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    render(a.run,a.out)
