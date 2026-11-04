"""Plot saved real-sample loss histories and fixed-V TD scores; no retraining."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from advantage.schema import Episode,file_hash
from advantage.targets import discounted_returns
from advantage.value import ValuePredictor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="artifacts/real_samples_smoke_v1")
    parser.add_argument("--output",default="reports/real_samples_smoke_v1/figures")
    args=parser.parse_args()
    source=Path(args.input).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/"report.json").read_text())
    with (source/"scores.csv").open() as f:score_rows=list(csv.DictReader(f))
    torch.set_num_threads(1)
    font=Path("/mnt/c/Windows/Fonts/msyh.ttc")
    chinese=font.exists()
    if chinese:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
        "axes.unicode_minus":False,"font.size":10,"axes.titlesize":11,"figure.facecolor":"white",
        "axes.labelcolor":"#334155","text.color":"#0f172a","axes.edgecolor":"#cbd5e1"})
    task_names={9:"放书入收纳盒",19:"打开柜子中层抽屉",26:"将沙拉酱放入篮子",33:"将黑碗放到盘子"}
    task_rows=[];loss_rows=[];stats=[]
    fig_loss,loss_axes=plt.subplots(2,2,figsize=(12,7.6))
    fig_td,td_axes=plt.subplots(2,2,figsize=(12,7.6))
    fig_fit,fit_axes=plt.subplots(2,2,figsize=(12,7.6))
    colors={"td_n1":"#2563eb","td_n5":"#e69f00","td_n10":"#009e73","td_n20":"#cc3377"}
    for task,ax_loss,ax_td,ax_fit in zip(report["tasks"],loss_axes.flat,td_axes.flat,fit_axes.flat):
        tid=task["task_index"];directory=source/f"task_{tid}"
        checkpoint_path=directory/"shared_v.pt"
        checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        if file_hash(checkpoint_path)!=task["v_checkpoint_sha256"]:raise ValueError("V checkpoint differs from score report")
        losses=np.asarray(checkpoint["losses"],float)
        if not np.isfinite(losses).all():raise ValueError("nonfinite loss history")
        manifest=json.loads((directory/"canonical/manifest.json").read_text())
        if len(manifest["episodes"])!=1:raise ValueError("this figure expects one episode per task")
        record=manifest["episodes"][0]
        ep=Episode.load(directory/"canonical"/record["path"])
        unique_raw=len(np.unique(ep.obs,axis=0))
        unique_augmented=len(np.unique(np.column_stack([ep.obs,ep.remaining_steps]),axis=0))
        success=bool(ep.metadata["success"])
        title=f"Task {tid} | "+(task_names[tid] if chinese else ep.task_id)+f" | {len(ep)} steps | "+("成功" if success and chinese else "失败" if chinese else "success" if success else "failure")
        ax_loss.plot(np.arange(1,len(losses)+1),losses,color="#93c5fd",lw=1,alpha=.65,label="原始 batch MSE" if chinese else "Raw batch MSE")
        window=min(10,len(losses))
        rolling=np.convolve(losses,np.ones(window)/window,mode="valid")
        ax_loss.plot(np.arange(window,len(losses)+1),rolling,color="#1d4ed8",lw=2,label="10步滑动平均" if chinese else "10-update moving average")
        if np.all(losses>0):ax_loss.set_yscale("log")
        ax_loss.set(title=title,xlabel="优化更新次数" if chinese else "Optimizer update",ylabel="V 训练 MSE（对数轴）" if chinese else "V training MSE (log scale)")
        ax_loss.grid(True,alpha=.22,which="major")
        ax_loss.legend(fontsize=8,frameon=False)
        for update,loss in enumerate(losses,1):loss_rows.append(dict(task_index=tid,update=update,train_batch_mse=float(loss)))
        for method,color in colors.items():
            rows=[r for r in score_rows if r["task_id"]==ep.task_id and r["method"]==method]
            rows.sort(key=lambda r:int(r["t"]))
            if len(rows)!=len(ep) or [int(r["t"]) for r in rows]!=list(range(len(ep))):raise ValueError("unaligned TD score rows")
            if any(r["v_checkpoint_sha256"]!=task["v_checkpoint_sha256"] for r in rows):raise ValueError("methods do not share V")
            values=np.array([float(r["score"]) for r in rows])
            ax_td.plot(np.arange(len(ep)),values,lw=1.15,color=color,alpha=.9,label=method.replace("td_n","TD n="))
            stats.append(dict(task_index=tid,method=method,states=len(values),mean=float(values.mean()),
                std=float(values.std()),minimum=float(values.min()),maximum=float(values.max()),
                positive_fraction=float((values>0).mean())))
        ax_td.axhline(0,color="#64748b",lw=.8,ls="--")
        ax_td.set(title=title,xlabel="轨迹时间步" if chinese else "Trajectory timestep",ylabel="Advantage 分数" if chinese else "Advantage score")
        ax_td.grid(True,alpha=.2);ax_td.legend(fontsize=8,frameon=False,ncol=2)
        actual=discounted_returns(ep,checkpoint["gamma"])
        prediction=ValuePredictor(checkpoint_path).predict(ep)
        ax_fit.plot(actual,color="#334155",lw=2,label="该轨迹 MC 回报" if chinese else "Observed MC return")
        ax_fit.plot(prediction,color="#2563eb",lw=1.2,label="拟合的 V" if chinese else "Fitted V")
        ax_fit.set(title=title,xlabel="轨迹时间步" if chinese else "Trajectory timestep",ylabel="回报 / V" if chinese else "Return / V")
        ax_fit.grid(True,alpha=.2);ax_fit.legend(fontsize=8,frameon=False)
        task_rows.append(dict(task_index=tid,task_id=ep.task_id,success=success,episodes=1,
            timestep_states=len(ep),unique_raw_state8_exact=unique_raw,unique_with_remaining_time=unique_augmented,
            raw_state_dim=ep.obs.shape[1],v_input_dim=checkpoint["input_dim"],optimizer_updates=len(losses),
            batch_size=128,v_loss_first20_mean=float(losses[:20].mean()),v_loss_last20_mean=float(losses[-20:].mean())))
    if chinese:
        titles=["真实样本：V 的训练 loss", "真实样本：固定同一个 V 的 TD / 多步 TD 比较", "补充：V 对同一条轨迹回报的拟合"]
        notes=["每任务一条轨迹、一个 V；这是同轨迹拟合，无独立验证集。浅色为逐次 loss，深色为滑动平均。",
            "每个任务内四种方法共用同一份 V 权重；各子图纵轴独立。分数差异不等于准确率差异。",
            "MC 回报是这一次实际轨迹的回报，不是状态期望价值真值；失败轨迹的目标回报全部为零。"]
    else:
        titles=["Real samples: V training loss","Real samples: TD / N-step with a fixed shared V","Supplement: V fit to the same observed trajectory"]
        notes=["One episode and one V per task; in-sample fit only, no validation set.",
            "Same V checkpoint for all spans within a task. Independent y scales. Scores are not accuracy.",
            "Observed returns are not true expected values; failed episodes have all-zero targets."]
    for fig,title,note,name in zip((fig_loss,fig_td,fig_fit),titles,notes,("v_loss.png","td_nstep.png","v_fit.png")):
        fig.suptitle(title,fontsize=16,fontweight="bold",y=.99)
        fig.text(.5,.015,note,ha="center",fontsize=9,color="#475569")
        fig.tight_layout(rect=[0,.045,1,.955]);fig.savefig(output/name,dpi=170,bbox_inches="tight");plt.close(fig)
    for name,rows in (("v_loss_history.csv",loss_rows),("td_summary.csv",stats)):
        with (output/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(scope="same_episode_smoke_only",total_timestep_states=sum(r["timestep_states"] for r in task_rows),
        total_episodes=len(task_rows),tasks=task_rows,notes=["Raw state uniqueness uses exact float equality, not physical or statistical independence.",
        "Terminal next observations are masked placeholders and are not counted as observed states.",
        "Loss history comes from existing saved checkpoints; no training was repeated."])
    (output/"state_counts.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
