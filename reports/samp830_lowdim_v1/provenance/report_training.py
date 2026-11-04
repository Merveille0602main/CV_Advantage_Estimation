"""Audit and report the fixed-budget four-task run; never tune on held-out scores."""
import argparse, csv, hashlib, importlib.metadata as md, json, math, shutil, subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from advantage.cli import read_split
from advantage.schema import file_hash, split_families
from advantage.targets import discounted_returns
from advantage.value import ValuePredictor
from advantage.qvalue import QPredictor

EXPECTED = "6c00b283950e6957e42c259688aa10ad94260495"
TASKS, SEEDS, NS = [0,13,22,31], [0,1,2], [1,5,10,20]
REPO = Path("/home/figo/CV_Advantage_Estimation")
RAW = Path("/mnt/d/CV/data/raw/samp830")
def save_json(p,value):
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def save_csv(p,rows):
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def windows(p):
    s=str(p)
    return "D:"+s[len("/mnt/d"):].replace("/",chr(92)) if s.startswith("/mnt/d/") else s
def rmse(pred,target):
    return float(np.sqrt(np.mean((np.asarray(pred)-target)**2)))
def main(run):
    if (run/"STATUS.txt").read_text().strip() != "training_and_scoring_complete":
        raise RuntimeError("training/scoring is not complete; do not report a partial run")
    assert subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()==EXPECTED
    subprocess.run(["git","-C",str(REPO),"diff","--exit-code"],check=True,capture_output=True)
    tests=(run/"logs/tests.log").read_text()
    assert "\nOK\n" in tests and "Ran 30 tests" in tests
    records={r["episode_index"]:r for r in map(json.loads,(RAW/"episode_manifest.jsonl").read_text().splitlines())}
    metrics,losses,score_stats,ckpts,audits,raw_checks=[],[],[],[],[],[]
    baselines={}
    raw_bytes=0
    score_row_total=0
    targets_by_task={}
    for task in TASKS:
        print("AUDIT task",task,flush=True)
        ds=run/"datasets"/f"task_{task}"
        manifest=json.loads((ds/"manifest.json").read_text())
        parts={s:read_split(ds,s) for s in ("train","validation","test")}
        eps=[ep for s in parts.values() for ep in s]
        assert len(eps)==25 and len({ep.episode_id for ep in eps})==25
        assert {s:len(v) for s,v in parts.items()}=={"train":17,"validation":3,"test":5}
        ordered=sorted(eps,key=lambda e:int(e.episode_id.rsplit(":",1)[-1]))
        assert split_families(ordered,12)==manifest["split"]
        for ep in ordered:
            eid=int(ep.episode_id.rsplit(":",1)[-1]); r=records[eid]
            assert r["task_index"]==task and len(ep)==r["length"]
            assert ep.obs.shape==(len(ep),8) and ep.action.shape==(len(ep),7)
            assert np.isfinite(ep.obs).all() and np.isfinite(ep.action).all()
            assert np.max(np.abs(ep.action))<=1.00001
            assert np.allclose(ep.next_obs[:-1],ep.obs[1:])
            assert not ep.reward[:-1].any() and ep.reward[-1]==float(r["success"])
            assert ep.terminated[-1] and ep.terminated.sum()==1 and not ep.truncated.any()
            assert np.array_equal(ep.remaining_steps,500-np.arange(len(ep)))
            assert r["success"] or len(ep)==500
            raw=RAW/"data/chunk-000"/f"episode_{eid:06d}.parquet"
            tab=pq.read_table(raw,columns=["state","actions","frame_index","episode_index","task_index","timestamp"])
            assert tab.num_rows==len(ep)
            assert np.allclose(np.asarray(tab["state"].to_pylist()),ep.obs)
            assert np.allclose(np.asarray(tab["actions"].to_pylist()),ep.action)
            assert np.array_equal(tab["frame_index"].to_numpy(),np.arange(len(ep)))
            assert np.all(tab["episode_index"].to_numpy()==eid)
            assert np.all(tab["task_index"].to_numpy()==task)
            ts=tab["timestamp"].to_numpy()
            assert np.isfinite(ts).all() and np.all(np.diff(ts)>0)
            raw_checks.append(dict(episode_id=eid,task_index=task,bytes=raw.stat().st_size,sha256=file_hash(raw)))
            raw_bytes+=raw.stat().st_size
        split_counts={s:dict(episodes=len(v),success=sum(records[int(e.episode_id.rsplit(":",1)[-1])]["success"] for e in v),transitions=sum(map(len,v))) for s,v in parts.items()}
        for c in split_counts.values(): c["failure"]=c["episodes"]-c["success"]
        audits.append(dict(task_index=task,task=records[int(ordered[0].episode_id.rsplit(":",1)[-1])]["task"],split=split_counts,manifest_sha256=file_hash(ds/"manifest.json"),family_identity_verified=False))
        train_returns=np.concatenate([discounted_returns(e,.99) for e in parts["train"]])
        train_constant=float(train_returns.mean())
        targets_by_task[task]=train_constant
        for split in ("validation","test"):
            baselines[task,split]=float(np.mean([rmse(train_constant,discounted_returns(e,.99)) for e in parts[split]]))
        for seed in SEEDS:
            prefix=f"task_{task}_seed{seed}"
            vp=ValuePredictor(run/"models"/f"{prefix}_v.pt")
            qp=QPredictor(run/"models"/f"{prefix}_q.pt")
            vhash=file_hash(run/"models"/f"{prefix}_v.pt")
            qhash=file_hash(run/"models"/f"{prefix}_q.pt")
            for label,predictor,h in (("V",vp,vhash),("Q",qp,qhash)):
                c=predictor.checkpoint
                assert c["gamma"]==.99 and c["seed"]==seed and c["updates"]==1000 and c["horizon"]==500
                assert len(c["losses"])==1000 and np.isfinite(c["losses"]).all()
                assert c["input_dim"]==(9 if label=="V" else 16)
                assert c["task_id"]==ordered[0].task_id and c["policy_id"]==ordered[0].policy_id
                ckpts.append(dict(task=task,seed=seed,model=label,path=f"models/{prefix}_{label.lower()}.pt",sha256=h,updates=1000,first_100_loss_mean=float(np.mean(c["losses"][:100])),last_100_loss_mean=float(np.mean(c["losses"][-100:]))))
                losses.extend(dict(task=task,seed=seed,model=label,update=i+1,mse=float(v)) for i,v in enumerate(c["losses"]))
            for split in ("validation","test"):
                selected=parts[split]
                lens={e.episode_id:len(e) for e in selected}
                for label,predictor in (("V",vp),("Q",qp)):
                    for ep in selected:
                        pred= predictor.predict(ep)
                        target=discounted_returns(ep,.99)
                        assert np.isfinite(pred).all()
                        metrics.append(dict(task=task,seed=seed,split=split,model=label,episode_id=ep.episode_id,transitions=len(ep),success=records[int(ep.episode_id.rsplit(":",1)[-1])]["success"],rmse=rmse(pred,target),mae=float(np.mean(np.abs(pred-target))),baseline_rmse=rmse(train_constant,target)))
                for kind in ("td","qv"):
                    path=run/"scores"/f"{prefix}_{split}_{kind}.csv"
                    with path.open() as f: rows=list(csv.DictReader(f))
                    methods=[f"n{n}" for n in NS] if kind=="td" else ["mc_q_minus_shared_v"]
                    counts=Counter()
                    keys=set()
                    for row in rows:
                        e,t,m=row["episode_id"],int(row["t"]),row["method"]
                        assert e in lens and 0<=t<lens[e] and m in methods
                        assert (e,t,m) not in keys; keys.add((e,t,m)); counts[e,m]+=1
                        assert math.isfinite(float(row["score"]))
                        if kind=="td": assert row["checkpoint_sha256"]==vhash
                        else: assert row["q_sha256"]==qhash and row["v_sha256"]==vhash
                    assert all(counts[e,m]==n for e,n in lens.items() for m in methods)
                    assert len(rows)==sum(lens.values())*len(methods)
                    score_row_total+=len(rows)
                    for method in methods:
                        a=np.array([float(r["score"]) for r in rows if r["method"]==method])
                        score_stats.append(dict(task=task,seed=seed,split=split,method=method,rows=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max())))
    assert len(ckpts)==24 and len(raw_checks)==100
    summary=[]
    for task in TASKS:
        for split in ("validation","test"):
            for model in ("V","Q"):
                vals=[float(np.mean([r["rmse"] for r in metrics if r["task"]==task and r["split"]==split and r["model"]==model and r["seed"]==seed])) for seed in SEEDS]
                summary.append(dict(task=task,split=split,model=model,episode_macro_rmse_mean=float(np.mean(vals)),seed_std=float(np.std(vals,ddof=1)),train_mean_baseline_rmse=baselines[task,split]))
    figdir=run/"figures";figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
    for j,task in enumerate(TASKS):
        for i,model in enumerate(("V","Q")):
            ax=axes[i,j]
            for seed in SEEDS:
                a=np.array([r["mse"] for r in losses if r["task"]==task and r["model"]==model and r["seed"]==seed])
                ax.plot(np.arange(25,1001),np.convolve(a,np.ones(25)/25,mode="valid"),label=f"seed {seed}")
            ax.set(title=f"Task {task}: {model}",yscale="log",xlabel="Optimizer updates",ylabel="Train MSE (25-update mean)")
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/"training_losses.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,split in zip(axes,("validation","test")):
        x=np.arange(4)
        for model,offset,color in (("V",-.17,"#3478b8"),("Q",.17,"#e38d3c")):
            rs=[next(r for r in summary if r["task"]==t and r["split"]==split and r["model"]==model) for t in TASKS]
            ax.bar(x+offset,[r["episode_macro_rmse_mean"] for r in rs],width=.32,yerr=[r["seed_std"] for r in rs],capsize=4,label=model,color=color)
        ax.scatter(x,[baselines[t,split] for t in TASKS],marker="x",color="black",label="Train-mean constant")
        ax.set(xticks=x,xticklabels=[str(t) for t in TASKS],xlabel="samp830 task index",ylabel="Mean per-episode RMSE",title=split.capitalize()+" observed-return prediction")
        ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(figdir/"return_prediction_rmse.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(13,12))
    for ax,task in zip(axes,TASKS):
        ds=run/"datasets"/f"task_{task}"
        ep=read_split(ds,"test")[0]
        for kind in ("td","qv"):
            with (run/"scores"/f"task_{task}_seed0_test_{kind}.csv").open() as f: rows=list(csv.DictReader(f))
            methods=[f"n{n}" for n in NS] if kind=="td" else ["mc_q_minus_shared_v"]
            for method in methods:
                rs=sorted((r for r in rows if r["episode_id"]==ep.episode_id and r["method"]==method),key=lambda r:int(r["t"]))
                ax.plot([int(r["t"]) for r in rs],[float(r["score"]) for r in rs],label=method,alpha=.85,linewidth=1)
        ax.axhline(0,color="black",linewidth=.6);ax.set(title=f"Task {task}; first test episode {ep.episode_id}; seed 0",xlabel="Recorded transition",ylabel="Estimated score")
        ax.legend(fontsize=8,ncol=5);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/"example_scores.png",dpi=160);plt.close(fig)
    save_csv(run/"trajectory_metrics.csv",metrics);save_csv(run/"metric_summary.csv",summary)
    save_csv(run/"loss_history.csv",losses);save_csv(run/"score_summary.csv",score_stats)
    save_json(run/"raw_checksums.json",raw_checks)
    old=Path("/home/figo/venvs/libero-cv/lib/python3.12/site-packages")
    from packaging.utils import canonicalize_name
    after={canonicalize_name(d.metadata["Name"]):d.version for d in md.distributions(path=[str(old)])}
    before=json.loads((run/"provenance/original-environment-before.json").read_text())
    assert before==after,"original environment package versions changed"
    save_json(run/"provenance/original-environment-after.json",after)
    subprocess.run(["git","-C",str(REPO),"archive","--format=tar.gz","--output="+str(run/"provenance/source-code.tar.gz"),EXPECTED,"src","scripts","tests","configs","docs","README.md","pyproject.toml",".gitignore"],check=True)
    shutil.copy2(__file__,run/"provenance/report_training.py")
    config=dict(repository_commit=EXPECTED,source_root=str(RAW),source_revision_declared="71a88ad0eaa5f5dc5237e6ede56e7cc3d4a3e9e8",tasks=TASKS,split_seed=12,train_seeds=SEEDS,gamma=.99,updates=1000,optimizer="Adam",learning_rate=3e-4,batch_size=128,hidden_layers=[256,256],state_dimensions=8,action_dimensions=7,lookaheads=NS,device="cuda",torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),reward="1 on successful final transition; otherwise 0",horizon=500,checkpoint_selection="final fixed-budget checkpoint; no test-driven tuning")
    save_json(run/"config.json",config)
    report=dict(status="complete",scope="lowdim_observed_return_diagnostics_only",tasks=audits,models=ckpts,metric_summary=summary,training_models=24,smoke_models=2,selected_episodes=100,selected_transitions=sum(c["transitions"] for a in audits for c in a["split"].values()),raw_bytes_hashed=raw_bytes,score_rows=score_row_total,tests=30,original_environment_preserved=True,raw_vs_canonical_exact_audited=True,independent_advantage_labels_available=False)
    save_json(run/"report.json",report)
    lines=["# samp830 四任务低维训练结果","",f"运行目录：{windows(run)}","", "**24 个正式 V／Q 模型全部完成训练，验证集与测试集的五种评分已生成，30 项仓库测试通过。**","",
    "本报告的误差衡量模型对离线轨迹实际折扣回报样本的预测，不是对真实 V／Q 或动作质量的准确率。","",
    "## 数据与配置","", "四个任务各 25 条轨迹；每任务训练／验证／测试为 17／3／5 条。划分种子 12，训练种子 0、1、2。γ=0.99，每模型 1000 次更新，Adam 3e-4，batch=128，两层 256 单元 MLP。训练使用 CUDA；评分沿用仓库默认 CPU 实现。","",
    "| 任务 | 任务内容 | 训练 成功/失败 | 验证 成功/失败 | 测试 成功/失败 |","|---|---|---|---|---|"]
    for a in audits:
        c=a["split"];lines.append(f'| {a["task_index"]} | {a["task"]} | {c["train"]["success"]}/{c["train"]["failure"]} | {c["validation"]["success"]}/{c["validation"]["failure"]} | {c["test"]["success"]}/{c["test"]["failure"]} |')
    lines+=["","任务编号采用 samp830 的全局任务编号；不能直接当作原生 LIBERO 套件内编号。任务 13、22、31 的验证集均为成功轨迹，验证误差代表性有限；未据此调整模型。","",
    "## 测试集实际回报预测误差","", "先计算每条轨迹的 RMSE，再按轨迹等权平均，最后汇总三个训练种子的均值和样本标准差。常数基线使用训练集所有 transition 的回报均值；不读取验证或测试数据拟合。","",
    "| 任务 | V RMSE 均值 ± 标准差 | Q RMSE 均值 ± 标准差 | 训练均值常数基线 |","|---|---|---|---|"]
    for task in TASKS:
        v=next(r for r in summary if r["task"]==task and r["split"]=="test" and r["model"]=="V")
        q=next(r for r in summary if r["task"]==task and r["split"]=="test" and r["model"]=="Q")
        lines.append(f'| {task} | {v["episode_macro_rmse_mean"]:.4f} ± {v["seed_std"]:.4f} | {q["episode_macro_rmse_mean"]:.4f} ± {q["seed_std"]:.4f} | {baselines[task,"test"]:.4f} |')
    lines+=["","训练损失下降不能替代留出集检验；若某模型误差大于常数基线，说明本设置的回报预测泛化尚未获得支持。Q 和 V 输入不同，Q 的误差较低也不能证明 Q−V 的 Advantage 估计更准确。","",
    "## 图表","","![训练损失](figures/training_losses.png)","","![留出回报预测误差](figures/return_prediction_rmse.png)","","![逐步评分样例](figures/example_scores.png)","",
    "样例固定取各任务 manifest 中第一条测试轨迹及训练种子 0；未根据曲线表现挑选。各评分没有独立动作质量标签，曲线分歧只能作为后续分析线索。","",
    "## 验收","",f"- 已核验 100 条原始 Parquet 的 SHA-256、维度、数值、时间索引、任务身份，并逐值核对转换后的状态和动作。共 {report['selected_transitions']:,} 个 transition。",
    f"- 已验证 24 个正式 checkpoint 的 1000 项损失有限、任务与策略匹配、可重新加载。生成并核对 {score_row_total:,} 行评分，检查重复、缺失、非有限值和 checkpoint 哈希。",
    "- 每个 N-step 方法使用同一个 V checkpoint；Q 与 V 的任务、策略、gamma 和 horizon 一致。",
    "- 原 LIBERO 环境的包版本前后核对一致；仓库训练源码保持固定提交且无改动；未上传 GitHub。","",
    "## 局限与下一步","",
    "- 只有机器人末端与夹爪状态，缺少物体、目标及视觉信息；模型输入未必足以判断任务进展。",
    "- 每任务训练轨迹只有 17 条，验证 3 条、测试 5 条；三个训练种子的波动不覆盖数据划分不确定性。",
    "- 轨迹家族身份尚未独立核实；当前按 episode 分组，不能保证不存在共享初始状态带来的相关性。",
    "- 成功末步奖励与失败时限终止按数据清单重建，未进行独立仿真回放核验。",
    "- 尚无独立逐步动作质量标签，不报告动作质量 AUROC、排序准确率或某种 Advantage 更准确的结论。",
    "- 后续优先检查误差大的轨迹、评估视觉特征的必要性，并制定独立评价标签协议；如据本次测试结果改方法，需要新的留出评价数据。","",
    "## 文件与复现","",
    "- config.json：固定训练配置；report.json：完整机器可读验收结果。",
    "- metric_summary.csv / trajectory_metrics.csv：汇总与逐轨迹回报预测误差。",
    "- models/：24 个正式模型；smoke/：独立短运行模型，不纳入正式结果。",
    "- scores/：每任务、种子和留出集合的 TD/N-step 与 Q−V 分数。",
    "- provenance/：固定源码压缩包、运行脚本、依赖版本和原环境保护记录。",
    "- 运行入口：D:\\CV\\training_tools\\samp830_lowdim_v1\\run_training.ps1。重新运行生成新目录，不覆盖本次结果。",""]
    (run/"结果说明.md").write_text("\n".join(lines),encoding="utf-8")
    (run/"STATUS.txt").write_text("complete\n",encoding="utf-8")
    print(json.dumps({"status":"complete","run":str(run),"models":len(ckpts),"episodes":100,"score_rows":score_row_total,"test_metrics":[r for r in summary if r["split"]=="test"]},indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path)
    a=p.parse_args()
    run=a.run or Path("/mnt/d/CV/training_runs/samp830_lowdim_v1/LATEST.txt").read_text().strip()
    main(Path(run))
