"""Reconstruct histories and generate independent training, reference and short scoring branches."""
import argparse
from pathlib import Path

import numpy as np
import torch
from advantage.schema import Episode, file_hash
from restore import restore
from collect import save_snapshot
from simulator import Simulation
from study import NAMES, ROOT, base_dir, eid, read, save_episode, split, task_dir, write


def train_families(shard):
    return split()["train"][shard::4][:3]


def pool_ids():
    return sorted({i for shard in range(4) for i in train_families(shard)} | set(split()["test"]))


def stem(init, frame):
    return f"{eid(init)}_t{frame:03}"


def load_snap(path):
    with np.load(path,allow_pickle=False) as z:
        return dict(z)


def candidate_set(sim, obs, state, original, seed, library, is_train):
    torch.manual_seed(seed)
    alternate = sim.action(obs)
    mask = library["train"] == is_train
    ids = np.flatnonzero(mask)
    distances = np.linalg.norm((library["state"][ids]-state)/library["scale"],axis=1)
    closest = int(np.argmin(distances))
    row = ids[closest]
    demo = sim.command(torch.from_numpy(library["action"][row].copy())[None])
    # The released tinyflow policy is trained directly on the LeRobot action field.
    # env_post is the same map used after undoing policy normalization in Simulation.action.
    mild = original.copy()
    mild[:3] *= .5
    strong = original.copy()
    strong[:3] = np.clip(-original[:3],-1,1)
    strong[6] = -1 if original[6]>=0 else 1
    return dict(zip(NAMES,[v.tolist() for v in (original,alternate,demo,mild,strong)])), dict(
        episode=int(library["episode"][row]), frame=int(library["frame"][row]), distance=float(distances[closest]),
        raw_action=library["action"][row].tolist(), donor_split="train" if is_train else "test")


def anchors(sim, worker, workers, library):
    out = task_dir(sim.task)
    frames = read(out/"demo.json")["frames"]
    for i, init in enumerate(pool_ids()):
        if i%workers!=worker:
            continue
        ep = Episode.load(base_dir(sim.task)/"episodes"/(eid(init)+".npz"))
        usable = [f for f in frames if f<len(ep)]
        is_train = init in split()["train"]
        if is_train and not usable:
            usable = [0]
        dests = [out/"anchors"/(stem(init,f)+".json") for f in usable]
        if all(p.exists() for p in dests):
            continue
        raw,(obs,state) = sim.start(init,0)
        # Independently reconstruct policy queues at every requested frame in one pass.
        for t in range(len(ep)):
            np.testing.assert_allclose(state, ep.obs[t],atol=1e-6,rtol=0)
            if t in usable:
                save_snapshot(out/"snapshots"/(stem(init,t)+".npz"),sim.base,sim.policy,raw,t,ep.metadata["policy_seed"])
            cmd = sim.action(obs)
            np.testing.assert_allclose(cmd,ep.action[t],atol=1e-6,rtol=0)
            raw,(obs,state),success = sim.step(cmd)
            assert success == bool(ep.reward[t])
        np.testing.assert_allclose(state,ep.next_obs[-1],atol=1e-6,rtol=0)
        for f,path in zip(usable,dests):
            snap = out/"snapshots"/(stem(init,f)+".npz")
            z=load_snap(snap)
            obs,state=restore(sim,ep,f,z)
            replay=sim.rollout(obs,state,f,sim.horizon)
            np.testing.assert_allclose(replay[0][1:],ep.next_obs[f:],atol=1e-6,rtol=0)
            np.testing.assert_allclose(replay[1],ep.action[f:],atol=1e-6,rtol=0)
            np.testing.assert_array_equal(replay[2],ep.reward[f:])
            obs,state=restore(sim,ep,f,z)
            seed=7000000+sim.task*100000+init*400+f
            candidates,donor=candidate_set(sim,obs,state,ep.action[f],seed,library,is_train)
            write(path,dict(task=sim.task,init=init,frame=f,split="train" if is_train else "test",
                  episode=eid(init),state=state.tolist(),candidates=candidates,donor=donor,candidate_seed=seed,
                  snapshot_sha256=file_hash(snap),replayed=len(replay[1]),
                  max_state_error=float(np.max(np.abs(replay[0][1:]-ep.next_obs[f:]))),
                  max_action_error=float(np.max(np.abs(replay[1]-ep.action[f:])))))
        print("restored",sim.task,init,usable,flush=True)


class Anchor:
    def __init__(self, task, path):
        self.row = read(path)
        self.frame = self.row["frame"]
        self.init = self.row["init"]
        self.episode = Episode.load(base_dir(task)/"episodes"/(eid(self.init)+".npz"))
        snap = task_dir(task)/"snapshots"/(path.stem+".npz")
        assert file_hash(snap)==self.row["snapshot_sha256"]
        self.z=load_snap(snap)

    def run(self, sim, name, seed, cap):
        obs,state = restore(sim,self.episode,self.frame,self.z)
        forced = np.array(self.row["candidates"][name],np.float32) if name else None
        return sim.rollout(obs,state,self.frame,cap,forced,seed)


def augment(sim, shard):
    out=task_dir(sim.task)
    families=train_families(shard)
    groups=[[Anchor(sim.task,p) for p in sorted((out/"anchors").glob(eid(init)+"_*.json"))] for init in families]
    assert len(groups)==3 and all(groups)
    for arm in ("plain","branch"):
        dest=out/"augmentation"/arm/f"shard{shard}"
        spent=slot=0
        while spent<1500:
            a=groups[slot%3][(slot//3+shard)%len(groups[slot%3])]
            path=dest/f"slot{slot:03}.npz"
            seed=11000000+sim.task*10000+shard*1000+slot
            name=NAMES[(slot+shard)%5] if arm=="branch" else None
            if path.exists():
                ep=Episode.load(path)
            else:
                history=a.run(sim,name,seed,1500-spent)
                ep=sim.episode(history,a.init,f"{arm}_{shard}_{slot}",a.frame,
                    dict(init_id=a.init,frame=a.frame,arm=arm,candidate=name,continuation_seed=seed,slot=slot,shard=shard,split="train"))
                save_episode(path,ep)
            assert ep.metadata["candidate"]==name and ep.metadata["init_id"]==a.init and ep.metadata["continuation_seed"]==seed
            spent+=len(ep)
            slot+=1
        assert spent==1500
        print(sim.task,arm,shard,spent,slot,flush=True)


def evaluate(sim, worker, workers):
    out=task_dir(sim.task)
    paths=[p for p in sorted((out/"anchors").glob("*.json")) if read(p)["split"]=="test"]
    assert {read(p)["init"] for p in paths}==set(split()["test"])
    # Schedule a whole paired repeat together. Reference output contains only the observed
    # terminal event and duration. Independently seeded score files retain the V inputs.
    anchors_by_path={}
    for rep in range(10):
        is_ref=rep<8
        for i,p in enumerate(paths):
            if (i+rep)%workers!=worker:
                continue
            group="ref" if is_ref else "score"
            dest=out/"evaluation"/f"{p.stem}_{group}{rep:02}.json"
            if dest.exists():
                continue
            if p not in anchors_by_path:
                anchors_by_path[p]=Anchor(sim.task,p)
            a=anchors_by_path[p]
            seed=(20000000 if is_ref else 30000000)+sim.task*100000+a.init*2000+a.frame*10+rep
            values={}
            for name in NAMES:
                states,actions,rewards=a.run(sim,name,seed,sim.horizon if is_ref else 20)
                steps=len(rewards)
                success=bool(rewards[-1])
                terminal=success or a.frame+steps==sim.horizon
                if is_ref:
                    assert terminal
                values[name]=dict(steps=steps,success=success,terminal=terminal)
                if not is_ref:
                    values[name].update(states=states.tolist(),actions=actions.tolist(),rewards=rewards.tolist())
            write(dest,dict(task=sim.task,init=a.init,frame=a.frame,seed=seed,group=group,repeat=rep,rows=values))
            print(sim.task,p.stem,group,rep,flush=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for key in ("task","worker","workers"):
        p.add_argument(key,type=int)
    p.add_argument("mode",choices=("anchors","augment","branches"))
    a=p.parse_args()
    assert (ROOT/"protocol.json").exists()
    sim=Simulation(a.task)
    try:
        if a.mode=="anchors":
            path=task_dir(a.task)/"demo_library.npz"
            assert file_hash(path)==read(task_dir(a.task)/"demo.json")["library_sha256"]
            anchors(sim,a.worker,a.workers,load_snap(path))
        elif a.mode=="augment":
            for shard in range(a.worker,4,a.workers):
                augment(sim,shard)
        else:
            evaluate(sim,a.worker,a.workers)
    finally:
        sim.env.close()


if __name__=="__main__":
    main()
