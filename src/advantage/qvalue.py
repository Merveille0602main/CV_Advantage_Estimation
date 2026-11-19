"""Monte Carlo Q: regress complete reference-policy returns on state/time/action."""
import numpy as np
from .networks import scalar_mlp
from .targets import discounted_returns


def fit_q(episodes, gamma=0.99, updates=1000, batch_size=128, seed=0, lr=3e-4, device="cpu"):
    import torch
    if not episodes or updates < 1 or batch_size < 1:
        raise ValueError("nonempty training episodes and positive budgets required")
    if len({e.task_id for e in episodes}) != 1 or len({e.policy_id for e in episodes}) != 1:
        raise ValueError("MC-Q requires a single task and fixed reference policy")
    for ep in episodes:
        ep.validate()
    # Only observed TRAIN rewards are supervision; no oracle or quality labels.
    returns = np.concatenate([discounted_returns(ep, gamma) for ep in episodes])
    horizon = max(int(ep.remaining_steps[0]) for ep in episodes)
    x = np.concatenate([np.column_stack([ep.obs, ep.remaining_steps/horizon, ep.action]) for ep in episodes])
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    tx = torch.as_tensor((x-mean)/scale,dtype=torch.float32,device=device)
    ty = torch.as_tensor(returns,dtype=torch.float32,device=device)
    model = scalar_mlp(x.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    losses=[]
    for step in range(updates):
        ix=torch.as_tensor(rng.integers(len(x),size=batch_size),device=device)
        loss=(model(tx[ix]).squeeze(-1)-ty[ix]).square().mean()
        optim.zero_grad(); loss.backward(); optim.step()
        losses.append(float(loss.detach().cpu()))
    return {"kind":"mc_q","state_dict":model.cpu().state_dict(),"input_dim":x.shape[1],
        "state_dim":episodes[0].obs.shape[1],"action_dim":episodes[0].action.shape[1],
        "mean":mean.tolist(),"scale":scale.tolist(),"horizon":horizon,
        "gamma":gamma,"task_id":episodes[0].task_id,"policy_id":episodes[0].policy_id,
        "seed":seed,"updates":updates,"losses":losses}


class QPredictor:
    def __init__(self,checkpoint,device="cpu"):
        import torch
        self.torch,self.device=torch,device
        self.checkpoint=torch.load(checkpoint,map_location="cpu",weights_only=True)
        c=self.checkpoint
        if c.get("kind") != "mc_q":
            raise ValueError("not an MC-Q checkpoint")
        self.model = scalar_mlp(c["input_dim"]).to(device)
        self.model.load_state_dict(c["state_dict"]);self.model.eval()

    def predict(self,episode,actions=None):
        c=self.checkpoint
        if episode.task_id != c["task_id"] or episode.policy_id != c["policy_id"]:
            raise ValueError("Q task/reference policy mismatch")
        actions=episode.action if actions is None else np.asarray(actions,dtype=float)
        if actions.shape != (len(episode),c["action_dim"]) or not np.isfinite(actions).all():
            raise ValueError("candidate action shape or values invalid")
        if episode.obs.shape[1] != c["state_dim"]:
            raise ValueError("Q observation feature dimension mismatch")
        x=np.column_stack([episode.obs,episode.remaining_steps/c["horizon"],actions])
        x=(x-np.asarray(c["mean"]))/np.asarray(c["scale"])
        with self.torch.no_grad():
            return self.model(self.torch.as_tensor(x,dtype=self.torch.float32,device=self.device)).squeeze(-1).cpu().numpy()


def qv_scores(q_predictor,v_predictor,episode,actions=None):
    """Learned Q minus frozen V: a comparison method, NEVER a ground-truth label."""
    for key in ("gamma","task_id","policy_id","horizon"):
        if q_predictor.checkpoint[key] != v_predictor.checkpoint[key]:
            raise ValueError(f"Q/V protocol mismatch: {key}")
    return q_predictor.predict(episode,actions)-v_predictor.predict(episode)
