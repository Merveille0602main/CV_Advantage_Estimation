"""Small PyTorch V learner for canonical low-dimensional data (optional torch).

Value targets are selectable so that the effect of the learning target can be
separated from the effect of the advantage construction:

  mc       complete discounted returns of the reference policy
  nstep    n-step bootstrapped targets using the current V
  lambda   TD(lambda) forward returns using the current V
  teacher  a supplied per-step target (e.g. the exact V), a diagnostic that
           removes bootstrapping error and leaves only approximation error
"""
import numpy as np
from .targets import n_step_targets, discounted_returns, lambda_targets

METHODS = ("mc", "nstep", "lambda", "teacher")
DEFAULT_HIDDEN = (256, 256)


def _architecture(input_dim, hidden):
    import torch
    layers, width = [], int(input_dim)
    for units in hidden:
        units = int(units)
        if units < 1:
            raise ValueError("hidden units must be positive")
        layers += [torch.nn.Linear(width, units), torch.nn.ReLU()]
        width = units
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)


def _teacher_targets(episodes, target_source):
    parts = []
    for episode in episodes:
        values = np.asarray(target_source(episode), dtype=np.float64)
        if values.shape != (len(episode),) or not np.isfinite(values).all():
            raise ValueError("target_source must return one finite value per transition")
        parts.append(values)
    return np.concatenate(parts)


def fit_value(episodes, n=1, gamma=0.99, updates=1000, batch_size=128,
              seed=0, lr=3e-4, target_refresh=20, device="cpu", method="mc",
              lambda_=0.9, hidden=DEFAULT_HIDDEN, target_source=None):
    import torch
    if not episodes or updates < 1 or batch_size < 1 or target_refresh < 1:
        raise ValueError("nonempty episodes and positive training budgets required")
    if len({e.policy_id for e in episodes}) != 1 or len({e.task_id for e in episodes}) != 1:
        raise ValueError("first version trains a single task and fixed reference policy")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "teacher" and target_source is None:
        raise ValueError("teacher targets require target_source")
    if method == "lambda" and not (np.isfinite(lambda_) and 0 <= lambda_ <= 1):
        raise ValueError("lambda_ must be finite and between zero and one")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    horizon = max(int(e.remaining_steps[0]) for e in episodes)
    def features(ep, nxt=False):
        obs = ep.next_obs if nxt else ep.obs
        remaining = ep.remaining_steps - int(nxt)
        return np.column_stack([obs, remaining/horizon])
    x = np.concatenate([features(e) for e in episodes])
    # All preprocessing is fitted on the supplied TRAIN episodes only.
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    def tensor(a):
        return torch.as_tensor((a-mean)/scale, dtype=torch.float32, device=device)
    tx = tensor(x)
    model = _architecture(x.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Fixed targets are built once; bootstrapped targets are refreshed with
    # the current V. The oracle is never read here unless the caller passes a
    # target_source explicitly (teacher diagnostic).
    fixed_targets = None
    if method == "mc":
        fixed_targets = np.concatenate([discounted_returns(e, gamma) for e in episodes])
    elif method == "teacher":
        fixed_targets = _teacher_targets(episodes, target_source)
    ty = None
    losses = []
    for step in range(updates):
        if fixed_targets is not None:
            if ty is None:
                ty = torch.as_tensor(fixed_targets, dtype=torch.float32, device=device)
        elif step == 0 or step % target_refresh == 0:
            with torch.no_grad():
                parts = []
                for episode in episodes:
                    next_values = model(tensor(features(episode, True))).squeeze(-1).cpu().numpy()
                    if method == "nstep":
                        parts.append(n_step_targets(episode, next_values, n, gamma))
                    else:
                        parts.append(lambda_targets(episode, next_values, gamma, lambda_))
                ty = torch.as_tensor(np.concatenate(parts), dtype=torch.float32, device=device)
        ix = torch.as_tensor(rng.integers(len(x), size=batch_size), device=device)
        loss = (model(tx[ix]).squeeze(-1)-ty[ix]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"state_dict": model.cpu().state_dict(), "input_dim": x.shape[1],
            "mean": mean.tolist(), "scale": scale.tolist(), "horizon": horizon,
            "method": method, "n": n if method == "nstep" else None,
            "lambda": lambda_ if method == "lambda" else None,
            "hidden": [int(units) for units in hidden],
            "target": method,
            "task_id": episodes[0].task_id, "policy_id": episodes[0].policy_id,
            "gamma": gamma, "seed": seed, "losses": losses,
            "updates": updates, "target_refresh": target_refresh}


class ValuePredictor:
    """Load a locally trained frozen V; use identical weights for every lookahead."""
    def __init__(self, checkpoint, device="cpu"):
        import torch
        self.torch = torch
        self.device = device
        self.checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        c = self.checkpoint
        if c.get("input_dim") is None or c.get("state_dict") is None:
            raise ValueError("not a V checkpoint")
        hidden = tuple(c.get("hidden", DEFAULT_HIDDEN))
        self.model = _architecture(c["input_dim"], hidden).to(device)
        self.model.load_state_dict(c["state_dict"])
        self.model.eval()

    def predict(self, episode, next_obs=False):
        c = self.checkpoint
        if episode.task_id != c["task_id"] or episode.policy_id != c["policy_id"]:
            raise ValueError("V task/reference policy does not match the episode")
        obs = episode.next_obs if next_obs else episode.obs
        remaining = episode.remaining_steps - int(next_obs)
        x = np.column_stack([obs, remaining/c["horizon"]])
        x = (x - np.asarray(c["mean"])) / np.asarray(c["scale"])
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(x,dtype=self.torch.float32,device=self.device)
            return self.model(tensor).squeeze(-1).cpu().numpy()
