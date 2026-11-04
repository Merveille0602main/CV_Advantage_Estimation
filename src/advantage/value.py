"""Small PyTorch V learner for canonical low-dimensional data (optional torch)."""
import numpy as np
from .targets import n_step_targets


def fit_value(episodes, n=1, gamma=0.99, updates=1000, batch_size=128,
              seed=0, lr=3e-4, target_refresh=20, device="cpu"):
    import torch
    if not episodes or updates < 1 or batch_size < 1 or target_refresh < 1:
        raise ValueError("nonempty episodes and positive training budgets required")
    if len({e.policy_id for e in episodes}) != 1 or len({e.task_id for e in episodes}) != 1:
        raise ValueError("first version trains a single task and fixed reference policy")
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
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(updates):
        if step % target_refresh == 0:
            with torch.no_grad():
                targets = np.concatenate([n_step_targets(
                    e, model(tensor(features(e, True))).squeeze(-1).cpu().numpy(), n, gamma) for e in episodes])
            ty = torch.as_tensor(targets, dtype=torch.float32, device=device)
        ix = torch.as_tensor(rng.integers(len(x), size=batch_size), device=device)
        loss = (model(tx[ix]).squeeze(-1)-ty[ix]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"state_dict": model.cpu().state_dict(), "input_dim": x.shape[1],
            "mean": mean.tolist(), "scale": scale.tolist(), "horizon": horizon,
            "n": n, "gamma": gamma, "seed": seed, "losses": losses,
            "updates": updates, "target_refresh": target_refresh}
