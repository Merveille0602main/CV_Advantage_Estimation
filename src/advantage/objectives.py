"""Three advantage representations over one shared MLP trunk.

continuous  regress the advantage itself (MSE)
binary      classify "better than the reference policy" (BCE with an ignore band)
pairwise    rank two actions at the same state (logistic pairwise loss)

All three receive the same features, capacity and update budget, so a
difference between them is attributable to the representation and loss rather
than to model size or data.
"""
import numpy as np

from .networks import DEFAULT_HIDDEN, scalar_mlp

KINDS = ("continuous", "binary", "pairwise")


def features(obs, remaining, action, horizon):
    obs = np.asarray(obs, dtype=np.float64)
    remaining = np.asarray(remaining, dtype=np.float64).reshape(-1)
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action.reshape(-1, 1)
    if len(obs) != len(remaining) or len(action) != len(remaining):
        raise ValueError("obs, remaining and action must share a length")
    return np.column_stack([obs, remaining/float(horizon), action])


def fit(kind, transition, pairs=None, horizon=24, updates=1000, batch_size=128,
        seed=0, lr=3e-4, hidden=DEFAULT_HIDDEN, device="cpu",
        pos_weight=None, min_gap=0.005):
    import torch
    import torch.nn.functional as F
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if updates < 1 or batch_size < 1:
        raise ValueError("positive training budget required")
    x = features(transition["obs"], transition["remaining"], transition["action"], horizon)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    tx = torch.as_tensor((x-mean)/scale, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = scalar_mlp(x.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    if kind == "continuous":
        targets = torch.as_tensor(transition["advantage"], dtype=torch.float32, device=device)
        for step in range(updates):
            index = torch.as_tensor(rng.integers(len(x), size=batch_size), device=device)
            loss = F.mse_loss(model(tx[index]).squeeze(-1), targets[index])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        stored_weight = None
    elif kind == "binary":
        from .labels import binary_labels
        labels = binary_labels(transition["advantage"], min_gap)
        keep = np.flatnonzero(labels >= 0)
        if len(keep) < 2 or len(np.unique(labels[keep])) < 2:
            raise ValueError("binary labels contain only one class after the ignore band")
        weight = pos_weight
        if weight is None:
            positive = float((labels[keep] == 1).sum())
            negative = float((labels[keep] == 0).sum())
            weight = negative / max(positive, 1.0)
        weight_tensor = torch.tensor([float(weight)], dtype=torch.float32, device=device)
        keep_tensor = torch.as_tensor(keep, device=device)
        labels_tensor = torch.as_tensor(labels[keep].astype(np.float32), device=device)
        for step in range(updates):
            choice = torch.as_tensor(rng.integers(len(keep), size=batch_size), device=device)
            index = keep_tensor[choice]
            logits = model(tx[index]).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels_tensor[choice],
                                                      pos_weight=weight_tensor)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        stored_weight = float(weight)
    else:
        if not pairs:
            raise ValueError("pairwise training requires same-state pairs")
        first = np.array([pair["i"] for pair in pairs], dtype=int)
        second = np.array([pair["j"] for pair in pairs], dtype=int)
        direction = np.array([1.0 if pair["label"] == 1 else -1.0 for pair in pairs],
                             dtype=np.float32)
        first_tensor = torch.as_tensor(first, device=device)
        second_tensor = torch.as_tensor(second, device=device)
        direction_tensor = torch.as_tensor(direction, device=device)
        for step in range(updates):
            choice = torch.as_tensor(rng.integers(len(pairs), size=batch_size), device=device)
            left = model(tx[first_tensor[choice]]).reshape(-1)
            right = model(tx[second_tensor[choice]]).reshape(-1)
            loss = F.softplus(-direction_tensor[choice]*(right-left)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        stored_weight = None

    return {"kind": kind, "state_dict": model.cpu().state_dict(), "input_dim": x.shape[1],
            "mean": mean.tolist(), "scale": scale.tolist(), "horizon": float(horizon),
            "hidden": [int(units) for units in hidden], "pos_weight": stored_weight,
            "min_gap": float(min_gap), "seed": seed, "updates": updates, "losses": losses}


class Predictor:
    """Score arbitrary (state, remaining, action) rows with a frozen model."""

    def __init__(self, checkpoint, device="cpu"):
        import torch
        self.torch = torch
        self.device = device
        self.checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        c = self.checkpoint
        if c.get("kind") not in KINDS:
            raise ValueError("not a representation checkpoint")
        self.model = scalar_mlp(c["input_dim"], tuple(c.get("hidden", DEFAULT_HIDDEN))).to(device)
        self.model.load_state_dict(c["state_dict"])
        self.model.eval()

    def score(self, obs, remaining, action):
        c = self.checkpoint
        x = features(obs, remaining, action, c["horizon"])
        x = (x - np.asarray(c["mean"])) / np.asarray(c["scale"])
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(x, dtype=self.torch.float32, device=self.device)
            return self.model(tensor).squeeze(-1).cpu().numpy()

    def probability(self, obs, remaining, action):
        if self.checkpoint["kind"] != "binary":
            raise ValueError("probability is defined only for the binary representation")
        return 1.0 / (1.0 + np.exp(-self.score(obs, remaining, action)))
