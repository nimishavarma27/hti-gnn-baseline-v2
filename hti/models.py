"""Models: R-GCN encoder with DistMult heads, two-tower network, and the
non-trained scorers (similarity kNN, popularity, random)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# message passing relations, always in this order
RELATIONS = [('herb', 'compound'), ('compound', 'herb'), ('compound', 'target'), ('target', 'compound')]


def standardise(x):
    """zero mean, unit variance per column (columns with no variance are left at zero)"""
    mu, sd = x.mean(0), x.std(0)
    return ((x - mu) / np.where(sd > 0, sd, 1)).astype(np.float32)


# ---------------------------------------------------------------- graph layers

class RelConv(nn.Module):
    """R-GCN layer: new = W_self x + sum over relations r of W_r (mean of neighbours under r)."""

    def __init__(self, dim, n_rel):
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_rel = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_rel)])

    def forward(self, x, edges):
        out = self.w_self(x)
        for w, (src, dst) in zip(self.w_rel, edges):
            if src.numel() == 0:
                continue
            total = torch.zeros_like(out).index_add_(0, dst, w(x[src]))
            count = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, torch.ones(len(dst), device=x.device))
            out = out + total / count.clamp(min=1)[:, None]
        return out


class GraphModel(nn.Module):
    """Encoder over the herb-compound-target graph plus a DistMult head for each link type.
    Node index layout: herbs, then compounds, then targets."""

    def __init__(self, prep, dim=128, dropout=0.2, feat_drop=0.0, target_feature='esm'):
        super().__init__()
        n_h, n_c, n_t = prep.n_herbs, prep.n_compounds, prep.n_targets
        self.ranges = {'herb': (0, n_h), 'compound': (n_h, n_h + n_c), 'target': (n_h + n_c, n_h + n_c + n_t)}
        target_x = prep.esm if target_feature == 'esm' else prep.aac
        self.register_buffer('family', prep.tensor(prep.family))
        self.register_buffer('compound_x', prep.tensor(prep.ecfp4))
        self.register_buffer('target_x', prep.tensor(standardise(target_x)))

        self.herb_emb = nn.Embedding(n_h, dim)
        self.herb_family = nn.Linear(self.family.shape[1], dim, bias=False)
        self.compound_in = nn.Linear(self.compound_x.shape[1], dim)
        self.target_in = nn.Linear(self.target_x.shape[1], dim)
        self.layer1, self.layer2 = RelConv(dim, len(RELATIONS)), RelConv(dim, len(RELATIONS))
        self.dropout, self.feat_drop = dropout, feat_drop
        self.rel = nn.ParameterDict({'ct': nn.Parameter(torch.ones(dim)), 'hc': nn.Parameter(torch.ones(dim))})

    def input_params(self):
        """the two large input projections, which get their own weight decay"""
        return list(self.compound_in.parameters()) + list(self.target_in.parameters())

    def forward(self, edges):
        comp = F.dropout(self.compound_x, self.feat_drop, self.training)
        x = torch.cat([self.herb_emb.weight + self.herb_family(self.family),
                       F.relu(self.compound_in(comp)),
                       F.relu(self.target_in(self.target_x))], dim=0)
        x = F.dropout(F.relu(self.layer1(x, edges)), self.dropout, self.training)
        return self.layer2(x, edges)

    def split(self, z):
        (h0, h1), (c0, c1), (t0, t1) = self.ranges['herb'], self.ranges['compound'], self.ranges['target']
        return z[h0:h1], z[c0:c1], z[t0:t1]

    def pair_scores(self, z, kind, rows, cols):
        zh, zc, zt = self.split(z)
        a, b = (zc, zt) if kind == 'ct' else (zh, zc)
        return (a[rows] * self.rel[kind] * b[cols]).sum(-1)

    def all_scores(self, z, kind):
        zh, zc, zt = self.split(z)
        a, b = (zc, zt) if kind == 'ct' else (zh, zc)
        return (a * self.rel[kind]) @ b.t()


def graph_edges(prep, hc_pairs, ct_pairs):
    """the four relation edge lists (source, destination) with absolute node indices"""
    n_h, n_c = prep.n_herbs, prep.n_compounds
    hc_pairs, ct_pairs = prep.tensor(hc_pairs), prep.tensor(ct_pairs)
    h, c1 = hc_pairs[0], hc_pairs[1] + n_h
    c2, t = ct_pairs[0] + n_h, ct_pairs[1] + n_h + n_c
    return [(h, c1), (c1, h), (c2, t), (t, c2)]


# ---------------------------------------------------------------- two-tower

class Tower(nn.Module):
    def __init__(self, d_in, d_out=256, hidden=512, drop=0.3, in_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(in_drop), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class TwoTower(nn.Module):
    """compound tower x target tower, scaled cosine similarity plus an optional per-target bias"""

    def __init__(self, compound_x, target_x, target_bias=True):
        super().__init__()
        self.register_buffer('compound_x', compound_x)
        self.register_buffer('target_x', target_x)
        self.compound_tower, self.target_tower = Tower(compound_x.shape[1]), Tower(target_x.shape[1])
        self.bias = nn.Parameter(torch.zeros(target_x.shape[0])) if target_bias else None
        self.scale = nn.Parameter(torch.tensor(10.0))

    def scores(self, compound_rows=None):
        xc = self.compound_x if compound_rows is None else self.compound_x[compound_rows]
        zc = F.normalize(self.compound_tower(xc), dim=-1)
        zt = F.normalize(self.target_tower(self.target_x), dim=-1)
        s = self.scale * zc @ zt.t()
        return s + self.bias if self.bias is not None else s


# ---------------------------------------------------------------- scorers without training

def tanimoto_matrix(bits):
    """compound x compound Tanimoto similarity on fingerprint bits"""
    inter = bits @ bits.t()
    count = bits.sum(1)
    return inter / (count[:, None] + count[None, :] - inter).clamp(min=1)


def knn_scores(similarity, train_pos, n_compounds, n_targets, k):
    """score(c, t) = summed similarity of the k most similar training compounds that hit t.
    Only training links are used. Target popularity is added at 1e-4 scale to break ties."""
    hits = torch.zeros(n_compounds, n_targets, device=similarity.device)
    hits[train_pos[0], train_pos[1]] = 1
    has = hits.sum(1) > 0
    sim = similarity[:, has]
    top = torch.topk(sim, min(k, sim.shape[1]), dim=1)
    weights = torch.zeros_like(sim).scatter_(1, top.indices, top.values)
    degree = hits.sum(0)
    return weights @ hits[has] + 1e-4 * degree / degree.max().clamp(min=1)


def popularity_scores(train_pos, n_rows, n_cols, seed):
    """score(row, col) = how many training rows link to col; ignores the row completely"""
    degree = np.bincount(train_pos[1].cpu().numpy(), minlength=n_cols).astype(np.float32)
    degree += np.random.default_rng(seed).random(n_cols).astype(np.float32) * 1e-3   # breaks ties
    return torch.as_tensor(np.tile(degree, (n_rows, 1)), device=train_pos.device)


def random_scores(n_rows, n_cols, seed, device):
    return torch.as_tensor(np.random.default_rng(seed).random((n_rows, n_cols), dtype=np.float32), device=device)


def row_ranks(scores):
    """per-row rank transform to [0, 1], so different scorers can be averaged"""
    return torch.argsort(torch.argsort(scores, dim=1), dim=1).float() / (scores.shape[1] - 1)
