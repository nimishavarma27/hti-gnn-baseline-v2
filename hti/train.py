"""Training loops for the graph model and the two-tower model.
Both stop early on validation MRR and return the full score matrix of the best epoch."""
import numpy as np
import torch
import torch.nn.functional as F

from .data import kind_of
from .metrics import val_mrr
from .models import GraphModel, TwoTower, graph_edges, standardise

GRAPH_CONFIGS = {
    # plain model, every training link is both a message edge and a scored link
    'plain': dict(msg_frac=1.0, hide_frac=0.0, feat_drop=0.0, wd_input=1e-5, emb_l2=0.0),
    # overfitting fixes: each epoch 30% of training links are taken out of the graph and only those are scored,
    # in the new compound / new target tasks 15% of training compounds / targets lose all their links the same way,
    # plus dropout on the fingerprint input and extra weight decay
    'fixes': dict(msg_frac=0.7, hide_frac=0.15, feat_drop=0.3, wd_input=1e-4, emb_l2=1e-4),
}


def sample_negatives(rows, train_known, allowed_cols, generator, tries=50):
    """one random column per row that is not a training positive of that row and is allowed.
    Validation and test positives can come up here: during training they are unknown, which is the
    positive-unlabeled setting, so excluding them would leak the held-out answers."""
    n_cols = train_known.shape[1]
    neg = torch.randint(0, n_cols, (len(rows),), device=rows.device, generator=generator)
    for _ in range(tries):
        bad = train_known[rows, neg]
        if allowed_cols is not None:
            bad |= ~allowed_cols[neg]
        if not bad.any():
            break
        neg[bad] = torch.randint(0, n_cols, (int(bad.sum()),), device=rows.device, generator=generator)
    return neg


def _train_masks(prep, fold, kind):
    n_rows, n_cols = (prep.n_compounds, prep.n_targets) if kind == 'ct' else (prep.n_herbs, prep.n_compounds)
    train_known = torch.zeros(n_rows, n_cols, dtype=torch.bool, device=prep.device)
    pos = fold['train']['pos']
    train_known[pos[0], pos[1]] = True
    return train_known, fold['train']['cols']


def train_graph_model(prep, fold, task, config='fixes', seed=0, max_epochs=300, patience=15,
                      batch=1024, check_every=2, verbose=False):
    cfg = GRAPH_CONFIGS[config]
    kind = kind_of(task)
    torch.manual_seed(seed)
    gen = torch.Generator(device=prep.device).manual_seed(seed)
    model = GraphModel(prep, feat_drop=cfg['feat_drop']).to(prep.device)
    inputs = set(model.input_params())
    opt = torch.optim.Adam([{'params': [p for p in model.parameters() if p not in inputs], 'weight_decay': 1e-5},
                            {'params': list(inputs), 'weight_decay': cfg['wd_input']}], lr=1e-3)

    train_pos = fold['train']['pos']
    train_known, allowed = _train_masks(prep, fold, kind)
    known = prep.known[kind]

    def edges_with(task_links):
        # the supervised link type uses only the given training links, the other type uses all of its links
        if kind == 'ct':
            return graph_edges(prep, prep.hc, task_links)
        return graph_edges(prep, task_links, prep.ct)

    full_edges = edges_with(train_pos)            # used for validation, test and the final scores
    hide_axis = {'ct_compound': 0, 'ct_target': 1}.get(task)
    units = torch.unique(train_pos[hide_axis]) if hide_axis is not None else None

    best_mrr, best_epoch, best_state, wait, curve = -1.0, 0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        scored = torch.zeros(train_pos.shape[1], dtype=torch.bool, device=prep.device)
        if cfg['msg_frac'] < 1:
            scored |= torch.rand(train_pos.shape[1], device=prep.device, generator=gen) >= cfg['msg_frac']
        if cfg['hide_frac'] > 0 and units is not None:
            n_hide = int(cfg['hide_frac'] * len(units))
            hidden = units[torch.randperm(len(units), device=prep.device, generator=gen)[:n_hide]]
            scored |= torch.isin(train_pos[hide_axis], hidden)
        if scored.any():
            edges, sup = edges_with(train_pos[:, ~scored]), train_pos[:, scored]
        else:
            edges, sup = full_edges, train_pos

        losses = []
        order = torch.randperm(sup.shape[1], device=prep.device, generator=gen)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            rows, pos_cols = sup[0, b], sup[1, b]
            neg_cols = sample_negatives(rows, train_known, allowed, gen)
            z = model(edges)
            s_pos = model.pair_scores(z, kind, rows, pos_cols)
            s_neg = model.pair_scores(z, kind, rows, neg_cols)
            loss = F.softplus(-(s_pos - s_neg)).mean()
            if cfg['emb_l2']:
                zh, zc, zt = model.split(z)
                a, b2 = (zc, zt) if kind == 'ct' else (zh, zc)
                loss = loss + cfg['emb_l2'] * (a[rows].pow(2).sum(1) + b2[pos_cols].pow(2).sum(1)
                                               + b2[neg_cols].pow(2).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if epoch % check_every == 0:
            model.eval()
            with torch.no_grad():
                scores = model.all_scores(model(full_edges), kind)
                v = val_mrr(scores, fold['val'], known)
                vp = fold['val']['pos']
                v_loss = float(F.softplus(-(scores[vp[0], vp[1]] - scores[vp[0], fold['val']['pair']])).mean())
            curve.append({'epoch': epoch, 'train loss': float(np.mean(losses)), 'val loss': v_loss, 'val MRR': v})
            if verbose:
                print(f'  epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val loss {v_loss:.4f}  val MRR {v:.4f}')
            if v > best_mrr:
                best_mrr, best_epoch, wait = v, epoch, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model.all_scores(model(full_edges), kind)
    return scores, {'best epoch': best_epoch, 'epochs run': epoch, 'val MRR at best': best_mrr}, curve


def two_tower_inputs(prep, target_feature='esm', herbs=True):
    compound_x = standardise(prep.chem)
    if herbs:
        compound_x = np.hstack([compound_x, prep.herb_member])
    target_x = standardise(prep.esm if target_feature == 'esm' else prep.aac)
    return prep.tensor(compound_x), prep.tensor(target_x)


def train_two_tower(prep, fold, target_feature='esm', herbs=True, seed=0, max_epochs=150, patience=12, batch=512,
                    verbose=False):
    """softmax over all allowed training targets; the compound's other training targets are masked out,
    because they are positives, not negatives"""
    torch.manual_seed(seed)
    gen = torch.Generator(device=prep.device).manual_seed(seed)
    compound_x, target_x = two_tower_inputs(prep, target_feature, herbs)
    model = TwoTower(compound_x, target_x).to(prep.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_pos = fold['train']['pos']
    train_known, allowed = _train_masks(prep, fold, 'ct')
    if allowed is None:
        allowed = torch.ones(prep.n_targets, dtype=torch.bool, device=prep.device)
    known = prep.known['ct']

    best_mrr, best_epoch, best_state, wait, curve = -1.0, 0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        order = torch.randperm(train_pos.shape[1], device=prep.device, generator=gen)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            c, t = train_pos[0, b], train_pos[1, b]
            logits = model.scores(c)
            other = train_known[c].clone()
            other[torch.arange(len(b), device=prep.device), t] = False
            logits = logits.masked_fill(other | ~allowed[None, :], -1e4)
            loss = F.cross_entropy(logits, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            scores = model.scores()
            v = val_mrr(scores, fold['val'], known)
            vp = fold['val']['pos']
            v_loss = float(F.softplus(-(scores[vp[0], vp[1]] - scores[vp[0], fold['val']['pair']])).mean())
        curve.append({'epoch': epoch, 'train loss': float(np.mean(losses)), 'val loss': v_loss, 'val MRR': v})
        if verbose:
            print(f'  epoch {epoch:3d}  train loss {np.mean(losses):.4f}  val loss {v_loss:.4f}  val MRR {v:.4f}')
        if v > best_mrr:
            best_mrr, best_epoch, wait = v, epoch, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model.scores()
    return scores, {'best epoch': best_epoch, 'epochs run': epoch, 'val MRR at best': best_mrr}, curve
