"""Metrics: PR-AUC, Hits@10, Hits@50, MRR and the paired BPR loss, for link sets and for herb-target sets."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERB_AGGREGATIONS = ['max', 'mean', 'top3', 'noisy_or', 'sum', 'top50']
POPULARITY_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def pr_auc(pos_scores, neg_scores):
    pos_scores, neg_scores = np.asarray(pos_scores), np.asarray(neg_scores)
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return float(average_precision_score(y, np.r_[pos_scores, neg_scores]))


def rank_summary(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    return {'Hits@10': float((r <= 10).mean()), 'Hits@50': float((r <= 50).mean()), 'MRR': float((1 / r).mean())}


@torch.no_grad()
def filtered_ranks(scores, rows, cols, known, allowed_cols=None, chunk=4096):
    """Rank of scores[row, col] inside its row.
    Other known positives of the row are skipped (filtered ranking), columns outside allowed_cols are skipped,
    and a tie counts as half a place, which is the expected rank if ties were broken at random."""
    out = []
    for i in range(0, len(rows), chunk):
        r, c = rows[i:i + chunk], cols[i:i + chunk]
        block = scores[r]
        true = block.gather(1, c[:, None])
        keep = ~known[r]                     # the true column is known, so it never counts against itself
        if allowed_cols is not None:
            keep &= allowed_cols[None, :]
        higher = ((block > true) & keep).sum(1).float()
        ties = ((block == true) & keep).sum(1).float()
        out.append(1 + higher + 0.5 * ties)
    return torch.cat(out).cpu().numpy() if out else np.zeros(0)


@torch.no_grad()
def score_link_set(scores, part, known, with_loss=True):
    """Metrics for one train / val / test part of a link task."""
    rows, cols = part['pos'][0], part['pos'][1]
    pos = scores[rows, cols]
    neg = scores[part['neg'][0], part['neg'][1]]
    res = {'PR-AUC': pr_auc(pos.cpu().numpy(), neg.cpu().numpy())}
    res.update(rank_summary(filtered_ranks(scores, rows, cols, known, part['cols'])))
    if with_loss:
        paired = scores[rows, part['pair']]
        res['loss'] = float(F.softplus(-(pos - paired)).mean())   # BPR: -log sigmoid(pos - neg)
    else:
        res['loss'] = float('nan')
    res['n_pos'], res['n_neg'] = int(len(rows)), int(part['neg'].shape[1])
    return res


@torch.no_grad()
def herb_scores(ct_scores, herb_compounds, how):
    """Turn compound x target scores into herb x target scores."""
    if how == 'top50':
        # for every compound, which targets are in its own top 50
        cutoff = ct_scores.topk(50, dim=1).values[:, -1:]
        in_top = (ct_scores >= cutoff).float()
    rows = []
    for idx in herb_compounds:
        block = ct_scores[idx]
        if how == 'max':
            rows.append(block.max(0).values)
        elif how == 'mean':
            rows.append(block.mean(0))
        elif how == 'top3':
            rows.append(block.topk(min(3, len(idx)), dim=0).values.mean(0))
        elif how == 'noisy_or':
            # 1 - prod(1 - sigmoid(s)) ranks the same as sum(softplus(s))
            rows.append(F.softplus(block).sum(0))
        elif how == 'sum':
            rows.append(block.sum(0))
        elif how == 'top50':
            # how many of the herb's compounds rank the target in their top 50; the mean score (scaled below 1e-3)
            # only breaks ties between equal counts
            mean = block.mean(0)
            tie_break = (mean - mean.min()) / (mean.max() - mean.min() + 1e-12) * 1e-3
            rows.append(in_top[idx].sum(0) + tie_break)
        else:
            raise ValueError(how)
    return torch.stack(rows)


@torch.no_grad()
def tied_ranks(x):
    """rank of every value in a 1-D tensor scaled to [0, 1]; equal values get equal ranks"""
    return (x[None, :] < x[:, None]).sum(1).float() / max(len(x) - 1, 1)


@torch.no_grad()
def herb_rows_to_ranks(herb_target_scores):
    return torch.stack([tied_ranks(row) for row in herb_target_scores])


@torch.no_grad()
def target_popularity(train_pos, n_targets):
    """rank of every target by how many training compounds hit it, scaled to [0, 1]"""
    degree = torch.bincount(train_pos[1], minlength=n_targets).float()
    return tied_ranks(degree)


@torch.no_grad()
def score_herb_set(herb_target_scores, positives, ht_known, ht_negative):
    """positives: herb x target mask of the pairs in this part."""
    h, t = torch.nonzero(positives, as_tuple=True)
    if len(h) == 0:
        return {'HT PR-AUC': float('nan'), 'HT Hits@10': float('nan'), 'HT Hits@50': float('nan'),
                'HT MRR': float('nan'), 'HT n_pos': 0}
    ranks = filtered_ranks(herb_target_scores, h, t, ht_known)
    res = {'HT PR-AUC': pr_auc(herb_target_scores[positives].cpu().numpy(),
                               herb_target_scores[ht_negative].cpu().numpy())}
    res.update({f'HT {k}': v for k, v in rank_summary(ranks).items()})
    res['HT n_pos'] = int(len(h))
    return res


def evaluate(scores, fold, prep, kind, with_loss=True, herb_popularity=True):
    """All metrics for the three parts of a fold.

    For C-T folds, herb-target scores are built from compound-target scores. Two versions are reported:
    - 'HT ... (model only)': the aggregation (max, mean, top3, noisy_or, sum, top50) with the best validation HT MRR
    - 'HT ...': the herb's model ranks blended with target popularity, (1 - w) * model rank + w * popularity rank,
      with the aggregation and w picked together on validation HT MRR. w = 1 is popularity alone.
    Popularity comes from the fold's training links only. herb_popularity=False (used for the reference scorers)
    reports the model-only version under the 'HT ...' names."""
    known = prep.known[kind]
    rows = {s: score_link_set(scores, fold[s], known, with_loss) for s in ['train', 'val', 'test']}
    if 'ht' not in fold['val']:
        return rows

    def val_herb_mrr(hs):
        return score_herb_set(hs, fold['val']['ht'], prep.ht_known, prep.ht_negative)['HT MRR']

    popularity = target_popularity(fold['train']['pos'], prep.n_targets)
    plain_how, plain_mrr = 'max', -1.0
    blend_how, blend_w, blend_mrr = 'max', 0.0, -1.0
    for how in HERB_AGGREGATIONS:
        hs = herb_scores(scores, prep.herb_compounds, how)
        mrr = val_herb_mrr(hs)
        if mrr > plain_mrr:
            plain_how, plain_mrr = how, mrr
        if herb_popularity:
            ranked = herb_rows_to_ranks(hs)
            for w in POPULARITY_WEIGHTS:
                mrr = val_herb_mrr((1 - w) * ranked + w * popularity[None, :])
                if mrr > blend_mrr:
                    blend_how, blend_w, blend_mrr = how, w, mrr

    plain_hs = herb_scores(scores, prep.herb_compounds, plain_how)
    if herb_popularity:
        blend_hs = (1 - blend_w) * herb_rows_to_ranks(herb_scores(scores, prep.herb_compounds, blend_how)) \
            + blend_w * popularity[None, :]
    for s in ['train', 'val', 'test']:
        plain = score_herb_set(plain_hs, fold[s]['ht'], prep.ht_known, prep.ht_negative)
        if herb_popularity:
            rows[s].update(score_herb_set(blend_hs, fold[s]['ht'], prep.ht_known, prep.ht_negative))
            rows[s]['HT aggregation'] = blend_how
            rows[s]['HT popularity weight'] = blend_w
            rows[s].update({f'{k} (model only)': v for k, v in plain.items() if k != 'HT n_pos'})
            rows[s]['HT aggregation (model only)'] = plain_how
        else:
            rows[s].update(plain)
            rows[s]['HT aggregation'] = plain_how
    return rows


def val_mrr(scores, part, known):
    """MRR on the validation part, used for early stopping and for choosing k and weights."""
    ranks = filtered_ranks(scores, part['pos'][0], part['pos'][1], known, part['cols'])
    return float((1 / ranks).mean())
