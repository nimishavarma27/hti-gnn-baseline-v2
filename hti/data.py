"""Paths and the loader for everything 02_prep writes to prep/."""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT                      # the dataset files sit in the project folder
PREP = ROOT / 'prep'
RESULTS = ROOT / 'results'

TASKS = ['ct_edge', 'ct_compound', 'ct_target', 'hc_edge']
TASK_NAMES = {
    'ct_edge': 'C-T, hidden links (known compounds)',
    'ct_compound': 'C-T, new compounds',
    'ct_target': 'C-T, new targets',
    'hc_edge': 'H-C, hidden links',
}
SETS = ['train', 'val', 'test']


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kind_of(task):
    """'ct' tasks score compound x target, 'hc' scores herb x compound."""
    return 'hc' if task == 'hc_edge' else 'ct'


class Prep:
    """Holds features, links and folds. Rows and columns of every score matrix:
    ct -> compounds x targets, hc -> herbs x compounds."""

    def __init__(self, device=None, folder=PREP):
        self.folder = Path(folder)
        self.device = device or get_device()
        meta = json.loads((self.folder / 'meta.json').read_text())
        self.meta = meta
        self.n_herbs, self.n_compounds, self.n_targets = meta['n_herbs'], meta['n_compounds'], meta['n_targets']
        self.k = meta['k']
        self.herb_ids, self.compound_ids, self.target_ids = meta['herb_ids'], meta['compound_ids'], meta['target_ids']

        feats = np.load(self.folder / 'features.npz')
        self.ecfp4 = feats['ecfp4'].astype(np.float32)
        self.chem = feats['chem'].astype(np.float32)
        self.aac = feats['aac'].astype(np.float32)
        self.esm = feats['esm'].astype(np.float32)
        self.family = feats['family'].astype(np.float32)
        self.herb_member = feats['herb_member'].astype(np.float32)

        links = np.load(self.folder / 'links.npz')
        self.hc = links['hc']      # (2, 5005)   herb, compound
        self.ct = links['ct']      # (2, 14166)  compound, target

        self.known = {
            'ct': self._mask(self.ct, self.n_compounds, self.n_targets),
            'hc': self._mask(self.hc, self.n_herbs, self.n_compounds),
        }
        # herb x compound membership, used to turn compound scores into herb scores
        self.herb_compounds = [torch.as_tensor(np.unique(self.hc[1][self.hc[0] == h]), device=self.device)
                               for h in range(self.n_herbs)]
        self.ht_known = self.reach(self.ct)               # every herb-target pair with a path
        self.ht_negative = ~self.ht_known

    def _mask(self, pairs, n_rows, n_cols):
        m = torch.zeros(n_rows, n_cols, dtype=torch.bool, device=self.device)
        m[torch.as_tensor(pairs[0]), torch.as_tensor(pairs[1])] = True
        return m

    def reach(self, ct_pairs):
        """herb x target mask: True where some compound of the herb has one of the given C-T links."""
        herb_has = np.zeros((self.n_herbs, self.n_compounds), dtype=np.float32)
        herb_has[self.hc[0], self.hc[1]] = 1
        links = np.zeros((self.n_compounds, self.n_targets), dtype=np.float32)
        links[ct_pairs[0], ct_pairs[1]] = 1
        return torch.as_tensor(herb_has @ links > 0, device=self.device)

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device)

    def fold(self, task, k):
        """train / val / test parts of fold k. Each part has
        pos  (2, n) positive pairs, neg (2, m) negatives for PR-AUC, pair (n,) one negative column per positive,
        cols (n_cols,) bool mask of columns that may be ranked (only restricts anything for new targets)."""
        z = np.load(self.folder / f'folds_{task}.npz')
        out = {}
        for s in ['train', 'val', 'test']:
            part = {'pos': self.tensor(z[f'f{k}_{s}_pos'].astype(np.int64)),
                    'neg': self.tensor(z[f'f{k}_{s}_neg'].astype(np.int64)),
                    'pair': self.tensor(z[f'f{k}_{s}_pair'].astype(np.int64))}
            cols = z[f'f{k}_{s}_cols']
            part['cols'] = None if cols.size == 0 else self.tensor(cols.astype(bool))
            out[s] = part
        if kind_of(task) == 'ct' and task != 'ct_target':
            # herb-target pairs per part: reachable through this part's links and not through an earlier part
            tr = self.reach(out['train']['pos'].cpu().numpy())
            va = self.reach(out['val']['pos'].cpu().numpy()) & ~tr
            te = self.reach(out['test']['pos'].cpu().numpy()) & ~tr & ~va
            out['train']['ht'], out['val']['ht'], out['test']['ht'] = tr, va, te
        return out
