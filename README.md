# Herb-Target Interaction (HTI) Prediction: GNN Baseline

Heterogeneous graph of 62 herbs, 2,319 compounds and 1,341 targets. Predicts compound-target (C-T), herb-compound (H-C) and herb-target (H-T) links.
The main model is an R-GCN with DistMult heads; similarity kNN, a two-tower network and their ensemble are added for comparison,
together with random and popularity reference scorers.

## How to run

1. Put the dataset files in the project folder (next to the notebooks):
   `herbs.tsv`, `compounds.tsv`, `herb_compound_edges.tsv`, `compound_target_labeled.csv`, `compound_target_unlabeled.csv`,
   `herb_compound_target.csv`, `compound_features_ecfp4.csv`, `target_features_aac.csv`, `target_features_esm2.csv`
2. Install: `pip install -r requirements.txt` (torch with CUDA for a GPU)
3. Run the notebooks in order (times on an RTX 4060):

```
01_data_check   checks every file and column, nothing written      ~1 min
02_prep         features, links, folds, negatives -> prep/         ~3 min
03_rgcn         R-GCN on all four tasks                             ~8 min  -> results/metrics_rgcn.csv
04_inductive    kNN, two-tower, ensemble                            ~7 min  -> results/metrics_inductive.csv
05_results      random and popularity, all tables and plots         ~2 min  -> results/summary_*.csv, test_metrics.png
06_predictions  hidden-link check and new candidates (5 folds)      ~5 min  -> results/hidden_link_*.csv, candidates_*.csv
```

The dataset files are not in this repository (the unlabeled file alone is 142 MB). The `results/` folder holds the outputs of the last run.
The full write-up is in `hti_gnn_report.md`; a step-by-step explanation of every part and notebook is in `explanation_guide.md`.

Shared code is in the `hti` package: `data.py` (paths, loading `prep/`), `models.py`, `train.py`, `metrics.py`.
Every notebook sets `KMP_DUPLICATE_LIB_OK` itself (through `import hti`), so the Windows OpenMP error does not appear.

## Data

| File | Rows | Used for |
|---|---|---|
| herbs.tsv | 62 | herb ids, names, plant family (herb feature) |
| compounds.tsv | 2,319 | compound ids, names, SMILES (all fingerprints are computed from it), InChIKey (stereoisomer groups) |
| herb_compound_edges.tsv | 8,396 rows, 5,005 unique pairs | H-C links (one row per plant part, de-duplicated) |
| compound_target_labeled.csv | 14,166 | known C-T links (positives), gene symbols |
| compound_target_unlabeled.csv | 3,095,613 | every other compound-target pair, the pool negatives are drawn from |
| herb_compound_target.csv | 126,830 rows, 24,835 herb-target pairs | check that herb-target pairs equal the H-C x C-T join |
| compound_features_ecfp4.csv | 2,319 | checked only, see below |
| target_features_aac.csv | 1,341 | target feature for the comparison model (20 values), protein names |
| target_features_esm2.csv | 1,341 | main target feature (ESM-2, 1,280 values) |

`01_data_check` lists every column of every file with what it is used for or why it is not.

**Fingerprint file defect.** In `compound_features_ecfp4.csv` the bit rows are in sorted compound id order but the `compound_id` column is not
(2,317 of 2,319 rows carry another compound's id). Fingerprints are therefore recomputed from SMILES with RDKit; they equal the file's rows
in sorted order exactly.

Only 726 of the 2,319 compounds have a known target. There are no confirmed negatives: every pair that is not a known link is unlabeled
(positive-unlabeled data).

## Graph and models

- Nodes: herbs, compounds, targets (3,722). Relations: herb to compound, compound to herb, compound to target, target to compound.
- Graph stored as edge lists per relation (the sparse form of the adjacency matrix).
- Inputs: compounds ECFP4 bits (2,048); targets ESM-2 (1,280, standardised); herbs a learned vector plus their plant family one-hot.
- **R-GCN**: 2 layers, hidden size 128. Each layer: own vector times W_self plus, for every relation, the mean of the neighbours times W_relation.
- **DistMult heads**: score(c, t) = sum of z_c × r_ct × z_t, and score(h, c) = sum of z_h × r_hc × z_c.
- **Loss**: BPR, -log sigmoid(score of a true link - score of a random negative for the same row).
- **R-GCN + fixes**: each epoch 30% of training links leave the graph and only those are scored (so the model cannot see the link it is
  scored on); in the new-compound and new-target tasks 15% of training compounds or targets also lose all their links;
  dropout 0.3 on the fingerprint input; extra weight decay on the input layers; small L2 penalty on embeddings.
- **Similarity kNN**: Tanimoto on ECFP4; score(c, t) = summed similarity of the k most similar training compounds that hit t (k picked on validation).
- **Two-tower**: compound tower (ECFP4 counts, FCFP4, MACCS, herb membership) and target tower (ESM-2, or AAC for comparison),
  scaled cosine similarity plus a per-target bias, softmax cross-entropy over training targets with the compound's other training targets masked.
- **Ensemble**: weighted average of per-compound ranks of two-tower (ESM-2) and kNN, weight picked on validation.
- **Herb-target scores**: from compound-target scores, using max, mean, top-3 mean, noisy-OR, sum, or the number of the herb's compounds
  that rank the target in their top 50, picked on validation H-T MRR ("model only" columns). The reported H-T columns also blend the herb's
  model ranks with target popularity from the training links, (1 - w) × model + w × popularity, with the rule and w picked together on
  validation (w = 1 is popularity alone). Random and popularity are never blended.
- **Reference scorers**: random scores, and popularity (how many training rows link to each column; ignores the row).

All trained models stop early on validation MRR.

## Tasks and cross validation

| Task | What is held out | Question |
|---|---|---|
| ct_edge | C-T links | find missing targets of known compounds |
| ct_compound | compounds with all their C-T links | predict targets of a new compound |
| ct_target | targets with all their C-T links | predict compounds for a new protein |
| hc_edge | H-C links | find missing compounds of a herb |

H-T is evaluated inside `ct_edge` and `ct_compound`.

**5-fold cross validation.** The units of a task (links, compounds or targets) are spread over 5 parts of about 20% of the links each.
In fold k, part k is the test set; from the other parts, 15% of all links go to validation and the rest to training (about 65 / 15 / 20).
Every unit is in the test set exactly once. Each fold trains new models from scratch.

**Negatives** (all drawn at random without replacement, train / val / test never share one):
- per part, 10 negatives per positive for train and val and 50 for test, for PR-AUC
- per positive, one paired negative for the same row, for the BPR loss on train, val and test
- C-T negatives come from the 3.1 million unlabeled pairs: for hidden links, unlabeled pairs of the 726 compounds with known targets
  (pairs of compounds with no known target would be too easy and inflate PR-AUC); for new compounds, pairs of that part's compounds;
  for new targets, pairs of that part's targets, and ranking is only among that part's targets
- H-C negatives: herb x compound pairs without a link (there is no unlabeled file for H-C)
- During training, negatives are drawn freshly from everything that is not a training positive of that row. Validation and test positives
  can be drawn there: during training they are unknown, and excluding them would tell the model where the answers are.

## Leakage controls

- Only training links of the predicted type are in the message-passing graph. For H-C, only training H-C links are in the graph.
- Validation picks early stopping, k, ensemble weight and herb aggregation; the test set is never used for a choice.
- New compounds: compounds with an identical ECFP4 fingerprint or the same InChIKey first block (stereoisomers) form one group and always
  land in the same part, so no test compound has a twin in training. 470 compounds share a fingerprint, 318 share an InChIKey block.
- New targets: no target is in two parts, and held-out targets are not used as training negatives.
- H-T: a herb-target pair is in the test part only if it is reachable through test links and not through any training or validation link.
  Herb-target pairs reachable through training links are counted in the train part instead.
- Filtered ranking: when a true link is ranked, the row's other known links are left out of the ranking.
- `02_prep` ends with checks (all must be True): parts cover all links once, every unit tested once, no negative is a known link,
  parts share no negative, no compound, compound group or target is in two parts.

## Metrics

For every model, task and fold, on **train, val and test**:
- **PR-AUC**: positives of the part against its sampled negatives
- **Hits@10, Hits@50**: share of true links ranked in the top 10 / 50 of their row (filtered ranking, ties count half)
- **MRR**: mean of 1 / rank
- **loss**: BPR loss on the part's positives and their paired negatives (trained models only)
- **HT PR-AUC, HT Hits@10, HT Hits@50, HT MRR**: the same for herb-target pairs (C-T tasks with herbs)

Reported as mean ± std over the 5 folds. Training curves (train loss, val loss, val MRR per epoch) are saved for every run.
ROC-AUC and accuracy are not used: with a huge unlabeled pool they look high even for a weak model.

## Outputs

- `results/metrics_rgcn.csv`, `metrics_inductive.csv`, `metrics_baselines.csv`: one row per task, model, fold and part
- `results/curves_rgcn.csv`, `curves_inductive.csv`: training curves
- `results/summary_<task>.csv`: mean ± std per model and part; `results/summary_test.csv`: test set of every task
- `results/test_metrics.png`, `results/curves_rgcn.png`
- `results/candidates_ct.csv`, `candidates_ht.csv`, `candidates_hc.csv`: highest scoring pairs not known anywhere

## Other files

- `hti_gnn_report.md`: full report (method, results for every task and split, findings, predictions)
- `explanation_guide.md`: explanation of every part of the project and every notebook cell, with likely questions and answers
