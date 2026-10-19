# Herb-Target Interaction Prediction: GNN Baseline Report

## Summary

The GNN baseline is an **R-GCN** (relational graph convolutional network) on a graph of herbs, compounds and protein targets, trained in
two settings: plain, and with **fixes** that stop it from memorising training links (section 3). Results are test-set means over 5-fold
cross validation, compared with a **popularity** baseline that predicts the targets most compounds hit, ignoring the compound.

| Task | Popularity (MRR) | R-GCN (MRR) | R-GCN + fixes (MRR) | Best non-graph model (MRR) |
|---|---|---|---|---|
| C-T, known compounds | 0.069 | **0.294** | 0.268 | 0.342 (two-tower) |
| C-T, new compounds | 0.069 | 0.036 | **0.072** | 0.140 (two-tower) |
| C-T, new targets | 0.026 | 0.093 | **0.157** | 0.206 (two-tower) |
| H-C | 0.052 | 0.053 | **0.079** | not run |
| H-T, known compounds (with blend) | 0.073 | 0.074 | **0.078** | 0.076 (two-tower) |
| H-T, new compounds (with blend) | 0.068 | 0.067 | 0.067 | 0.068 (two-tower) |

- The plain R-GCN is the best GNN setting for known compounds (0.294, four times popularity), but falls below
  popularity for new compounds (0.036) and only ties it on H-C (0.053).
- With the fixes, the R-GCN beats popularity on every compound-target task and on H-C; for new compounds only just
  (0.072 against 0.069). The fixes matter whenever the test compounds, targets
  or links are new to the graph.
- The non-graph two-tower model with protein language model features is the strongest compound-target model; it is included to show
  where the graph helps and where it does not.
- Herb-target prediction from compound scores alone does not beat popularity; mixed with popularity (the **blend**, section 3) it is
  slightly better than popularity for known compounds and equal for new compounds. This is the main open problem.

### Terms used in this report
- **GNN baseline**: the R-GCN, in its two settings "R-GCN" and "R-GCN + fixes". kNN and two-tower are non-graph comparison models.
- **Fixes**: three changes to R-GCN training that stop it from memorising the links it is trained on (details in section 3):
  (1) each epoch, 30% of the training links are removed from the graph and the model is trained to predict only those, so it never
  sees the link it is asked to score; (2) in the new-compound and new-target tasks, each epoch 15% of the training compounds (or targets)
  lose all their links, so training looks like the test situation; (3) dropout and stronger weight decay on the input features.
- **Blend**: a way to build herb-target scores. For every herb, targets are ranked by the model and by popularity, and the two ranks
  are mixed: score = (1 - w) × model rank + w × popularity rank, with w between 0 (model only) and 1 (popularity only) chosen on
  validation. If the chosen w is below 1 and the result beats popularity, the model adds information beyond popularity.
- **Popularity**: how many training compounds hit a target (for H-C, how many training herbs contain a compound).

## 1. Goal

Predict which links are missing between Ayurvedic herbs, the compounds they contain and the proteins (targets) those compounds act on:
- **C-T**: which targets does a compound act on
- **H-C**: which compounds does a herb contain
- **H-T**: which targets does a herb affect, through its compounds

The main model is a graph neural network (R-GCN). Two models that work from chemistry and protein sequence (similarity kNN and a
two-tower network) and two reference scorers (random and popularity) are run on exactly the same folds for comparison.

## 2. Data

| Item | Count |
|---|---|
| Herbs | 62 (38 plant families) |
| Compounds | 2,319 (726 with at least one known target) |
| Targets (proteins) | 1,341 |
| Herb-compound links | 5,005 |
| Known compound-target links | 14,166 |
| Unlabeled compound-target pairs | 3,095,613 (every other pair) |
| Herb-target pairs (through compounds) | 24,835 |

There are no confirmed negative pairs; every pair that is not a known link is unlabeled (positive-unlabeled data).

Features used:
- **Compounds**: fingerprints computed from each compound's SMILES with RDKit: ECFP4 (2,048 bits) for the graph model and kNN;
  ECFP4 counts, FCFP4 and MACCS keys (4,263 values) plus the herbs the compound is found in, for the two-tower model.
  The ECFP4 values in `compound_features_ecfp4.csv` match these exactly by row order, but that file's id column is in the wrong order,
  so fingerprints are always computed from SMILES.
- **Targets**: ESM-2 protein language model embeddings (1,280 values per protein). Amino acid composition (AAC, 20 values) is used in one
  comparison model.
- **Herbs**: a learned vector plus their plant family.

## 3. Method

### Graph model (R-GCN)
- Nodes: 62 herbs, 2,319 compounds, 1,341 targets (3,722). Four relations: herb to compound, compound to herb, compound to target,
  target to compound. The graph is stored as edge lists per relation (the sparse form of the adjacency matrix).
- Each node's input is projected to 128 numbers. Two R-GCN layers follow: a node's new vector is its own vector times W_self plus, for every
  relation, the average of its neighbours under that relation times that relation's own weight matrix.
- A DistMult head scores a pair: score(c, t) = sum of z_c × r × z_t, with one learned vector r for compound-target and one for herb-compound.
- Loss: BPR, which pushes the score of a true link above the score of a random pair for the same compound (or herb).
- Adam, learning rate 0.001, early stopping on validation MRR.

| Setting | Value |
|---|---|
| Hidden size | 128 |
| R-GCN layers | 2, ReLU between them, dropout 0.2 |
| Optimiser | Adam, learning rate 0.001, weight decay 1e-5 (input layers 1e-4 with fixes) |
| Batch | 1,024 training links |
| Epochs | at most 300, validation MRR checked every 2 epochs, stop after 15 checks without improvement |
| Seeds | fold k uses seed k |
| Hardware  | NVIDIA RTX 4060 |

### The two R-GCN settings and what the fixes are
**R-GCN (plain).** Every training link is used twice: as an edge in the graph (messages pass along it) and as a link the loss asks the
model to score. Because the link being scored is visible in the graph, the model can learn to recognise links it can already see. It then
scores training links very high and new links much lower: this is memorisation. It shows in the results as a large gap between training
and test (H-C training MRR 0.815 against test 0.053).

**R-GCN + fixes.** Three changes, applied during training only. Validation and test always use the full training graph.

| Fix | What happens in every epoch | Why |
|---|---|---|
| 1. Message / supervision split | 30% of the training links are taken out of the graph; the loss is computed only on those 30% | the model is never scored on a link it can see, as on the test set |
| 2. Hide compounds or targets (new-compound and new-target tasks only) | 15% of the training compounds (or targets) lose all their links in the graph, and those links are scored | test compounds (or targets) have no links in the graph, so training should contain that situation too |
| 3. Feature regularisation | dropout 0.3 on the fingerprint input, weight decay 1e-4 on the input layers, a small L2 penalty on the node vectors | the model leans less on memorising individual fingerprints |

With the fixes the training-test gap shrinks (H-C training MRR 0.092 against test 0.079) and
the test results improve on every task where the test units are new to the graph.

### Comparison models
- **Similarity kNN**: score(c, t) = summed Tanimoto similarity of the k most similar training compounds that hit t (k chosen on validation).
- **Two-tower network**: one network for the compound, one for the target, score = scaled cosine similarity plus a per-target bias,
  trained with softmax over all training targets. Run with ESM-2 and, for comparison, with AAC target features.
- **Popularity**: score = how many training compounds hit the target (for H-C: how many training herbs contain the compound).
  It ignores the compound, so a model is only useful if it beats it.
- **Random**: random scores, chance level.

An ensemble of two-tower and kNN was also tested; validation chose two-tower only in every fold, so it is the same as the two-tower model
and is not listed separately.

### Herb-target scores and what the blend is
The models only score compound-target pairs; a herb's score for a target is built from the scores of its compounds, in two ways.

**Model only.** One of six rules over the herb's compounds, chosen on validation: max (the herb's best compound), mean, mean of the top 3,
noisy-OR (the chance that at least one compound hits the target), sum, or the number of the herb's compounds that rank the target in their
own top 50.

**Blend with popularity.** Model-only herb scores lose to popularity, because a herb with many compounds almost always reaches the
proteins that most compounds hit. The blend mixes the two:
1. for each herb, its targets are ranked by the model-only score and turned into a rank between 0 (worst) and 1 (best);
2. every target also gets a popularity rank between 0 and 1, from how many training compounds hit it;
3. herb score = (1 - w) × model rank + w × popularity rank.

w is chosen on validation from 0, 0.1, 0.25, 0.5, 0.75, 0.9 and 1, together with the rule. w = 0 is the model alone and w = 1 is popularity
alone. For example, with w = 0.75 a target that the model ranks first but that few compounds hit can still rise above a very popular
target the model ranks low. Popularity is counted from the fold's training links only and w is never chosen on test data, so the blend
does not leak. Random and popularity themselves are never blended. The report shows both versions: "model only" (what the model does by
itself) and "blend" (model plus popularity); the chosen w for every fold is listed in the results.

## 4. Evaluation

### Four tasks
| Task | What is held out | Question |
|---|---|---|
| C-T, known compounds | C-T links | find missing targets of compounds that already have known targets |
| C-T, new compounds | compounds with all their C-T links | predict targets of a compound the model has never seen linked |
| C-T, new targets | targets with all their C-T links | predict compounds for a protein the model has never seen linked |
| H-C | H-C links | find missing compounds of a herb |

H-T is evaluated inside the two compound tasks.

### Cross validation
- 5 folds. The links (or compounds, or targets) are spread over 5 parts of about 20% of the links each. In fold k, part k is the test set;
  from the rest, 15% of all links go to validation and the remainder to training (about 65 / 15 / 20). Every link, compound or target is in
  the test set exactly once, and every fold trains new models from scratch.
- Results are the mean ± standard deviation over the 5 folds.

### Negatives
- Drawn at random, without replacement, from the 3.1 million unlabeled pairs (H-C: from herb-compound pairs without a link).
  Train and validation get 10 negatives per positive, test gets 50. Train, validation and test never share a negative.
- Known compounds: negatives only from the 726 compounds with known targets. New compounds: only from that part's compounds.
  New targets: only from that part's targets, and ranking is only among that part's targets.
- One extra negative per positive, for the same compound or herb, gives a loss value on train, validation and test.

### Leakage controls (all checked in `02_prep.ipynb`, all passed)
- Only training links of the predicted type are in the graph.
- Test data is never used to choose anything; early stopping, k and the herb aggregation are chosen on validation.
- Compounds with an identical fingerprint or the same InChIKey first block (stereoisomers) always land in the same part, so a test compound
  never has a twin in training (470 compounds share a fingerprint).
- No compound or target is in two parts; held-out targets are never used as training negatives.
- A herb-target pair counts as test only if it is reachable through test links and not through any training or validation link.

### Metrics
- **PR-AUC**: positives against the sampled negatives (random gives about 0.02 at 1:50)
- **Hits@10 / Hits@50**: share of true links ranked in the top 10 / 50 of their compound (or herb), with the other known links left out of the ranking
- **MRR**: mean of 1 / rank
- **loss**: BPR loss on the part's positives and paired negatives (trained models only)
- **HT ...**: the same for herb-target pairs

## 5. Results

### 5.1 C-T, known compounds
Test set: about 2,833 positive links and 141,660 negatives per fold;
2,553 herb-target test pairs per fold.

**Compound-target (test)**

| Model | PR-AUC | Hits@10 | Hits@50 | MRR | loss |
|---|---|---|---|---|---|
| Random | 0.020 ± 0.001 | 0.009 ± 0.001 | 0.042 ± 0.003 | 0.006 ± 0.000 | n/a |
| Popularity | 0.113 ± 0.004 | 0.153 ± 0.007 | 0.420 ± 0.009 | 0.069 ± 0.004 | n/a |
| R-GCN | 0.135 ± 0.016 | 0.446 ± 0.004 | 0.666 ± 0.006 | 0.294 ± 0.012 | 0.329 ± 0.026 |
| R-GCN + fixes | 0.219 ± 0.027 | 0.412 ± 0.024 | 0.669 ± 0.014 | 0.268 ± 0.019 | 0.166 ± 0.011 |
| kNN | 0.186 ± 0.009 | 0.208 ± 0.011 | 0.462 ± 0.011 | 0.107 ± 0.008 | n/a |
| Two-tower (AAC) | 0.216 ± 0.016 | 0.426 ± 0.016 | 0.621 ± 0.014 | 0.285 ± 0.012 | 0.248 ± 0.011 |
| Two-tower (ESM-2) | 0.238 ± 0.015 | 0.510 ± 0.012 | 0.729 ± 0.003 | 0.342 ± 0.015 | 0.168 ± 0.009 |

**Herb-target, blend with popularity (test)**

| Model | HT PR-AUC | HT Hits@10 | HT Hits@50 | HT MRR | popularity weight w (5 folds) |
|---|---|---|---|---|---|
| Random | 0.049 ± 0.005 | 0.011 ± 0.002 | 0.056 ± 0.005 | 0.009 ± 0.002 | not blended |
| Popularity | 0.099 ± 0.011 | 0.146 ± 0.017 | 0.303 ± 0.024 | 0.073 ± 0.010 | not blended |
| R-GCN | 0.101 ± 0.005 | 0.149 ± 0.017 | 0.311 ± 0.021 | 0.074 ± 0.009 | 0.9, 0.75, 0.9, 0.75, 0.75 |
| R-GCN + fixes | 0.105 ± 0.008 | 0.153 ± 0.020 | 0.322 ± 0.023 | 0.078 ± 0.009 | 0.75, 0.9, 0.75, 0.75, 0.75 |
| kNN | 0.093 ± 0.008 | 0.147 ± 0.020 | 0.306 ± 0.025 | 0.073 ± 0.011 | 0.9, 0.9, 0.9, 0.9, 1 |
| Two-tower (AAC) | 0.095 ± 0.007 | 0.148 ± 0.017 | 0.306 ± 0.023 | 0.074 ± 0.010 | 0.9, 0.9, 0.9, 0.9, 0.9 |
| Two-tower (ESM-2) | 0.095 ± 0.007 | 0.148 ± 0.018 | 0.310 ± 0.021 | 0.076 ± 0.010 | 0.9, 0.9, 0.9, 0.9, 0.9 |

**Herb-target, model only (test)**

| Model | HT PR-AUC | HT Hits@10 | HT Hits@50 | HT MRR |
|---|---|---|---|---|
| Random | 0.049 ± 0.005 | 0.011 ± 0.002 | 0.056 ± 0.005 | 0.009 ± 0.002 |
| Popularity | 0.099 ± 0.011 | 0.146 ± 0.017 | 0.303 ± 0.024 | 0.073 ± 0.010 |
| R-GCN | 0.081 ± 0.009 | 0.103 ± 0.011 | 0.282 ± 0.025 | 0.049 ± 0.007 |
| R-GCN + fixes | 0.117 ± 0.024 | 0.088 ± 0.009 | 0.249 ± 0.020 | 0.052 ± 0.006 |
| kNN | 0.131 ± 0.023 | 0.100 ± 0.014 | 0.282 ± 0.034 | 0.051 ± 0.009 |
| Two-tower (AAC) | 0.075 ± 0.007 | 0.102 ± 0.011 | 0.246 ± 0.025 | 0.050 ± 0.006 |
| Two-tower (ESM-2) | 0.092 ± 0.010 | 0.114 ± 0.018 | 0.297 ± 0.018 | 0.056 ± 0.009 |

### 5.2 C-T, new compounds
Test set: about 2,833 positive links and 141,660 negatives per fold;
2,501 herb-target test pairs per fold.

**Compound-target (test)**

| Model | PR-AUC | Hits@10 | Hits@50 | MRR | loss |
|---|---|---|---|---|---|
| Random | 0.020 ± 0.001 | 0.009 ± 0.002 | 0.043 ± 0.007 | 0.006 ± 0.001 | n/a |
| Popularity | 0.114 ± 0.015 | 0.157 ± 0.022 | 0.425 ± 0.052 | 0.069 ± 0.008 | n/a |
| R-GCN | 0.045 ± 0.009 | 0.073 ± 0.016 | 0.236 ± 0.021 | 0.036 ± 0.008 | 0.524 ± 0.050 |
| R-GCN + fixes | 0.079 ± 0.015 | 0.142 ± 0.021 | 0.376 ± 0.041 | 0.072 ± 0.010 | 0.416 ± 0.093 |
| kNN | 0.188 ± 0.033 | 0.217 ± 0.040 | 0.482 ± 0.065 | 0.106 ± 0.020 | n/a |
| Two-tower (AAC) | 0.093 ± 0.025 | 0.212 ± 0.033 | 0.383 ± 0.045 | 0.134 ± 0.015 | 0.496 ± 0.084 |
| Two-tower (ESM-2) | 0.130 ± 0.028 | 0.229 ± 0.045 | 0.439 ± 0.060 | 0.140 ± 0.022 | 0.423 ± 0.085 |

**Herb-target, blend with popularity (test)**

| Model | HT PR-AUC | HT Hits@10 | HT Hits@50 | HT MRR | popularity weight w (5 folds) |
|---|---|---|---|---|---|
| Random | 0.044 ± 0.017 | 0.011 ± 0.004 | 0.057 ± 0.008 | 0.009 ± 0.002 | not blended |
| Popularity | 0.090 ± 0.029 | 0.141 ± 0.042 | 0.310 ± 0.061 | 0.068 ± 0.016 | not blended |
| R-GCN | 0.092 ± 0.026 | 0.141 ± 0.038 | 0.307 ± 0.053 | 0.067 ± 0.018 | 0.9, 0.9, 1, 0.5, 1 |
| R-GCN + fixes | 0.094 ± 0.026 | 0.142 ± 0.038 | 0.307 ± 0.057 | 0.067 ± 0.015 | 0.9, 0.9, 1, 0.9, 0.9 |
| kNN | 0.090 ± 0.027 | 0.143 ± 0.038 | 0.313 ± 0.049 | 0.067 ± 0.017 | 1, 0.1, 1, 0.25, 0.75 |
| Two-tower (AAC) | 0.093 ± 0.025 | 0.135 ± 0.039 | 0.300 ± 0.063 | 0.068 ± 0.017 | 1, 1, 0.9, 1, 0.5 |
| Two-tower (ESM-2) | 0.092 ± 0.026 | 0.139 ± 0.043 | 0.300 ± 0.062 | 0.068 ± 0.017 | 1, 1, 0.9, 1, 0.25 |

**Herb-target, model only (test)**

| Model | HT PR-AUC | HT Hits@10 | HT Hits@50 | HT MRR |
|---|---|---|---|---|
| Random | 0.044 ± 0.017 | 0.011 ± 0.004 | 0.057 ± 0.008 | 0.009 ± 0.002 |
| Popularity | 0.090 ± 0.029 | 0.141 ± 0.042 | 0.310 ± 0.061 | 0.068 ± 0.016 |
| R-GCN | 0.073 ± 0.018 | 0.056 ± 0.012 | 0.189 ± 0.034 | 0.029 ± 0.005 |
| R-GCN + fixes | 0.081 ± 0.024 | 0.074 ± 0.028 | 0.218 ± 0.056 | 0.036 ± 0.012 |
| kNN | 0.109 ± 0.071 | 0.091 ± 0.035 | 0.277 ± 0.069 | 0.049 ± 0.023 |
| Two-tower (AAC) | 0.068 ± 0.011 | 0.081 ± 0.027 | 0.190 ± 0.034 | 0.047 ± 0.020 |
| Two-tower (ESM-2) | 0.083 ± 0.012 | 0.097 ± 0.021 | 0.255 ± 0.040 | 0.054 ± 0.009 |

### 5.3 C-T, new targets
Test set: about 2,833 positive links and 141,660 negatives per fold. Ranking is among the
held-out targets only (about 270), so random already reaches Hits@10 of 0.041. Similarity kNN cannot score a
protein with no training links.

| Model | PR-AUC | Hits@10 | Hits@50 | MRR | loss |
|---|---|---|---|---|---|
| Random | 0.020 ± 0.000 | 0.041 ± 0.003 | 0.197 ± 0.002 | 0.026 ± 0.002 | n/a |
| Popularity | 0.020 ± 0.002 | 0.033 ± 0.016 | 0.218 ± 0.042 | 0.026 ± 0.009 | n/a |
| R-GCN | 0.062 ± 0.017 | 0.197 ± 0.029 | 0.487 ± 0.020 | 0.093 ± 0.019 | 0.652 ± 0.047 |
| R-GCN + fixes | 0.140 ± 0.021 | 0.335 ± 0.035 | 0.630 ± 0.055 | 0.157 ± 0.017 | 0.522 ± 0.101 |
| kNN | 0.020 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.008 ± 0.000 | n/a |
| Two-tower (AAC) | 0.080 ± 0.025 | 0.248 ± 0.027 | 0.495 ± 0.014 | 0.124 ± 0.014 | 0.660 ± 0.047 |
| Two-tower (ESM-2) | 0.192 ± 0.029 | 0.375 ± 0.033 | 0.657 ± 0.026 | 0.206 ± 0.017 | 0.535 ± 0.063 |

### 5.4 H-C (herb-compound)
Test set: about 1,001 positive links and 50,050 negatives per fold. Each herb's hidden
compounds are ranked among all 2,319 compounds.

| Model | PR-AUC | Hits@10 | Hits@50 | MRR | loss |
|---|---|---|---|---|---|
| Random | 0.020 ± 0.000 | 0.004 ± 0.002 | 0.024 ± 0.005 | 0.004 ± 0.001 | n/a |
| Popularity | 0.086 ± 0.004 | 0.106 ± 0.008 | 0.237 ± 0.005 | 0.052 ± 0.004 | n/a |
| R-GCN | 0.105 ± 0.016 | 0.110 ± 0.012 | 0.261 ± 0.018 | 0.053 ± 0.006 | 0.746 ± 0.067 |
| R-GCN + fixes | 0.136 ± 0.026 | 0.155 ± 0.011 | 0.371 ± 0.012 | 0.079 ± 0.006 | 0.289 ± 0.012 |

### 5.5 Train, validation and test
MRR and loss for every part show how much each model memorises its training links. For known compounds and H-C, validation and test
are almost equal. For new compounds and new targets, validation is somewhat higher than test: early stopping picks the epoch that is best
on those same validation compounds or targets, which makes the validation number slightly optimistic. The test numbers are the ones to
report, and the test set is never used for any choice.

**C-T, known compounds: MRR**

| Model | train MRR | val MRR | test MRR |
|---|---|---|---|
| Random | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.006 ± 0.000 |
| Popularity | 0.072 ± 0.002 | 0.069 ± 0.001 | 0.069 ± 0.004 |
| R-GCN | 0.530 ± 0.048 | 0.295 ± 0.019 | 0.294 ± 0.012 |
| R-GCN + fixes | 0.331 ± 0.021 | 0.271 ± 0.021 | 0.268 ± 0.019 |
| kNN | 0.506 ± 0.089 | 0.108 ± 0.006 | 0.107 ± 0.008 |
| Two-tower (AAC) | 0.668 ± 0.047 | 0.285 ± 0.007 | 0.285 ± 0.012 |
| Two-tower (ESM-2) | 0.770 ± 0.045 | 0.342 ± 0.003 | 0.342 ± 0.015 |

**C-T, known compounds: loss**

| Model | train loss | val loss | test loss |
|---|---|---|---|
| R-GCN | 0.025 ± 0.004 | 0.327 ± 0.048 | 0.329 ± 0.026 |
| R-GCN + fixes | 0.079 ± 0.013 | 0.166 ± 0.014 | 0.166 ± 0.011 |
| Two-tower (AAC) | 0.066 ± 0.016 | 0.253 ± 0.010 | 0.248 ± 0.011 |
| Two-tower (ESM-2) | 0.024 ± 0.008 | 0.167 ± 0.008 | 0.168 ± 0.009 |

**C-T, new compounds: MRR**

| Model | train MRR | val MRR | test MRR |
|---|---|---|---|
| Random | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.006 ± 0.001 |
| Popularity | 0.073 ± 0.004 | 0.067 ± 0.009 | 0.069 ± 0.008 |
| R-GCN | 0.347 ± 0.143 | 0.050 ± 0.016 | 0.036 ± 0.008 |
| R-GCN + fixes | 0.287 ± 0.043 | 0.082 ± 0.010 | 0.072 ± 0.010 |
| kNN | 0.628 ± 0.142 | 0.125 ± 0.038 | 0.106 ± 0.020 |
| Two-tower (AAC) | 0.549 ± 0.034 | 0.172 ± 0.033 | 0.134 ± 0.015 |
| Two-tower (ESM-2) | 0.610 ± 0.043 | 0.178 ± 0.037 | 0.140 ± 0.022 |

**C-T, new targets: MRR**

| Model | train MRR | val MRR | test MRR |
|---|---|---|---|
| Random | 0.009 ± 0.000 | 0.032 ± 0.004 | 0.026 ± 0.002 |
| Popularity | 0.093 ± 0.006 | 0.023 ± 0.005 | 0.026 ± 0.009 |
| R-GCN | 0.366 ± 0.104 | 0.163 ± 0.034 | 0.093 ± 0.019 |
| R-GCN + fixes | 0.339 ± 0.102 | 0.235 ± 0.063 | 0.157 ± 0.017 |
| kNN | 0.722 ± 0.005 | 0.011 ± 0.002 | 0.008 ± 0.000 |
| Two-tower (AAC) | 0.490 ± 0.169 | 0.176 ± 0.041 | 0.124 ± 0.014 |
| Two-tower (ESM-2) | 0.739 ± 0.113 | 0.274 ± 0.061 | 0.206 ± 0.017 |

**H-C: MRR**

| Model | train MRR | val MRR | test MRR |
|---|---|---|---|
| Random | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.004 ± 0.001 |
| Popularity | 0.065 ± 0.002 | 0.054 ± 0.005 | 0.052 ± 0.004 |
| R-GCN | 0.815 ± 0.040 | 0.058 ± 0.008 | 0.053 ± 0.006 |
| R-GCN + fixes | 0.092 ± 0.006 | 0.089 ± 0.008 | 0.079 ± 0.006 |

**H-C: loss**

| Model | train loss | val loss | test loss |
|---|---|---|---|
| R-GCN | 0.004 ± 0.003 | 0.739 ± 0.042 | 0.746 ± 0.067 |
| R-GCN + fixes | 0.322 ± 0.022 | 0.297 ± 0.013 | 0.289 ± 0.012 |

Full train / validation / test tables for every metric are in `results/summary_<task>.csv`.

## 6. Findings

1. **Known compounds.** The R-GCN reaches test MRR 0.294 and Hits@10 0.446,
   against 0.069 and 0.153 for popularity. The two-tower model with ESM-2
   is the best model: MRR 0.342, Hits@10 0.510, Hits@50 0.729.
2. **New compounds.** The plain R-GCN falls below popularity (MRR 0.036 against
   0.069), because it learns each compound from its links and a new compound has none. The fixes bring it to
   0.072. The two-tower model reaches 0.140 (Hits@10 0.229),
   twice popularity.
3. **New targets.** R-GCN + fixes reaches MRR 0.157 and the two-tower model 0.206, against
   0.026 for random. ESM-2 is clearly better than AAC here (two-tower MRR 0.206 against
   0.124).
4. **Herb-compound.** R-GCN + fixes is the best: MRR 0.079, Hits@10 0.155, against
   0.052 and 0.106 for popularity.
5. **Overfitting fixes.** They lower the gap between training and test (H-C training MRR 0.815 without
   fixes, 0.092 with fixes) and improve every task where the test units are new (new compounds, new targets,
   H-C).
6. **Herb-target from compound scores alone loses to popularity on ranking.** Model-only herb scores stay below popularity on HT
   Hits@10, Hits@50 and MRR for every model (known compounds: best HT Hits@10 0.114 against
   0.146; new compounds: best 0.097 against
   0.141). On HT PR-AUC some models are above popularity (known compounds: kNN
   0.131 and R-GCN + fixes 0.117 against
   0.099). The extra rules (sum, top 50 count) were rarely chosen on validation and did not change this.
7. **Blending with popularity gives a small, consistent gain for known compounds.** R-GCN + fixes with the blend reaches HT Hits@10
   0.153, Hits@50 0.322 and MRR 0.078, against
   0.146, 0.303 and 0.073
   for popularity. It is better than popularity in 5 of 5 folds on Hits@10,
   5 of 5 on Hits@50 and 4 of 5 on MRR. Validation chose
   w between 0.75 and 0.9, so popularity carries most of the herb-level signal and the model adds a smaller part on top.
8. **For herbs through new compounds the blend only matches popularity** (HT MRR 0.067 for R-GCN + fixes
   against 0.068); validation chose w of 0.9 or 1 in most folds, so the model adds almost nothing
   here. Herb-target prediction remains the main open
   problem; the next step is to train the graph model directly on herb-target pairs (a herb-to-target head trained together with the
   compound-target head), instead of building herb scores afterwards.

## 7. Predictions

`06_predictions.ipynb` uses the models of all 5 folds: R-GCN + fixes and two-tower ESM-2 for C-T (averaged by rank within each compound,
so the C-T and H-T lists combine the GNN with the non-graph two-tower model) and R-GCN + fixes alone for H-C. First, each fold's models are checked on that fold's hidden test links. Then the 5 fold models are averaged
to list new links that are not known anywhere; each fold model was trained on a different 65% of the links, so together they use all
of them. Every candidate also gets a count of how many of the 5 fold models put it in their own top 10, which shows how stable it is.
Scores rank targets (or compounds) within one compound (or herb); they are not probabilities, so the lists are per compound and per herb.

### 7.1 Do the models find hidden links?
For every test compound of every fold, the top 10 targets from that fold's models (training links left out) were compared with the
links hidden from them. "Expected by chance" is the number of test links a random top 10 would contain.

| Fold | test compounds | test hits in top 10 | expected by chance | validation links in top 10 | unlabeled in top 10 | compounds with at least one test hit |
|---|---|---|---|---|---|---|
| 0 | 470 | 1.76 | 0.05 | 1.10 | 7.15 | 68% |
| 1 | 456 | 1.77 | 0.05 | 1.13 | 7.09 | 69% |
| 2 | 463 | 1.65 | 0.05 | 1.17 | 7.18 | 66% |
| 3 | 459 | 1.74 | 0.05 | 1.23 | 7.03 | 70% |
| 4 | 459 | 1.67 | 0.05 | 1.19 | 7.14 | 66% |
| mean |  | 1.72 | 0.05 | 1.16 | 7.12 | 68% |

On average 1.72 of the top 10 are hidden test links, against 0.05 by chance, and another
1.16 are validation links (also real, also hidden from training). 68% of
test compounds have at least one hidden test link in their top 10. The three test compounds of fold 0 with the most hidden links:

| Compound | hidden test targets | test hits in top 10 | validation links in top 10 | unlabeled in top 10 | test hits found |
|---|---|---|---|---|---|
| Quercetin | 79 | 7 | 2 | 1 | MAPK3, NPY1R, HTR2A, SLC6A2, AKR1B1, CYP1A2, CHRM3 |
| Apigenin | 58 | 5 | 5 | 0 | PIM1, ATAD5, SLCO2B1, GABRG3, AURKB |
| Genistein | 51 | 5 | 5 | 0 | SLC6A3, CYP2E1, PDE5A, OPRM1, CYP2C19 |

The same ranking that recovers hidden links produces the candidate lists below, which is why high scoring unknown pairs are worth
following up.

### 7.2 How stable are the candidates?

| List | candidates | found by 5 of 5 fold models | found by at least 3 of 5 |
|---|---|---|---|
| C-T (top 5 per compound) | 11,595 | 29% | 85% |
| H-T (top 10 per herb) | 620 | 19% | 60% |
| H-C (top 10 per herb) | 620 | 2% | 33% |

### 7.3 New compound-target candidates
`results/candidates_ct.csv` holds the 5 best new targets for every one of the 2,319 compounds (11,595 rows), with the number of
known targets of each compound and the fold agreement. Examples for well-known phytochemicals:

| Compound | known targets | top 5 new targets (folds in top 10) |
|---|---|---|
| Quercetin | 354 | NFE2L2 (4/5), FYN (4/5), ERBB2 (4/5), IDH1 (2/5), ATXN2 (2/5) |
| Curcumin | 193 | ATXN2 (4/5), AKR1B1 (4/5), SLCO1B1 (3/5), PTPN1 (4/5), CBX1 (4/5) |
| Withaferin A | 14 | NFKBIA (3/5), CYP1A2 (4/5), PRKCA (3/5), NFE2L2 (1/5), AKT2 (3/5) |
| Piperine | 39 | GMNN (5/5), USP1 (5/5), MAPT (5/5), NFKB1 (4/5), TP53 (2/5) |
| Andrographolide | 15 | NFE2L2 (5/5), ALDH1A1 (4/5), TP53 (4/5), JUN (4/5), APEX1 (4/5) |
| Eugenol | 113 | HTR1A (5/5), PGR (4/5), HDAC6 (5/5), PDE3A (3/5), HRH3 (4/5) |
| Berberine | 35 | NFE2L2 (4/5), TDP1 (3/5), AKR1B1 (3/5), ALOX5 (3/5), GMNN (3/5) |
| Asiatic acid | 20 | PTPN6 (5/5), EGLN1 (5/5), PTPN11 (5/5), PTPRE (4/5), EGLN2 (4/5) |

For the 1,593 compounds with no known target at all, these lists are the only target hypotheses available;
they come from chemistry alone and are the least certain.

### 7.4 New herb-target and herb-compound candidates
`results/candidates_ht.csv` holds the 10 best new targets for every herb (herb-target pairs with no path through any known link),
using the model-only rule (max), so popularity does not decide the list. `results/candidates_hc.csv` holds the 10 best new compounds
for every herb.

| Herb | compounds | top 10 new targets (folds in top 10) | top 5 new compounds (folds in top 10) |
|---|---|---|---|
| Withania somnifera | 79 | UGT1A6 (3/5), FGFR3 (4/5), UGT1A7 (1/5), HSPD1 (3/5), CA4 (1/5), HSPA9 (3/5), ALOX12 (0/5), POLH (1/5), BRCA1 (0/5), SNCA (1/5) | Lupeol (2/5), beta-Sitosterol-beta-D-glucosi (2/5), Campesterol (1/5), Quercetin (1/5), D-Galactose (1/5) |
| Curcuma longa | 189 | GABRB3 (5/5), UGT1A7 (5/5), GABRG1 (5/5), GABRB1 (4/5), PTPN6 (4/5), GABRA3 (5/5), PTPN11 (4/5), GABRA5 (3/5), PTPRE (4/5), DUSP3 (3/5) | alpha-Copaene (3/5), Elemol (1/5), (+)-gamma-Cadinene (2/5), gamma-Eudesmol (1/5), gamma-Muurolene (1/5) |
| Zingiber officinale | 343 | UGT1A7 (5/5), CYP2A13 (4/5), UGT2B4 (4/5), CYP2C18 (2/5), CYP2A7 (2/5), UGT2B10 (1/5), CYP2F1 (1/5), CYP1A1 (2/5), RELA (1/5), ATXN2 (2/5) | (-)-alpha-Cadinol (4/5), 2-Cyclohexen-1-ol, 1-methyl-4- (3/5), trans-Sabinene hydrate (3/5), Guaiol (2/5), (Z)-alpha-Bisabolene (2/5) |
| Ocimum tenuiflorum | 257 | UGT2B4 (4/5), UGT2B10 (4/5), AKT2 (2/5), TARDBP (2/5), GABRR3 (0/5), GABRR2 (2/5), PRKCI (1/5), PRKCH (0/5), GPR84 (2/5), PRKCG (1/5) | (1S,2S,6S,7R,8R)-1,3-dimethyl- (4/5), delta-Cadinol (3/5), 3-Carene (3/5), Cedrol (3/5), Copaene (2/5) |
| Bacopa monnieri | 35 | PTPN2 (4/5), PTPN6 (3/5), USP7 (2/5), TRPV1 (3/5), DUSP3 (1/5), KDM1A (2/5), PTPN11 (0/5), CES2 (1/5), CES1 (2/5), PTPRF (0/5) | Lupeol (4/5), Palmitic acid (3/5), Stearic acid (3/5), Quercetin (4/5), Oleic acid (1/5) |
| Glycyrrhiza glabra | 312 | PTPRE (5/5), UGT1A7 (5/5), PTPRA (4/5), UGT2B7 (4/5), PTPRF (3/5), USP7 (3/5), GABRR2 (1/5), CHRNB4 (3/5), GABRR1 (1/5), GABRR3 (0/5) | Palmitic acid (2/5), Arachidic acid (2/5), Linoleic acid (0/5), Docosanoic acid (1/5), Stearic acid (1/5) |

The lists are built from the average of the 5 fold models. A candidate can therefore be in the averaged top 10 while no single fold model
had it in its own top 10 (shown as 0/5): it was ranked fairly high, but not top 10, by most of them. Candidates with 4/5 or 5/5 are the
most reliable.

Herb-level predictions come in protein families rather than single proteins, because related proteins have similar ESM-2 features: GABA-A receptor subunits for *Curcuma longa* (GABRB3, GABRG1, GABRB1, GABRA3, GABRA5), UGT and cytochrome P450 enzymes for *Zingiber officinale*, protein tyrosine phosphatases for *Bacopa monnieri* and *Glycyrrhiza glabra*, AKT and protein kinase C isoforms for *Ocimum tenuiflorum*, and heat shock proteins (HSPD1, HSPA9) and FGFR3 for *Withania somnifera*. UGT and cytochrome P450 enzymes appear for many herbs (UGT1A7 is in the top 10 of 32% of herbs); these enzymes metabolise a wide range of plant compounds, so they are expected to be shared across herbs. Each herb-level candidate is a hypothesis until it is checked in the literature. Herb-target candidates are less stable than compound-target candidates (19% found by all 5 fold models against 29%).

### 7.5 What the predictions show about the data
**Popular targets.** The same few proteins appear in the top 5 of many compounds:

| Target | share of compounds with known targets | share of compounds without known targets |
|---|---|---|
| NFE2L2 | 41% | 65% |
| TDP1 | 35% | 49% |
| HDAC6 | 29% | 54% |
| ALDH1A1 | 28% | 24% |
| AR | 24% | 48% |
| PPARD | 18% | 18% |
| GMNN | 14% | 6% |
| SLCO1B1 | 14% | 46% |

Only 339 of the 1,341 targets appear anywhere in the compound top 5 lists. The most likely reason is how the known links
were collected: many links in ChEMBL and NPASS come from large screening panels that test thousands of natural products against the same
proteins, so these proteins look popular in the data and the models inherit that. For compounds with no known target the bias is
stronger, because the model has nothing else to go on. This is also why herb-target prediction stays close to the popularity baseline:
a herb with many compounds almost always reaches these heavily screened proteins.

**Missing herb-compound annotations.** The new H-C candidates are mostly common plant metabolites:

| Compound | share of herbs where it is a top 10 new compound |
|---|---|
| Daucosterol | 53% |
| Stigmasterol | 45% |
| Stearic acid | 40% |
| Palmitic acid | 40% |
| Kaempferol | 37% |
| Quercetin | 37% |
| Linoleic acid | 34% |
| Oleic acid | 34% |

These compounds (plant sterols such as daucosterol and stigmasterol, fatty acids, kaempferol, quercetin, lupeol) occur in many plants, so the H-C predictions mainly point to compounds a herb probably contains but that are not recorded for it in IMPPAT. For herbs with a characteristic chemistry the candidates follow it, for example sesquiterpenes (copaene, elemol, cadinene, eudesmol) for *Curcuma longa* and *Zingiber officinale*. H-C candidates are the least stable of the three lists (only 2% are found by all 5 fold models), so they are more useful for pointing at likely gaps in the data than as individual predictions.


