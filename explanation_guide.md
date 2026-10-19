# HTI GNN Baseline: Explanation Guide

A complete walk through the project: the problem, the data, the model, how it is tested, what every notebook and cell does,
what the results mean, what is weak about it, and answers to likely questions. Numbers are from the latest run
(test set, mean over 5 folds) unless stated otherwise.

---

## 1. The problem

Ayurvedic herbs contain chemical compounds, and compounds act on proteins in the body (targets):

```
Herb  --contains-->  Compound  --acts on-->  Target (protein)
```

Many of these links are unknown, and testing every compound against every protein in a lab is impossible
(2,319 compounds × 1,341 proteins = 3.1 million pairs). The project trains models on the known links to predict which unknown links
are likely real, so researchers know what to test first.

Three things are predicted:

| Short name | Question |
|---|---|
| C-T (compound-target) | Which proteins does a compound act on? |
| H-C (herb-compound) | Which compounds does a herb contain? |
| H-T (herb-target) | Which proteins does a herb affect, through its compounds? |

In one sentence: herb-target interaction is treated as **link prediction on a graph** of herbs, compounds and proteins.

---

## 2. The data

Source: IMPPAT.

| File | What it holds |
|---|---|
| `herbs.tsv` | 62 herbs (names, plant family) |
| `compounds.tsv` | 2,319 compounds, including SMILES (a text formula of the molecule) and InChIKey |
| `herb_compound_edges.tsv` | 8,396 rows, 5,005 unique herb-compound links (one row per plant part) |
| `compound_target_labeled.csv` | 14,166 known compound-target links |
| `compound_target_unlabeled.csv` | 3,095,613 pairs with no known link |
| `herb_compound_target.csv` | 24,835 herb-target pairs, derived through compounds |
| `target_features_aac.csv` | amino acid composition of each protein (20 values) |
| `target_features_esm2.csv` | ESM-2 embedding of each protein (1,280 values) |
| `compound_features_ecfp4.csv` | compound fingerprints (has an id defect, see below) |

Three facts that shape the whole project:

1. **Positive-unlabeled (PU) data.** There are no confirmed "does not bind" pairs. A pair that is not in the known list is untested,
   not proven negative. This decides the loss (BPR), how negatives are drawn and which metrics are used.
2. **Only 726 of the 2,319 compounds have any known target.** The other 1,593 have none.
3. **The fingerprint file has a defect.** Its `compound_id` column is in the wrong order (2,317 of 2,319 rows carry another compound's id),
   while the fingerprint rows themselves are in sorted id order. Fingerprints are therefore recomputed from SMILES with RDKit, and
   `01_data_check` proves they equal the file's rows exactly.

---

## 3. Turning the data into numbers (features)

A model cannot read names, so every node is described by numbers.

- **Compounds**: fingerprints computed from SMILES with RDKit.
  - **ECFP4**: 2,048 yes/no switches; each one says whether the molecule contains a certain small chemical piece. Similar molecules
    have similar switches on. Used by the R-GCN and by kNN.
  - For the two-tower model also **ECFP4 counts, FCFP4** (pieces described by chemical role) and **MACCS keys** (166 standard chemical
    questions): 4,263 numbers, plus which herbs the compound is found in.
- **Proteins**: **ESM-2**, a protein language model from Meta trained on millions of protein sequences. It turns a protein sequence into
  1,280 numbers that capture its family, structure and function. **AAC** (share of each of the 20 amino acids) is much weaker and is used
  only as a comparison.
- **Herbs**: no chemical description exists, so each herb gets a **learned vector** (numbers the model adjusts during training) plus its
  **plant family** (38 families).

---

## 4. The graph

- **Nodes**: 62 herbs + 2,319 compounds + 1,341 targets = **3,722 nodes**.
- **Four link types (relations)**: herb to compound, compound to herb, compound to target, target to compound. Each link type exists in
  both directions so information can flow both ways.
- **Storage**: edge lists (for each relation, an array of "from" nodes and an array of "to" nodes). This is the sparse form of an
  adjacency matrix: the same maths without storing 13.8 million mostly-zero cells.

---

## 5. The GNN model: R-GCN

### 5.1 Same size for every node
Compounds start with 2,048 numbers, proteins with 1,280, herbs with 128. A small learned layer turns each into **128 numbers**.

### 5.2 Message passing (the graph part)
In each layer every node updates its vector from its neighbours:

```
new vector = W_self × (own vector)
           + W_r × (average of the neighbours under relation r), for every relation r
```

- **Relational** (the R in R-GCN): every link type has its own weight matrix, because "contains" and "acts on" mean different things.
  A plain GCN would treat all links the same.
- **Average, not sum**, so a herb with 343 compounds does not get huge numbers.
- **Two layers**: information travels two steps (for example herb to compound to target), so a compound knows about its herbs, its
  targets, and other compounds that share them.

The output is a final 128-number vector (embedding) for every node.

### 5.3 Scoring a pair: DistMult

```
score(c, t) = sum over the 128 positions of  z_c × r × z_t
```

r is a learned vector for "compound acts on target" (a second r scores "herb contains compound"). A higher score means a more likely
link. Scores are used for ranking only; they are not probabilities.

### 5.4 Learning: BPR loss
Without true negatives the model is not asked "is this link true?" but "does the true target score higher than a random target for
the same compound?":

```
loss = -log sigmoid( score(c, true target) - score(c, random target) )
```

When both scores are equal the loss is 0.69 (where training starts); when the true one is clearly higher the loss is near 0.
The random target is drawn fresh each time from everything that is not a training link of that compound. Validation and test links can
be drawn: during training they are unknown, and excluding them would secretly tell the model where the answers are.

### 5.5 Training settings

| Setting | Value |
|---|---|
| Hidden size | 128 |
| Layers | 2 R-GCN layers, ReLU between them, dropout 0.2 |
| Optimiser | Adam, learning rate 0.001, weight decay 1e-5 (input layers 1e-4 with fixes) |
| Batch | 1,024 training links |
| Epochs | at most 300; validation MRR checked every 2 epochs; stop after 15 checks without improvement; keep the best weights |
| Time | about 10 seconds per model on an RTX 4060 (the graph is small) |

### 5.6 Plain R-GCN and R-GCN + fixes

**The problem with the plain model.** Every training link is used twice: as an edge in the graph and as a link the loss scores. The model
can see the link it is scored on, so it memorises training links. Evidence (H-C): training MRR 0.815 against test 0.053.

**The fixes** (training only; validation and test always use the full training graph):

| Fix | What happens every epoch | Why |
|---|---|---|
| 1. Message / supervision split | 30% of the training links are removed from the graph and only those are scored | the model is never scored on a link it can see, as on the test set |
| 2. Hide compounds or targets (new-compound and new-target tasks) | 15% of training compounds (or targets) lose all their links | test compounds have no links, so training contains that situation too |
| 3. Feature regularisation | dropout 0.3 on fingerprints, more weight decay, small L2 penalty on node vectors | less memorising of individual fingerprints |

With the fixes the gap shrinks (H-C training MRR 0.092 against test 0.079) and results improve on every task where the test items are new.

---

## 6. Comparison models and baselines

A GNN number means nothing without a reference point.

| Model | How it works | Role |
|---|---|---|
| Random | random scores | chance level; also checks the metrics are correct |
| Popularity | score = how many training compounds hit the target (H-C: how many training herbs contain the compound); ignores the compound | **the bar to beat** |
| Similarity kNN | find the k most similar compounds (Tanimoto on ECFP4) and use their known targets; k chosen on validation | classic "similar molecules, similar targets" |
| Two-tower | network A turns compound fingerprints into a vector, network B turns ESM-2 into a vector; score = similarity of the two | strongest non-graph model; works for brand-new compounds and proteins |

The two-tower model is **not a GNN**; it shows where the graph helps and where it does not. An ensemble of kNN and two-tower was tested;
validation always chose two-tower alone, so it is not reported separately.

---

## 7. Herb-target scores and the blend

The models only score compound-target pairs, so a herb's score for a target is built from its compounds.

**Model only**: one of six rules, chosen on validation: max (the herb's best compound), mean, mean of the top 3, noisy-OR (the chance that
at least one compound hits the target), sum, or the number of the herb's compounds that rank the target in their own top 50.

Model-only herb scores lose to popularity: a herb with hundreds of compounds almost always reaches the proteins most compounds hit.

**Blend with popularity**:

```
herb score = (1 - w) × model rank + w × popularity rank
```

- Both are turned into ranks between 0 and 1 so they can be mixed.
- w = 0 is the model alone, w = 1 is popularity alone; w is chosen on validation.
- Popularity is counted from the fold's training links only, so nothing leaks. Random and popularity are never blended.

Result: for known compounds w = 0.75 to 0.9 and the blend beats popularity in all 5 folds on Hits@10 (0.153 against 0.146): the model adds
a little on top of popularity. For new compounds w = 0.9 to 1 and the blend only ties popularity.

---

## 8. Transductive and inductive

These words describe what the model is allowed to know about the things it is tested on.

Think of an exam: **transductive** asks about students who were in the class (the teacher knows them, just not every answer);
**inductive** asks about new students who never attended (the teacher must judge from general rules).

| | Transductive | Inductive (cold-start) |
|---|---|---|
| What is hidden | some links | whole nodes, with all their links |
| Is the test node in training? | yes, with its other links | no links at all, only its features |
| Our tasks | C-T known compounds (`ct_edge`), H-C (`hc_edge`) | C-T new compounds (`ct_compound`), C-T new targets (`ct_target`) |
| Real use | complete gaps for known compounds | predict for a brand-new compound or protein |
| Model relies on | the node's links and features | features only (fingerprint, ESM-2) |
| Difficulty | easier | harder |

**Example.** Transductive: quercetin keeps about 280 of its 354 known targets in training and the model must find the hidden ~70.
Inductive: compound X has all its targets removed; the model only knows its structure and which herbs contain it.

Our inductive split is the cold-start kind: a held-out compound keeps its fingerprint and its herb links, a held-out protein keeps its
ESM-2 features; only their compound-target links are removed. Twin compounds (identical fingerprint or stereoisomers) are kept on the
same side, and held-out proteins are never used as training negatives.

Why both matter:

| Model | Transductive (known compounds) | Inductive (new compounds) |
|---|---|---|
| Popularity | 0.069 | 0.069 |
| R-GCN | **0.294** | 0.036 |
| R-GCN + fixes | 0.268 | 0.072 |
| Two-tower | 0.342 | **0.140** |

The R-GCN learns a lot from each node's links, so it is strong transductively and weak inductively. Tested only transductively, it would
wrongly look ready for new compounds.

---

## 9. How the models are tested

### 9.1 Four tasks

| Task | What is hidden | Question |
|---|---|---|
| C-T, known compounds | some compound-target links | find missing targets of known compounds |
| C-T, new compounds | whole compounds with all their links | predict for a compound never seen linked |
| C-T, new targets | whole proteins with all their links | predict for a protein never seen linked |
| H-C | some herb-compound links | find missing compounds of a herb |

H-T is measured inside the two compound tasks.

### 9.2 5-fold cross validation
- Items are spread over 5 parts of about 20% of the links each. In fold k, part k is the test set; 15% of all links from the other parts
  go to validation; the remaining ~65% is training.
- Every item is tested exactly once; every fold trains new models from scratch; results are mean ± standard deviation.
- Why: one split can be lucky or unlucky; five give a stable average and show how much the result varies.
- Note: the team lead's protocol says 70/15/15. Classic 5-fold gives 20% test; in exchange every item is tested exactly once.

### 9.3 Negatives
- Drawn at random from the 3.1 million unlabeled pairs, without repeats: 10 per positive for train and validation, 50 for test.
  Train, validation and test never share a negative.
- Known compounds: only from the 726 compounds with known targets (pairs of compounds with no target at all would be too easy).
  New compounds: only that part's compounds. New targets: only that part's targets, and ranking is only among them.
- H-C: from herb-compound pairs without a link.
- One paired negative per positive (same compound or herb) gives a loss value on train, validation and test.

### 9.4 Leakage controls (all checked in `02_prep`, all passed)
- Only training links of the predicted type are in the graph.
- The test set is never used to choose anything (early stopping, k, herb rule and w all use validation).
- Compounds with an identical fingerprint or the same InChIKey skeleton (470 share a fingerprint) are kept in the same part.
- No compound or target is in two parts.
- A herb-target pair is a test pair only if it is reachable through test links alone, never through a training or validation link.

---

## 10. Metrics

| Metric | Meaning | Random level |
|---|---|---|
| PR-AUC | how well true pairs score above the sampled negatives | about 0.02 at 1:50 |
| Hits@10 | share of true links in the top 10 of their compound (out of 1,341 targets) | about 0.007 |
| Hits@50 | same for the top 50 | about 0.04 |
| MRR | average of 1 / rank (rank 1 gives 1.0, rank 4 gives 0.25) | about 0.006 |
| loss | BPR loss on each part (trained models only) | |

- **Filtered ranking**: when a true link is ranked, the compound's other known targets are removed, so they do not count against it.
- **Why not accuracy or ROC-AUC**: with millions of unlabeled pairs they look high even for a weak model (predicting "no link" everywhere
  would give 99.5% accuracy).

---

## 11. The notebooks, cell by cell

### The shared `hti` package

| File | What is inside |
|---|---|
| `hti/__init__.py` | sets `KMP_DUPLICATE_LIB_OK` so the Windows OpenMP error does not appear; every notebook starts with `import hti` |
| `hti/data.py` | paths, the 4 task names, `Prep` (loads everything in `prep/`), `prep.fold(task, k)` (train / val / test of fold k) |
| `hti/models.py` | R-GCN layer, `GraphModel` (encoder + DistMult), `TwoTower`, kNN, popularity, random |
| `hti/train.py` | `train_graph_model` and `train_two_tower`, both with early stopping on validation MRR |
| `hti/metrics.py` | PR-AUC, Hits@10/50, MRR, loss, herb-target scores and the blend |

```
01 (check data) -> 02 (prep/) -> 03 (R-GCN)      -> 05 (all tables)
                              -> 04 (two-tower)  -> 05
                              -> 06 (predictions, uses 03's results file)
```

### 01_data_check (writes nothing)

| Cell | What it does |
|---|---|
| 1 Setup | imports, data folder |
| 2 Load all files | reads the 9 files, table of rows and columns |
| 3 Every column | every column of every file with a decision: used for what, or not used and why (constant, copy of another column, external identifier) |
| 4 Ids and links | about 14 True/False checks: unique ids, links use known ids, every compound in a herb, every label 1 |
| 5 Unlabeled pairs | no duplicates, no overlap with known links, exactly every pair that is not known |
| 6 Herb-target pairs | the herb-target file equals the H-C × C-T join (24,835) |
| 7 Fingerprint file | RDKit fingerprints from SMILES match the file's rows by sorted position (2,319) but not by its id column (2) |
| 8 Look-alike compounds | 470 share a fingerprint, 318 share an InChIKey skeleton, 0 duplicate proteins |
| 9 Summary | final counts |

### 02_prep (writes `prep/`)

| Cell | What it does |
|---|---|
| 1 Setup | seed 42, 5 folds, 15% validation, negatives 10 / 10 / 50 per positive; clears old files |
| 2 Load | reads the dataset files |
| 3 Node index | sorts herbs, compounds and targets by id; the position is the node number |
| 4 Compound features | RDKit: ECFP4 bits, ECFP4 counts, FCFP4, MACCS; asserts ECFP4 equals the file's rows |
| 5 Target and herb features | AAC and ESM-2 in target order; plant family one-hot for herbs |
| 6 Links | de-duplicates H-C rows (8,396 to 5,005); link arrays; which herbs each compound is in; known-link tables |
| 7 Negative pools | loads the 3.1M unlabeled pairs and asserts they are exactly the non-known pairs; H-C pool (138,773) |
| 8 Compound groups | joins compounds with the same fingerprint or InChIKey skeleton into groups |
| 9 Fold assignment | spreads groups over 5 balanced parts; per fold: test part, 15% validation, rest training |
| 10 Negative functions | random draws without repeats; paired negatives; index helpers |
| 11 Build folds | for each task and fold: split positives, draw negatives from the right pool, paired negatives, save |
| 12 Save | `features.npz`, `links.npz`, `meta.json` |
| 13 Leakage checks | parts cover all links once, no negative is known, parts share no negative, no compound / group / target in two parts, every link tested once; stops if any fails |
| 14 H-T test pairs | herb-target pairs per part (test only if reachable through test links alone) |

### 03_rgcn (writes `results/metrics_rgcn.csv`, `curves_rgcn.csv`, `curves_rgcn.png`)

| Cell | What it does |
|---|---|
| 1 Setup | loads `Prep()`, prints `cuda` and counts |
| 2 One run, step by step | trains one R-GCN + fixes on fold 0, printing train loss, val loss and val MRR every 2 epochs, then the train / val / test table; the cell to show that training works |
| 3 All tasks | 4 tasks × 2 settings × 5 folds = 40 models; one line each; saves every result and curve |
| 4 Mean and std | mean ± std per task, model and part |
| 5 Curves | train loss, val loss and val MRR over epochs for fold 0 |

Inside `train_graph_model`: build model and optimiser; every epoch choose which links are scored (the fixes); for every batch run the GNN,
score true and random targets, BPR loss, update; every 2 epochs validation MRR, keep the best, stop after 15 checks without improvement;
return the 2,319 × 1,341 score matrix. Inside `evaluate`: all metrics for train, val and test, plus herb-target model-only and blend.

### 04_inductive (writes `results/metrics_inductive.csv`, `curves_inductive.csv`)

| Cell | What it does |
|---|---|
| 1 Setup | loads `Prep()`; Tanimoto similarity between all compounds for kNN |
| 2 All C-T tasks | per task and fold: kNN (k chosen on validation), two-tower ESM-2, two-tower AAC, ensemble (weight chosen on validation; always two-tower only); evaluate and save |
| 3 Mean and std | same table format as 03 |

### 05_results (writes `metrics_baselines.csv`, `summary_<task>.csv`, `summary_test.csv`, `test_metrics.png`)

| Cell | What it does |
|---|---|
| 1 Setup | metric column list |
| 2 Baselines | random and popularity on every fold (popularity from that fold's training links; never blended) |
| 3 All results | combines baselines, 03 and 04; `summary(task)` table; saves per-task CSVs |
| 4 to 7 | one train / val / test table per task |
| 8 Test only | main results table, `summary_test.csv` |
| 9 Beats popularity | per task and model: test MRR and whether it beats popularity (also for H-T) |
| 10 Bar chart | test MRR and Hits@10 per task with error bars |

### 06_predictions (writes `hidden_link_check.csv`, `hidden_link_examples.csv`, `candidates_ct/ht/hc.csv`)

| Cell | What it does |
|---|---|
| 1 Setup | readable names for herbs, compounds and targets |
| 2 Train 5 folds | per fold: R-GCN + fixes and two-tower for C-T (averaged by rank), R-GCN + fixes for H-C |
| 3 Hidden-link check | for every test compound of every fold: test hits, validation links and unlabeled in the top 10, against chance; 3 examples from fold 0 |
| 4 Average and agreement | averages the 5 fold models; "folds in top 10" (0 to 5) per pair |
| 5 C-T candidates | 5 best new targets per compound |
| 6 H-T candidates | 10 best new targets per herb (no path through any known link) |
| 7 H-C candidates | 10 best new compounds per herb |
| 8 Stability | share of candidates agreed by 5 of 5 and by at least 3 of 5 fold models |

---

## 12. Results and how to read them

| Task (test MRR) | Popularity | R-GCN | R-GCN + fixes | Two-tower |
|---|---|---|---|---|
| C-T, known compounds | 0.069 | **0.294** | 0.268 | 0.342 |
| C-T, new compounds | 0.069 | 0.036 | **0.072** | 0.140 |
| C-T, new targets | 0.026 | 0.093 | **0.157** | 0.206 |
| H-C | 0.052 | 0.053 | **0.079** | not run |
| H-T, known compounds (blend) | 0.073 | 0.074 | **0.078** | 0.076 |
| H-T, new compounds (blend) | 0.068 | 0.067 | 0.067 | 0.068 |

1. **Known compounds**: the R-GCN is about 4 times popularity; Hits@10 0.446 means 45% of hidden true targets land in the top 10 of 1,341.
2. **New compounds**: the plain R-GCN falls below popularity (a new compound has no links to learn from); the fixes bring it just above;
   the two-tower model, which works from chemistry directly, is twice popularity.
3. **New targets**: R-GCN + fixes works well. ESM-2 matters: two-tower with ESM-2 0.206, with AAC 0.124.
4. **H-C**: R-GCN + fixes clearly beats popularity (Hits@10 0.155 against 0.106).
5. **H-T**: only with the blend, and only slightly, for known compounds. This is the open problem.

**Predictions** (`06`): on average 1.72 of a test compound's top 10 are hidden true links, against 0.05 by chance (about 35 times), and 68%
of test compounds have at least one in their top 10. Candidates come from the 5 fold models averaged: 85% of C-T candidates are agreed by at
least 3 of 5 folds (H-T 60%, H-C 33%). Herb candidates come in protein families (GABA-A receptors for turmeric, UGT and cytochrome P450
enzymes for ginger). H-C candidates are mostly common plant metabolites (sterols, fatty acids, quercetin), pointing to compounds IMPPAT has
probably not recorded for those herbs.

---

## 13. What is weak or limited (say these first)

1. **The best C-T model is not the GNN.** The two-tower model beats the R-GCN on every C-T task. Message: for C-T, good features (ESM-2,
   fingerprints) matter more than graph structure; the graph helps most for H-C.
2. **Herb-target is not solved.** Alone the model loses to popularity; with the blend the gain is small.
3. **Popular-target bias.** NFE2L2 is in the top 5 of 41% of compounds; only 339 of 1,341 targets appear in the top 5 lists. The likely
   cause is that many known links come from screening panels testing the same proteins.
4. **"Negatives" may contain real links.** Unlabeled does not mean negative, so the metrics probably underestimate true performance.
5. **Split is 65 / 15 / 20, not the protocol's 70 / 15 / 15**, in exchange for every item being tested exactly once.
6. **Validation is slightly optimistic for new compounds and new targets** because early stopping picks the best epoch on validation;
   test is what is reported and is never used for choices.
7. **Hyperparameters were not re-tuned** in this pipeline (an earlier tuning round found no gain). Differences under about 0.01 are GPU noise.
8. **No new-herb test** (herbs never seen by the model); it needs careful design because herbs share compounds.
9. **Predictions are not yet checked against the literature**; they are hypotheses.
10. **Only 62 herbs**, which is small for herb-level conclusions.

---

## 14. Likely questions and answers

**Why a GNN?**
The data is naturally a graph (herb, compound, target), and a GNN passes information along those links, for example from a herb's other
compounds.

**Why R-GCN and not GCN?**
GCN treats all links the same; ours have different meanings, and R-GCN gives each link type its own weights.

**Why BPR loss and not cross-entropy?**
There are no confirmed negatives. BPR only needs "the true link should rank above a random one", which fits positive-unlabeled data.

**What is transductive and inductive here?**
Transductive: some links are hidden but every node is in the training graph (C-T known compounds, H-C). Inductive (cold-start): whole
compounds or proteins have all their links removed and must be predicted from their features (C-T new compounds, new targets).

**What are the fixes?**
Three training changes against memorisation: score only links removed from the graph, hide whole compounds or targets during training in
the inductive tasks, and regularise the input features.

**What is the blend? Is it cheating?**
Herb score = (1 - w) × model rank + w × popularity rank, with w chosen on validation. Not cheating: popularity comes from training links
only, w is never chosen on test, and model-only numbers are reported separately. It shows how much the model adds beyond popularity.

**Why is a non-GNN model better?**
For C-T the information is mostly in the chemistry and the protein sequence, and the two-tower model uses them directly. The R-GCN learns
much from links, which new compounds do not have. That is the baseline result the next model (MAMGN-HTI) should improve on.

**How do you know there is no leakage?**
The checks run in code and all pass: test links never enter the graph, the test set is never used for choices, twin compounds stay
together, and herb-target test pairs have no path through training links.

**Why not ROC-AUC?**
With 3 million unlabeled pairs ROC-AUC is always high. PR-AUC, Hits@K and MRR show whether true links really reach the top.

**Why are the numbers low?**
Each true link competes with 1,341 targets, and many "negatives" may be untested positives. Random MRR is 0.006; the R-GCN reaches 0.294
for known compounds, about 50 times random and 4 times popularity.

**What did you fix in the data?**
The fingerprint file's id column was in the wrong order; fingerprints are recomputed from SMILES and checked against the file.

**What is next?**
Train the GNN directly on herb-target pairs; a new-herb test; check top predictions against the literature; MAMGN-HTI next semester.

---

## 15. Short versions to say out loud

**30 seconds, the project**
"We predict links between herbs, compounds and protein targets as link prediction on a graph. The GNN baseline is an R-GCN with a DistMult
scorer trained with BPR loss, because the data has only positives and unlabeled pairs. We test it with 5-fold cross validation in four
tasks, transductive and inductive, with negatives from the 3.1 million unlabeled pairs and checked leakage controls, against random and
popularity baselines and a non-graph two-tower model."

**30 seconds, the results**
"For known compounds the R-GCN reaches MRR 0.29, four times popularity. For new compounds and new proteins it needs our anti-memorisation
fixes, and a two-tower model with protein language model features is stronger. Herb-compound prediction clearly beats popularity.
Herb-target prediction only matches or slightly beats popularity, which is the main open problem."

**30 seconds, the notebooks**
"01 checks every file and column and finds the fingerprint id defect. 02 computes features, builds 5-fold splits for four tasks with
negatives from the unlabeled pairs, and verifies there is no leakage. 03 trains the R-GCN, plain and with fixes. 04 trains the non-graph
comparison models. 05 adds the baselines and builds all tables. 06 checks that the models recover hidden links and lists new candidates
with agreement across the 5 folds."
