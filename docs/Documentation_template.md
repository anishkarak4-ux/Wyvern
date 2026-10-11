# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Wyvern  
**Team Members:** Anish Karak, Soumya Gupta, Souvik Ghosh  
**Submission Date:** October 2, 2026  

---

## 1. Executive Summary

Team Wyvern developed a high-recall, precision-optimized machine learning solution for large-scale multi-source commercial entity resolution across three independent data sources (`Source 1`, `Source 2`, and `Source 3`). Operating strictly under the non-external lookup constraint, the solution pairs a high-recall multi-key blocking engine—**Portfolio 2 (P2)** combining 8 deterministic, country-isolated indexing rules—with a 46-feature pairwise gradient boosted decision tree classifier (**XGBoost**) and a precision-maximizing post-model decision policy (**Rule F**).

The confirmed final submission variant—**Emergency Best / Rule F**—operates directly on the Portfolio 2 candidate universe of $336,928,889$ candidate pairs ($194.47$ candidates/entity). It retains the rank-1 leading prediction above base threshold $\tau = 0.90$, and conditionally admits secondary candidates up to rank 5 only when their probability gap from the leader satisfies $(p_1 - p_i) \le 0.08$. This targeted pruning eliminates over $2.31\text{ Million}$ runaway false-positive branch distractors, boosting precision from $87.0\%$ to **$88.3\%+$**, elevating held-out validation **Macro $F_{0.5}$ to $0.80433$ – $0.80526$**, and achieving a verified **$0.780$ on the public competition leaderboard**. Evaluated across the full hidden test set of $1,732,544$ Source 1 entities, the pipeline emitted $4,195,824$ matches ($2.422$ matches/entity), $207,395$ Source 1 entities received zero predicted matches, and the outputs passed all official validator checks with zero errors and zero warnings.

---

## 2. Methodology & Problem Formulation

### 2.1 Problem Analysis & Formulation
Commercial entities in this challenge are distributed across three distinct sources:
- **`Source 1` ($S_1$):** Deduplicated reference dataset ($1,732,544$ test entities).
- **`Source 2` ($S_2$) & `Source 3` ($S_3$):** Target datasets containing zero, one, or multiple records corresponding to an $S_1$ entity.

The goal is to predict the set of target IDs $\mathcal{M}(e_1) \subseteq S_2 \cup S_3$ matching each $e_1 \in S_1$. The competition metric is **Macro $F_{0.5}$**, defined as:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
computed per Source 1 entity and averaged across all entities, including unlinked entities ($\mathcal{M}(e_1) = \emptyset$). Correctly predicting an empty set for an unlinked entity yields $1.0$, while a single false match on an unlinked entity drops its score immediately to $0.0$. Because precision is weighted $2\times$ over recall, avoiding false merges on multi-branch commercial entities while maintaining broad candidate coverage is the paramount engineering objective.

**Key Dataset Challenges Discovered During EDA:**
1. **Severe Combinatorial Scale:** $1.73 \times 10^6$ Source 1 entities against $3.5 \times 10^6$ Source 2/3 entities implies $\approx 6.06 \times 10^{12}$ potential pairs. Exhaustive evaluation is impossible; candidate generation must achieve extreme reduction ($>99.99\%$) while retaining high recall.
2. **Jurisdictional & Geographic Shifts:** Training data covers `US` and `India`, whereas the test set introduces `France` as an unseen open-set country label. The pipeline cannot rely on hard-coded geographic dictionaries and must generalize across varying international naming and address conventions.
3. **Severe Textual Noise & Branch Expansions:** Indian entities frequently exhibit landmark-based addresses (e.g., "Near SBI ATM"), missing postal PIN codes, and localized corporate branch names (e.g., "State Bank of India Chandni Chowk Branch" vs. "State Bank of India"). US entities feature complex suite numbers, corporate legal suffix variations (e.g., "Corp", "Inc", "LLC"), and phonetic misspellings.

### 2.2 Solution Strategy
We adopt a decoupled three-stage architecture:
1. **Candidate Generation (Blocking):** An inverted-index multi-key blocking engine with per-country isolation (Portfolio 2) that reduces the search space from $6 \times 10^{12}$ pairs to $3.37 \times 10^8$ candidates ($\approx 99.994\%$ reduction ratio).
2. **Pairwise Classifier:** A 46-feature XGBoost histogram-gradient boosting classifier trained with hard-negative mining to predict $P(\text{match} \mid e_1, e_{\text{target}})$.
3. **Precision Optimization (Rule F):** An entity-level decision policy that dynamically filters candidate lists by rank and relative probability gap from the top prediction to suppress false merges on commercial chains.

---

## 3. Data Preprocessing & Candidate Generation (Blocking)

### 3.1 Preprocessing & Normalization
Implemented in `src/preprocessing/normalization.py`, all fields undergo deterministic cleaning:
- **Unicode & Case Normalization:** Unicode NFKC normalization, lowercasing, replacing non-alphanumeric characters with spaces while preserving multilingual Unicode characters (no destructive diacritic/accent stripping), and collapsing redundant whitespace.
- **Legal Entity Designation Stripping:** Regex-based removal and standardization of jurisdictional legal forms (`pvt`, `ltd`, `inc`, `corp`, `llc`, `gmbh`, `sa`, `sarl`, `co`, `limited`, `private limited`).
- **Address Expansion & Numeric Tokenization:** Standardizing road types (`rd` $\to$ `road`, `st` $\to$ `street`, `ave` $\to$ `avenue`), separating numerical tokens, and extracting numbers $\ge 3$ digits (e.g., building numbers, street numbers, postal codes).

### 3.2 Candidate Generation Strategy & Portfolio 2
To recover matches missed by conventional single-key blocking without suffering combinatorial explosion, we developed **Portfolio 2 (P2)**, combining 8 deterministic, country-isolated indexing rules:

| Rule Name | Blocking Keys | Bucket Cap | Design Rationale & Recovery Target |
|:---|:---|:---:|:---|
| `rule_stripped_name` | `(country, stripped_name)` | Uncapped | Exact match on stripped alphanumeric name string |
| `rule_first_2_tokens` | `(country, first_2_tokens)` | 200 | Multi-word names sharing primary brand prefix |
| `rule_name_pfx8` | `(country, name_pfx8)` | 200 | Robust to trailing typos, legal variations, branch suffixes |
| `rule_addr_num_initial` | `(country, addr_num_ge_3, name_initial)` | 100 | Street address number + name initial letter |
| `rule_fw_addr_num` | `(country, first_word_ge_3, addr_num_ge_3)` | 100 | Primary brand word + street address number |
| `rule_sorted_tok2` | `(country, sorted_first_2_tokens)` | 200 | Inverted word order (e.g., "McDonald's Restaurant" vs "Restaurant McDonald's") |
| `rule_fw4` | `(country, first_word_ge_4)` | 50 | Unique corporate names where address is missing/noisy |
| `rule_rare_tok_addr_num` | `(country, rare_name_tok_df_le_100, addr_num_ge_3)` | 100 | Rare catalog name token ($DF \le 100$, len $\ge 4$) + street number |

**Candidate Generation Validation Performance (10,002 Held-Out Entities):**
- Baseline Blocking (4 rules): $71.68\%$ recall on held-out validation ($24,858 / 34,677$ GT matches recovered).
- Combo 5 (7 rules): $74.30\%$ recall ($25,764 / 34,677$ GT matches recovered, $+906$ over baseline).
- Portfolio 2 (8 rules): **$75.09\%$ candidate blocking recall** ($26,040 / 34,677$ GT matches recovered, $+1,182$ over baseline), establishing the candidate universe of $336,928,889$ pairs for the full test set.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Engineering (46 Features)
Dynamic pairwise feature extraction (`src/features/pair_features.py`) constructs exactly 46 dense, interpretable features per candidate pair:

1. **Name Lexical & Token Similarity (Features 1–15):**
   - Exact string equality, stripped name equality, character length difference, character length ratio, stripped length difference.
   - Jaro-Winkler string similarity, normalized Levenshtein distance.
   - RapidFuzz token metrics: `token_sort_ratio`, `token_set_ratio`, `partial_ratio`.
   - Structural indicators: `first_two_tokens_match`, `name_prefix_8_match`, `first_word_match`.
   - Token overlap: shared token count, token Jaccard similarity.

2. **Address & Numeric Consistency (Features 16–28):**
   - Exact address match, address length difference, Jaro-Winkler similarity, normalized Levenshtein distance.
   - Address token metrics: `token_sort_ratio`, `token_set_ratio`, shared token count, token Jaccard similarity.
   - Numeric token features: shared numeric token count, shared numeric token ratio, exact numeric match flag.
   - Postal code features: postal code pair presence flag, exact postal code match flag.

3. **Cross-Field Interaction Features (Features 29–34):**
   - `name_addr_sim_product`: Multiplicative interaction between name Jaro-Winkler and address Jaro-Winkler.
   - `name_exact_and_addr_sim`: High-confidence interaction indicating identical name supported by address similarity.
   - `name_sim_and_shared_num`: Name similarity confirmed by shared street/building number.
   - `both_fields_agree_strong`: Binary indicator that both name and address similarities exceed $0.85$.
   - `name_only_agrees` & `addr_only_agrees`: Discrepancy flags identifying partial matches (crucial for distinguishing separate businesses sharing a commercial complex).

4. **Metadata & Blocking Provenance (Features 35–46):**
   - `same_country` indicator, `is_source3` indicator.
   - Missing data indicators: $S_1$ missing name, Target missing name, $S_1$ missing address, Target missing address, joint address presence.
   - Provenance rule flags: `blocking_rule_stripped`, `blocking_rule_tok2`, `blocking_rule_pfx8`, `blocking_rule_addr_num`, and total distinct rule hits (`num_blocking_rules`).

### 4.2 Model Architecture & Training
- **Model Type:** XGBoost (`xgb.XGBClassifier`) with histogram-based tree building (`tree_method='hist'`).
- **Hyperparameters:** `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=1.0`, `reg_alpha=0.0`.
- **Training Strategy:** Trained on labeled pairs generated from the training catalog using blocking candidates and hard negative mining. Final production weights were frozen at `data/models/xgb_final_stage5.json` ($4,884,794$ bytes / $\sim 4.66\text{ MB}$).

---

## 5. Post-Processing Decision Logic: Emergency Best / Rule F

### 5.1 The Multi-Branch Precision Bottleneck
In large-scale commercial entity resolution, high-frequency corporate chains (banks, retail chains, telecommunications offices) create severe precision degradation. Standard probability thresholding ($\tau = 0.90$) produced entities with up to **717 matching predictions**, where almost all secondary candidates were different branches of the same entity sharing identical brand names. Because Macro $F_{0.5}$ weights precision $2\times$ over recall, predicting 50 matches for an entity with 4 true matches drops that entity's score from near $1.0$ down to $0.098$.

Analysis of the training ground-truth distributions revealed that **$99.78\%$ of entities have $\le 8$ true matches**, and **$88.56\%$ have $\le 5$ matches** (with only $0.22\%$ exceeding 8 matches).

### 5.2 Rule F Formulation
To eliminate false merges without sacrificing legitimate multi-source matches:
1. For each $S_1$ entity, candidates scoring $p \ge 0.90$ are sorted in descending order: $p_1 \ge p_2 \ge \dots \ge p_k$.
2. **Rank 1:** The top candidate is unconditionally retained ($p_1 \ge 0.90$).
3. **Ranks 2–5:** Secondary candidates are retained **only if** their probability gap from the leader satisfies:
   $$(p_1 - p_i) \le 0.08$$
4. **Rank > 5:** Candidates beyond rank 5 are pruned.

### 5.3 Empirical Impact on Validation Set (10,002 Entities)
- **Baseline ($\tau = 0.90$ Uncapped):** Macro $F_{0.5} = 0.79831$, Precision = $0.8703$, Recall = $0.6868$, False Positives = $3,416$.
- **Emergency Best (Rule F):** Macro $F_{0.5} = \mathbf{0.80433 – 0.80526}$, Precision = $\mathbf{0.8818 – 0.8827}$, Recall = $0.6758 – 0.6761$, False Positives = $\mathbf{1,853 – 1,896}$ (**$-44.5\%$ reduction in false merges**).

---

## 6. Final Submission & Test Verification Statistics

### 6.1 Test Set Verification Summary

| Metric | Stage 7 Baseline | Stage 9 Portfolio 2 (Uncapped) | Final Submission (Emergency Best / Rule F) |
|:---|:---:|:---:|:---:|
| **$S_1$ Entities Processed** | $1,732,544$ | $1,732,544$ | **$1,732,544$** ($100.0\%$) |
| **Total Candidates Evaluated** | $296,419,213$ | $336,928,889$ | **$336,928,889$** |
| **Average Candidates / $S_1$** | $171.09$ | $194.47$ | **$194.47$** |
| **Total Accepted Matches** | $6,219,897$ | $6,557,986$ | **$4,195,824$** |
| **Average Matches / $S_1$** | $3.590$ | $3.785$ | **$2.422$** |
| **Zero-Match Entities (Empty Predictions)** | $231,210$ ($13.35\%$) | $207,395$ ($11.97\%$) | **$207,395$** ($11.97\%$) |
| **Validation Macro $F_{0.5}$ (10,002 Entities)** | $0.77853$ | $0.79804$ | **$0.80433 – 0.80526$** |
| **Public Leaderboard Score** | $0.764$ | *Unsubmitted* | **$0.780$** |
| **Subset Violations ($\mathcal{M} \subseteq \mathcal{C}$)** | $0$ | $0$ | **$0$** ($100\%$ verified) |
| **Missing / Duplicate $S_1$ Rows** | $0$ | $0$ | **$0$** |

### 6.2 Official Validator Result
Executed `utils/validate_submission.py` with full ID-level verification (`--check-ids`):
```text
ML Challenge 2026 — submission validator
  test dir: data/raw/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (207395 empty, 1525149 non-empty).
  candidate_pairs.tsv: 1732544 rows (1818 empty, 1730726 non-empty).

PASS — no blocking issues found. Safe to submit.
```
* **Exit Code:** `0` (Success)
* **Subset Containment:** $\mathcal{M}(e_1) \subseteq \mathcal{C}(e_1)$ verified across all $1,732,544$ rows ($0$ violations).
* **Target ID Integrity:** $100\%$ of predicted target IDs exist in `test_source2.tsv` or `test_source3.tsv`.

---

## 7. Operational, Compliance & Technical Considerations

### 7.1 Computational Considerations & Scalability
- **Country-Isolated Partitioning:** Processing was partitioned strictly by country (`France`: 259k, `US`: 663k, `India`: 810k), guaranteeing zero cross-country predictions and capping RAM usage to $< 24\text{ GB}$.
- **Vectorized String Operations:** Pairwise feature generation utilizes RapidFuzz C++ bindings and vectorized string matching, achieving high throughput without GPU requirements.
- **Histogram-Based Inference:** XGBoost's `tree_method='hist'` enabled efficient batch scoring across $3.37 \times 10^8$ pairs.

### 7.2 Academic Integrity & Prohibited External Data
- **Strict Prohibition Adherence:** Zero external databases, commercial ER APIs, government business registries, geocoding lookups, or internet data augmentations were used.
- **Model Constraints:** XGBoost is open-source (Apache 2.0). The model consists of 1,000 trees ($4,884,794$ bytes / $\sim 4.66\text{ MB}$ JSON), well below the 8 Billion parameter ceiling.

---

## Appendix: Reproduction & Code Artifacts

The complete runnable pipeline is located in `code/business_entity_resolution/`:
- **Inference Entry Point:** `python -m src.models.run_stage9_test_inference` (generates Portfolio 2 candidates and baseline predictions).
- **Rule F Post-Processing:** `python -m src.models.apply_rule_f` (produces final `matching_results.tsv`).
- **Validation Script:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids`.
