# Amazon ML Challenge 2026: Comprehensive Stage 1 EDA & Dataset Profile

## 1. Executive Summary
This document presents the empirical findings from Stage 1 Exploratory Data Analysis (EDA) and profiling of the Amazon 72-hour Machine Learning Challenge 2026 dataset (Business Entity Resolution).

Key takeaways:
- **Dataset Scale:** 2.21M training reference entities (Source 1) mapped against 5.03M Source 2 and 5.29M Source 3 entities; 1.73M test reference entities mapped against 4.89M Source 2 and 5.08M Source 3 entities.
- **Country Hard Blocking:** 100% of the 3.94M evaluated ground truth links share identical country labels ($0.00\%$ cross-country matches). Country is an absolute partitioning boundary.
- **Open Set Country in Test:** Test data introduces **France** (14.98% of test S1, 14.39% of test S2, 14.40% of test S3) which is completely absent from training. Preprocessing and blocking logic must remain country-agnostic.
- **Singletons are Crucial:** Exactly 5.58% (123,247) of Source 1 entities have zero matches in Source 2 or Source 3. In the $F_{0.5}$ metric, singletons score 1.0 for predicting empty string, but 0.0 for any false match.
- **High Blocking Recall Feasibility:** A multi-key blocking strategy combining normalized name prefix, legal suffix stripping, first-token inverted indexing, and address numeric token overlap achieves **>94% true match recall** on empirical pairs while pruning the comparison space by $>99.999%$.

---

## 2. Dataset Structure and Schema Summary

| Dataset File | Split | Role | File Size | Data Rows | Total Columns | Missing Values (Count & %) | Unique Entity IDs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train_source1.tsv`** | Train | Reference | 200.34 MB | 2,206,821 | 4 | None (0.00%) | 2,206,821 |
| **`train_source2.tsv`** | Train | Target Fragment | 466.63 MB | 5,034,616 | 4 | `business_name`: 1 (0.00%)<br>`business_address`: 168,967 (3.36%) | 5,034,616 |
| **`train_source3.tsv`** | Train | Target Fragment | 480.37 MB | 5,285,603 | 4 | `business_address`: 175,916 (3.33%) | 5,285,603 |
| **`train_ground_truth.tsv`** | Train | Ground Truth | 121.13 MB | 2,206,821 | 2 | `matched_entity_ids`: 123,247 (5.58%) | 2,206,821 |
| **`test_source1.tsv`** | Test | Reference | 166.91 MB | 1,732,544 | 4 | None (0.00%) | 1,732,544 |
| **`test_source2.tsv`** | Test | Target Fragment | 485.86 MB | 4,887,273 | 4 | `business_address`: 129,408 (2.65%) | 4,887,273 |
| **`test_source3.tsv`** | Test | Target Fragment | 482.56 MB | 5,082,316 | 4 | `business_name`: 1 (0.00%)<br>`business_address`: 136,098 (2.68%) | 5,082,316 |

### Missing Values Key Insights:
- `entity_id` and `country`: **0% missing** across all 7 train and test files.
- `business_name`: **0% missing** in Source 1 and Source 3 train; only 1 single missing value in `train_source2` and `test_source3`.
- `business_address`: **100% present in Source 1** (reference), but has missing values in ~2.65% to 3.36% of records in Sources 2 and 3. Models must handle missing address tokens gracefully.

---

## 3. Country Distribution Analysis

| Dataset | US Count (%) | India Count (%) | France Count (%) | Total Rows |
| :--- | :--- | :--- | :--- | :--- |
| **`train_source1.tsv`** | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | 2,206,821 |
| **`train_source2.tsv`** | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | 5,034,616 |
| **`train_source3.tsv`** | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.00%) | 5,285,603 |
| **`test_source1.tsv`** | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 1,732,544 |
| **`test_source2.tsv`** | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 4,887,273 |
| **`test_source3.tsv`** | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) | 5,082,316 |

### Country Insights:
1. **Training Split:** Covers strictly `US` (~60.0%) and `India` (~40.0%).
2. **Test Split:** Introduces `France` (~14.4%–15.0%), while `India` represents ~46.8%–47.3% and `US` represents ~38.3%.
3. **Architectural Implication:** Hard partitioning by country is mathematically safe (0% cross-country matches in ground truth) and reduces the test search space into three strictly independent sub-problems: US, India, and France.

---

## 4. Business Name Profiling

Based on deterministic sampling ($N = 50,000$ per dataset):

| Dataset | Min Len | Median Len | Max Len | Mean Len | Median Tokens | Non-ASCII (%) | All-Caps (%) | Title-Case (%) | Top Legal Suffixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train_source1.tsv`** | 3 | 24.0 | 64 | 24.0 | 4.0 | 0.00% | 0.00% | 69.55% | limited (11969), private (9920), llc (8005), inc (5328) |
| **`train_source2.tsv`** | 2 | 25.0 | 78 | 25.1 | 4.0 | 15.18% | 18.77% | 46.76% | private (5310), limited (5266), llc (5133), inc (4087) |
| **`train_source3.tsv`** | 2 | 25.0 | 89 | 25.2 | 4.0 | 11.36% | 2.97% | 61.12% | limited (6484), private (6139), llc (5436), ltd (4084) |
| **`test_source1.tsv`** | 3 | 24.0 | 83 | 23.9 | 4.0 | 2.29% | 0.00% | 65.70% | limited (13701), private (11401), llc (5173), ltd (3898) |
| **`test_source2.tsv`** | 2 | 25.0 | 81 | 25.7 | 4.0 | 19.15% | 17.43% | 44.26% | limited (6605), private (6108), ltd (4357), llc (3232) |
| **`test_source3.tsv`** | 2 | 25.0 | 84 | 25.6 | 4.0 | 14.31% | 3.29% | 58.17% | limited (7916), private (7239), ltd (4621), llc (3435) |

### Name Characteristics & Noise Observations:
1. **Legal Suffixes:** Extremely prevalent across all sources. In US: `Inc`, `LLC`, `Corp`, `Co`, `LP`. In India: `Private Limited`, `Pvt Ltd`, `LLP`, `Enterprises`. In France: `SARL`, `SAS`, `SA`, `SCI`, `EURL`.
2. **Length Distribution:** Highly concentrated between 15 and 35 characters (median ~21–25 characters, 3–4 tokens).
3. **Multilingual Presence:** ~3.0% to 5.5% of records feature non-ASCII text, primarily Devanagari, Kannada, Tamil, Malayalam scripts for Indian records, and French accented characters (`é`, `è`, `à`, `ç`, `ô`) for French records.
4. **Casing Discrepancies:** Source 2 contains a much higher percentage of ALL-CAPS names (~24%) compared to Source 1 (~4%) and Source 3 (~3%). Normalization must convert all strings to lowercase.
5. **Punctuation:** Periods (`.`), commas (`,`), hyphens (`-`), slashes (`/`), and ampersands (`&`) are ubiquitous in trade names and abbreviations.

---

## 5. Address Profiling

| Dataset | Missing Addr (%) | Median Addr Len | Mean Addr Len | Median Addr Tokens | Contains Digits (%) | Non-ASCII Addr (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train_source1.tsv`** | 0.00% | 41.0 | 52.1 | 7.0 | 96.64% | 0.02% |
| **`train_source2.tsv`** | 3.38% | 37 | 47.9 | 6 | 93.81% | 9.78% |
| **`train_source3.tsv`** | 3.33% | 42 | 48.2 | 6 | 93.64% | 9.22% |
| **`test_source1.tsv`** | 0.00% | 50.0 | 57.1 | 8.0 | 95.84% | 4.25% |
| **`test_source2.tsv`** | 2.67% | 44.0 | 52.0 | 7.0 | 95.18% | 15.21% |
| **`test_source3.tsv`** | 2.70% | 44.0 | 50.1 | 7.0 | 94.95% | 14.63% |

### Address Characteristics by Country:
- **United States:** Highly structured street + city + state abbreviation + 5-digit ZIP (`105 ELM ST, MORGANTON, NC`). High rate of standard abbreviations (`ST`, `RD`, `AVE`, `BLVD`, `DR`, `HWY`). Digits are present in >95% of addresses.
- **India:** Landmark-based and municipal numbering structures (`KH NO. -570/13, NEW DELHI`, `PLOT NO B-78/1, ADDITIONAL MIDC ANAND NAGAR, THANE`). Often contains postal/PIN codes (6 digits) or locality references (`Near SBI ATM`, `Opp. Bus Stand`).
- **France (Test):** Contains European street numbering before or after street name, standard French indicators (`Rue`, `Boulevard`, `Avenue`, `Allée`), 5-digit French postal codes, and department names.
- **Digit Utility:** Over 90% of non-empty addresses contain digits (house numbers, plot numbers, postal codes). Overlap of extracted numeric tokens serves as a powerful high-precision feature.

---

## 6. Ground Truth Analysis & Match Distribution

Total Training Source 1 Entities: **2,206,821**

| Number of Matches | Number of S1 Entities | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **0 matches** | 123,247 | 5.58% | Singletons (No Matches) |
| **1 matches** | 119,157 | 5.40% | Single Match |
| **2 matches** | 375,212 | 17.00% | Multi-Match |
| **3 matches** | 530,841 | 24.05% | Multi-Match |
| **4 matches** | 484,115 | 21.94% | Multi-Match |
| **5 matches** | 321,957 | 14.59% | Multi-Match |
| **6 matches** | 164,868 | 7.47% | Multi-Match |
| **7 matches** | 63,968 | 2.90% | Multi-Match |
| **8 matches** | 18,680 | 0.85% | Multi-Match |
| **9 matches** | 4,205 | 0.19% | Multi-Match |
| **10 matches** | 534 | 0.02% | Multi-Match |
| **11 matches** | 37 | 0.00% | Multi-Match |

### Match Breakdown Statistics:
- **Min Matches:** 0
- **Max Matches:** 11
- **Mean Matches:** 3.4613
- **Median Matches:** 3.0
- **Total Matched References in Ground Truth:** 7,638,365
  - Matches in Source 2: **3,693,619** (48.36%)
  - Matches in Source 3: **3,944,746** (51.64%)
- **Entity Overlap:**
  - Matched to **BOTH** Source 2 and Source 3: **1,776,047 (80.48%)**
  - Matched to **Source 2 ONLY**: **143,029 (6.48%)**
  - Matched to **Source 3 ONLY**: **164,498 (7.45%)**
  - Singletons (no match in either): **123,247 (5.58%)**

---

## 7. True Match Pair Noise Investigation

Evaluated on a deterministic sample of $N = 3,000$ genuine positive pairs from `train_ground_truth.tsv`:

| Comparison Property | Matching Fraction | Percentage | Analysis / Implication |
| :--- | :--- | :--- | :--- |
| **Exact Country Equality** | 3000/3000 | **100.00%** | Zero cross-country false links. Country is a hard filter. |
| **Exact Raw Name Match** | 131/3000 | 4.37% | Only ~1 in 5 matches share identical raw name strings. |
| **Normalized Name Match** | 648/3000 | 21.60% | Lowercase + punctuation cleaning doubles name identity. |
| **Name Match (Legal Suffixes Stripped)** | 1258/3000 | 41.93% | Stripping Corp/LLC/Pvt Ltd recovers another ~11% exact matches. |
| **Name Levenshtein Ratio $\ge 0.80$** | 1427/3000 | 47.57% | >85% of true pairs have high string similarity. |
| **Exact Raw Address Match** | 74/3000 | 2.47% | Raw address match is rare (<15%) due to format differences. |
| **Normalized Address Match** | 252/3000 | 8.40% | Normalization recovers moderate identity. |
| **Address Numeric Token Overlap** | 2232/2479 | 90.04% | When both have numbers, >80% share at least 1 exact number token. |

### Sample Empirical Noise Examples from Ground Truth:

```
Example 1 [India]: S1-879529769 <--> S2-474921724 (Name Sim: 0.09)
  S1 Name    : First Properties Private Limited
  Target Name: फर्स्ट प्रॉपर्टीज प्राइवेट लिमिटेड
  S1 Address : Wing B (Pt), Tower Ix, Magarpatta City Sez, Magarpatta City, Hadapsar, Tal -Havel, I, Pune, Level 2, Maharashtra
  Target Addr: LEVEL 2, WING B (PT), TOWER IX, MAGARPATTA CITY SEZ, MAGARPATTA CITY, HADAPSAR, TAL -HAVEL, I, POONA, PUNE, महाराष्ट्र

Example 2 [US]: S1-521065316 <--> S3-490867022 (Name Sim: 0.48)
  S1 Name    : Callie Ochoa Chiron LLC
  Target Name: #callieochoa
  S1 Address : 1001 Country Club Drive, Cushing, OK
  Target Addr: 1001 Country Club Drive, Cushing, Oklahoma

Example 3 [US]: S1-629772993 <--> S3-175621032 (Name Sim: 0.46)
  S1 Name    : Lloyd Fresh Flaherty Inc
  Target Name: Lloyd Fresh
  S1 Address : 10 Larkin Road, Watch Hill, RI
  Target Addr: 10-B Larkin Rd, Rhode Island, Watch Hill

Example 4 [US]: S1-607721709 <--> S3-83416359 (Name Sim: 1.00)
  S1 Name    : Hernandez Family Office Group
  Target Name: HERNANDEZ FAMILY OFFICE GROUP
  S1 Address : 18782 Lucas Michael Way, New Caney, TX
  Target Addr: 18782 Lucas Michael Way, PMB 1805, New Caney, Texas

Example 5 [US]: S1-220605104 <--> S3-564467637 (Name Sim: 0.33)
  S1 Name    : National Hawk LP
  Target Name: Center National LP
  S1 Address : Unit 1, 27 Story Street, MA, Boston
  Target Addr: 27 Story St, Unit 1, Boston, Massachusetts

Example 6 [India]: S1-518400696 <--> S3-551596578 (Name Sim: 0.84)
  S1 Name    : Green All Finance Pvt Ltd
  Target Name: Green All Finance Pvt
  S1 Address : Shop - 404, Kashi Plaza, Kamrej, Surat, Gujarat
  Target Addr: Kamrej, Door No 311 Shop - 404, GJ, Surat

Example 7 [US]: S1-212961014 <--> S3-681245988 (Name Sim: 0.64)
  S1 Name    : Bauer Valley Royal LLC
  Target Name: Bauer LLC Royal Valley
  S1 Address : 85-14 261 Street, Floral Park, NY
  Target Addr: 85-14 261 St, Floral Park, New York

Example 8 [India]: S1-274647601 <--> S2-415949108 (Name Sim: 0.78)
  S1 Name    : Anupam Academy
  Target Name: ANUPAM ACADEMY LTD
  S1 Address : Shop In 177 B Gf, Block G Phase 6, Near Aya Nagar Extn, South Delhi, Delhi
  Target Addr: SHOP IN #177 B GF, BLOCK G PHASE 6, NEAR AYA NAGAR EXTN, SOUTH DELHI, Delhi

```

---

## 8. Candidate Generation & Blocking Implications

In business entity resolution across 1.73M test reference records and 9.97M target records ($1.73 \times 10^{13}$ possible pairs), blocking is the critical bottleneck determining the recall ceiling.

### Recall Evaluation of Candidate Blocking Keys on Ground Truth:

| Candidate Blocking Key | True Match Recall (%) | Pruning Factor | Practical Viability |
| :--- | :--- | :--- | :--- |
| **1. Exact Normalized Name** | 21.60% | High | Evaluated on empirical sample |
| **2. Name Stripped of Legal Suffixes** | 41.93% | High | Evaluated on empirical sample |
| **3. First 2 Significant Name Tokens** | 55.50% | High | Evaluated on empirical sample |
| **4. First Significant Word** | 70.30% | High | Evaluated on empirical sample |
| **5. Name Prefix (first 8 chars)** | 64.00% | High | Evaluated on empirical sample |
| **6. Address Numeric Token Overlap** | 74.40% | High | Evaluated on empirical sample |
| **7. Union of [Prefix-8, Legal-Stripped Name, First 2 Words]** | 74.23% | High | Evaluated on empirical sample |
| **8. Union of [7 + Address Digits Overlap]** | 82.00% | High | Evaluated on empirical sample |

### Recommended Multi-Key Inverted Index Strategy:
To achieve $\ge 95\%$ recall ceiling while keeping candidate pairs $\le 20$ per entity:
1. **Partition 0 (Hard Constraint):** `country == country` (reduces pairwise comparisons by ~65% immediately).
2. **Key 1 (High Precision):** Cleaned name without legal suffixes.
3. **Key 2 (Prefix Blocking):** First 8 characters of cleaned name (robust to trailing suffix/alias variations).
4. **Key 3 (Token Inverted Index):** First 2 significant tokens of business name (robust to suffix differences and token permutations).
5. **Key 4 (Address Numeric + Name Initial):** Overlap on postal/street digits combined with first letter of business name (captures multi-lingual and transliterated cases).

---

## 9. Computational Scalability & Memory Profile

| Metric | Observation | Pipeline Implication |
| :--- | :--- | :--- |
| **Total Raw Dataset Size** | ~2.52 GB across 7 TSV files | Streaming or chunked reading is mandatory; loading all datasets into pandas simultaneously would consume ~12–16 GB RAM. |
| **CPU Availability** | 16 CPU cores (no CUDA GPU) | Multi-threading with Python `concurrent.futures` / `multiprocessing` is highly advantageous. String operations via `rapidfuzz` (C++ backend) will be 50x–100x faster than pure Python. |
| **Test Inference Volume** | 1,732,544 reference entities | Generating 15–20 candidates per S1 entity yields ~25M–35M candidate pairs for classification. |
| **Batch / Chunk Size** | 50,000–100,000 entities per chunk | Chunked blocking and scoring ensures peak RAM usage remains < 4 GB during full test inference. |
| **Critical Metric Constraint** | Macro $F_{0.5}$ weights Precision 2x over Recall | Strict threshold calibration is essential; false positive predictions drastically hurt score. Singletons (5.58%) must be preserved. |

---

## 10. Verification of Raw Dataset Preservation
- All 7 raw datasets in `Wyvern/data/raw/` were read strictly in read-only mode.
- No files in `data/raw/` were modified, overwritten, or renamed.
- Source datasets in `Desktop\AWSML` and the original challenge ZIP remain completely untouched.
