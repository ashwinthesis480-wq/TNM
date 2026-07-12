# Onco-TCGA: Automated TNM Staging Extraction from Pathology Reports

A hybrid information extraction pipeline built on The Cancer Genome Atlas (TCGA) pathology reports.
The system extracts **Tumor (T)**, **Node (N)**, and **Metastasis (M)** staging values from free-text clinical reports and maps them to an overall AJCC cancer stage.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Extraction Rules & Decision Basis](#extraction-rules--decision-basis)
   - [Rule 1 – Combined TNM Pattern Matching](#rule-1--combined-tnm-pattern-matching)
   - [Rule 2 – Individual Component Fallback Regex](#rule-2--individual-component-fallback-regex)
   - [Rule 3 – Tumor Size Inference (T stage)](#rule-3--tumor-size-inference-t-stage)
   - [Rule 4 – Keyword-Based T Stage Detection](#rule-4--keyword-based-t-stage-detection)
   - [Rule 5 – Keyword-Based N Stage Detection](#rule-5--keyword-based-n-stage-detection)
   - [Rule 6 – Node Count Inference (N stage)](#rule-6--node-count-inference-n-stage)
   - [Rule 7 – Keyword-Based M Stage Detection](#rule-7--keyword-based-m-stage-detection)
   - [Rule 8 – Hybrid Confidence-Gated Pipeline](#rule-8--hybrid-confidence-gated-pipeline)
   - [Rule 9 – TNM-to-Stage Conversion Table](#rule-9--tnm-to-stage-conversion-table)
4. [Basis for Design Decisions](#basis-for-design-decisions)
5. [Current Limitations](#current-limitations)
6. [Enhancement Roadmap: Better Methods & Techniques](#enhancement-roadmap-better-methods--techniques)
   - [NLP & NER Techniques](#1-nlp--ner-techniques)
   - [Transformer-Based Deep Learning](#2-transformer-based-deep-learning)
   - [Clinical NLP Toolkits](#3-clinical-nlp-toolkits)
   - [Structured Information Extraction](#4-structured-information-extraction)
   - [Ensemble & Voting Methods](#5-ensemble--voting-methods)
   - [Active Learning & Annotation](#6-active-learning--annotation)
   - [Retrieval-Augmented Generation (RAG)](#7-retrieval-augmented-generation-rag)
   - [Graph-Based Reasoning](#8-graph-based-reasoning)
   - [Evaluation & Validation Improvements](#9-evaluation--validation-improvements)
   - [Post-Processing & Normalization](#10-post-processing--normalization)
7. [Quick Start](#quick-start)
8. [Data Files](#data-files)

---

## Project Overview

Clinical pathology reports in TCGA are free-text documents. Staging information is embedded in varied natural language, abbreviations, and structured tables depending on the cancer type and reporting institution. This project builds a pipeline to:

1. Parse 9,523 TCGA pathology reports.
2. Extract T, N, M components using deterministic rules and regex.
3. Fall back to a large language model (GPT-4o-mini) when rules are insufficient.
4. Map the extracted TNM combination to an AJCC overall stage (0 – IV).
5. Expose results via a Streamlit web application supporting both single-report and batch-CSV modes.

---

## Repository Structure

```
Onco-TCGA/
├── TCGA_Reports.csv                       # Raw TCGA pathology reports (9 523 rows)
├── TCGA_Complete_TNM_Extraction.csv       # Reports with complete T+N+M extraction
├── TCGA_Partial_TNM_Extraction.csv        # Reports with at least one component found
├── TCGA_Reports_with_TNM_Extraction.csv   # Full dataset with all extraction flags
├── TCGA_TNM_Extraction_Summary.csv        # Coverage statistics & metrics
├── T_stage_1_reports.csv                  # Filtered export: T stage = 1
├── N_stage_0_reports.csv                  # Filtered export: N stage = 0
├── M_stage_0_reports.csv                  # Filtered export: M stage = 0
├── TCGA_EDA.ipynb                         # Exploratory data analysis notebook
└── streamlit-app/
    └── app.py                             # Interactive Streamlit UI
```

---

## Extraction Rules & Decision Basis

The extraction logic is split across two layers:

- **Notebook (`TCGA_EDA.ipynb`)** – bulk regex extraction for the full 9,523-report dataset.
- **Streamlit app (`app.py`)** – single-report interactive extraction with an LLM fallback.

Both layers share the same conceptual rule set, described below.

---

### Rule 1 – Combined TNM Pattern Matching

**Where used:** `TCGA_EDA.ipynb` → `extract_tnm_components()`

**What it does:**
The extractor first tries to match all three components in a single pass using the combined pattern:

```
(?<![A-Z])([PCYR]?T(?:IS|X|[0-4][A-C]?))[\s,]*([PCYR]?N(?:X|[0-3][A-C]?))[\s,]*([PCYR]?M[01X])(?![A-Z0-9])
```

This matches formats like:

| Input Text | Matched |
|---|---|
| `T2 N1 M0` | T=2, N=1, M=0 |
| `pT2N1M0` | T=2, N=1, M=0 |
| `T2,N1,M0` | T=2, N=1, M=0 |
| `pT2 pN1 pM0` | T=2, N=1, M=0 |
| `yT3N2M1` | T=3, N=2, M=1 |

**Why combined-first:**
When all three components appear near each other the combined match is more precise and avoids the risk of the individual patterns accidentally picking up unrelated letters elsewhere in the report.

**Regex anchoring decisions:**
- `(?<![A-Z])` – negative lookbehind prevents matching T/N/M embedded inside other words (e.g., `DEPTH3`, `EXTENT2`).
- `[\s,]*` – allows zero or more spaces/commas between components, handling both concatenated (`T2N1M0`) and spaced (`T2 N1 M0`) representations.
- `(?![A-Z0-9])` – lookahead at the end prevents false continuation of the match.
- Prefix group `[PCYR]?` covers: `p` (pathological), `c` (clinical), `y` (post-treatment), `r` (recurrence).

---

### Rule 2 – Individual Component Fallback Regex

**Where used:** `TCGA_EDA.ipynb` → `extract_tnm_components()` (fallback branch)

**What it does:**
When the combined pattern fails (components are scattered across the report), each component is searched independently.

```python
# T stage
r'(?<![A-Z])([PCYR]?T)(IS|X|[0-4][A-C]?)(?![A-BD-LO-WY-Z0-9])'

# N stage
r'(?<![A-Z])([PCYR]?N)(X|[0-3][A-C]?)(?![A-BD-LO-WY-Z0-9])'

# M stage
r'(?<![A-Z])([PCYR]?M)([01X])(?![A-Z0-9])'
```

**Lookahead design for T and N:**
The lookahead `(?![A-BD-LO-WY-Z0-9])` deliberately allows `C`, `M`, `N`, `X` to follow, so that a text like `T2N1` (partial concatenation without M) still correctly extracts T=2 and then N=1 separately.

**Uppercase normalization:**
All text is uppercased before regex application so that lowercase variants (`pt1`, `Pt1`, `pT1`) are all captured identically.

---

### Rule 3 – Tumor Size Inference (T stage)

**Where used:** `app.py` → `extract_tumor_stage()`

**What it does:**
When no explicit T-code is found in the text, tumor size in centimeters is parsed and converted to a T stage using standard AJCC size cutoffs:

| Tumor Size | Assigned T Stage |
|---|---|
| 0 cm | T0 |
| ≤ 2 cm | T1 |
| > 2 cm and ≤ 5 cm | T2 |
| > 5 cm and ≤ 7 cm | T3 |
| > 7 cm | T4 |

**Basis:**
These cutoffs correspond to the AJCC 8th Edition breast cancer T-stage size criteria, which are also used as conservative approximations for other solid tumors when cancer-type-specific thresholds are not available.

**Pattern used:**
```python
r'(\d+\.?\d*)\s*(?:cm|centimeter)'
```

---

### Rule 4 – Keyword-Based T Stage Detection

**Where used:** `app.py` → `extract_tumor_stage()`

**What it does:**
Fallback after both explicit regex and size inference fail. Keywords are matched against the lowercased report text:

| Matched Keyword | Assigned T Stage |
|---|---|
| `no tumor`, `no evidence of tumor`, `tumor-free` | T0 |
| `carcinoma in situ`, `in situ` | Tis |

**Basis:**
These phrases are standard reporting conventions in surgical pathology. `Tis` (tumor in situ) is an important special case indicating pre-invasive disease with a different prognosis from T1.

---

### Rule 5 – Keyword-Based N Stage Detection

**Where used:** `app.py` → `extract_node_stage()`

**What it does:**
Keyword scanning after explicit N-code regex fails:

**N0 keywords** (no nodal involvement):
```
"no lymph node", "lymph nodes negative", "no nodal involvement",
"nodes are negative", "0 out of", "no regional lymph"
```

**N-positive keywords** (nodal involvement present):
```
"lymph node involvement", "lymph node metastasis", "positive lymph node",
"nodal involvement", "nodes positive"
```

**Basis:**
Pathologists reliably use these phrases when lymph node status is discussed. Negative node reports almost always include explicit language such as "lymph nodes negative" or "0 out of N nodes positive," making keyword matching highly reliable for the N0 assignment.

---

### Rule 6 – Node Count Inference (N stage)

**Where used:** `app.py` → `extract_node_stage()`

**What it does:**
When N-positive keywords are detected, the number of positive nodes is extracted using:

```python
r'(\d+)\s*(?:out of|/)\s*\d+\s*(?:lymph\s*)?nodes?\s*positive'
```

The count is then mapped to an N sub-stage:

| Positive Node Count | Assigned N Stage |
|---|---|
| 0 | N0 |
| 1 – 3 | N1 |
| 4 – 9 | N2 |
| ≥ 10 | N3 |

**Basis:**
AJCC 8th Edition (breast cancer) and most solid tumor guidelines define N stage by positive lymph node count in this way. Using the count directly is more precise than relying solely on presence/absence keywords.

---

### Rule 7 – Keyword-Based M Stage Detection

**Where used:** `app.py` → `extract_metastasis_stage()`

**What it does:**
Keyword lists are used after the explicit M-code regex fails:

**M0 keywords** (no distant metastasis):
```
"no metastasis", "no distant metastasis", "no evidence of metastasis",
"metastasis: none", "no distant spread", "no evidence of distant"
```

**M1 keywords** (distant metastasis present):
```
"distant metastasis", "metastatic", "metastasis present",
"metastasis to", "metastases"
```

**Basis:**
Metastasis status is typically stated explicitly in TCGA pathology reports and the vocabulary used is highly standardized. Keyword matching for M stage has the highest precision of the three components because the language used is less ambiguous than for T or N.

---

### Rule 8 – Hybrid Confidence-Gated Pipeline

**Where used:** `app.py` → `perform_extraction()`

**What it does:**
The pipeline applies a confidence gate to decide when to invoke the LLM:

```
Rule-Based Extraction
        │
        ▼
All 3 components found? ──YES──► Return result (confidence = high, method = rule-based)
        │
       NO
        ▼
LLM Extraction (GPT-4o-mini)
        │
        ▼
LLM returns valid T+N+M? ──YES──► Merge rules + LLM ──► Return (method = hybrid or llm)
        │
       NO
        ▼
Return partial result with TX/NX/MX placeholders (confidence = low)
```

**Merge logic:**
When rules extracted some components but not all, rule-based values take precedence over LLM values for those components (rule values are considered higher-precision for the specific patterns they match).

**Basis:**
Deterministic rules are extremely fast and highly precise for explicit TNM codes. The LLM is reserved for ambiguous reports where inference and contextual reasoning are needed, balancing cost and latency against coverage.

---

### Rule 9 – TNM-to-Stage Conversion Table

**Where used:** `app.py` → `tnm_to_stage()`

**What it does:**
Converts the extracted (T, N, M) tuple into an AJCC overall stage using a hard-coded decision table:

| Condition | Stage |
|---|---|
| M = M1 (any T, any N) | Stage IV |
| Tis + N0 | Stage 0 |
| T0 or T1 + N0 | Stage I |
| T1 + N1/N1a/N1b | Stage IIA |
| T2 + N0 | Stage IIA |
| T2 + N1/N1a/N1b | Stage IIB |
| T3 + N0 | Stage IIB |
| T3 + N1 or N2 | Stage IIIA |
| T4 + N0 or N1 | Stage IIIA |
| T3/T4 + N2 | Stage IIIB |
| Any T + N3 | Stage IIIC |
| T4 (remaining) | Stage IIIB |
| N2 (remaining) | Stage III |

**Basis:**
The table follows AJCC 8th Edition breast cancer staging as a cross-cancer approximation. Full per-cancer staging requires cancer-type-specific tables, which is noted as a future enhancement.

---

## Basis for Design Decisions

| Decision | Rationale |
|---|---|
| Regex over full NLP parsing | No external NLP dependency; fast batch processing over 9,523 reports |
| Combined pattern tried first | Reduces ambiguity — components close together are almost certainly related |
| Negative lookbehind `(?<![A-Z])` | Prevents false positives from words like `EXTENT`, `DEPTH`, `INVASION` |
| Uppercase normalization | TCGA reports mix casing; normalization avoids duplicated patterns |
| Rule-based before LLM | LLM adds cost and latency; reserved as fallback to minimize both |
| Hard-coded stage table | Deterministic, auditable, and free from hallucination risk |
| Three-tier export (complete / partial / none) | Downstream analysts can choose quality level appropriate to their study |

---

## Current Limitations

1. **Single staging schema:** The TNM-to-stage table is primarily calibrated for breast cancer. Other cancers (lung, colon, prostate, etc.) have different size cutoffs and node count thresholds.
2. **No negation handling:** Phrases like "no T2 involvement" could erroneously trigger a T2 match if only the standalone regex is used.
3. **No context window:** A T-code found in a comparison sentence ("previously staged T3, now...") could overwrite the current staging.
4. **Abbreviation collisions:** Letters N and M appear in many medical abbreviations; the lookahead/lookbehind anchors reduce but do not eliminate false matches.
5. **Alternative staging systems:** Reports using Dukes, Astler-Coller, or Clark staging are not captured.
6. **No uncertainty quantification:** Confidence is a three-level categorical label (high/medium/low) rather than a calibrated probability.

---

## Enhancement Roadmap: Better Methods & Techniques

### 1. NLP & NER Techniques

**Named Entity Recognition (NER)**
- Train a token-level NER model (BIO tagging) where each token is labelled as `B-T_STAGE`, `I-T_STAGE`, `B-N_STAGE`, `I-N_STAGE`, `B-M_STAGE`, `I-M_STAGE`, or `O`.
- This handles multi-token staging mentions and automatically learns contextual boundaries without hand-crafted lookbehinds.

**Negation Detection**
- Apply `NegEx` or `pyConTextNLP` before pattern matching to detect negated mentions.
- Negation scope detection prevents "no T2 evidence" from being extracted as T2.

**Section Segmentation**
- Parse report structure into labeled sections: `GROSS DESCRIPTION`, `MICROSCOPIC FINDINGS`, `STAGING`, `DIAGNOSIS`, etc.
- Apply TNM extraction only within staging-relevant sections to reduce false positives from narrative text.

**Coreference Resolution**
- Resolve pronouns and noun phrases back to their referent so that "the lesion (3.5 cm) ... it invades" correctly attributes the size to the primary tumor.

---

### 2. Transformer-Based Deep Learning

**BioBERT / PubMedBERT**
- Pre-trained on PubMed abstracts and biomedical literature.
- Fine-tune on a labelled TCGA TNM annotation dataset for sequence labelling or span extraction.
- Significantly outperforms regex on ambiguous, implicit, or paraphrased staging language.

**ClinicalBERT / BlueBERT**
- Trained on MIMIC-III clinical notes; better suited to discharge summaries and pathology report style than general-domain BERT.

**Longformer / BigBird**
- TCGA reports can be several thousand tokens long. Longformer's sparse attention mechanism handles documents beyond the 512-token BERT window without truncation.

**GPT-4 with Structured Output (JSON Mode)**
- Use `response_format={"type": "json_object"}` with a strict JSON schema to guarantee well-formed TNM output and eliminate the regex post-processing step currently applied to LLM responses.

**T5 / BART Seq2Seq**
- Frame TNM extraction as a text-to-text generation problem: input = full report, output = structured JSON string.
- Seq2seq models can learn to aggregate evidence across the entire document in one pass.

---

### 3. Clinical NLP Toolkits

| Toolkit | Strengths | Relevant Features |
|---|---|---|
| **Apache cTAKES** | UIMA-based, production-tested | TNM and cancer staging annotators built-in |
| **MedCAT** | Active learning, concept linking | SNOMED-CT / UMLS concept normalisation for stage values |
| **CLAMP** | HIPAA-compliant cloud + local | Named entity recognition, relation extraction |
| **scispaCy** | Drop-in spaCy replacement | Biomedical NER models (`en_ner_bc5cdr_md`, etc.) |
| **medspaCy** | spaCy extension | Context detection (negation, uncertainty, family history) |

---

### 4. Structured Information Extraction

**Relation Extraction**
- Go beyond entity detection to extract (entity, relation, entity) triples.
- Example: `(tumor, has_size, 3.2 cm)` and `(3.2 cm, maps_to, T2)` as two linked facts.

**Template Filling**
- Define a structured template `{T: ?, N: ?, M: ?, size_cm: ?, nodes_positive: ?, nodes_total: ?}`.
- Use a reading comprehension model (e.g., fine-tuned BERT on SQuAD) to fill each slot from the document.

**OpenIE / Information Extraction**
- Extract open-domain triples to discover implicit staging evidence not captured by hand-crafted keyword lists.

---

### 5. Ensemble & Voting Methods

- Run multiple independent extractors in parallel (regex, NER, LLM, cTAKES).
- Apply a **voting scheme** (majority vote or weighted vote based on past precision) to select the final label per component.
- Use **confidence-based arbitration**: if two methods agree with high confidence, accept; if they disagree, escalate to the highest-precision method or request human review.
- **Stacked generalization**: train a meta-classifier whose features are the outputs of all base extractors; the meta-classifier learns which extractor to trust in which context.

---

### 6. Active Learning & Annotation

- Use **uncertainty sampling**: route reports where the model's confidence is lowest to a human annotator.
- Annotated reports are added back to the training set, iteratively improving the model with minimal annotation cost.
- Tools: **Label Studio**, **Prodigy** (spaCy's annotation tool), **CVAT** (for structured tasks).
- Establish a gold-standard evaluation set of 500+ manually annotated TCGA reports to measure precision/recall improvements objectively.

---

### 7. Retrieval-Augmented Generation (RAG)

- Build a vector store of AJCC staging guidelines and cancer-type-specific criteria (e.g., breast, lung, colon, prostate).
- At inference time, retrieve the relevant guideline section for the cancer type mentioned in the report.
- Provide the retrieved guideline as context in the LLM prompt so staging decisions are grounded in the correct disease-specific criteria rather than a generic cross-cancer approximation.

**Benefit:** Removes the need for hard-coded per-cancer stage tables; the LLM dynamically applies the correct AJCC edition rules.

---

### 8. Graph-Based Reasoning

**Knowledge Graphs**
- Build a medical knowledge graph with nodes for cancer types, T/N/M values, staging criteria, and prognosis data.
- Use **Graph Attention Networks (GAT)** or **Graph Convolutional Networks (GCN)** to propagate evidence across the graph and resolve ambiguous staging.

**Multi-Hop Reasoning**
- When staging evidence is spread across multiple sentences (e.g., size in sentence 1, node count in sentence 5), a graph-based reading comprehension model can link these pieces across hops.

---

### 9. Evaluation & Validation Improvements

| Technique | Description |
|---|---|
| **Token-level F1** | Measure precision, recall, F1 for each TNM component independently |
| **Exact-match accuracy** | Require all three components to be correct simultaneously |
| **Confusion matrix per stage value** | Identify which T/N/M values are most commonly confused |
| **Inter-annotator agreement (IAA)** | Measure Cohen's kappa between human annotators to establish a performance ceiling |
| **Calibration curves** | Plot predicted confidence against actual accuracy to assess if "high confidence" truly means high accuracy |
| **Error analysis taxonomy** | Categorise errors as: missed explicit code, wrong context, negation error, format variant, alternative staging system |

---

### 10. Post-Processing & Normalization

**Cancer-Type-Specific Stage Tables**
- Load AJCC 8th Edition tables per cancer type (breast, lung, colon, prostate, kidney, etc.) and apply the correct table based on the cancer type detected in the report.

**UMLS / SNOMED Normalisation**
- Map extracted free-text staging values to UMLS CUI codes for interoperability with EHR systems.

**Duplicate and Conflict Resolution**
- When multiple TNM codes appear in a single report (e.g., clinical and pathological staging), apply a priority rule: `pTNM > cTNM > ycTNM > rTNM`.

**Temporal Filtering**
- Use date parsing to extract the most recent staging episode when a patient's report includes re-staging after treatment.

**Confidence Calibration**
- Replace the current three-level categorical confidence with a calibrated probability score using Platt scaling or isotonic regression on a held-out validation set.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
pip install streamlit pandas openai

# 3. Run the Streamlit app
cd streamlit-app
streamlit run app.py
```

For LLM-enhanced extraction, set environment variables before launching:

```powershell
$env:AI_INTEGRATIONS_OPENAI_BASE_URL = "https://your-endpoint"
$env:AI_INTEGRATIONS_OPENAI_API_KEY  = "your-api-key"
```

For the EDA notebook, open `TCGA_EDA.ipynb` in VS Code or JupyterLab and run cells sequentially.

---

## Data Files

| File | Rows | Description |
|---|---|---|
| `TCGA_Reports.csv` | 9,523 | Raw pathology reports (`patient_filename`, `text`) |
| `TCGA_Complete_TNM_Extraction.csv` | ~2,902 | Reports with T + N + M all extracted |
| `TCGA_Partial_TNM_Extraction.csv` | ~4,623 | Reports with at least one component |
| `TCGA_Reports_with_TNM_Extraction.csv` | 9,523 | Full dataset + boolean flags |
| `TCGA_TNM_Extraction_Summary.csv` | – | Coverage metrics and statistics |
| `T_stage_1_reports.csv` | – | Filtered: T stage = 1 |
| `N_stage_0_reports.csv` | – | Filtered: N stage = 0 |
| `M_stage_0_reports.csv` | – | Filtered: M stage = 0 |

---

*Dataset: The Cancer Genome Atlas (TCGA) Pathology Reports*  
*Staging Standard: AJCC 8th Edition (cross-cancer approximation)*  
*Extraction Method: Regex + Keyword Rules + GPT-4o-mini LLM Fallback*
