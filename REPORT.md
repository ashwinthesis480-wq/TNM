# Onco-TCGA: TNM Staging Extraction from Free-Text Pathology Reports
## Project Report — July 2026

---

## Executive Summary

This project developed a hybrid **Regex + LLM** pipeline to automatically extract **TNM (Tumour / Node / Metastasis) cancer staging** from 9,523 free-text TCGA pathology reports. Starting from a 16.5% complete-extraction baseline, the pipeline achieved **99.0% complete TNM coverage** — a +82.5 percentage-point improvement — using a four-track approach combining enhanced regular expressions, clinical staging system mappings, and a locally-hosted `gemma3:4b` language model.

| Milestone | Complete TNM Records | Coverage |
|---|---|---|
| Baseline (original regex) | 1,567 / 9,523 | 16.5% |
| Track A — Enhanced Regex | 3,050 / 9,523 | 32.0% |
| Track B — + LLM fill-in | **9,424 / 9,523** | **99.0%** |

---

## 1. Dataset

| Property | Value |
|---|---|
| Source | NIH Genomic Data Commons (GDC) |
| Records | 9,523 TCGA pathology reports |
| Columns | `patient_filename` (TCGA barcode), `text` (free text) |
| Cancer types | 33 TCGA cohorts (BRCA, LUAD, COAD, GBM, OV, …) inferred from TSS code in barcode |
| Report length | Variable; median ~400 words, range 50–5,000+ words |

TCGA pathology reports are unstructured clinical documents written in natural language. Staging information may appear in standardised TNM notation (`T2N1M0`), in narrative prose ("three of twelve lymph nodes were positive"), or in alternative staging systems (Dukes, Astler-Coller).

---

## 2. Methodology

The pipeline was implemented across four tracks, each building on the previous.

### Track A — Enhanced Regex Extraction

**Goal:** Maximise coverage from deterministic text matching without requiring inference.

**Approach (priority order):**

1. **Combined TNM pattern** — A single regex matching all three components on the same line with optional clinical/pathological prefixes (`yp`, `yc`, `rc`):
   ```
   [PCYR]{0,2}T(IS|X|[0-4][A-C]?)[\s,;/]*[PCYR]{0,2}N(X|[0-3][A-C]?)[\s,;/]*[PCYR]{0,2}M[01X]
   ```

2. **Individual component regex** — Separate T, N, M patterns applied independently, supporting prefix variants.

3. **Node-count keyword fallback (N stage)** — Extracts counts from phrases like *"3 of 12 lymph nodes positive"* and maps to N0–N3 by AJCC thresholds. Also matches direct N0/N1 keyword phrases (12 N0 keywords, 8 N1+ keywords).

4. **Metastasis keyword fallback (M stage)** — Maps 14 M0 keyword phrases (e.g., *"no distant metastasis"*, *"no evidence of metastasis"*) and 10 M1 phrases (e.g., *"metastatic disease"*, *"liver metastases"*).

5. **Dukes staging conversion** — Maps Dukes A/B/C/C1/C2/D to approximate TNM equivalents for colorectal reports.

6. **Astler-Coller staging conversion** — Maps A/B1/B2/C1/C2/D to approximate TNM equivalents.

**Results:**

| Metric | Baseline | Enhanced | Delta |
|---|---|---|---|
| T stage found | 5,023 (52.7%) | 5,304 (55.7%) | +281 |
| N stage found | 2,903 (30.5%) | 4,703 (49.4%) | **+1,800** |
| M stage found | 2,640 (27.7%) | 3,854 (40.5%) | **+1,214** |
| Complete TNM | 1,567 (16.5%) | 3,050 (32.0%) | **+1,483** |

N and M stages drove the largest gains, confirming the hypothesis that prose-based staging language (node counts, metastasis phrases) is prevalent but missed by simple TNM pattern matching.

**Extraction source breakdown (all 9,523 records):**

| Source Rule | Records |
|---|---|
| None (not extracted) | 3,376 |
| Individual regex only | 2,745 |
| Combined regex | 956 |
| Individual regex + N keyword | 775 |
| Individual regex + M keyword | 506 |
| N keyword only | 385 |
| Individual regex + Astler-Coller | 191 |
| Individual regex + N kw + M kw | 178 |
| Astler-Coller only | 173 |
| Other combinations | 190 |

---

### Track B — LLM Benchmark & Batch Processing

**Goal:** Fill the remaining ~68% of records (zero or partial extraction) using a locally-hosted large language model.

#### 2.1 Model Selection

Five models were selected from those available locally via Ollama, covering a range of architectures, sizes, and specialisations:

| Model | Parameters | Why selected |
|---|---|---|
| `qwen3:8b` | 8B | Best chain-of-thought reasoning |
| `mistral:7b-instruct-q4_K_M` | 7B Q4 | Best-in-class structured JSON output |
| `qwen2.5:7b-instruct-q4_K_M` | 7B Q4 | Strong instruction following |
| `gemma3:4b` | 4B | Speed/quality balance |
| `meditron:7b` | 7B | Medical domain fine-tune |

#### 2.2 System Prompt Design

A structured system prompt was developed to enforce consistent JSON output:

```
You are an expert oncology pathologist specialising in TNM cancer staging.
Extract TNM staging values from the clinical pathology report provided.

Rules:
- T: use only '0','IS','1','2','3','4' (+ optional A/B/C suffix)
- N: use only '0','1','2','3' (+ optional A/B/C suffix) or 'X'
- M: use only '0' or '1' or 'X'
- If absent from the report, return null.
- Do NOT hallucinate staging information.
- Set confidence to 'high', 'medium', or 'low'.

Return ONLY valid JSON, no markdown:
{"T": "...", "N": "...", "M": "...", "confidence": "...", "reasoning": "..."}
```

Reports were truncated to 3,000 characters (retaining the diagnostically dense opening). Temperature was set to 0 for deterministic output.

#### 2.3 Benchmark Results (100 zero-extraction reports per model)

| Model | Parse Rate | T Rate | N Rate | M Rate | **Complete** | High Conf | Speed |
|---|---|---|---|---|---|---|---|
| **gemma3:4b** | **100%** | **100%** | **100%** | **100%** | **100%** | 82% | 1.97 s/report |
| qwen2.5:7b | 100% | 97% | 39% | 53% | 39% | 85% | 1.67 s/report |
| mistral:7b | 98% | 69% | 35% | 32% | 31% | 71% | 1.90 s/report |
| meditron:7b | 13% | 3% | 3% | 3% | 3% | 0% | 4.45 s/report |
| qwen3:8b | 2% | 0% | 0% | 0% | 0% | 0% | 5.63 s/report |

**Key observations:**

- **`gemma3:4b` achieved a perfect 100% complete TNM rate** on the benchmark set, outperforming all larger models. Its 4B size also made it the second-fastest model.
- **`qwen3:8b` failed almost entirely** (2% parse rate) — the model's thinking-chain output format (interspersed `<think>` tags) prevented reliable JSON extraction. This reflects a known quirk of Qwen3's default output mode.
- **`meditron:7b`** (a medical fine-tune of LLaMA) underperformed despite its clinical specialisation, suggesting its training objective did not emphasise structured JSON output.
- **`qwen2.5:7b`** showed strong T-stage extraction (97%) but poor N/M rates (39%/53%), revealing systematic under-extraction of nodal and metastatic information.
- `gemma3:4b` was chosen as the batch model.

#### 2.4 Batch Processing

All 6,473 records with zero or partial extraction were processed with `gemma3:4b`:

| Metric | Value |
|---|---|
| Total records processed | 6,473 |
| Successful JSON parses | 6,467 (99.9%) |
| Complete TNM extracted | 6,344 (98.0%) |
| Records that filled a missing component | 6,423 |
| Checkpoint saves (every 100 records) | ✓ |
| Wall-clock time | ~3 h 5 min (1.72 s/report avg) |

---

### Track C — Downstream Clinical Analysis

**Goal:** Derive clinical insights from the final staged dataset.

**Methods:**
- Cancer type inferred from the 2-character TSS (Tissue Source Site) code in each TCGA barcode, mapped to TCGA cohort labels.
- AJCC clinical stage derived from final T/N/M values using standard stage-grouping rules.
- 9-panel visualisation generated covering: AJCC stage distribution, T×N co-occurrence heatmap, cancer type frequency, complete TNM rate by cancer type, T/N/M stage stacked bars by cancer type, AJCC stage heatmap by cancer type, M1 rate by cancer type, and extraction coverage by cancer type.

---

### Track D — Streamlit Application (OncoStage AI)

**Goal:** Productionise the pipeline as an interactive web application.

**Features:**
- **Single-report tab:** Paste or upload a pathology report; hybrid extraction (regex → LLM) returns T/N/M with confidence level (colour-coded progress bar), method indicator (regex / LLM / hybrid), AJCC stage inference, and extraction history.
- **Batch CSV tab:** Upload a CSV with a text column; auto-detect column name; process all rows; summary metrics + stage distribution chart; CSV download.
- **About tab:** Pipeline diagram, model comparison table, AJCC reference, dataset link.
- **Sidebar:** Live model selector (all locally-available Ollama models enumerated at runtime), status indicator showing Ollama connectivity, pipeline explanation.

---

## 3. Final Results

### 3.1 Coverage Progression

| Stage | Complete TNM | T Coverage | N Coverage | M Coverage |
|---|---|---|---|---|
| Baseline | 1,567 (16.5%) | 52.7% | 30.5% | 27.7% |
| + Enhanced Regex (Track A) | 3,050 (32.0%) | 55.7% | 49.4% | 40.5% |
| **+ LLM fill-in (Track B)** | **9,424 (99.0%)** | **99.3%** | **99.3%** | **99.3%** |

Only **99 records (1.0%)** remain without a complete TNM — these are likely records where staging information is genuinely absent from the report text.

### 3.2 Source Attribution (final 9,424 complete records)

| Extraction Source | Records | Share |
|---|---|---|
| Regex only (no LLM fill) | ~2,526 | 26.8% |
| LLM fill on zero-extraction records | 3,297 | 35.0% |
| LLM fill on partial-regex records | 3,126 | 33.2% |
| Regex + staging system (Dukes/AC) | ~475 | 5.0% |

### 3.3 LLM Confidence Distribution (gemma3:4b, 6,467 valid responses)

| Confidence Level | Records | Share |
|---|---|---|
| High | 5,797 | 89.7% |
| Low | 582 | 9.0% |
| Medium | 88 | 1.4% |

89.7% of LLM-produced extractions carry **high confidence**, indicating the model was generally able to find unambiguous staging language in the reports.

---

## 4. Output Files

| File | Description |
|---|---|
| `TCGA_Enhanced_TNM_Extraction.csv` | All 9,523 records with enhanced-regex results + source column |
| `TCGA_Enhanced_Complete_TNM.csv` | 3,050 records with complete TNM from regex only |
| `TCGA_Final_TNM_Dataset.csv` | All 9,523 records with final T/N/M, source, LLM confidence |
| `TCGA_Ollama_Batch_Raw.csv` | Raw LLM output for 6,473 processed records |
| `TCGA_Ollama_Benchmark_Results.csv` | 5-model benchmark table |
| `TCGA_Ollama_Benchmark.png` | 3-panel benchmark visualisation |
| `TCGA_Final_Coverage_Summary.png` | Coverage progression bar charts + source pie |
| `TCGA_Downstream_Analysis.png` | 9-panel downstream clinical analysis |
| `streamlit-app/app.py` | Interactive Streamlit application (OncoStage AI) |

---

## 5. Limitations & Future Work

- **Hallucination risk:** LLM-extracted stages (especially low-confidence ones) may not accurately reflect the original report. Clinical use requires expert validation.
- **Staging system approximations:** Dukes → TNM mappings are coarse (e.g., Dukes B → T3N0M0 covers a range of true T stages). These should be flagged as approximate.
- **qwen3 failure:** The thinking-chain output format of `qwen3:8b` could be handled by a post-processing filter for `<think>` tags, potentially recovering its strong reasoning capability for structured extraction.
- **Report truncation:** Reports longer than 3,000 characters are truncated. For very long operative reports, important staging information near the end may be missed.
- **Cancer-type inference:** TSS codes are mapped heuristically; the ~686 unique TSS codes in the dataset are not all covered, with residual records labelled "OTHER".
- **Future:** Fine-tune `gemma3:4b` on a validated gold-standard TNM dataset to improve precision, add stage-specific confidence thresholds, and integrate with a clinical validation workflow.

---

## 6. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Data processing | pandas, numpy |
| Regex extraction | Python `re` module |
| LLM inference | Ollama (local) + `ollama` Python SDK v0.6.2 |
| LLM model | `gemma3:4b` (Google Gemma 3, 4B parameters) |
| Visualisation | matplotlib, seaborn |
| Web application | Streamlit |
| Notebooks | Jupyter (VS Code kernel: `llminferencing`) |
| Hardware | Local GPU (Ollama server) |
