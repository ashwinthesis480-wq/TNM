import streamlit as st
import pandas as pd
import re
import json
import io
import os
from openai import OpenAI

# ─────────────────────────────────────────────
# Rule-Based TNM Extractor
# ─────────────────────────────────────────────

def extract_tumor_stage(text: str) -> str | None:
    """Extract T stage using regex and keyword matching."""
    lower = text.lower()

    # Look for explicit T staging (e.g., T2, T3a)
    explicit = re.search(r'\b(t[0-4][a-c]?)\b', lower)
    if explicit:
        return explicit.group(1).upper()

    # Infer from tumor size in cm
    size_match = re.search(r'(\d+\.?\d*)\s*(?:cm|centimeter)', text, re.IGNORECASE)
    if size_match:
        size = float(size_match.group(1))
        if size <= 0:
            return "T0"
        elif size <= 2:
            return "T1"
        elif size <= 5:
            return "T2"
        elif size <= 7:
            return "T3"
        else:
            return "T4"

    # Keyword-based detection
    if any(kw in lower for kw in ["no tumor", "no evidence of tumor", "tumor-free"]):
        return "T0"
    if any(kw in lower for kw in ["carcinoma in situ", "in situ"]):
        return "Tis"

    return None


def extract_node_stage(text: str) -> str | None:
    """Extract N stage using regex and keyword matching."""
    lower = text.lower()

    # Look for explicit N staging
    explicit = re.search(r'\b(n[0-3][a-c]?)\b', lower)
    if explicit:
        return explicit.group(1).upper()

    # Keywords for N0
    n0_keywords = [
        "no lymph node", "lymph nodes negative",
        "no nodal involvement", "nodes are negative",
        "0 out of", "no regional lymph"
    ]
    if any(kw in lower for kw in n0_keywords):
        return "N0"

    # Keywords for N1+
    n_positive_keywords = [
        "lymph node involvement", "lymph node metastasis",
        "positive lymph node", "nodal involvement", "nodes positive"
    ]
    if any(kw in lower for kw in n_positive_keywords):
        count_match = re.search(
            r'(\d+)\s*(?:out of|/)\s*\d+\s*(?:lymph\s*)?nodes?\s*positive', lower
        )
        if count_match:
            count = int(count_match.group(1))
            if count == 0:
                return "N0"
            elif count <= 3:
                return "N1"
            elif count <= 9:
                return "N2"
            else:
                return "N3"
        return "N1"

    return None


def extract_metastasis_stage(text: str) -> str | None:
    """Extract M stage using regex and keyword matching."""
    lower = text.lower()

    # Look for explicit M staging
    explicit = re.search(r'\b(m[0-1][a-c]?)\b', lower)
    if explicit:
        return explicit.group(1).upper()

    # Keywords for M0
    m0_keywords = [
        "no metastasis", "no distant metastasis",
        "no evidence of metastasis", "metastasis: none",
        "no distant spread", "no evidence of distant"
    ]
    if any(kw in lower for kw in m0_keywords):
        return "M0"

    # Keywords for M1
    m1_keywords = [
        "distant metastasis", "metastatic",
        "metastasis present", "metastasis to", "metastases"
    ]
    if any(kw in lower for kw in m1_keywords):
        return "M1"

    return None


def rule_based_extraction(report_text: str) -> dict:
    """Run full rule-based TNM extraction."""
    T = extract_tumor_stage(report_text)
    N = extract_node_stage(report_text)
    M = extract_metastasis_stage(report_text)
    confident = T is not None and N is not None and M is not None
    return {"T": T, "N": N, "M": M, "confident": confident}


# ─────────────────────────────────────────────
# TNM → Stage Converter
# ─────────────────────────────────────────────

def tnm_to_stage(T: str, N: str, M: str) -> str:
    """Convert TNM values to overall cancer stage."""
    if M == "M1":
        return "Stage IV"

    t = T.upper()
    n = N.upper()

    if t == "TIS" and n == "N0":
        return "Stage 0"
    if t in ("T0", "T1") and n == "N0":
        return "Stage I"
    if t == "T1" and n in ("N1", "N1A", "N1B"):
        return "Stage IIA"
    if t == "T2" and n == "N0":
        return "Stage IIA"
    if t == "T2" and n in ("N1", "N1A", "N1B"):
        return "Stage IIB"
    if t == "T3" and n == "N0":
        return "Stage IIB"
    if t == "T3" and n in ("N1", "N2"):
        return "Stage IIIA"
    if t == "T4" and n in ("N0", "N1"):
        return "Stage IIIA"
    if t in ("T3", "T4") and n == "N2":
        return "Stage IIIB"
    if n == "N3":
        return "Stage IIIC"
    if t == "T4":
        return "Stage IIIB"
    if n == "N2":
        return "Stage III"
    if t in ("T0", "TIS"):
        return "Stage 0"
    if t == "T1":
        return "Stage I"
    if t == "T2":
        return "Stage II"
    if t == "T3":
        return "Stage III"

    return "Unknown"


# ─────────────────────────────────────────────
# LLM-Based TNM Extractor
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an oncology expert specializing in TNM cancer staging classification.

Extract TNM staging from the clinical report provided by the user.

Rules:
- Use standard TNM classification (T0-T4, N0-N3, M0-M1)
- T stage: T0 (no tumor), Tis (in situ), T1 (<=2cm), T2 (2-5cm), T3 (>5cm), T4 (any size with extension)
- N stage: N0 (no nodes), N1 (1-3 nodes), N2 (4-9 nodes), N3 (10+ nodes)
- M stage: M0 (no metastasis), M1 (distant metastasis)
- If uncertain, choose the closest conservative stage
- Do NOT hallucinate or invent information not present in the report
- Assess your confidence as "high", "medium", or "low"

Return ONLY valid JSON with no markdown formatting:
{"T": "", "N": "", "M": "", "confidence": "", "explanation": ""}"""


def llm_extraction(report_text: str) -> dict | None:
    """Use OpenAI LLM to extract TNM staging."""
    #base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    #api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    #api_key = "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"

    #if not base_url or not api_key:
    #    return None

    try:
        #client = OpenAI(api_key="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")#, base_url=base_url)
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Clinical Report:\n{report_text}"},
            ],
        )
        content = response.choices[0].message.content.strip()
        cleaned = re.sub(r'```json\n?', '', content)
        cleaned = re.sub(r'```\n?', '', cleaned).strip()
        return json.loads(cleaned)
    except Exception as e:
        st.warning(f"LLM extraction failed: {e}")
        return None


# ─────────────────────────────────────────────
# Hybrid Extraction (Rule-Based + LLM)
# ─────────────────────────────────────────────

def perform_extraction(report_text: str) -> dict:
    """Hybrid extraction: rules first, LLM fallback."""
    rules = rule_based_extraction(report_text)

    # If rules are fully confident, use them directly
    if rules["confident"]:
        stage = tnm_to_stage(rules["T"], rules["N"], rules["M"])
        return {
            "T": rules["T"],
            "N": rules["N"],
            "M": rules["M"],
            "stage": stage,
            "confidence": "high",
            "method": "rule-based",
            "explanation": (
                f"Extracted via pattern matching: "
                f"Tumor stage {rules['T']} based on size/keywords, "
                f"Node stage {rules['N']} based on lymph node findings, "
                f"Metastasis {rules['M']} based on metastasis keywords."
            ),
        }

    # Try LLM fallback
    llm_result = llm_extraction(report_text)

    if llm_result and llm_result.get("T") and llm_result.get("N") and llm_result.get("M"):
        T = rules["T"] or llm_result["T"]
        N = rules["N"] or llm_result["N"]
        M = rules["M"] or llm_result["M"]
        stage = tnm_to_stage(T, N, M)
        is_hybrid = rules["T"] or rules["N"] or rules["M"]
        return {
            "T": T,
            "N": N,
            "M": M,
            "stage": stage,
            "confidence": "medium" if is_hybrid else llm_result.get("confidence", "medium"),
            "method": "hybrid" if is_hybrid else "llm",
            "explanation": llm_result.get("explanation", ""),
        }

    # Partial rule-based fallback
    T = rules["T"] or "TX"
    N = rules["N"] or "NX"
    M = rules["M"] or "MX"
    stage = tnm_to_stage(T, N, M)
    return {
        "T": T,
        "N": N,
        "M": M,
        "stage": stage,
        "confidence": "low",
        "method": "rule-based",
        "explanation": "Partial extraction via rules only (LLM unavailable). Some staging values could not be determined.",
    }


# ─────────────────────────────────────────────
# Sample Reports
# ─────────────────────────────────────────────

SAMPLE_REPORTS = {
    "Clear TNM Case - Breast Cancer": """SURGICAL PATHOLOGY REPORT
Patient: Jane Doe, 58F
Specimen: Left breast mastectomy

DIAGNOSIS: Invasive ductal carcinoma, left breast

TUMOR SIZE: 3.2 cm in greatest dimension
HISTOLOGIC GRADE: Grade 2 (moderately differentiated)
MARGINS: All margins negative, closest margin 0.8 cm

LYMPH NODE STATUS: 2 out of 12 axillary lymph nodes positive for metastatic carcinoma
LYMPHOVASCULAR INVASION: Present

DISTANT METASTASIS: No evidence of distant metastasis

TNM STAGING: T2 N1 M0
HORMONE RECEPTORS: ER positive (95%), PR positive (80%)
HER2: Negative (1+ by IHC)""",

    "Ambiguous Case - Lung Cancer": """PATHOLOGY REPORT
Patient: John Smith, 67M
Specimen: Right upper lobe wedge resection

DIAGNOSIS: Non-small cell lung carcinoma, adenocarcinoma subtype

FINDINGS:
A mass measuring approximately 4.5 cm was identified in the right upper lobe.
The tumor appears to invade the visceral pleura. Some areas show possible chest wall involvement, though this is difficult to assess definitively on the current specimen.

Several hilar lymph nodes were sampled. There is suspicious cellular activity in at least one lymph node, but definitive metastatic involvement is equivocal.

Additional imaging studies have shown a small indeterminate lesion in the liver measuring 0.6 cm. Clinical correlation is recommended to rule out metastatic disease.

MARGINS: Negative""",

    "No Metastasis Case - Colon Cancer": """SURGICAL PATHOLOGY REPORT
Patient: Maria Garcia, 72F
Specimen: Sigmoid colectomy

DIAGNOSIS: Adenocarcinoma of the sigmoid colon

GROSS DESCRIPTION: An ulcerated mass measuring 2.8 cm is identified in the sigmoid colon.

MICROSCOPIC FINDINGS:
- Moderately differentiated adenocarcinoma
- Tumor invades through the muscularis propria into pericolorectal tissues
- No lymphovascular invasion identified
- Perineural invasion: Not identified

LYMPH NODES: 0 out of 18 lymph nodes positive for malignancy. All lymph nodes are negative for metastatic carcinoma.

DISTANT METASTASIS: No metastasis. PET scan and CT imaging show no evidence of distant spread.

MARGINS: Proximal and distal margins negative. Radial margin negative.""",
}


# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="OncoStage AI - TNM Extractor",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 OncoStage AI - TNM Staging Extractor")
st.markdown("Extract **Tumor (T)**, **Node (N)**, and **Metastasis (M)** staging from clinical pathology reports using a hybrid rule-based + LLM approach.")
st.divider()

# Tabs for different modes
tab1, tab2, tab3 = st.tabs(["📝 Single Report", "📊 Batch CSV", "📜 Code Reference"])

# ─── Tab 1: Single Report ───
with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Clinical Report Input")

        # Sample report buttons
        st.markdown("**Quick Fill with Sample Reports:**")
        sample_cols = st.columns(3)
        for i, (title, text) in enumerate(SAMPLE_REPORTS.items()):
            with sample_cols[i]:
                if st.button(title.split(" - ")[0], key=f"sample_{i}", use_container_width=True):
                    st.session_state["report_text"] = text

        report_text = st.text_area(
            "Paste your clinical pathology report below:",
            value=st.session_state.get("report_text", ""),
            height=400,
            key="report_input",
        )

        # File upload
        uploaded_file = st.file_uploader("Or upload a TXT file:", type=["txt"])
        if uploaded_file is not None:
            report_text = uploaded_file.read().decode("utf-8")
            st.session_state["report_text"] = report_text

        extract_btn = st.button("🔍 Extract TNM", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Extraction Results")

        if extract_btn and report_text.strip():
            with st.spinner("Analyzing report..."):
                result = perform_extraction(report_text)

            # Store result in session state
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].insert(0, {
                "report": report_text[:100] + "...",
                **result,
            })

            # Display TNM badges
            tnm_cols = st.columns(3)
            with tnm_cols[0]:
                st.metric("Tumor (T)", result["T"])
            with tnm_cols[1]:
                st.metric("Node (N)", result["N"])
            with tnm_cols[2]:
                st.metric("Metastasis (M)", result["M"])

            st.divider()

            # Stage and confidence
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Overall Stage", result["stage"])
            with info_cols[1]:
                confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.metric("Confidence", f"{confidence_emoji.get(result['confidence'], '')} {result['confidence'].title()}")
            with info_cols[2]:
                st.metric("Method", result["method"].replace("-", " ").title())

            st.divider()

            # Explanation
            st.markdown("**Explanation:**")
            st.info(result["explanation"])

            # JSON output
            with st.expander("📋 Raw JSON Output"):
                st.json(result)

            # Download result
            result_csv = pd.DataFrame([result])
            csv_data = result_csv.to_csv(index=False)
            st.download_button(
                "⬇️ Download Result as CSV",
                data=csv_data,
                file_name="tnm_result.csv",
                mime="text/csv",
            )

        elif extract_btn:
            st.warning("Please enter or upload a clinical report first.")
        else:
            st.info("Enter a clinical report and click **Extract TNM** to see results.")

# ─── Tab 2: Batch CSV ───
with tab2:
    st.subheader("Batch Processing")
    st.markdown("Upload a CSV file with a `report_text` column to process multiple reports at once.")

    csv_file = st.file_uploader("Upload CSV file:", type=["csv"], key="csv_upload")

    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write(f"Found **{len(df)}** rows in the CSV.")

        if "report_text" not in df.columns:
            st.error("CSV must contain a `report_text` column.")
        else:
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Process All Reports", type="primary"):
                results = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    with st.spinner(f"Processing report {i+1}/{len(df)}..."):
                        try:
                            result = perform_extraction(row["report_text"])
                            results.append({
                                "row": i + 1,
                                "T": result["T"],
                                "N": result["N"],
                                "M": result["M"],
                                "stage": result["stage"],
                                "confidence": result["confidence"],
                                "method": result["method"],
                                "explanation": result["explanation"],
                            })
                        except Exception as e:
                            results.append({
                                "row": i + 1,
                                "T": "Error",
                                "N": "Error",
                                "M": "Error",
                                "stage": "Unknown",
                                "confidence": "low",
                                "method": "error",
                                "explanation": str(e),
                            })
                    progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)
                st.success(f"Processed {len(results)} reports!")
                st.dataframe(results_df, use_container_width=True)

                csv_output = results_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download All Results as CSV",
                    data=csv_output,
                    file_name="tnm_batch_results.csv",
                    mime="text/csv",
                )

# ─── Tab 3: Code Reference ───
with tab3:
    st.subheader("Python Code Reference")
    st.markdown("Here's the core Python logic for TNM extraction that powers this application.")

    with st.expander("🔧 Rule-Based Extractor (`rule_based_extraction`)", expanded=True):
        st.code('''
import re

def extract_tumor_stage(text: str) -> str | None:
    """Extract T stage using regex and keyword matching."""
    lower = text.lower()

    # Look for explicit T staging (e.g., T2, T3a)
    explicit = re.search(r\'\\b(t[0-4][a-c]?)\\b\', lower)
    if explicit:
        return explicit.group(1).upper()

    # Infer from tumor size in cm
    size_match = re.search(r\'(\\d+\\.?\\d*)\\s*(?:cm|centimeter)\', text, re.IGNORECASE)
    if size_match:
        size = float(size_match.group(1))
        if size <= 0: return "T0"
        elif size <= 2: return "T1"
        elif size <= 5: return "T2"
        elif size <= 7: return "T3"
        else: return "T4"

    # Keyword-based detection
    if any(kw in lower for kw in ["no tumor", "no evidence of tumor"]):
        return "T0"
    if "carcinoma in situ" in lower:
        return "Tis"

    return None


def extract_node_stage(text: str) -> str | None:
    """Extract N stage using regex and keyword matching."""
    lower = text.lower()

    explicit = re.search(r\'\\b(n[0-3][a-c]?)\\b\', lower)
    if explicit:
        return explicit.group(1).upper()

    if any(kw in lower for kw in ["no lymph node", "lymph nodes negative"]):
        return "N0"

    if any(kw in lower for kw in ["lymph node involvement", "nodes positive"]):
        count_match = re.search(r\'(\\d+)\\s*(?:out of|/)\\s*\\d+.*?positive\', lower)
        if count_match:
            count = int(count_match.group(1))
            if count == 0: return "N0"
            elif count <= 3: return "N1"
            elif count <= 9: return "N2"
            else: return "N3"
        return "N1"

    return None


def extract_metastasis_stage(text: str) -> str | None:
    """Extract M stage using regex and keyword matching."""
    lower = text.lower()

    explicit = re.search(r\'\\b(m[0-1][a-c]?)\\b\', lower)
    if explicit:
        return explicit.group(1).upper()

    if any(kw in lower for kw in ["no metastasis", "no distant metastasis"]):
        return "M0"

    if any(kw in lower for kw in ["distant metastasis", "metastatic", "metastases"]):
        return "M1"

    return None
''', language="python")

    with st.expander("🧠 LLM Extractor (`llm_extraction`)"):
        st.code('''
from openai import OpenAI
import json

SYSTEM_PROMPT = """You are an oncology expert.
Extract TNM staging from the clinical report.
Rules:
- Use standard TNM classification (T0-T4, N0-N3, M0-M1)
- If uncertain, choose the closest conservative stage
- Do NOT hallucinate

Return ONLY JSON:
{"T": "", "N": "", "M": "", "confidence": "", "explanation": ""}"""

def llm_extraction(report_text: str) -> dict:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical Report:\\n{report_text}"},
        ],
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)
''', language="python")

    with st.expander("🔀 Hybrid Extraction (`perform_extraction`)"):
        st.code('''
def perform_extraction(report_text: str) -> dict:
    """Hybrid extraction: rules first, LLM fallback."""
    rules = rule_based_extraction(report_text)

    # If rules are fully confident, use them directly
    if rules["confident"]:
        stage = tnm_to_stage(rules["T"], rules["N"], rules["M"])
        return {
            "T": rules["T"], "N": rules["N"], "M": rules["M"],
            "stage": stage,
            "confidence": "high",
            "method": "rule-based",
            "explanation": "Extracted via pattern matching.",
        }

    # Try LLM fallback
    llm_result = llm_extraction(report_text)
    if llm_result:
        T = rules["T"] or llm_result["T"]
        N = rules["N"] or llm_result["N"]
        M = rules["M"] or llm_result["M"]
        stage = tnm_to_stage(T, N, M)
        return {
            "T": T, "N": N, "M": M,
            "stage": stage,
            "confidence": "medium",
            "method": "hybrid",
            "explanation": llm_result.get("explanation", ""),
        }

    # Partial fallback
    return {
        "T": rules["T"] or "TX",
        "N": rules["N"] or "NX",
        "M": rules["M"] or "MX",
        "stage": "Unknown",
        "confidence": "low",
        "method": "rule-based",
        "explanation": "Partial extraction only.",
    }
''', language="python")

    with st.expander("📊 TNM to Stage Converter (`tnm_to_stage`)"):
        st.code('''
def tnm_to_stage(T: str, N: str, M: str) -> str:
    """Convert TNM values to overall cancer stage."""
    if M == "M1":
        return "Stage IV"

    t, n = T.upper(), N.upper()

    if t == "TIS" and n == "N0": return "Stage 0"
    if t in ("T0", "T1") and n == "N0": return "Stage I"
    if t == "T2" and n == "N0": return "Stage IIA"
    if t == "T2" and n == "N1": return "Stage IIB"
    if t == "T3" and n == "N0": return "Stage IIB"
    if t == "T3" and n in ("N1", "N2"): return "Stage IIIA"
    if t == "T4" and n in ("N0", "N1"): return "Stage IIIA"
    if t in ("T3", "T4") and n == "N2": return "Stage IIIB"
    if n == "N3": return "Stage IIIC"

    return "Unknown"
''', language="python")

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### About OncoStage AI")
    st.markdown("""
    **OncoStage AI** extracts TNM cancer staging from 
    clinical pathology reports using a hybrid approach:
    
    1. **Rule-Based Layer** - Pattern matching with regex 
       for tumor size, lymph node keywords, and metastasis indicators
    2. **LLM Layer** - AI fallback using GPT for ambiguous cases
    3. **Hybrid** - Combines both for best results
    """)

    st.divider()
    st.markdown("### TNM Quick Reference")
    st.markdown("""
    | Stage | Description |
    |-------|-------------|
    | **T0** | No tumor |
    | **T1** | ≤ 2 cm |
    | **T2** | 2-5 cm |
    | **T3** | > 5 cm |
    | **T4** | Any size + extension |
    | **N0** | No node involvement |
    | **N1** | 1-3 nodes |
    | **N2** | 4-9 nodes |
    | **N3** | 10+ nodes |
    | **M0** | No metastasis |
    | **M1** | Distant metastasis |
    """)

    # History
    if st.session_state.get("history"):
        st.divider()
        st.markdown("### Extraction History")
        for i, entry in enumerate(st.session_state["history"][:5]):
            with st.expander(f"#{i+1}: {entry['T']}{entry['N']}{entry['M']} → {entry['stage']}"):
                st.write(f"**Confidence:** {entry['confidence']}")
                st.write(f"**Method:** {entry['method']}")
                st.write(f"**Report:** {entry['report']}")
