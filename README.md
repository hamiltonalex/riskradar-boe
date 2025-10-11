# RiskRadar - AI-Powered Early Warning System for Financial Risk Detection

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://riskradar-boe.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)]()

**Multi-module LLM system for analyzing bank financial disclosures to detect early warning signals of financial risk**

Developed by Group 9 - **Overfit and Underpaid** - Cambridge Data Science Career Accelerator Program

---

## Project Statement

This project was developed as part of a data science challenge in collaboration with the **Bank of England (BoE)**. The objective was to assess whether large language models (LLMs) can be used to analyze public financial documents - such as bank annual reports and earnings call transcripts - to flag early signs of financial risk.

Financial regulators face the challenge of processing hundreds of pages of complex financial disclosures from multiple institutions in a timely manner. Manual review is time-consuming, resource-intensive, and may miss subtle patterns that emerge across documents. RiskRadar explores whether state-of-the-art LLMs can assist supervisors by automatically extracting structured risk signals, identifying red flags, and synthesizing comprehensive risk assessments aligned with the CAMELS supervisory framework.

The system processes 400+ page documents, extracts quantitative metrics (capital ratios, liquidity, credit quality), qualitative signals (sentiment, management confidence, analyst concerns), and produces actionable risk reports with full citation traceability back to source pages.

---

## Hypothesis

**The BoE aims to explore whether publicly available financial disclosures can be analyzed using state-of-the-art LLMs to generate early warning signals about financial risk.**

**Research Question:** 

By feeding annual reports and earnings call transcripts into large language models, can we identify red flags - such as deteriorating capital buffers, evasive management language, hidden off-balance-sheet exposures, or emerging litigation risks-before a crisis materializes?

**What this could deliver:**

- Automate first-pass reviews of lengthy documents, freeing up supervisors for higher-value analysis
- Spot subtle patterns across multiple reports that might be missed in manual review
- Apply consistent evaluation criteria systematically across institutions
- Every finding backed by citations to specific pages in source documents
- Faster regulatory review cycles without sacrificing thoroughness

---

## Implementation Approach

### Modular Multi-Specialist Architecture
We designed a **16-module system** where each specialized module analyzes a specific financial risk dimension:

**Tier 1: Linguistic Analysis Modules (4 modules)**

- Sentiment tracking - tone analysis, hedging language patterns, emotional signals
- Topic evolution - emerging themes, narrative shifts, what's being downplayed
- Management confidence - response quality, evasiveness indicators, credibility assessment
- Analyst concerns - reading between the lines of Q&A sessions to spot skepticism

**Tier 2: Quantitative Risk Modules (9 modules)**

- Capital adequacy - CET1, Tier 1, leverage ratios, MDA headroom
- Liquidity & funding - LCR, NSFR, deposit concentrations, wholesale funding reliance
- Market & interest rate risk - IRRBB sensitivities, unrealized losses, hedging gaps
- Credit quality - NPL ratios, Stage 2/3 migration trends, ECL coverage
- Earnings quality - ROE, NIM, cost efficiency, sustainability of profits
- Governance & controls - material weaknesses, auditor qualifications, board oversight
- Legal & regulatory - enforcement actions, pending litigation, breach disclosures
- Business model risks - revenue concentration, unsustainable growth, strategic pivots
- Off-balance-sheet - commitments, guarantees, derivatives, SPV relationships

**Tier 3: Pattern Detection Module (1 module)**

- Red flag scanner - searches for critical warning phrases ("going concern," "covenant breach," "material uncertainty")

**Tier 4: Meta-Analysis Modules (2 modules - sequential execution)**

- Discrepancy auditor - cross-checks for contradictions and missing critical metrics
- CAMELS synthesizer - aggregates findings into final risk assessment with traffic-light ratings (🟢 🟡 🔴)

### Execution Strategy

**2-Phase Pipeline:**

1. **Phase 1 (Parallel):** 14 modules run concurrently with rate limiting and backoff
2. **Phase 2 (Sequential):** Discrepancy Auditor -> CAMELS Fuser process aggregated results

### Dual Interface Design

**1. Streamlit Web Application**
([Live Demo](https://riskradar-boe.streamlit.app))

- **Purpose:** Interactive demo and proof-of-concept visualization
- Visual CAMELS risk heatmap with 🟢 Green / 🟡 Amber / 🔴 Red signals (⚪ for failed responses)
- Document upload and model selection interface
- Real-time analysis execution monitoring
- **Dual analysis modes:**
  - Standard 16-module analysis (text truncated for demo responsiveness)
  - RAG Analysis tab with Qdrant vector store (full document retrieval for Q&A)
- **Note:** For comprehensive full-document analysis with all 16 modules, use the Jupyter notebook. For semantic Q&A on full documents, use the RAG Analysis tab in Streamlit.

**2. Jupyter Notebooks**

Two notebook implementations are provided:

**a) Standard Multi-Module Implementation** **Recommended for Full Analysis**

[`Group9_CAM_EP_Assignment3_notebook.ipynb`](notebook/Group9_CAM_EP_Assignment3_notebook.ipynb)

- Processes entire 450-page documents (2M+ characters) using intelligent chunking
- 16-module architecture with parallel execution
- Near production-ready architecture (pending domain expert validation)
- Direct LLM prompting approach (no RAG infrastructure required)
- Offline, reproducible, and auditable execution
- Step-by-step documentation with extended comments
- Suitable for Google Colab with minimal setup
- Runtime: ~12-19 minutes for comprehensive full-document analysis
- Use case: Pilot deployment, regulatory review, comprehensive risk assessment

**b) RAG-Based Implementation**

[`Group9_CAM_EP_Assignment3_notebook_RAG.ipynb`](notebook/Group9_CAM_EP_Assignment3_notebook_RAG.ipynb)

- Retrieval-Augmented Generation approach
- Qdrant vector database for document storage and retrieval
- Embedding-based semantic search for targeted document sections
- Chunking strategy to handle 400+ page documents
- More efficient for large document processing
- Requires Qdrant setup (local or cloud)

### Interface Comparison

| Feature | Jupyter Notebook | Streamlit App |
|---------|------------------|---------------|
| **Purpose** | Production-capable analysis engine | Interactive demo & visualization |
| **Document Processing** | Full 450-page documents (2M+ chars) | Standard tab: truncated / RAG tab: full retrieval |
| **Analysis Depth** | Comprehensive, all modules | Standard: demo / RAG: semantic Q&A |
| **RAG Support** | RAG notebook available | Dedicated RAG Analysis tab |
| **Reproducibility** | Fully reproducible with logging | Interactive, not reproducible |
| **Deployment Target** | Regulatory review, pilot production | Stakeholder demos, concept validation |
| **Setup Complexity** | Medium (Jupyter/Colab) | Low (web browser only) |
| **Best For** | Actual risk assessment work | Understanding the approach + RAG Q&A |

**Recommendation:** For comprehensive regulatory analysis with all 16 modules, use the Jupyter notebook. For interactive Q&A on full documents, use the Streamlit RAG tab.

### Multi-LLM Provider Support

- **OpenAI:** GPT-4o, GPT-4o-mini, GPT-5-mini
- **Anthropic:** Claude Sonnet, Claude Opus
- **Google:** Gemini Pro, Gemini Flash

Configurable via unified `config.py` prompt registry with model-specific API routing.

### Configurable Prompt Framework
Each analysis module operates via a structured prompt template defined in [config.py](config.py):

- **JSON-enforced output format** for structured data extraction
- **Citation requirements** mandate source page references for every finding
- **Risk scoring rules** with explicit severity thresholds
- **Plug-and-play modularity** - new modules can be added without modifying core orchestration logic

### Two Implementation Approaches

**1. Standard Multi-Module Approach** (Primary Implementation)

- All 16 modules analyze the full document text
- Documents are passed in their entirety (with chunking for large files)
- Each module receives relevant sections based on its domain
- Suitable for most use cases
- **Available in:**
  - Jupyter Notebook: `Group9_CAM_EP_Assignment3_notebook.ipynb`
  - Streamlit App: Standard analysis tab (demo mode with truncation)

**2. RAG-Enhanced Approach**

- Documents ingested into Qdrant vector database
- Semantic search retrieves only relevant sections per query
- More efficient for very large documents (>400 pages)
- Reduces token consumption and enables full-document Q&A
- Requires additional infrastructure (Qdrant)
- **Available in:**
  - Jupyter Notebook: `Group9_CAM_EP_Assignment3_notebook_RAG.ipynb`
  - Streamlit App: Dedicated "RAG Analysis" tab (full document retrieval)

---

## Main Challenges

### 1. Document Length vs. Token Limits

Annual reports often exceed 400 pages, far beyond what most LLMs can handle in a single prompt. We addressed this through intelligent chunking with overlap, selective section extraction based on module needs, and an optional RAG module that retrieves only relevant passages.

### 2. LLM Rate Limits & Reliability

API rate caps (especially OpenAI's TPM limits) and inconsistent JSON formatting across providers created bottlenecks. We implemented semaphore-based throttling, exponential backoff with retry logic, and JSON repair utilities that clean up malformed responses.

### 3. Structured Output Validation

LLMs don't always return perfect JSON - sometimes they miss required fields or add extra commentary. We validate against JSON schemas, automatically retry on failures, fall back to regex extraction when needed, and log everything for debugging.

### 4. Cross-Platform Compatibility

Running across VSCode, Google Colab, and Streamlit Cloud meant dealing with different dependency management systems. We unified configuration through `config.py` with environment variable overrides, kept dependencies minimal, and added platform-specific files (`packages.txt`, `runtime.txt`) for Streamlit Cloud.

### 5. Citation Traceability

For regulatory use, every finding needs to trace back to a specific page in the source document. We built a custom `SourceTracker` utility, enforce citations in prompts, and use a consistent format: `(source_title p. page)`.

### 6. Reproducibility

LLMs are inherently non-deterministic, which is problematic for reproducible research. We set temperature=0.3, use seed parameters where supported, log all prompts and responses, and snapshot model versions.

---

## Advantages of Our Implementation

### Modular Architecture

The system uses a base class hierarchy where all analysis modules inherit shared functionality. Adding a new risk dimension means writing a new prompt and creating a subclass - no need to touch the core orchestration logic. Prompts live in `config.py`, completely separate from execution code.

### Flexibility & Scalability

Want to switch from OpenAI to Claude? Change one config line. The system is model-agnostic. We run 14 modules in parallel with a configurable thread pool, and there's an optional RAG module if you need more sophisticated document retrieval.

### Logging & Auditability

Everything gets logged: prompts, responses, token counts, execution times. Each module's JSON output is archived per run. The Streamlit app even has a real-time debug console that streams logs as analysis runs.

### Visual Risk Dashboard

The CAMELS heatmap uses traffic-light signals (🟢 Green / 🟡 Amber / 🔴 Red) for each of the 6 regulatory dimensions. Failed module responses show as ⚪ white/gray but still display available results. You can track trends across documents or over time, and every finding links back to the source document page number.

### Performance Optimization

We respect API rate limits through semaphore-based throttling, handle transient failures with exponential backoff and jitter, and support optional response caching for repeated queries.

### Explainability & Transparency

Every finding cites its source page. Structured JSON outputs enable programmatic analysis downstream. Confidence assessments flag missing critical metrics, and the discrepancy auditor catches contradictions between modules.

---

## Folder Structure

```
riskradar-boe/
├── app.py                                  # Streamlit web application entry point
├── config.py                               # Unified configuration and all 16 module prompts
├── requirements.txt                        # Python dependencies
├── packages.txt                            # System dependencies for Streamlit Cloud
├── runtime.txt                             # Python version specification
│
├── notebook/
│   ├── Group9_CAM_EP_Assignment3_notebook.ipynb      # Main 16-module implementation
│   └── Group9_CAM_EP_Assignment3_notebook_RAG.ipynb  # RAG-based implementation
│
├── src/
│   ├── agents/                             # Note: "agents" folder kept for backward compatibility
│   │   ├── base_agent.py                  # Base class with shared LLM routing logic
│   │   ├── orchestrator.py                # Risk synthesizer (16-module orchestration)
│   │   ├── sentiment_agent.py             # Sentiment and tone analysis module
│   │   ├── topic_agent.py                 # Topic evolution tracking module
│   │   ├── confidence_agent.py            # Management confidence evaluation module
│   │   ├── analyst_agent.py               # Analyst concern extraction module
│   │   ├── capital_agent.py               # Capital buffers module (CET1, leverage)
│   │   ├── liquidity_agent.py             # Liquidity & funding module (LCR, NSFR)
│   │   ├── market_irrbb_agent.py          # Market & interest rate risk module
│   │   ├── credit_agent.py                # Credit quality module (NPL, Stage 2/3)
│   │   ├── earnings_agent.py              # Earnings quality module (ROE, NIM)
│   │   ├── governance_agent.py            # Governance & controls module
│   │   ├── legal_agent.py                 # Legal & regulatory risk module
│   │   ├── business_model_agent.py        # Business model analysis module
│   │   ├── off_balance_sheet_agent.py     # Off-balance-sheet exposures module
│   │   ├── red_flags_agent.py             # Red flag pattern detection module
│   │   ├── discrepancy_agent.py           # Cross-module discrepancy audit
│   │   ├── camels_fusion_agent.py         # CAMELS fusion & final report generator
│   │   └── rag_module.py                  # Optional RAG system (Qdrant + embeddings)
│   │
│   ├── data/
│   │   ├── data_loader.py                 # Transcript and document loaders
│   │   ├── document_processor.py          # PDF text extraction
│   │   └── transcript_processor.py        # Transcript-specific processing
│   │
│   ├── models/
│   │   └── model_evaluator.py             # Model comparison utilities
│   │
│   └── utils/
│       ├── debug_logger.py                # Centralized logging system
│       ├── source_tracker.py              # Citation tracking utilities
│       ├── document_viewer.py             # Document preview for Streamlit
│       └── document_validator.py          # File validation and sanitization
│
├── data/
│   ├── transcripts/                        # Earnings call transcripts (PDF)
│   └── financial_docs/                     # Annual reports (PDF)
│
├── docs/
│   ├── notebook/                           # Architecture diagrams (Mermaid exports)
│   │   ├── RiskRadar_1_System_Architecture_Overview.pdf
│   │   ├── RiskRadar_2_Main_Execution_Pipeline_Flow.pdf
│   │   ├── RiskRadar_3_Multi-module_Architecture_And_Routing.pdf
│   │   ├── RiskRadar_4_Parallel_Execution_Sequence_with_Rate_Limiting.pdf
│   │   └── RiskRadar_5_Aggregation_Strategy_Decision_Flow.pdf
│   └── streamlit/                          # Streamlit-specific documentation
│
├── slides/                                 # Assignment presentations
│   ├── Group9_CAM_EP_Assignment1_Plan.pdf
│   ├── Group9_CAM_EP_Assignment2_Presentation.pdf
│   └── Group9_CAM_EP_Assignment3_Report.pdf
│
└── videos/                                 # Demo recordings
    ├── RiskRadar_Analysis_Streamlit_app_DEMO_short.mp4
    ├── RiskRadar_Analysis_Notebook_DEMO_short.mp4
    └── RAG_Streamlit_single_prompt.mp4
```

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)

### Local Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/hamiltonalex/riskradar-boe.git
   cd riskradar-boe
   ```

2. **Create virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API keys:**

   Create a `.env` file in the root directory:
   
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

5. **Or open the Jupyter notebook:**

   ```bash
   # Standard multi-module approach:
   jupyter notebook notebook/Group9_CAM_EP_Assignment3_notebook.ipynb

   # OR RAG-based approach (requires Qdrant):
   jupyter notebook notebook/Group9_CAM_EP_Assignment3_notebook_RAG.ipynb
   ```

### Google Colab Execution

**Standard Multi-Module Approach:**

- Upload [`Group9_CAM_EP_Assignment3_notebook.ipynb`](notebook/Group9_CAM_EP_Assignment3_notebook.ipynb) to Colab
- Add API keys to Colab secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)
- Run cells sequentially (runtime: ~12-19 minutes for full analysis with 8-14 modules in parallel)

**RAG-Based Approach:**

- Upload [`Group9_CAM_EP_Assignment3_notebook_RAG.ipynb`](notebook/Group9_CAM_EP_Assignment3_notebook_RAG.ipynb) to Colab
- Set up Qdrant (can use Qdrant Cloud or local Docker instance)
- Add API keys and Qdrant configuration to Colab secrets (`QDRANT_URL`, `QDRANT_API_KEY`)
- Run cells sequentially for RAG-enhanced document analysis

---

## Key Features

- 16 specialized modules cover both qualitative (sentiment, topics, management tone) and quantitative (capital, liquidity, credit quality) risk dimensions
- Works with OpenAI, Claude, or Gemini - just change one config setting
- All outputs in structured JSON with schema validation and mandatory source citations
- 14 modules run in parallel with intelligent rate limiting and backoff
- Visual CAMELS dashboard shows traffic-light risk ratings (🟢 🟡 🔴) across 6 regulatory dimensions
- Every finding links back to a specific page in the source document
- Complete audit trail: logged prompts, responses, token usage, execution times
- Two interfaces: Streamlit (demo + RAG Q&A), Jupyter notebook (production analysis)
- RAG support in both Streamlit (dedicated tab) and Jupyter (separate notebook) with Qdrant vector store
- Fully reproducible with documented seeds, versions, and dependencies
- The notebook processes entire 450-page documents (2M+ characters) using intelligent chunking

---

## Documentation

### Architecture Diagrams

- [System Architecture Overview](docs/notebook/RiskRadar_1_System_Architecture_Overview.pdf)
- [Main Execution Pipeline Flow](docs/notebook/RiskRadar_2_Main_Execution_Pipeline_Flow.pdf)
- [Multi-module Architecture and Routing](docs/notebook/RiskRadar_3_Multi-module_Architecture_And_Routing.pdf)
- [Parallel Execution Sequence with Rate Limiting](docs/notebook/RiskRadar_4_Parallel_Execution_Sequence_with_Rate_Limiting.pdf)
- [Aggregation Strategy Decision Flow](docs/notebook/RiskRadar_5_Aggregation_Strategy_Decision_Flow.pdf)

### Presentations

- [Assignment 1: Project Plan](slides/Group9_CAM_EP_Assignment1_Plan.pdf)
- [Assignment 2: Preliminary Solution Pitch](slides/Group9_CAM_EP_Assignment2_Presentation.pdf)
- [Assignment 3: Final Report & Presentation](slides/Group9_CAM_EP_Assignment3_Report.pdf)

### Demo Videos

- [Streamlit App Demo (Short)](videos/RiskRadar_Analysis_Streamlit_app_DEMO_short.mp4)
- [Jupyter Notebook Demo (Short)](videos/RiskRadar_Analysis_Notebook_DEMO_short.mp4)
- [RAG Module Single Prompt Demo](videos/RAG_Streamlit_single_prompt.mp4)

---

## Team

**Overfit and Underpaid**
Cambridge Data Science Career Accelerator - Employer Project P1 2025

**Academic Partner:** Bank of England

**Course:** CAM DS 401 - Employer Project

---

## Acknowledgements

This project was developed in collaboration with the **Bank of England** as part of the Cambridge Data Science Career Accelerator program. We thank the BoE supervisors and Cambridge instructors for their guidance, feedback, and domain expertise.

Special thanks to:

- Bank of England Prudential Regulation Authority for regulatory context
- Cambridge Data Science program faculty
- Open-source LLM providers (OpenAI, Anthropic, Google) for API access

---

## Disclaimer

**For Academic and Research Purposes Only**

This system is a **proof-of-concept** developed for educational purposes. It is **not intended for production use** in actual regulatory supervision without extensive validation, human oversight, and compliance with applicable regulations.

**Limitations:**

- LLM outputs may contain hallucinations or inaccuracies
- Incomplete or missing source data will result in degraded analysis quality
- System requires manual verification of all findings before regulatory action
- Not all regulatory frameworks or document types have been tested
- Performance varies significantly across LLM providers and model versions
- **Streamlit web app has dual modes:** standard analysis tab (truncates for demo performance) and RAG Analysis tab (full document retrieval for Q&A)
- **Notebook implementation is near production-ready** but requires domain expert validation before operational deployment

**Regulatory Compliance:**

- Users are responsible for ensuring compliance with data protection regulations (GDPR, etc.)
- Sensitive or confidential financial documents should not be uploaded to third-party LLM APIs without proper security controls
- Always consult qualified financial supervisors and legal counsel before relying on automated risk assessments

---

## License

This project is licensed for academic use.

---

**Live Demo:** [https://riskradar-boe.streamlit.app](https://riskradar-boe.streamlit.app)

---

*Last Updated: October 2025*
