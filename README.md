# ⚡ Telecom Conversation Intelligence & Quality Assurance Microservice

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)](tests/)
[![Groundedness](https://img.shields.io/badge/Quote%20Grounding-100%25-success.svg)](benchmarks/)
[![Latency P95](https://img.shields.io/badge/Live%20Stream%20P95-11.98ms-blueviolet.svg)](benchmarks/)

A production-grade conversation intelligence microservice for enterprise telecom contact centers. Designed to replace manual 2% sample audits with **100% automated, explainable, real-time conversation analytics, live agent assist, and supervisor quality rollups**.

---

## Table of Contents
1. [Architecture Diagram](#1-architecture-diagram)
2. [Key Capabilities & Features](#2-key-capabilities--features)
3. [Full Executable Codebase & Execution Guide](#3-full-executable-codebase--execution-guide)
4. [Additional Exploration: Acoustic Dynamics, CPNI Redaction & HITL](#4-additional-exploration)
5. [Evals on System Health & Benchmarks](#5-evals-on-system-health--benchmarks)
6. [Production Scale Considerations (200,000 Calls/Day)](#6-production-scale-considerations)
7. [Repository Structure](#7-repository-structure)

---

## 1. Architecture Diagram

The system employs a dual-engine architecture: a **real-time stream engine** delivering sub-300ms live next-best-actions (NBA) and instant CPNI compliance interventions during calls, paired with a **post-call grounded QA & analytics engine** that generates auditable scorecards, sentiment arcs, churn predictions, and supervisor rollups.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Gateway Layer"]
        A["Telecom Corpus (8,300+ Multi-Turn Dialogue CSV Dataset)"] --> B["CorpusLoader (Dialogue Parsing, Caching & Search Indexing)"]
        B --> C["FastAPI Microservice Gateway (Port 8000)"]
    end

    subgraph RealTime ["2. Real-Time Live Assist Engine (< 300ms SLA)"]
        C -->|POST /analyze/stream-turn (Turn-by-Turn Payload)| E1["Dialogue State Tracker (DST)"]
        E1 --> E2["Turn Polarity Meter & Emotion Nudge"]
        E1 --> E3["CPNI Compliance Guardrail (Masked PIN Detection)"]
        E1 --> E4["Next Best Action (NBA) Generator"]
        E4 & E3 --> E5["Live Assist Stream Response"]
    end

    subgraph PostCall ["3. Post-Call Analytics & Grounded QA Engine"]
        C -->|POST /analyze/batch (Full Conversation Payload)| F1["Post-Call Analytics Pipeline"]
        F1 --> F2["Multi-Label Reason Classifier (TF-IDF + OvR ML)"]
        F1 --> F3["Sentiment Arc Analyzer (Start → Mid → End)"]
        F1 --> F4["Churn Risk & Resolution Detector (Calibrated ML)"]
        F1 --> F5["Executive Summarizer & Action Generator"]
        F1 --> F6["QA Checklist Evaluator (6 Criteria Rubric)"]
        F6 --> F7["Anti-Hallucination Grounding Guardrail"]
    end

    subgraph Persistence ["4. Rollup & Metrics Warehousing"]
        F2 & F3 & F4 & F5 & F7 --> G1["Agent Scorecard Aggregator"]
        G1 --> G2["Team Rollup Manager (5 Telecom Teams)"]
        G1 --> G3["Prometheus Metrics Exporter (/metrics)"]
        G1 --> G4["System Health Monitor (/health)"]
    end

    subgraph Presentation ["5. Interactive Streamlit Dashboard (Port 8501)"]
        E5 & G2 & G4 --> H1["⚡ Streamlit Enterprise Dashboard"]
        H1 --> H2["Tab 1: Live Assist Stream Simulator"]
        H1 --> H3["Tab 2: Post-Call & Grounded QA Scorecards"]
        H1 --> H4["Tab 3: Supervisor Team Leaderboards"]
        H1 --> H5["Tab 4: System Health, Benchmarks & Evals"]
    end
```

### Architectural Layer Responsibilities
* **Ingestion Gateway (FastAPI)**: Validates incoming payloads against strict Pydantic v2 schemas (`TranscriptInput`, `LiveTurnInput`), handles async I/O, and exposes REST endpoints.
* **Dialogue State Tracker (DST)**: Tracks conversational phases (`GREETING`, `AUTHENTICATION`, `DISCOVERY`, `NEGOTIATION`, `CLOSING`), identifies competitors, and monitors customer authentication.
* **Grounded QA Guardrail**: Ensures every cited quote in the evaluation scorecard maps directly to a verbatim transcript substring. Unsubstantiated or hallucinated citations are mathematically rejected.
* **Rollup Manager**: Aggregates metrics across 5 specialized contact center teams: *General Telecom Support*, *Retention Team Alpha*, *Billing & Retention Team Beta*, *Technical Support Tier-1*, and *Compliance Specialists*.
* **Streamlit Presentation Tier**: Provides interactive search across 8,300+ transcripts, live audio playback simulation, and interactive supervisor analytics.

---

## 2. Key Capabilities & Features

### 2.1 Post-Call Conversation Intelligence (`POST /analyze/batch`)
* **Executive Summaries**: Synthesizes customer intent, actions taken, and final disposition into audit-ready executive summaries.
* **Multi-Label Call Reasons**: Classifies primary and secondary drivers (*Service Cancellation*, *Network Coverage & Call Quality*, *Pricing & Cost*, *Competitor Switching*, *Plan Modification*, *Device Return*, *Identity Verification Difficulty*) with confidence scores and turn citations.
* **Sentiment Arc Trajectory**: Computes windowed sentiment shifts ($Start \rightarrow Middle \rightarrow End$) to identify *Positive Recovery*, *Negative Escalation*, *Consistently Negative*, or *Neutral* trajectories with closing courtesy de-biasing.
* **Calibrated Churn Risk & Resolution Detector**: Evaluates cancellation intent against retention proposals and warm transfers. Accurately distinguishes retained customers (low churn ~18%) from escalated transfers or unauthenticated cancellations (critical churn 85–100%).

### 2.2 Grounded QA Checklist Scoring (`src/qa/`)
Scores agents against 6 standardized contact center criteria:
1. `greeting` (10%): Courteous introduction, agent name, and Union Mobile branding.
2. `identity_verification` (25%, Critical): Strict CPNI authentication (PIN, account number, billing address) prior to account access.
3. `empathy` (15%): Acknowledgment of customer frustration and service quality issues.
4. `correct_disclosure` (20%, Critical): Mandatory disclosure of final bills, cancellation fees, and equipment return rules.
5. `no_prohibited_promises` (20%, Critical): Zero unauthorized promotional offers or deceptive promises.
6. `proper_closure` (10%): Offer of additional assistance, appreciation, and professional signoff.

### 2.3 Live Turn-by-Turn Assist Stream (`POST /analyze/stream-turn`)
* Executes on every utterance with **< 12ms latency** (SLA target: < 300ms).
* Delivers real-time Next Best Actions (NBA) with recommended agent scripts.
* Triggers instantaneous compliance interventions (e.g. blocking unauthenticated cancellations before FCC compliance violations occur).

---

## 3. Full Executable Codebase & Execution Guide

### Prerequisites
* Python 3.11 or higher
* Git

### Step 1: Clone & Install Dependencies
```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd prodapt
pip install -r requirements.txt
```
*(If installing manually: `pip install fastapi uvicorn pydantic pandas numpy scikit-learn joblib requests streamlit pytest httpx`)*

### Step 2: Run the Unit & Integration Test Suite
```powershell
python -m pytest -v
```
**Result**: 17 passing test cases covering analytics, streaming, CPNI guardrails, edge cases, and API routes.

### Step 3: Run the System Benchmark Evaluation Suite
```powershell
python benchmarks/run_evals.py
```
Executes batch analysis and live streaming benchmarks over 25 multi-turn corpus conversations, measuring P50/P95/P99 latencies and 100% quote grounding fidelity.

### Step 4: Launch the Microservice Backend (FastAPI)
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive Swagger (OpenAPI) Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **System Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Prometheus Metrics**: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

### Step 5: Launch the Streamlit Enterprise Dashboard
```powershell
python -m streamlit run streamlit_app.py --server.port 8501
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to access:
* **Corpus Search Bar**: Search across 8,300+ transcripts by Agent name (e.g., *Mariela*, *Heath*, *Zachary*, *Ami*), Customer name, Keyword (*rural area*), or Conversation ID.
* **Live Turn Simulator**: Step turn-by-turn through active calls with real-time NBA guidance.
* **Post-Call & Grounded QA**: View verbatim scorecards, sentiment arcs, and churn analysis.
* **Supervisor Team Rollups**: View leaderboards, quality heatmaps, and automated agent coaching plans.

---

## 4. Additional Exploration

### 4.1 Audio Acoustic Dynamics vs. Text-Only Analysis
While text transcripts capture semantic meaning, telephony audio contains prosodic cues critical for telecom QA:
1. **Sarcasm & Pitch Inflection**: Customers stating *"Oh, wonderful, another dropped call"* exhibit negative $F_0$ pitch contours and vocal tension that text-only sentiment classifiers can misinterpret.
2. **Talk-Over & Cross-Talk**: Frequent agent interruptions are strong drivers of customer escalation not captured in serialized text.
3. **Dead Air Tracking**: Pauses exceeding 8–10 seconds indicate agent desktop navigation friction or knowledge base gaps.
* **Proposed Extension**: An audio prosody classifier extracting RMS Energy (volume), Fundamental Frequency ($F_0$ jitter), and speech-to-silence ratios fused with text embeddings to calibrate emotional severity.

### 4.2 CPNI & PII Regulatory Data Redaction Layer
Under FCC Customer Proprietary Network Information (CPNI) rules and PCI-DSS, raw account credentials must be masked before long-term storage:
* **Account PINs**: 4–6 digit numeric strings (`PIN is 1234` $\rightarrow$ `PIN is [REDACTED_PIN]`).
* **Payment Card Data**: 16-digit card numbers or CVVs $\rightarrow$ `[REDACTED_PCI]`.
* **Government Identifiers**: SSNs and driver's licenses $\rightarrow$ `[REDACTED_GOV_ID]`.
* **Asynchronous Middleware**: Implemented as an inbound stream interceptor prior to database persistence to maintain zero audit exposure.

### 4.3 Human-in-the-Loop (HITL) Dispute & Calibration Workflow
To build operational trust in 100% automated QA:
1. **Agent Dispute Queue**: Agents can flag contested scorecard items directly in their dashboard.
2. **Supervisor Review**: Supervisors review verbatim highlighted evidence alongside rubric criteria to affirm or override scores.
3. **Continuous Calibration**: Overridden interactions are versioned into a benchmark calibration set to evaluate inter-rater reliability via Cohen's Kappa ($\kappa > 0.85$).

---

## 5. Evals on System Health & Benchmarks

The system was evaluated against multi-turn conversations from the 8,300+ record telecom corpus using `benchmarks/run_evals.py`:

| Evaluation Metric | Target SLA | Benchmark Result | Status |
| :--- | :--- | :--- | :--- |
| **Quote Grounding Fidelity** | 100.0% | **100.0%** (128 / 128 verified) | ✅ PASS |
| **Live Assist Stream Latency (P95)** | < 300 ms | **11.98 ms** | ✅ PASS |
| **Live Assist Stream Latency (P50)** | < 100 ms | **6.00 ms** | ✅ PASS |
| **Post-Call Batch Latency (P95)** | < 1,500 ms | **39.69 ms** | ✅ PASS |
| **Post-Call Batch Latency (P50)** | < 500 ms | **22.11 ms** | ✅ PASS |
| **Churn Risk Sensitivity** | > 70% | **76.0%** | ✅ PASS |
| **Test Suite Pass Rate** | 100% | **100%** (17 of 17 passing) | ✅ PASS |

### Prometheus Health & Observability Metrics (`/metrics`)
The microservice exposes standard Prometheus metrics for Grafana dashboards and Datadog monitoring:
* `telecom_calls_processed_total`: Cumulative counter of evaluated interactions.
* `telecom_live_turns_processed_total`: Cumulative counter of real-time stream turns.
* `telecom_groundedness_ratio`: Gauge tracking verified vs. unverified evidence (maintained at 1.0).
* `telecom_compliance_violations_total`: Gauge tracking critical regulatory infractions.
* `telecom_batch_latency_seconds`: Summary tracking P50, P90, P95, and P99 execution latency.

---

## 6. Production Scale Considerations

### 6.1 Workload Sizing: 200,000 Calls / Day
Processing an enterprise contact center corpus of **200,000 calls per day** introduces distinct throughput and scaling requirements:
* **Interactions**: 200,000 calls/day.
* **Turns**: Average 15–20 turns per call $\rightarrow$ **3,000,000 to 4,000,000 turns/day**.
* **Peak Load Distribution**: 70% of call volume occurs in an 8-hour window:
  $$\text{Peak Calls per Second} = \frac{200,000 \times 0.70}{8 \times 3,600} \approx 4.86\text{ completed calls/sec}$$
  $$\text{Peak Live Stream Turns per Second} \approx 4.86 \times 18 \approx 87.5\text{ turns/sec}$$

### 6.2 Cloud Economics: Frontier LLM vs. Our Hybrid Architecture

| Dimension | Frontier LLM (GPT-4o / Claude 3.5) | Self-Hosted SLM (Llama-3-8B) | Hybrid Engine (Our Architecture) |
| :--- | :--- | :--- | :--- |
| **Tokens per Call** | ~2,500 tokens | ~2,500 tokens | 0 external tokens |
| **Daily Inference Cost** | **$2,500 / day** ($912,500 / yr) | ~$100 / day ($36,500 / yr) | **< $5 / day** (Standard CPU Pods) |
| **Live Assist Latency (P95)** | 600 ms – 1,200 ms (Violates SLA) | 150 ms – 300 ms | **11.98 ms** (Ultra Low Latency) |
| **Batch Latency (P95)** | 2,500 ms – 5,000 ms | 400 ms – 900 ms | **39.69 ms** |
| **Quote Hallucination Risk** | 5% – 12% citation drift | 3% – 8% | **0.0% (Mathematically Verified)** |
| **Infrastructure Required** | Cloud API Keys | High-End GPU Cluster (A10G/A100) | Standard 4-Core CPU Pods |

### 6.3 Recommended Production Deployment Topology
```
                          [Telephony WebRTC / SIP Trunk]
                                         │
                                         ▼
                             [Load Balancer / Ingress]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [FastAPI Live Worker Pod 1]                     [FastAPI Live Worker Pod 2]
     (Horizontal Pod Autoscaler)                     (Horizontal Pod Autoscaler)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            [Redis Stream Event Bus]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    [Batch Analytics Consumer 1]                    [Batch Analytics Consumer 2]
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     [PostgreSQL / ClickHouse Analytics Warehouse]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   [Streamlit Supervisor UI]                       [Prometheus + Grafana Alerts]
```

---

## 7. Repository Structure

```
├── benchmarks/
│   ├── run_evals.py                 # Automated benchmark test runner
│   └── EVALUATION_REPORT.md         # Generated benchmark evaluation report
├── docs/
│   ├── ARCHITECTURE.md              # Detailed architecture & design rationale
│   └── ADDITIONAL_EXPLORATION.md    # Scale economics, acoustics & CPNI masking
├── models/
│   ├── telecom_churn_model.joblib   # Trained ML model for churn probability
│   ├── telecom_sentiment_model.joblib # Trained ML sentiment model
│   └── telecom_call_reason_model.joblib # Trained multi-label reason classifier
├── src/
│   ├── analytics/
│   │   ├── churn_detector.py        # Calibrated churn risk & resolution engine
│   │   ├── dialogue_state_tracker.py# Multi-turn conversation phase tracker
│   │   ├── live_assist.py           # Real-time Next Best Action stream engine
│   │   ├── reason_classifier.py     # Multi-label reason classifier
│   │   ├── sentiment_analyzer.py    # Sentiment arc & turn polarity engine
│   │   └── summarizer.py            # Executive summarizer & follow-up generator
│   ├── api/
│   │   ├── main.py                  # FastAPI application & lifecycle handlers
│   │   ├── routes_analyze.py        # Batch & streaming REST endpoints
│   │   ├── routes_health.py         # /health and /metrics Prometheus endpoints
│   │   └── routes_qa.py             # QA rubric config & supervisor rollup routes
│   ├── data/
│   │   └── corpus_loader.py         # 8,300+ record corpus loader with search
│   ├── models/
│   │   └── schemas.py               # Pydantic v2 schemas and API contracts
│   └── qa/
│       ├── checklist.py             # Configurable QA rubric manager
│       ├── evaluator.py             # 6-criteria grounded QA evaluation engine
│       ├── grounding.py             # Substring verification anti-hallucination guardrail
│       └── rollups.py               # Team and agent scorecard aggregator
├── tests/
│   ├── test_analytics.py            # Churn, sentiment, and assist unit tests
│   ├── test_api_endpoints.py        # FastAPI integration and endpoint tests
│   └── test_qa_and_grounding.py     # Anti-hallucination grounding tests
├── streamlit_app.py                 # Streamlit interactive enterprise dashboard
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Enterprise project documentation
```
