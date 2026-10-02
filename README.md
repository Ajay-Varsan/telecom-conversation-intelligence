# Telecom Contact Center Conversation Analytics & Quality Intelligence Microservice

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/Tests-13%20Passed-emerald.svg)](tests/)
[![Groundedness](https://img.shields.io/badge/Quote%20Grounding-100%25-brightgreen.svg)](benchmarks/)

A production-grade conversation intelligence microservice for enterprise telecom contact centers. Designed to replace manual 2% sample audits with **100% automated, explainable, real-time conversation analytics, live agent assist, and supervisor quality rollups**.

---

## Architecture Overview

```
                      [Telephony Audio / Chat Stream]
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Ingestion & Gateway Layer    │
                    │  FastAPI (Async I/O + Schemas)  │
                    └────────────────┬────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼ (Live Turn Event)                     ▼ (Call Completed)
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │   Live Assist Engine    │             │   Post-Call Analytics   │
    │  - Turn Polarity Meter  │             │  - Executive Summary    │
    │  - Next Best Action     │             │  - Multi-Label Reasons  │
    │  - Compliance Alerts    │             │  - Sentiment Arc        │
    │  - Sub-300ms SLA        │             │  - Churn & Resolution   │
    └─────────────────────────┘             └────────────┬────────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │  Grounded QA Checklist  │
                                            │  - Verbatim Quotations  │
                                            │  - Substring Guardrail  │
                                            │  - Compliance Severity  │
                                            └────────────┬────────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │  Rollup & Metrics Store │
                                            │  - Agent Scorecards     │
                                            │  - Team Leaderboards    │
                                            │  - Prometheus /metrics  │
                                            └─────────────────────────┘
```

---

## Key Features

1. **Post-Call Multi-Dimensional Analytics (`POST /analyze/batch`)**:
   * **Concise Executive Summary**: Structures customer intent, actions taken by the agent, and final outcome.
   * **Multi-Label Call Reasons**: Classifies reasons (e.g., *Service Cancellation*, *Network Quality*, *Pricing & Cost*, *Competitor Switching*) with confidence scores and turn citations.
   * **Sentiment Arc Trajectory**: Computes emotional movement ($Start \rightarrow Middle \rightarrow End$) to flag *Positive Recovery*, *Negative Escalation*, or *Persistent Dissatisfaction*.
   * **Resolution & Churn Risk**: Classifies resolution (`RESOLVED`, `UNRESOLVED`, `ESCALATED`) and computes churn risk score (0.0 to 1.0) with risk factor breakdown.
   * **Automated Follow-Up Actions**: Triggers live workflow actions (e.g. equipment return kits, prorated refunds, engineering tickets).

2. **Grounded QA Checklist Scoring (`src/qa/`)**:
   * Scores agent adherence against 6 configurable criteria:
     * `greeting`: Professional greeting and company branding.
     * `identity_verification`: Mandatory CPNI identity authentication before account actions.
     * `empathy`: Acknowledgment of customer frustration and service issues.
     * `correct_disclosure`: Mandatory cancellation fee and equipment return policy disclosure.
     * `no_prohibited_promises`: Zero deceptive or unauthorized promotional promises ("it's on us" free phone).
     * `proper_closure`: Offer of further assistance and courteous wrap-up.
   * **Strict Anti-Hallucination Grounding**: Every attributed quote is verified against the raw transcript turns. Unverified or fabricated quotes are rejected.

3. **Live Turn-by-Turn Assist Stream (`POST /analyze/stream-turn`)**:
   * Runs turn-by-turn during active customer calls with sub-300ms latency.
   * Emits dynamic Next Best Actions (NBA) with recommended verbatim scripts.
   * Fires instant compliance warning banners when agents make unauthorized promises or miss disclosures.

4. **Supervisor Team Rollups & Agent Scorecards (`GET /qa/rollups/*`)**:
   * Aggregates quality scores, pass rates, and critical compliance violations across agents and teams.
   * Generates coaching tips tailored to each agent's top failure modes.
   * Team performance leaderboards and checklist category breakdown heatmaps.

5. **Operational Health & Observability (`GET /health`, `GET /metrics`)**:
   * Prometheus exposition metrics tracking throughput, latency percentiles, and violation rates.
   * 100% groundedness verification monitoring.

---

## Benchmark & Evaluation Results

Evaluated on real multi-turn conversations from the telecom corpus:

| Evaluation Metric | Target SLA | Achieved Result |
| :--- | :--- | :--- |
| **Quote Grounding Fidelity** | 100% | **100.0%** (119/119 quotes verified verbatim) |
| **Live Assist Latency (P95)** | < 300 ms | **0.46 ms** |
| **Batch Analysis Latency (P95)** | < 1,500 ms | **16.07 ms** |
| **Churn Risk Sensitivity** | > 80% | **88.0%** |
| **Test Suite Pass Rate** | 100% | **100%** (13 passed, 0 warnings) |

---

## Quickstart & Installation

### 1. Requirements
* Python 3.11+
* Dependencies: `fastapi`, `uvicorn`, `pydantic`, `pandas`, `pytest`, `httpx`

### 2. Run Tests
```powershell
python -m pytest -v
```

### 3. Run Benchmark Evaluation Suite
```powershell
python benchmarks/run_evals.py
```

### 4. Start the Microservice
```powershell
python -m src.api.main
```
Or with uvicorn directly:
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access Interactive Dashboard & API Docs
* **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
* **OpenAPI Interactive Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)
* **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## Repository Structure

```
├── benchmarks/
│   ├── run_evals.py             # Evaluation benchmark runner
│   └── EVALUATION_REPORT.md     # Auto-generated eval & health report
├── docs/
│   ├── ARCHITECTURE.md          # Full architecture diagram & design decisions
│   └── ADDITIONAL_EXPLORATION.md# Production scale math, prosody, CPNI redaction
├── src/
│   ├── analytics/
│   │   ├── churn_detector.py    # Churn risk & resolution detector
│   │   ├── live_assist.py       # Live turn streaming & next best action engine
│   │   ├── reason_classifier.py # Multi-label call reason classifier
│   │   ├── sentiment_analyzer.py# Sentiment arc & turn polarity engine
│   │   └── summarizer.py        # Executive summarizer & follow-ups
│   ├── api/
│   │   ├── main.py              # FastAPI app & lifespan configuration
│   │   ├── routes_analyze.py    # Batch analysis & stream-turn endpoints
│   │   ├── routes_health.py     # Health & Prometheus metrics endpoints
│   │   └── routes_qa.py         # QA config & team rollup endpoints
│   ├── data/
│   │   └── corpus_loader.py     # Ingests and formats telecom corpus CSVs
│   ├── models/
│   │   └── schemas.py           # Pydantic v2 schemas and data contracts
│   └── static/
│       ├── app.js               # Reactive frontend application logic
│       ├── index.html           # Modern supervisor & agent dashboard
│       └── style.css            # Dark mode glassmorphic styling
├── tests/
│   ├── test_analytics.py        # Analytics unit tests
│   ├── test_api_endpoints.py    # FastAPI integration tests
│   └── test_qa_and_grounding.py # Anti-hallucination grounding tests
├── .gitignore
└── README.md
```
