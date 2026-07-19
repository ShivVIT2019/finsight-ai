# FinSight AI — Multi-Agent Financial Intelligence

> 🟢 **Live demo:** 
> - Dashboard: https://finsight-ai-dashboard-727430679800.us-central1.run.app (interactive UI)
> - API: https://finsight-ai-4lfjlhbw2q-uc.a.run.app/docs (Swagger, try `POST /analyze`)
>
> First request may take ~30–40s (Cloud Run scale-to-zero cold start).


A **LangGraph-powered multi-agent system** where specialized AI agents collaborate to deliver real-time financial analysis. Two agents — a Research Agent and a Risk Agent — share state, invoke real financial tools through a function-calling loop, and produce auditable investment briefs through an orchestrated pipeline.

Built as a portfolio project demonstrating **agentic AI architecture, tool-use orchestration via the Model Context Protocol (MCP), and production ML system design** for financial applications.

---

## Architecture

```
User Query (natural language)
        │
        ▼
┌─────────────────────────────┐
│   LangGraph Orchestrator    │
│   (State Machine + Routing) │
└──────┬──────────────┬───────┘
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│  Research    │ │    Risk      │
│  Agent       │ │    Agent     │
│              │ │              │
│  Tools:      │ │  Tools:      │
│  • market    │ │  • risk calc │
│    data      │ │  • scorer    │
│  • financials│ │              │
│  • peers     │ │              │
└──────┬───────┘ └──────┬───────┘
       │                │
       │   Gemini function-calling loop
       │   (tools also exposed via MCP server)
       ▼                ▼
┌─────────────────────────────┐
│     Synthesis Node          │
│  (Combines + formats output)│
└──────────────┬──────────────┘
               │
               ▼
       FastAPI Response (JSON)
```

### How It Works

1. **User submits a query** like "Should I invest in NVIDIA right now?"
2. **LangGraph orchestrator** initializes shared state and routes to agents
3. **Research Agent** fetches real-time market data, financial summaries, and peer comparisons by calling tools through a Gemini function-calling loop, then writes a research brief
4. **Risk Agent** reads context from shared state, calls the risk-metrics tool (volatility, Sharpe ratio, max drawdown, beta), and writes a risk assessment
5. **Synthesis Node** combines both agents' outputs into a structured investment brief with a confidence level
6. **FastAPI** returns the complete analysis as JSON

### Tool-Use via MCP

The four financial tools are exposed two ways from a single source of truth:

- **In-process function calling** — the agents call tools directly through Gemini's function-calling loop
- **MCP server** (`mcp_server/server.py`) — the same tools are served over the Model Context Protocol, so any MCP client (Claude Desktop, Cursor, the MCP Inspector) can invoke them

### Key Design Decisions

- **Shared State over Direct Communication**: Agents communicate through a typed state object (`FinState`), not direct messages — testable, debuggable, extensible. `tools_called` uses an additive reducer so both agents accumulate their tool history without overwriting each other.
- **Sequential Execution**: The Risk Agent benefits from the Research Agent's context, so they run sequentially.
- **Explicit (manual) tool-use loop**: Automatic function calling is disabled so the agent loop is visible and structured tool outputs can be captured into state.
- **Typed State Schema**: `TypedDict` with explicit fields makes state flow visible in code review.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Orchestration | **LangGraph** | Multi-agent state machine |
| LLM Backbone | **Google Gemini 2.5 Flash** (`google-genai`) | Research, risk narratives, synthesis |
| Tool Protocol | **MCP (FastMCP)** | Financial tools exposed over Model Context Protocol |
| Market Data | **yfinance** | Real-time prices, fundamentals, financials |
| Backend API | **FastAPI** | REST endpoint serving the pipeline |
| Risk Analytics | **NumPy / Pandas** | Volatility, Sharpe ratio, drawdown, beta |
| Containerization | **Docker** | Reproducible deployment |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
git clone https://github.com/ShivVIT2019/finsight-ai.git
cd finsight-ai

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

export GEMINI_API_KEY="your-key-here"   # never commit this
```

### Run the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
# opens at http://localhost:8501
```

### Run the MCP Server (optional, for MCP clients)

```bash
pip install -r mcp_server/requirements.txt
python mcp_server/server.py
# then connect the MCP Inspector to http://localhost:8001/mcp
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Run full multi-agent analysis |
| `GET` | `/healthz` | Health check |
| `GET` | `/` | Service info |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I invest in NVIDIA right now?", "symbol": "NVDA"}'
```

### Example Response

```json
{
  "query": "Should I invest in NVIDIA right now?",
  "symbol": "NVDA",
  "final_answer": "## Investment Analysis: NVDA ...",
  "confidence": "Medium",
  "market_data": {
    "symbol": "NVDA",
    "current_price": 135.42,
    "pe_ratio": 65.3,
    "sector": "Technology"
  },
  "risk_metrics": {
    "volatility": 0.4521,
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.1845,
    "risk_tier": "Aggressive",
    "risk_score": 72.5
  },
  "tools_called": [
    {"tool": "get_market_data"},
    {"tool": "get_financial_summary"},
    {"tool": "compare_peers"},
    {"tool": "calculate_risk_metrics"}
  ],
  "processing_time_seconds": 12.7
}
```

---

## Docker Deployment

```bash
docker build -t finsight-ai .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here finsight-ai
```

---

## Project Structure

```
finsight-ai/
├── agents/
│   ├── research_agent.py    # Gemini function-calling loop: market data, financials, peers
│   ├── risk_agent.py        # Gemini function-calling loop: risk metrics + assessment
│   └── mcp_tools.py         # Tool schemas + real yfinance implementations (shared)
├── graph/
│   ├── state.py             # Typed shared state schema (FinState) with reducer
│   └── pipeline.py          # LangGraph definition + orchestration + synthesis
├── mcp_server/
│   ├── server.py            # FastMCP server exposing the financial tools over MCP
│   └── requirements.txt
├── api/
│   └── main.py              # FastAPI endpoints
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Risk Metrics Explained

| Metric | What It Measures | How It's Calculated |
|--------|-----------------|-------------------|
| **Volatility** | Price unpredictability | Annualized std dev of daily returns |
| **Sharpe Ratio** | Risk-adjusted return | (Return − Risk-free rate) / Volatility |
| **Max Drawdown** | Worst peak-to-trough decline | Max(peak − trough) / peak |
| **Beta** | Market sensitivity | Covariance with SPY / SPY variance |
| **Risk Score** | Composite risk (0–100) | Weighted combination of the above |

---

## Roadmap

- [x] Week 1: LangGraph multi-agent pipeline with Research + Risk agents (Gemini)
- [x] Week 2: Function-calling tool-use + MCP server exposing financial tools
- [x] Week 3a: Streamlit dashboard over the pipeline
- [x] Week 3b: Deployed FastAPI + Streamlit dashboard on Cloud Run (300s timeout, 2GB/2vCPU)
- [x] Week 3c: Migrated from AI Studio API key to Vertex AI service-account auth (no rate limits, no secret to manage)
- [ ] Week 4: Add RAG over financial filings (ChromaDB) for document-grounded Q&A

---

## Resume Bullets

> Architected a multi-agent LangGraph pipeline with two specialized AI agents (Research + Risk) collaborating through typed shared state; agents invoke real financial tools (live market data, risk metrics, peer comparison) via a Gemini function-calling loop, producing auditable investment briefs

> Built a Model Context Protocol (MCP) server with FastMCP exposing four financial tools, enabling tool-use both in-process and from any MCP client; backed by a FastAPI service and composite risk scoring (volatility, Sharpe ratio, max drawdown, beta)

> Deployed the FastAPI service to Google Cloud Run with the Gemini API key managed via Secret Manager (mounted at runtime as an env var, never baked into the image); configured 300s request timeout and 2GB / 2 vCPU for the multi-agent workload, with the deployment provisioned from a single `gcloud run deploy --source` command.

> Migrated the deployed pipeline from an API-key backend (AI Studio Gemini) to Vertex AI with service-account (Application Default Credentials) authentication; eliminated the free-tier 20-requests/day cap, removed the Secret Manager mount, and cut the interview-demo failure mode where recruiter clicks hit rate limits.




---

## Author

**Sivasai Atchyut Akella**
MS Computer Science (AI), Binghamton University

- Portfolio: [portfolio-nine-rho-36.vercel.app](https://portfolio-nine-rho-36.vercel.app)
- GitHub: [github.com/ShivVIT2019](https://github.com/ShivVIT2019)
- LinkedIn: [linkedin.com/in/atchyut](https://linkedin.com/in/atchyut)

---

## License

MIT
