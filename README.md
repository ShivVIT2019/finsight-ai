# FinSight AI — Multi-Agent Financial Intelligence

A **LangGraph-powered multi-agent system** where specialized AI agents collaborate to deliver real-time financial analysis. Two agents — a Research Agent and a Risk Agent — share state, invoke tools, and produce auditable investment briefs through an orchestrated pipeline.

Built as a portfolio project demonstrating **agentic AI architecture, tool-use orchestration, and production ML system design** for financial applications.

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
│  • yfinance  │ │  • Risk calc │
│  • ChromaDB  │ │  • Scorer    │
└──────┬───────┘ └──────┬───────┘
       │                │
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
3. **Research Agent** extracts the stock symbol, fetches real-time market data via yfinance, and generates a research summary using Gemini
4. **Risk Agent** reads market data from shared state, computes risk metrics (volatility, Sharpe ratio, max drawdown, beta), and generates a risk assessment
5. **Synthesis Node** combines both agents' outputs into a structured investment brief with confidence level
6. **FastAPI** returns the complete analysis as a JSON response

### Key Design Decisions

- **Shared State over Direct Communication**: Agents communicate through a typed state object (`FinState`), not direct messages. This makes the pipeline testable, debuggable, and extensible.
- **Sequential Execution**: Risk Agent depends on market data from Research Agent, so they run sequentially. Future: add parallel branches for independent data sources.
- **Tool-Use Pattern**: Each agent has specific tools (yfinance, risk calculator) invoked during execution, following the agent → tool → state update pattern.
- **Typed State Schema**: Using `TypedDict` with explicit fields ensures compile-time type safety and makes state flow visible in code review.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Orchestration | **LangGraph** | Multi-agent state machine with tool-use |
| LLM Backbone | **Google Gemini 2.0 Flash** | Research summaries, risk narratives, synthesis |
| Vector Store | **ChromaDB** | Financial document retrieval (RAG) |
| Market Data | **yfinance** | Real-time stock prices and fundamentals |
| Backend API | **FastAPI** | REST endpoints serving the pipeline |
| Embeddings | **sentence-transformers** | Semantic search for financial docs |
| Containerization | **Docker** | Reproducible deployment |
| Risk Analytics | **NumPy / Pandas** | Volatility, Sharpe ratio, drawdown |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/ShivVIT2019/finsight-ai.git
cd finsight-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### Run a Quick Test

```bash
python -m graph.pipeline
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Run full multi-agent analysis |
| `GET` | `/api/quote/{symbol}` | Quick market data lookup |
| `GET` | `/api/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I invest in NVIDIA right now?"}'
```

### Example Response

```json
{
  "query": "Should I invest in NVIDIA right now?",
  "final_answer": "## Investment Analysis: NVDA\n\n**Overall Assessment**: ...",
  "confidence": "High",
  "market_data": {
    "symbol": "NVDA",
    "current_price": 135.42,
    "pe_ratio": 65.3,
    "market_cap": 3320000000000,
    "sector": "Technology"
  },
  "risk_metrics": {
    "volatility": 0.4521,
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.1845,
    "risk_tier": "Aggressive",
    "risk_score": 72.5
  },
  "processing_time_seconds": 8.34
}
```

---

## Docker Deployment

```bash
# Build
docker build -t finsight-ai .

# Run
docker run -p 8000:8000 --env-file .env finsight-ai
```

### Deploy to Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/finsight-ai
gcloud run deploy finsight-ai \
  --image gcr.io/YOUR_PROJECT/finsight-ai \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=your_key_here
```

---

## Project Structure

```
finsight-ai/
├── agents/
│   ├── research_agent.py    # Market data retrieval + LLM analysis
│   ├── risk_agent.py        # Risk metrics computation + assessment
│   └── tools.py             # Shared tools (yfinance, risk calculator)
├── graph/
│   ├── state.py             # Typed shared state schema (FinState)
│   └── pipeline.py          # LangGraph graph definition + orchestration
├── vectorstore/
│   ├── ingest.py            # Ingest financial docs into ChromaDB
│   └── retriever.py         # Query ChromaDB for relevant context
├── api/
│   └── main.py              # FastAPI endpoints
├── data/
│   └── financial_docs/      # Sample financial documents for RAG
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Risk Metrics Explained

| Metric | What It Measures | How It's Calculated |
|--------|-----------------|-------------------|
| **Volatility** | Price unpredictability | Annualized std dev of daily returns |
| **Sharpe Ratio** | Risk-adjusted return | (Return - Risk-free rate) / Volatility |
| **Max Drawdown** | Worst peak-to-trough decline | Max(peak - trough) / peak |
| **Beta** | Market sensitivity | Stock volatility / Market volatility |
| **Risk Score** | Composite risk (0-100) | Weighted combination of above metrics |

---

## Roadmap

- [x] Week 1: LangGraph multi-agent pipeline with Research + Risk agents
- [ ] Week 2: Swap Gemini for Claude API + add MCP server for tool-use
- [ ] Week 3: Deploy on AWS Lambda + add Streamlit dashboard
- [ ] Week 4: Fine-tune small LLM with LoRA for financial Q&A

---

## Resume Bullets

> Architected multi-agent LangGraph pipeline with two specialized AI agents (Research + Risk) collaborating through shared state for real-time financial analysis; integrated yfinance market data, ChromaDB RAG retrieval, and composite risk-scoring across agent boundaries
>
> Built FastAPI backend serving multi-agent financial intelligence system with real-time market data ingestion, risk metrics computation (volatility, Sharpe ratio, max drawdown), and LLM-powered investment brief synthesis

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
