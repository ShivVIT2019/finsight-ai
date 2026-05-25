"""
FinSight AI - FastAPI Backend
=============================
REST API serving the multi-agent LangGraph pipeline.

Endpoints:
  POST /api/analyze     - Run full analysis for a stock query
  GET  /api/health      - Health check
  GET  /api/quote/{sym} - Quick market data lookup (no agent analysis)
"""

import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from graph.pipeline import run_analysis
from agents.tools import fetch_market_data

app = FastAPI(
    title="FinSight AI",
    description="Multi-agent financial intelligence powered by LangGraph + Gemini",
    version="1.0.0",
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class AnalysisRequest(BaseModel):
    """Request model for the /analyze endpoint."""
    query: str = Field(
        ...,
        description="Natural language investment question",
        examples=["Should I invest in NVIDIA?", "Is Tesla a good buy right now?"],
    )


class AnalysisResponse(BaseModel):
    """Response model for the /analyze endpoint."""
    query: str
    final_answer: str
    confidence: str
    market_data: dict | None = None
    risk_metrics: dict | None = None
    research_summary: str | None = None
    risk_assessment: str | None = None
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str
    version: str
    agents: list[str]


# --- Endpoints ---

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents=["research_agent", "risk_agent", "synthesis_node"],
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Run the full multi-agent analysis pipeline.
    
    The pipeline executes three nodes sequentially:
    1. Research Agent → market data + analysis
    2. Risk Agent → risk metrics + assessment
    3. Synthesis → combined investment brief
    
    Typical response time: 5-15 seconds depending on LLM latency.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY not configured. Set it in .env file.",
        )
    
    start_time = time.time()
    
    try:
        result = run_analysis(request.query)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {str(e)}",
        )
    
    processing_time = round(time.time() - start_time, 2)
    
    return AnalysisResponse(
        query=result["query"],
        final_answer=result["final_answer"],
        confidence=result["confidence"],
        market_data=result.get("market_data"),
        risk_metrics=result.get("risk_metrics"),
        research_summary=result.get("research_summary"),
        risk_assessment=result.get("risk_assessment"),
        processing_time_seconds=processing_time,
    )


@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Quick market data lookup without running the full agent pipeline."""
    data = fetch_market_data(symbol.upper())
    
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
