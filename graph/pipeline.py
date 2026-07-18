"""
FinSight AI — LangGraph Pipeline
Orchestrates Research Agent -> Risk Agent -> Synthesis using shared FinState.
Week 2: Gemini (google-genai) backbone; agents call real financial tools via
manual function calling. MCP server (mcp_server/server.py) exposes the same
tools over MCP for any external MCP client.
"""

import json
import os
import time

from google import genai
from langgraph.graph import StateGraph, END

from graph.state import FinState
from agents.research_agent import run_research_agent
from agents.risk_agent import run_risk_agent


client = genai.Client(vertexai=True, project="project-13b1a293-3e33-4184-8d9", location="us-central1")
MODEL = "gemini-2.5-flash"


def extract_symbol(state: FinState) -> dict:
    """Extract the stock ticker from the query if not explicitly provided."""
    if state.get("symbol"):
        # Return the symbol explicitly (delta only) so the channel is set.
        return {"symbol": state["symbol"]}

    response = client.models.generate_content(
        model=MODEL,
        contents=(
            "Extract the stock ticker symbol from this query. Return ONLY the "
            f"ticker, nothing else.\n\nQuery: {state['query']}"
        ),
    )
    extracted = "".join(c for c in (response.text or "").strip().upper() if c.isalpha())
    return {"symbol": extracted or "SPY"}


def synthesize(state: FinState) -> dict:
    """Combine research + risk outputs into a final investment brief."""
    research = state.get("research_summary", "No research data available.")
    risk = state.get("risk_assessment", "No risk data available.")
    risk_metrics = state.get("risk_metrics", {})
    market_data = state.get("market_data", {}) or {}
    symbol = state["symbol"]

    prompt = f"""You are the Synthesis Agent. Combine the research and risk analyses
below into a final investment brief for {symbol}.

## Research Agent Output
{research}

## Risk Agent Output
{risk}

## Raw Risk Metrics
{json.dumps(risk_metrics, indent=2) if risk_metrics else 'N/A'}

## Raw Market Data
Price: ${market_data.get('current_price', 'N/A')}
PE: {market_data.get('pe_ratio', 'N/A')}
Sector: {market_data.get('sector', 'N/A')}

---

Write a unified investment brief with:
1. Executive Summary (2-3 sentences)
2. Key Findings (specific numbers)
3. Risk/Reward Assessment (integrate both agents)
4. Recommendation (Buy/Hold/Sell + confidence High/Medium/Low + time horizon)
5. Position Sizing Guidance (based on risk score)

Be direct and actionable. Target 300-400 words."""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    final_text = response.text or ""

    confidence = "Medium"
    lt = final_text.lower()
    if "high confidence" in lt or "strong buy" in lt:
        confidence = "High"
    elif "low confidence" in lt or "insufficient" in lt:
        confidence = "Low"

    return {"final_answer": final_text, "confidence": confidence}


def build_pipeline():
    graph = StateGraph(FinState)
    graph.add_node("extract_symbol", extract_symbol)
    graph.add_node("research", run_research_agent)
    graph.add_node("risk", run_risk_agent)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("extract_symbol")
    graph.add_edge("extract_symbol", "research")
    graph.add_edge("research", "risk")
    graph.add_edge("risk", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


pipeline = build_pipeline()


def run_analysis(query: str, symbol: str = "") -> dict:
    """Run the full analysis pipeline and return the complete FinState."""
    start = time.time()
    initial_state: FinState = {
        "query": query, "symbol": symbol,
        "research_summary": None, "market_data": None,
        "financial_summary": None, "peer_comparison": None,
        "risk_metrics": None, "risk_assessment": None,
        "final_answer": None, "confidence": None,
        "processing_time_seconds": None, "model_used": None,
        "tools_called": [],
    }
    result = pipeline.invoke(initial_state)
    result["processing_time_seconds"] = round(time.time() - start, 2)
    return result
