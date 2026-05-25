"""
FinSight AI - LangGraph Pipeline
=================================
This is the core orchestration layer. It defines a LangGraph state machine
that routes user queries through:

  1. Research Agent → fetches market data + generates analysis
  2. Risk Agent → computes risk metrics + generates assessment
  3. Synthesis Node → combines both agents' outputs into a final answer

The pipeline uses sequential execution (Research → Risk → Synthesis)
because the Risk Agent depends on market data from the Research Agent.

Future extension: Add conditional routing (e.g., skip risk analysis
for pure information queries) and parallel agent execution for
independent data gathering.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import FinState
from agents.research_agent import research_node
from agents.risk_agent import risk_node

load_dotenv()


def synthesize_node(state: FinState) -> dict:
    """
    Synthesis node that combines Research Agent and Risk Agent outputs
    into a final, structured investment analysis.
    
    This node reads from all state fields populated by the two agents
    and produces a comprehensive final answer with clear sections
    and an overall confidence level.
    """
    market_data = state.get("market_data", {})
    research_summary = state.get("research_summary", "No research data available.")
    risk_metrics = state.get("risk_metrics", {})
    risk_assessment = state.get("risk_assessment", "No risk data available.")
    
    # Determine confidence based on data quality
    has_market_data = market_data and "error" not in market_data
    has_risk_data = risk_metrics and "error" not in risk_metrics
    
    if has_market_data and has_risk_data:
        confidence = "High"
    elif has_market_data or has_risk_data:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Generate final synthesis using Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    symbol = market_data.get("symbol", "Unknown")
    company = market_data.get("company_name", "Unknown")
    
    synthesis_prompt = f"""You are a senior financial advisor synthesizing research from two 
specialist analysts. Combine their findings into a clear, actionable investment brief.

RESEARCH ANALYST'S FINDINGS:
{research_summary}

RISK ANALYST'S FINDINGS:
{risk_assessment}

INVESTOR'S ORIGINAL QUESTION:
{state['query']}

KEY DATA POINTS:
- Stock: {company} ({symbol})
- Current Price: ${market_data.get('current_price', 'N/A')}
- Risk Tier: {risk_metrics.get('risk_tier', 'Unknown')}
- Risk Score: {risk_metrics.get('risk_score', 'N/A')}/100
- Volatility: {risk_metrics.get('volatility', 'N/A')}
- Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 'N/A')}

Create a structured investment brief with these exact sections:

## Investment Analysis: {symbol}

**Overall Assessment**: [One sentence — bullish/bearish/neutral with reasoning]

**Market Position**: [2-3 sentences from research findings]

**Risk Profile**: [2-3 sentences from risk findings]

**Recommendation**: [Clear action — Buy/Hold/Sell with conditions and position sizing]

**Key Risks to Monitor**: [2-3 bullet points]

**Confidence Level**: {confidence}

Be direct, specific, and actionable. No generic disclaimers."""

    response = llm.invoke(synthesis_prompt)
    final_answer = response.content
    
    return {
        "final_answer": final_answer,
        "confidence": confidence,
        "messages": [{"role": "assistant", "content": f"Synthesis: Completed investment brief for {symbol}"}]
    }


def build_pipeline():
    """
    Construct and compile the LangGraph pipeline.
    
    Graph structure:
        research_node → risk_node → synthesize_node → END
    
    Returns:
        Compiled LangGraph application ready to invoke.
    """
    graph = StateGraph(FinState)
    
    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("risk", risk_node)
    graph.add_node("synthesize", synthesize_node)
    
    # Define edges (sequential flow)
    graph.set_entry_point("research")
    graph.add_edge("research", "risk")
    graph.add_edge("risk", "synthesize")
    graph.add_edge("synthesize", END)
    
    # Compile
    return graph.compile()


# Module-level pipeline instance
pipeline = build_pipeline()


def run_analysis(query: str) -> dict:
    """
    Run the full analysis pipeline for a user query.
    
    Args:
        query: Natural language investment question
              (e.g., "Should I invest in NVIDIA?")
    
    Returns:
        Dict with final_answer, confidence, market_data, and risk_metrics
    """
    result = pipeline.invoke({
        "query": query,
        "messages": [],
        "market_data": None,
        "research_context": None,
        "research_summary": None,
        "risk_metrics": None,
        "risk_assessment": None,
        "final_answer": None,
        "confidence": None,
    })
    
    return {
        "query": query,
        "final_answer": result.get("final_answer", "Analysis failed."),
        "confidence": result.get("confidence", "Low"),
        "market_data": result.get("market_data"),
        "risk_metrics": result.get("risk_metrics"),
        "research_summary": result.get("research_summary"),
        "risk_assessment": result.get("risk_assessment"),
    }


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("FinSight AI - Multi-Agent Financial Intelligence")
    print("=" * 60)
    
    test_query = "Should I invest in NVIDIA right now?"
    print(f"\nQuery: {test_query}")
    print("-" * 60)
    
    result = run_analysis(test_query)
    
    print(f"\nConfidence: {result['confidence']}")
    print(f"\n{result['final_answer']}")
