"""
FinSight AI - Shared State Schema
=================================
This module defines the shared state that flows between agents in the
LangGraph pipeline. Both the Research Agent and Risk Agent read from
and write to this state, enabling multi-agent collaboration without
direct agent-to-agent communication.

Key design decision: Using TypedDict with Annotated fields so LangGraph
can merge state updates from parallel or sequential agent executions.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class MarketData(TypedDict):
    """Structured market data fetched by the Research Agent."""
    symbol: str
    current_price: float
    pe_ratio: Optional[float]
    market_cap: Optional[float]
    fifty_two_week_high: float
    fifty_two_week_low: float
    avg_volume: int
    sector: str
    industry: str
    price_history: list  # Last 30 days closing prices


class RiskMetrics(TypedDict):
    """Risk metrics computed by the Risk Agent."""
    volatility: float          # Annualized standard deviation of returns
    sharpe_ratio: float        # Risk-adjusted return metric
    max_drawdown: float        # Maximum peak-to-trough decline
    beta: float                # Market sensitivity
    risk_tier: str             # Conservative / Moderate / Aggressive
    risk_score: float          # 0-100 composite risk score


class FinState(TypedDict):
    """
    The central state object passed through the LangGraph pipeline.
    
    Flow:
    1. User query enters → query field populated
    2. Research Agent → fills market_data, research_context, research_summary
    3. Risk Agent → reads market_data, fills risk_metrics
    4. Synthesis Node → reads all fields, writes final_answer
    
    Messages field uses LangGraph's add_messages annotation to
    accumulate the conversation history across all agent interactions.
    """
    # Input
    query: str
    
    # Conversation history (accumulated across agents)
    messages: Annotated[list, add_messages]
    
    # Research Agent outputs
    market_data: Optional[MarketData]
    research_context: Optional[list]    # Retrieved docs from ChromaDB
    research_summary: Optional[str]     # Agent's analysis of market data
    
    # Risk Agent outputs
    risk_metrics: Optional[RiskMetrics]
    risk_assessment: Optional[str]      # Agent's risk narrative
    
    # Synthesis output
    final_answer: Optional[str]
    confidence: Optional[str]           # High / Medium / Low
