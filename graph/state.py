"""
FinSight AI — Shared State Schema
TypedDict for LangGraph shared state between Research and Risk agents.
Week 2: Gemini backbone. tools_called uses an additive reducer so both
agents can append without overwriting each other.
"""

import operator
from typing import TypedDict, Optional, Annotated


class FinState(TypedDict):
    # Input
    query: str
    symbol: str

    # Research Agent outputs
    research_summary: Optional[str]
    market_data: Optional[dict]
    financial_summary: Optional[dict]
    peer_comparison: Optional[dict]

    # Risk Agent outputs
    risk_metrics: Optional[dict]
    risk_assessment: Optional[str]

    # Final synthesis
    final_answer: Optional[str]
    confidence: Optional[str]

    # Metadata
    processing_time_seconds: Optional[float]
    model_used: Optional[str]
    # Additive: each node appends its tool calls; reducer concatenates lists.
    tools_called: Annotated[list, operator.add]
