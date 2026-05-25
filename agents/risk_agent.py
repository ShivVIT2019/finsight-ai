"""
FinSight AI - Risk Agent
========================
The Risk Agent reads market data from the shared state (written by the
Research Agent) and computes risk metrics including volatility, Sharpe
ratio, max drawdown, and a composite risk score.

It then generates a narrative risk assessment using Gemini, explaining
what the numbers mean for an investor.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import FinState
from agents.tools import calculate_risk_metrics


def risk_node(state: FinState) -> dict:
    """
    LangGraph node for the Risk Agent.
    
    1. Reads market_data from shared state (written by Research Agent)
    2. Computes risk metrics from price history
    3. Generates a risk narrative using Gemini
    4. Writes risk_metrics and risk_assessment to shared state
    
    Args:
        state: Current FinState with market_data populated
    
    Returns:
        Dict with state updates (risk_metrics, risk_assessment)
    """
    market_data = state.get("market_data")
    
    if not market_data or "error" in market_data:
        return {
            "risk_metrics": {
                "volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "beta": 0.0, "risk_tier": "Unknown", "risk_score": 50.0,
            },
            "risk_assessment": "Unable to compute risk metrics: no market data available.",
            "messages": [{"role": "assistant", "content": "Risk Agent: Skipped - no market data"}]
        }
    
    # Step 1: Calculate risk metrics from price history
    price_history = market_data.get("price_history", [])
    risk_metrics = calculate_risk_metrics(price_history)
    
    if "error" in risk_metrics:
        return {
            "risk_metrics": risk_metrics,
            "risk_assessment": f"Risk calculation error: {risk_metrics['error']}",
            "messages": [{"role": "assistant", "content": f"Risk Agent: Error - {risk_metrics['error']}"}]
        }
    
    # Step 2: Generate risk narrative using Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    risk_prompt = f"""You are a portfolio risk analyst. Based on the following risk metrics,
provide a clear risk assessment for an investor considering {market_data['symbol']}.

Risk Metrics:
- Annualized Volatility: {risk_metrics['volatility']:.2%}
- Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}
- Max Drawdown: {risk_metrics['max_drawdown']:.2%}
- Beta: {risk_metrics['beta']}
- Risk Tier: {risk_metrics['risk_tier']}
- Risk Score: {risk_metrics['risk_score']}/100

Stock Context:
- Current Price: ${market_data['current_price']}
- 52-Week Range: ${market_data['fifty_two_week_low']} - ${market_data['fifty_two_week_high']}
- Sector: {market_data['sector']}

Investor's Question: {state['query']}

Provide a concise risk assessment (100-150 words) covering:
1. What the risk tier means for this investor
2. Key risk factors (volatility, drawdown exposure)
3. Position sizing suggestion (what % of portfolio is appropriate for this risk level)
4. One specific risk to watch

Be direct and practical. No disclaimers."""

    response = llm.invoke(risk_prompt)
    risk_assessment = response.content
    
    return {
        "risk_metrics": risk_metrics,
        "risk_assessment": risk_assessment,
        "messages": [{"role": "assistant", "content": f"Risk Agent: Completed risk assessment - {risk_metrics['risk_tier']}"}]
    }
