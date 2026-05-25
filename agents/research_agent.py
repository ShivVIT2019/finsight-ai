"""
FinSight AI - Research Agent
============================
The Research Agent is responsible for gathering and analyzing market data.
It has access to two tools:
  1. Market Data Tool (yfinance) - fetches real-time stock data
  2. RAG Retrieval Tool (ChromaDB) - retrieves relevant financial context

The agent writes its findings to the shared state so the Risk Agent
can use them for risk assessment.
"""

import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import FinState
from agents.tools import fetch_market_data, format_large_number


def extract_symbol(query: str) -> str:
    """
    Extract a stock ticker symbol from a natural language query.
    Uses Gemini to interpret the query if no obvious ticker is found.
    
    Examples:
        "Should I invest in NVIDIA?" → "NVDA"
        "Is AAPL a good buy?" → "AAPL"
        "Tell me about Tesla stock" → "TSLA"
    """
    # Direct ticker pattern (all caps, 1-5 chars)
    ticker_pattern = r'\b([A-Z]{1,5})\b'
    matches = re.findall(ticker_pattern, query)
    
    # Common words that look like tickers but aren't
    stop_words = {"I", "A", "AI", "AM", "AN", "AS", "AT", "BE", "BY", "DO",
                  "GO", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON",
                  "OR", "SO", "TO", "UP", "US", "WE", "THE", "AND", "FOR",
                  "NOT", "BUT", "ARE", "CAN", "HAD", "HAS", "HER", "HIM",
                  "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "OUR",
                  "OWN", "SAY", "SHE", "TOO", "USE", "WAY", "WHO", "HOW",
                  "SHOULD", "GOOD", "BUY", "SELL", "STOCK", "INVEST"}
    
    valid_tickers = [m for m in matches if m not in stop_words]
    
    if valid_tickers:
        return valid_tickers[0]
    
    # Company name to ticker mapping (common ones)
    company_map = {
        "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL",
        "microsoft": "MSFT", "amazon": "AMZN", "tesla": "TSLA",
        "nvidia": "NVDA", "meta": "META", "facebook": "META",
        "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
        "disney": "DIS", "spotify": "SPOT", "uber": "UBER",
        "airbnb": "ABNB", "palantir": "PLTR", "snowflake": "SNOW",
        "coinbase": "COIN", "shopify": "SHOP", "salesforce": "CRM",
        "adobe": "ADBE", "oracle": "ORCL", "ibm": "IBM",
        "walmart": "WMT", "jpmorgan": "JPM", "goldman": "GS",
        "boeing": "BA", "nike": "NKE", "coca-cola": "KO",
        "pepsi": "PEP", "visa": "V", "mastercard": "MA",
    }
    
    query_lower = query.lower()
    for company, ticker in company_map.items():
        if company in query_lower:
            return ticker
    
    # Default fallback
    return "AAPL"


def research_node(state: FinState) -> dict:
    """
    LangGraph node for the Research Agent.
    
    1. Extracts stock symbol from the user query
    2. Fetches market data via yfinance
    3. Generates a research summary using Gemini
    4. Writes market_data and research_summary to shared state
    
    Args:
        state: Current FinState with the user query
    
    Returns:
        Dict with state updates (market_data, research_summary)
    """
    query = state["query"]
    symbol = extract_symbol(query)
    
    # Step 1: Fetch market data
    market_data = fetch_market_data(symbol)
    
    if "error" in market_data:
        return {
            "market_data": market_data,
            "research_summary": f"Unable to fetch market data: {market_data['error']}",
            "messages": [{"role": "assistant", "content": f"Research Agent: Error - {market_data['error']}"}]
        }
    
    # Step 2: Generate research summary using Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    research_prompt = f"""You are a financial research analyst. Analyze the following market data 
and provide a concise research summary for an investor.

Stock: {market_data['company_name']} ({market_data['symbol']})
Current Price: ${market_data['current_price']}
P/E Ratio: {market_data.get('pe_ratio', 'N/A')}
Market Cap: {format_large_number(market_data.get('market_cap'))}
52-Week Range: ${market_data['fifty_two_week_low']} - ${market_data['fifty_two_week_high']}
Sector: {market_data['sector']}
Industry: {market_data['industry']}

Recent Price Trend (last 5 days): {market_data['price_history'][-5:]}

User's Question: {query}

Provide a structured analysis covering:
1. Current market position (where is the price relative to 52-week range?)
2. Valuation signal (is P/E ratio reasonable for this sector?)
3. Recent momentum (what does the 5-day trend suggest?)
4. Key factors to consider

Keep it concise (150-200 words). Be factual, not promotional."""

    response = llm.invoke(research_prompt)
    research_summary = response.content
    
    return {
        "market_data": market_data,
        "research_summary": research_summary,
        "messages": [{"role": "assistant", "content": f"Research Agent: Completed analysis for {symbol}"}]
    }
