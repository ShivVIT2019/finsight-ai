"""
FinSight AI - Shared Tools
==========================
Tools that agents can invoke during their execution. Each tool is a
standalone function that takes specific inputs and returns structured
outputs. LangGraph agents call these via tool-use patterns.

Design: Tools are stateless functions. All state management happens
in the LangGraph state object, not inside tools.
"""

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def fetch_market_data(symbol: str) -> dict:
    """
    Fetch real-time market data for a stock symbol using yfinance.
    
    Args:
        symbol: Stock ticker (e.g., 'NVDA', 'AAPL', 'GOOGL')
    
    Returns:
        Dictionary with current price, fundamentals, and 30-day history.
        Returns error dict if symbol is invalid or API fails.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        # Fetch 30-day price history
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            return {"error": f"No data found for symbol: {symbol}"}
        
        price_history = hist["Close"].tolist()
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(info.get("currentPrice", price_history[-1]), 2),
            "pe_ratio": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "fifty_two_week_low": round(info.get("fiftyTwoWeekLow", 0), 2),
            "avg_volume": info.get("averageVolume", 0),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price_history": [round(p, 2) for p in price_history],
            "company_name": info.get("longName", symbol.upper()),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}


def calculate_risk_metrics(price_history: list, risk_free_rate: float = 0.05) -> dict:
    """
    Calculate risk metrics from historical price data.
    
    Args:
        price_history: List of daily closing prices (at least 5 data points)
        risk_free_rate: Annual risk-free rate (default 5% for current T-bill rate)
    
    Returns:
        Dictionary with volatility, Sharpe ratio, max drawdown, and risk tier.
    """
    if not price_history or len(price_history) < 5:
        return {
            "error": "Insufficient price data for risk calculation",
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "beta": 0.0,
            "risk_tier": "Unknown",
            "risk_score": 50.0,
        }
    
    prices = np.array(price_history)
    
    # Daily returns
    returns = np.diff(prices) / prices[:-1]
    
    # Annualized volatility (assuming 252 trading days)
    volatility = float(np.std(returns) * np.sqrt(252))
    
    # Annualized return
    total_return = (prices[-1] - prices[0]) / prices[0]
    annualized_return = total_return * (252 / len(prices))
    
    # Sharpe ratio
    sharpe_ratio = float(
        (annualized_return - risk_free_rate) / volatility
        if volatility > 0 else 0.0
    )
    
    # Max drawdown
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    max_drawdown = float(np.max(drawdown))
    
    # Beta approximation (using S&P 500 proxy — simplified)
    # In production, you'd fetch SPY data and compute covariance
    beta = round(volatility / 0.16, 2)  # 0.16 is approximate annual SPY volatility
    
    # Composite risk score (0-100, higher = riskier)
    risk_score = min(100, max(0, 
        volatility * 100 +           # Weight volatility heavily
        max_drawdown * 80 +          # Drawdown matters
        abs(beta - 1) * 20 -         # Deviation from market
        max(0, sharpe_ratio) * 10    # Good Sharpe reduces risk
    ))
    
    # Risk tier assignment
    if risk_score < 30:
        risk_tier = "Conservative"
    elif risk_score < 60:
        risk_tier = "Moderate"
    else:
        risk_tier = "Aggressive"
    
    return {
        "volatility": round(volatility, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "beta": beta,
        "risk_tier": risk_tier,
        "risk_score": round(risk_score, 2),
        "annualized_return": round(annualized_return, 4),
        "daily_returns_std": round(float(np.std(returns)), 6),
    }


def format_large_number(num: Optional[float]) -> str:
    """Format large numbers for display (e.g., 2.5T, 300B, 45M)."""
    if num is None:
        return "N/A"
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num:,.2f}"
