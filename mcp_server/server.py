"""
FinSight AI — MCP Server
Exposes financial tools (market data, risk metrics, peer comparison, financials)
via Model Context Protocol using FastMCP.

This server is model-agnostic: any MCP client (Claude, Gemini-backed agents via an
MCP client, MCP Inspector) can call these tools. The tool implementations are shared
with agents/mcp_tools.py.

Run locally:
    python mcp_server/server.py
Then connect MCP Inspector to http://localhost:8001/mcp
"""

import json
import math
from datetime import datetime

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "FinSight Financial Tools",
    description="Real-time market data, risk analytics, and financials via MCP",
)


@mcp.tool()
def get_market_data(symbol: str, period: str = "6mo") -> str:
    """
    Fetch real-time market data for a stock symbol.

    Args:
        symbol: Stock ticker (e.g., AAPL, NVDA, TSLA)
        period: Data period — 1mo, 3mo, 6mo, 1y, 2y, 5y (default: 6mo)
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        hist = ticker.history(period=period)
        if hist.empty:
            return json.dumps({"error": f"No data found for {symbol}"})

        recent = hist.tail(5)
        price_history = [
            {"date": idx.strftime("%Y-%m-%d"),
             "close": round(row["Close"], 2),
             "volume": int(row["Volume"])}
            for idx, row in recent.iterrows()
        ]
        return json.dumps({
            "symbol": symbol.upper(),
            "current_price": round(info.get("currentPrice", info.get("regularMarketPrice", 0)), 2),
            "previous_close": round(info.get("previousClose", 0), 2),
            "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
            "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else None,
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "fifty_two_week_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "fifty_two_week_low": round(info.get("fiftyTwoWeekLow", 0), 2),
            "avg_volume": info.get("averageVolume"),
            "dividend_yield": round(info.get("dividendYield", 0), 2) if info.get("dividendYield") else None,
            "beta": round(info.get("beta", 0), 3) if info.get("beta") else None,
            "recent_prices": price_history,
            "data_period": period,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


@mcp.tool()
def calculate_risk_metrics(symbol: str, period: str = "1y", risk_free_rate: float = 0.05) -> str:
    """
    Compute risk metrics: volatility, Sharpe ratio, max drawdown, beta vs SPY,
    and a composite risk score (0-100).

    Args:
        symbol: Stock ticker
        period: Historical period — 6mo, 1y, 2y (default: 1y)
        risk_free_rate: Annual risk-free rate for Sharpe calc (default: 0.05)
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        if len(hist) < 30:
            return json.dumps({"error": f"Insufficient data for {symbol} (need 30+ days)"})

        closes = hist["Close"]
        daily_returns = closes.pct_change().dropna()
        volatility = float(daily_returns.std() * math.sqrt(252))
        total_return = float((closes.iloc[-1] / closes.iloc[0]) - 1)
        trading_days = len(closes)
        annualized_return = float((1 + total_return) ** (252 / trading_days) - 1)
        sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        rolling_max = closes.cummax()
        drawdown = (closes - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        spy = yf.Ticker("SPY").history(period=period)
        if len(spy) >= 30:
            spy_returns = spy["Close"].pct_change().dropna()
            min_len = min(len(daily_returns), len(spy_returns))
            stock_r = daily_returns.iloc[-min_len:]
            market_r = spy_returns.iloc[-min_len:]
            covariance = float(stock_r.cov(market_r))
            market_var = float(market_r.var())
            beta = covariance / market_var if market_var > 0 else 1.0
        else:
            beta = 1.0

        vol_score = min(volatility / 0.8, 1.0) * 30
        dd_score = min(abs(max_drawdown) / 0.5, 1.0) * 25
        beta_score = min(abs(beta) / 2.0, 1.0) * 25
        sharpe_penalty = max(0, (1 - sharpe) / 2) * 20
        risk_score = round(vol_score + dd_score + beta_score + sharpe_penalty, 1)

        if risk_score < 25:
            risk_tier = "Conservative"
        elif risk_score < 50:
            risk_tier = "Moderate"
        elif risk_score < 75:
            risk_tier = "Aggressive"
        else:
            risk_tier = "Speculative"

        return json.dumps({
            "symbol": symbol.upper(),
            "volatility": round(volatility, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_drawdown, 4),
            "beta": round(beta, 3),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "calculation_params": {
                "period": period,
                "risk_free_rate": risk_free_rate,
                "trading_days_used": trading_days,
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


@mcp.tool()
def compare_peers(symbol: str, peers: str = "") -> str:
    """
    Compare a stock against sector peers on PE, market cap, beta, margins,
    and revenue growth. Auto-detects peers if not specified.

    Args:
        symbol: Primary stock ticker
        peers: Comma-separated peer tickers (e.g., "MSFT,GOOG,META"). Auto if empty.
    """
    try:
        primary_info = yf.Ticker(symbol.upper()).info
        if peers:
            peer_list = [p.strip().upper() for p in peers.split(",")]
        else:
            sector = primary_info.get("sector", "")
            sector_peers = {
                "Technology": ["AAPL", "MSFT", "GOOG", "META", "NVDA", "AMZN"],
                "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
                "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABT", "LLY"],
                "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
            }
            candidates = sector_peers.get(sector, [])
            peer_list = [p for p in candidates if p != symbol.upper()][:4]

        comparisons = []
        for ticker_symbol in [symbol.upper()] + peer_list:
            try:
                info = yf.Ticker(ticker_symbol).info
                comparisons.append({
                    "symbol": ticker_symbol,
                    "price": round(info.get("currentPrice", 0), 2),
                    "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
                    "market_cap_B": round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None,
                    "beta": round(info.get("beta", 0), 3) if info.get("beta") else None,
                    "profit_margin": round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") else None,
                    "revenue_growth": round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
                })
            except Exception:
                continue
        return json.dumps({
            "primary": symbol.upper(),
            "sector": primary_info.get("sector", "N/A"),
            "comparisons": comparisons,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_financial_summary(symbol: str) -> str:
    """
    Get financial summary: revenue, net income, margins, debt, cash flow,
    and valuation ratios for the most recent fiscal year.

    Args:
        symbol: Stock ticker
    """
    try:
        info = yf.Ticker(symbol.upper()).info
        return json.dumps({
            "symbol": symbol.upper(),
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "employees": info.get("fullTimeEmployees"),
            "financials": {
                "revenue": info.get("totalRevenue"),
                "net_income": info.get("netIncomeToCommon"),
                "profit_margin": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else None,
                "operating_margin": round(info.get("operatingMargins", 0) * 100, 2) if info.get("operatingMargins") else None,
                "revenue_growth_yoy": round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
            },
            "balance_sheet": {
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "debt_to_equity": round(info.get("debtToEquity", 0), 2) if info.get("debtToEquity") else None,
                "current_ratio": round(info.get("currentRatio", 0), 2) if info.get("currentRatio") else None,
            },
            "valuation": {
                "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
                "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else None,
                "price_to_book": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else None,
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


@mcp.resource("market://status")
def market_status() -> str:
    """Current US market status (open / pre-market / after-hours / closed)."""
    now = datetime.utcnow()
    et_hour = (now.hour - 4) % 24
    weekday = now.weekday()
    if weekday >= 5:
        status, reason = "closed", "Weekend"
    elif (9 < et_hour < 16) or (et_hour == 9 and now.minute >= 30):
        status, reason = "open", "Regular trading hours"
    elif 4 <= et_hour < 9:
        status, reason = "pre-market", "Pre-market session"
    elif 16 <= et_hour < 20:
        status, reason = "after-hours", "After-hours session"
    else:
        status, reason = "closed", "Outside trading hours"
    return json.dumps({"status": status, "reason": reason, "utc_time": now.isoformat()})


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
