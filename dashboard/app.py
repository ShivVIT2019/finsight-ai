"""
FinSight AI — Streamlit Dashboard (thin client)
Calls the deployed FinSight AI API. Phase 1 layout: sidebar controls,
main workspace, welcome state before first analysis.
"""

import os

import requests
import streamlit as st
from streamlit_searchbox import st_searchbox
import plotly.graph_objects as go
import pandas as pd


def search_yahoo_tickers(query: str) -> list:
    """
    Live-search Yahoo Finance for matching tickers.
    Returns list of (display_label, symbol) tuples for st_searchbox.
    """
    if not query or len(query.strip()) < 1:
        return []
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query.strip(), "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=4,
        )
        if r.status_code != 200:
            return []
        results = []
        for q in r.json().get("quotes", []):
            sym = q.get("symbol")
            name = q.get("shortname") or q.get("longname") or ""
            exch = q.get("exchDisp") or q.get("exchange") or ""
            qtype = q.get("quoteType") or ""
            if not sym:
                continue
            label = f"{sym} — {name}" + (f"  ({exch})" if exch else "")
            if qtype and qtype not in ("EQUITY",):
                label += f"  ·{qtype.lower()}"
            results.append((label, sym))
        return results
    except Exception:
        return []


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_API = "https://finsight-ai-4lfjlhbw2q-uc.a.run.app"
API_URL = os.environ.get("FINSIGHT_API_URL", DEFAULT_API).rstrip("/")
REQUEST_TIMEOUT = 120





def _md_safe(text: str) -> str:
    """
    Escape $ so Streamlit's markdown/mathjax doesn't interpret $...$ as LaTeX.
    Applied to model-generated text that talks about dollar amounts.
    """
    if not text:
        return text
    return text.replace("$", "\\$")


# ── CHART HELPERS ──────────────────────────────────────────────────────────────

def _risk_gauge(score, tier):
    """Semicircle gauge showing composite risk score (0-100)."""
    if score is None:
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"font": {"size": 42, "color": "white"}},
        title={"text": f"Risk Tier: <b>{tier or '—'}</b>",
               "font": {"size": 14, "color": "#aaa"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666",
                     "tickfont": {"color": "#888"}},
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],  "color": "#0e7d3f"},   # green – conservative
                {"range": [25, 50], "color": "#c8a415"},   # yellow – moderate
                {"range": [50, 75], "color": "#c96419"},   # orange – aggressive
                {"range": [75, 100],"color": "#a11f1f"},   # red – speculative
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": float(score),
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def _price_history_chart(recent_prices, symbol):
    """Line chart of the last few closes."""
    if not recent_prices:
        return None
    df = pd.DataFrame(recent_prices)
    if "date" not in df or "close" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines+markers",
        line=dict(color="#4ea1ff", width=3),
        marker=dict(size=8, color="#4ea1ff"),
        hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{symbol} — recent closes", font=dict(size=14, color="#ddd")),
        height=260,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#888"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                   color="#888", tickprefix="$"),
        showlegend=False,
    )
    return fig


def _peer_bar_chart(peer_comparison, symbol):
    """Grouped bar chart comparing PE, profit margin, revenue growth across peers."""
    if not peer_comparison or not peer_comparison.get("comparisons"):
        return None
    peers = peer_comparison["comparisons"]
    if len(peers) < 2:
        return None
    symbols = [p.get("symbol", "?") for p in peers]
    pe = [p.get("pe_ratio") for p in peers]
    margin = [p.get("profit_margin") for p in peers]
    growth = [p.get("revenue_growth") for p in peers]
    colors_pe = ["#4ea1ff" if s == symbol else "#2f4a6b" for s in symbols]
    colors_pm = ["#7bd88f" if s == symbol else "#2f5c3f" for s in symbols]
    colors_gr = ["#f2a65a" if s == symbol else "#6b4a2c" for s in symbols]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="P/E ratio", x=symbols, y=pe, marker_color=colors_pe))
    fig.add_trace(go.Bar(name="Profit margin %", x=symbols, y=margin, marker_color=colors_pm))
    fig.add_trace(go.Bar(name="Revenue growth %", x=symbols, y=growth, marker_color=colors_gr))
    fig.update_layout(
        title=dict(text=f"{symbol} vs peers", font=dict(size=14, color="#ddd")),
        barmode="group",
        height=300,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#888"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", color="#888"),
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
if "query" not in st.session_state:
    st.session_state.query = "Should I invest in this stock right now?"


def check_backend():
    """Return (is_healthy, message)."""
    try:
        r = requests.get(f"{API_URL}/", timeout=8)
        if r.status_code == 200:
            return True, "Connected"
        return False, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, "Unreachable"


def run_analysis(query: str, symbol: str):
    """Call the API. Returns (result_dict, error_message)."""
    try:
        resp = requests.post(
            f"{API_URL}/analyze",
            json={"query": query, "symbol": symbol},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return None, f"Timed out after {REQUEST_TIMEOUT}s. Try again."
    except requests.RequestException as e:
        return None, f"Failed to reach the API: {e}"

    if resp.status_code != 200:
        return None, f"API returned {resp.status_code}: {resp.text[:300]}"

    try:
        return resp.json(), None
    except ValueError:
        return None, "API response was not valid JSON."


# ── Sidebar: control panel ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 FinSight AI")
    st.caption("Multi-agent financial analysis")

    st.divider()

    # Backend health
    healthy, msg = check_backend()
    if healthy:
        st.success(f"● Backend {msg}", icon="🟢")
    else:
        st.error(f"● Backend {msg}", icon="🔴")

    st.divider()

    # Input controls — live Yahoo Finance search
    st.markdown("**Analysis Setup**")

    st.caption("Search any stock, ETF, or crypto")
    selected_symbol = st_searchbox(
        search_function=search_yahoo_tickers,
        placeholder="Type a company or ticker (e.g. Apple, BTC, SPY)…",
        key="ticker_searchbox",
        clear_on_submit=False,
    )
    if selected_symbol and selected_symbol != st.session_state.ticker:
        st.session_state.ticker = selected_symbol
        st.rerun()

    with st.expander("Or type a symbol directly"):
        ticker_manual = st.text_input(
            "Ticker",
            value=st.session_state.ticker,
            placeholder="AAPL",
            max_chars=15,
            label_visibility="collapsed",
        ).upper().strip()
        if ticker_manual and ticker_manual != st.session_state.ticker:
            st.session_state.ticker = ticker_manual

    ticker = st.session_state.ticker
    st.caption(f"Selected: **{ticker}**" if ticker else "No ticker selected")

    query = st.text_area(
        "Your question",
        value=st.session_state.query,
        height=80,
        placeholder="e.g. Is NVIDIA a good long-term hold?",
    )

    run = st.button(
        "▶ Analyze",
        type="primary",
        use_container_width=True,
        disabled=not healthy,
    )

    if not healthy:
        st.caption("⚠️ Backend must be healthy to run analysis")

    st.divider()

    with st.expander("How it works"):
        st.markdown(
            "1. **Research Agent** pulls market data, financials, peers\n"
            "2. **Risk Agent** computes volatility, Sharpe, drawdown, beta\n"
            "3. **Synthesis** combines both into an investment brief\n\n"
            "Tools are called via a Gemini function-calling loop and are also "
            "exposed over MCP."
        )


# ── Main area ─────────────────────────────────────────────────────────────────

# Trigger analysis
if run:
    if not query.strip() or not ticker.strip():
        st.warning("Please enter both a ticker and a question.")
    else:
        st.session_state.ticker = ticker
        st.session_state.query = query
        with st.spinner(
            f"Analyzing {ticker} — Research Agent → Risk Agent → Synthesis. "
            "First request may take ~40s (cold start)."
        ):
            result, error = run_analysis(query.strip(), ticker.strip())
        if error:
            st.error(error)
            st.session_state.result = None
        else:
            st.session_state.result = result
            st.rerun()


# Welcome state (no analysis yet)
if st.session_state.result is None:
    st.markdown("# 📈 FinSight AI")
    st.markdown(
        "#### Multi-agent financial intelligence powered by real-time market data"
    )
    st.markdown("")

    # Feature cards row
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("##### 🔍 Research Agent")
            st.caption(
                "Pulls live market data, financial statements, and peer comparisons "
                "for the target ticker via the Gemini function-calling loop."
            )
    with c2:
        with st.container(border=True):
            st.markdown("##### ⚖️ Risk Agent")
            st.caption(
                "Computes volatility, Sharpe ratio, max drawdown, beta, and a "
                "composite 0–100 risk score against SPY."
            )
    with c3:
        with st.container(border=True):
            st.markdown("##### 📝 Synthesis")
            st.caption(
                "Combines research + risk into a structured investment brief with "
                "an actionable recommendation and confidence level."
            )

    st.markdown("")
    with st.container(border=True):
        st.markdown("##### Get started")
        st.markdown(
            "Use the sidebar on the left to pick a ticker and ask a question — "
            "the analysis will appear here. Try one of the popular tickers to see "
            "the pipeline in action, or type in your own."
        )

# Results state (after analysis)
else:
    result = st.session_state.result
    md = result.get("market_data") or {}
    rm = result.get("risk_metrics") or {}

    # Header row
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f"# {result.get('symbol', ticker)} Analysis")
        st.caption(
            f"{md.get('sector', '—')} · {md.get('industry', '—')} · "
            f"Confidence: **{result.get('confidence', '—')}** · "
            f"Processed in {result.get('processing_time_seconds', '—')}s"
        )
    with hcol2:
        if st.button("← New analysis", use_container_width=True):
            st.session_state.result = None
            st.rerun()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${md.get('current_price', '—')}")
    m2.metric("P/E", md.get("pe_ratio", "—"))
    m3.metric("Risk Score", rm.get("risk_score", "—"))
    m4.metric("Risk Tier", rm.get("risk_tier", "—"))

    st.markdown("")

    # ── Charts row: risk gauge (left) + price history (right) ────────────────
    chart_left, chart_right = st.columns([1, 2])
    with chart_left:
        gauge = _risk_gauge(rm.get("risk_score"), rm.get("risk_tier"))
        if gauge is not None:
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    with chart_right:
        price_fig = _price_history_chart(md.get("recent_prices"), result.get("symbol", ""))
        if price_fig is not None:
            st.plotly_chart(price_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Peer comparison row ──────────────────────────────────────────────────
    peer_fig = _peer_bar_chart(result.get("peer_comparison"), result.get("symbol", ""))
    if peer_fig is not None:
        st.plotly_chart(peer_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # Investment brief (the model returns its own H2 heading)
    st.markdown(_md_safe(result.get("final_answer") or "_No brief generated._"))

    st.markdown("---")

    # Detail panels
    with st.expander("Risk Metrics (detail)"):
        st.json({
            "volatility": rm.get("volatility"),
            "sharpe_ratio": rm.get("sharpe_ratio"),
            "max_drawdown": rm.get("max_drawdown"),
            "beta": rm.get("beta"),
            "annualized_return": rm.get("annualized_return"),
            "risk_score": rm.get("risk_score"),
            "risk_tier": rm.get("risk_tier"),
        })

    with st.expander("Tools Called (tool-use trace)"):
        tools = result.get("tools_called") or []
        for i, t in enumerate(tools, 1):
            st.write(f"{i}. `{t.get('tool')}`  {t.get('input', {})}")

    with st.expander("Research Agent output"):
        st.markdown(_md_safe(result.get("research_summary") or "_None_"))

    with st.expander("Risk Agent output"):
        st.markdown(_md_safe(result.get("risk_assessment") or "_None_"))

    with st.expander("Raw API response (debug)"):
        st.json(result)
