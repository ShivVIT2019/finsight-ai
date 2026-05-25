"""
FinSight AI — Streamlit Dashboard
Interactive UI over the multi-agent analysis pipeline.

Run with:
    streamlit run dashboard/app.py

Requires GEMINI_API_KEY in the environment.
"""

import os
import sys

# Make the project root importable when Streamlit runs this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from graph.pipeline import run_analysis


st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="wide")

st.title("📈 FinSight AI")
st.caption(
    "Multi-agent financial intelligence — a Research Agent and a Risk Agent "
    "collaborate through a LangGraph pipeline, calling real market-data tools."
)

# ── Sidebar: API key status + about ───────────────────────────────────────────
with st.sidebar:
    st.header("Status")
    if os.environ.get("GEMINI_API_KEY"):
        st.success("GEMINI_API_KEY is set")
    else:
        st.error("GEMINI_API_KEY not set")
        st.code("export GEMINI_API_KEY='your-key'", language="bash")

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Research Agent pulls market data, financials, and peers\n"
        "2. Risk Agent computes volatility, Sharpe, drawdown, beta\n"
        "3. Synthesis combines both into an investment brief\n\n"
        "Tools are called via a Gemini function-calling loop and are also "
        "exposed over MCP."
    )

# ── Input ──────────────────────────────────────────────────────────────────────
col_q, col_s = st.columns([3, 1])
with col_q:
    query = st.text_input(
        "Your question",
        value="Should I invest in this stock right now?",
        placeholder="e.g. Is NVIDIA a good long-term hold?",
    )
with col_s:
    symbol = st.text_input("Ticker", value="AAPL", placeholder="AAPL")

run = st.button("Analyze", type="primary", use_container_width=True)

# ── Run pipeline ────────────────────────────────────────────────────────────────
if run:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Set GEMINI_API_KEY in your environment before running.")
        st.stop()
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Agents are analyzing — fetching data, computing risk, synthesizing..."):
        try:
            result = run_analysis(query=query.strip(), symbol=symbol.strip())
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ── Top metrics row ──────────────────────────────────────────────────────────
    md = result.get("market_data") or {}
    rm = result.get("risk_metrics") or {}

    st.subheader(f"Results for {result.get('symbol', symbol)}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${md.get('current_price', '—')}")
    m2.metric("P/E", md.get("pe_ratio", "—"))
    m3.metric("Risk Score", rm.get("risk_score", "—"))
    m4.metric("Risk Tier", rm.get("risk_tier", "—"))

    # ── Final brief ────────────────────────────────────────────────────────────
    st.markdown("### Investment Brief")
    confidence = result.get("confidence", "—")
    st.caption(f"Confidence: **{confidence}**  ·  "
               f"Processing time: {result.get('processing_time_seconds', '—')}s")
    st.markdown(result.get("final_answer") or "_No brief generated._")

    # ── Detail panels ───────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        with st.expander("Risk Metrics", expanded=True):
            if rm:
                st.json({
                    "volatility": rm.get("volatility"),
                    "sharpe_ratio": rm.get("sharpe_ratio"),
                    "max_drawdown": rm.get("max_drawdown"),
                    "beta": rm.get("beta"),
                    "annualized_return": rm.get("annualized_return"),
                    "risk_score": rm.get("risk_score"),
                    "risk_tier": rm.get("risk_tier"),
                })
            else:
                st.write("No risk metrics returned.")

    with right:
        with st.expander("Tools Called (tool-use trace)", expanded=True):
            tools = result.get("tools_called") or []
            if tools:
                for i, t in enumerate(tools, 1):
                    st.write(f"{i}. `{t.get('tool')}`  {t.get('input', {})}")
            else:
                st.write("No tools were called.")

    # ── Agent outputs ───────────────────────────────────────────────────────────
    with st.expander("Research Agent output"):
        st.markdown(result.get("research_summary") or "_None_")
    with st.expander("Risk Agent output"):
        st.markdown(result.get("risk_assessment") or "_None_")

    # ── Raw state (debug) ───────────────────────────────────────────────────────
    with st.expander("Raw pipeline state (debug)"):
        st.json(result)
