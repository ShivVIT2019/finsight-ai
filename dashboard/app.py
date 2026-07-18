"""
FinSight AI — Streamlit Dashboard (thin client)
Calls the deployed FinSight AI API instead of running the pipeline in-process.

Why a thin client:
- The Gemini API key lives only in the API's Secret Manager, not here
- This container is small (just Streamlit + requests), so builds are fast
- Mirrors a real product architecture: frontend talks to backend over HTTP

Configuration:
    Set FINSIGHT_API_URL to point at the deployed API
    (defaults to the live Cloud Run service)

Run locally:
    streamlit run dashboard/app.py
"""

import os

import requests
import streamlit as st


# Live Cloud Run API by default; override with env var for local testing.
DEFAULT_API = "https://finsight-ai-4lfjlhbw2q-uc.a.run.app"
API_URL = os.environ.get("FINSIGHT_API_URL", DEFAULT_API).rstrip("/")
REQUEST_TIMEOUT = 120  # seconds; matches the API's 300s ceiling with margin


st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="wide")

st.title("📈 FinSight AI")
st.caption(
    "Multi-agent financial intelligence — a Research Agent and a Risk Agent "
    "collaborate through a LangGraph pipeline, calling real market-data tools."
)

# ── Sidebar: backend status + about ───────────────────────────────────────────
with st.sidebar:
    st.header("Backend")
    st.code(API_URL, language=None)
    try:
        r = requests.get(f"{API_URL}/", timeout=10)
        if r.status_code == 200:
            st.success("API is healthy")
        else:
            st.warning(f"API returned {r.status_code}")
    except requests.RequestException as e:
        st.error(f"API unreachable: {e}")

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

# ── Run analysis (via API) ─────────────────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner(
        "Agents are analyzing — fetching data, computing risk, synthesizing... "
        "First request may take ~40s while the backend warms up."
    ):
        try:
            resp = requests.post(
                f"{API_URL}/analyze",
                json={"query": query.strip(), "symbol": symbol.strip()},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.Timeout:
            st.error(
                f"Request timed out after {REQUEST_TIMEOUT}s. "
                "Try again — cold starts can be slow."
            )
            st.stop()
        except requests.RequestException as e:
            st.error(f"Failed to reach the API: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text[:500]}")
        st.stop()

    try:
        result = resp.json()
    except ValueError:
        st.error("API response was not valid JSON.")
        st.stop()

    if isinstance(result, dict) and result.get("error"):
        st.error(result["error"])
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
    st.caption(
        f"Confidence: **{confidence}**  ·  "
        f"Processing time: {result.get('processing_time_seconds', '—')}s"
    )
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
    with st.expander("Raw API response (debug)"):
        st.json(result)
