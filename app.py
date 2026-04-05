import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import json
import re
from dotenv import load_dotenv

# Import your CrewAI logic
from agents.council import investment_council

load_dotenv()

# --- 1. SESSION STATE ---
if 'ticker' not in st.session_state: st.session_state.ticker = "NVDA"
if 'amount' not in st.session_state: st.session_state.amount = 10000
if 'duration' not in st.session_state: st.session_state.duration = "1y"
if 'trace_by_agent' not in st.session_state: st.session_state.trace_by_agent = {}
if 'final_plan' not in st.session_state: st.session_state.final_plan = ""
# Track who spoke last to avoid repeating the name prefix
if 'last_agent' not in st.session_state: st.session_state.last_agent = None

# --- 2. PLOTLY HELPERS ---
def get_plots(ticker, amount, duration):
    try:
        df = yf.download(ticker, period=duration, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None, None, None, None
        
        p1 = go.Figure(data=[go.Bar(name='Price', x=['Current', 'Fair'], y=[df['Close'].iloc[-1], df['Close'].iloc[-1]*0.85], marker_color=['#1f77b4', '#2ca02c'])])
        p2 = go.Figure(go.Scatter(x=df.index, y=df['Close'].pct_change().rolling(20).std(), line=dict(color='#ff7f0e')))
        p3 = go.Figure(data=[go.Pie(labels=['Equity', 'Debt'], values=[70, 30], hole=.6, marker_colors=['#2ca02c', '#d62728'])])
        shares = amount / df['Close'].iloc[0]
        p4 = go.Figure(go.Scatter(x=df.index, y=df['Close'] * shares, fill='tozeroy', line=dict(color='#58a6ff')))
        
        for p in [p1, p2, p3, p4]: p.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        return p1, p2, p3, p4
    except: return None, None, None, None

# --- 3. UI SETUP ---
st.set_page_config(page_title="AI Investment Council", layout="wide")
st.markdown("""
    <style>
    .agent-active-card { border: 2px solid #58a6ff; background: #161b22; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .trace-container { 
        background-color: #0d1117; padding: 15px; border-radius: 5px; 
        font-family: 'Courier New', monospace; color: #d1d5da; font-size: 0.85rem;
        max-height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5;
    }
    .ticker-badge { background: #1e1e1e; color: #58a6ff; padding: 5px 10px; border-radius: 5px; border: 1px solid #58a6ff; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

tab_strategy, tab_insights, tab_trace = st.tabs(["🚀 Strategy & Plan", "📊 Insights", "⚙️ Trace"])

# --- TAB 1: STRATEGY & PLAN ---
with tab_strategy:
    st.title("🏦 AI Investment Council")
    st.markdown(f"**Targeting:** <span class='ticker-badge'>{st.session_state.ticker}</span>", unsafe_allow_html=True)
    
    query = st.text_area("Investment Goal", placeholder="e.g., apple", height=100)
    
    if st.button("🚀 Convene the Council", use_container_width=True):
        if query:
            st.session_state.trace_by_agent = {} 
            st.session_state.final_plan = ""
            st.session_state.last_agent = None # Reset
            
            full_capture = "" 
            manual_plan_capture = ""
            status_ui = st.empty()
            
            try:
                res = investment_council.kickoff(inputs={'user_prompt': query})
                
                # --- CLEAN STREAMING LOOP ---
                for chunk in res:
                    role = getattr(chunk, 'agent_role', "Council Member")
                    content = getattr(chunk, 'content', str(chunk))
                    full_capture += content
                    
                    if any(x in role.lower() for x in ["advisor", "visualizer", "strategist"]):
                        manual_plan_capture += content

                    # LOGIC: Only add the prefix if the agent has changed
                    if role not in st.session_state.trace_by_agent:
                        st.session_state.trace_by_agent[role] = ""
                    
                    if role != st.session_state.last_agent:
                        st.session_state.trace_by_agent[role] += f"\n\n--- {role.upper()} ---\n"
                        st.session_state.last_agent = role
                    
                    st.session_state.trace_by_agent[role] += content

                    status_ui.markdown(f"<div class='agent-active-card'>🔵 <b>ACTIVE:</b> {role.upper()}</div>", unsafe_allow_html=True)

                # Sync ticker & plan (same as previous working logic)
                match = re.search(r'(?:symbol|ticker)["\s:]+([A-Z]{2,5})', full_capture, re.IGNORECASE)
                if match: st.session_state.ticker = match.group(1).upper()

                official_raw = getattr(res, 'raw', "")
                if official_raw and "CrewStreamingOutput" not in str(official_raw):
                    st.session_state.final_plan = official_raw
                else:
                    st.session_state.final_plan = re.sub(r'\{.*?\}', '', manual_plan_capture, flags=re.DOTALL).strip()

                status_ui.empty()
                st.rerun()

            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if st.session_state.final_plan:
        st.divider()
        st.subheader("📜 Council Investment Plan")
        st.markdown(st.session_state.final_plan)

# --- TAB 2: INSIGHTS ---
with tab_insights:
    t = st.session_state.ticker
    st.subheader(f"📊 Visual Justification: {t}")
    p1, p2, p3, p4 = get_plots(t, st.session_state.amount, st.session_state.duration)
    if p1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(p1, use_container_width=True, key=f"v_{t}")
        c2.plotly_chart(p2, use_container_width=True, key=f"m_{t}")
        st.divider()
        c3, c4 = st.columns(2)
        c3.plotly_chart(p3, use_container_width=True, key=f"s_{t}")
        c4.plotly_chart(p4, use_container_width=True, key=f"g_{t}")

# --- TAB 3: TRACE (CLEANED) ---
with tab_trace:
    st.subheader("System Execution Trace")
    if not st.session_state.trace_by_agent:
        st.info("No logs yet. Run an analysis.")
    else:
        for agent, full_text in st.session_state.trace_by_agent.items():
            with st.expander(f"🕵️ Trace: {agent}", expanded=True):
                # Using the <pre> tag and CSS for a clean terminal output
                st.markdown(f"<div class='trace-container'>{full_text}</div>", unsafe_allow_html=True)