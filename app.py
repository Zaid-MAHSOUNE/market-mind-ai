import streamlit as st
import time
from agents.researcher import Researcher
from agents.analyst import Analyst

# --- Page Configuration ---
st.set_page_config(
    page_title="MarketMind AI | Financial Investigator",
    page_icon="📈",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .reasoning-box { padding: 15px; border-left: 5px solid #007bff; background-color: #f8f9fa; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- Sidebar: Step Titles & History ---
st.sidebar.title("🧠 Reasoning Journey")
# Placeholder for dynamic step highlighting
step_placeholder = st.sidebar.empty()

st.sidebar.divider()
st.sidebar.title("📜 Investigation History")
if st.session_state.history:
    for stock in st.session_state.history:
        st.sidebar.button(f"🔍 {stock}", key=f"hist_{stock}")
else:
    st.sidebar.info("No stocks discussed yet.")

def main():
    st.title("📈 MarketMind AI: Financial Investigator")
    st.markdown("### Autonomous Trend Analysis & Investment Recommendations")
    st.divider()

    # --- User Input Section ---
    col1, col2 = st.columns([2, 1])
    with col1:
        company_input = st.text_input("Enter Company Name or Ticker:", placeholder="e.g., Enedis, Nvidia, Apple")
    with col2:
        invest_goal = st.selectbox("Investment Profile:", ["Aggressive Growth", "Balanced", "Conservative"])

    if st.button("Start Investigation", type="primary"):
        if not company_input:
            st.error("Please enter a company name or a ticker symbol.")
            return
        
        # Update History & State
        st.session_state.analysis_done = False
        if company_input not in st.session_state.history:
            st.session_state.history.append(company_input)

        # Initialize Agents
        researcher = Researcher()
        analyst = Analyst()

        # --- STAGE 1: ReAct Loop (Researcher) ---
        step_placeholder.markdown("### 🏃 Current Step:\n**Phase 1: Research & Discovery**")
        
        with st.status("🚀 Phase 1: ReAct Loop (Researcher)...", expanded=True) as status:
            st.write("🏃 **Action:** Executing `GeopoliticalSearch`...")
            
            # Action: Tool usage
            raw_data = researcher.get_raw_data(company_input)
            
            st.write("🧠 **Thought:** Processing market news and prices...")
            # Streaming the internal monologue DIRECTLY in the status box
            research_summary = st.write_stream(
                researcher.execute_stream(f"Perform a ReAct analysis (Thought/Action/Observation) on this data: {raw_data}")
            )
            
            status.update(label="✅ Research Phase Complete!", state="complete", expanded=False)
            st.session_state.research_summary = research_summary

        # --- STAGE 2: Chain of Thought & Self-Correction (Analyst) ---
        step_placeholder.markdown("### 🏃 Current Step:\n**Phase 2: Advanced Reasoning**")
        
        with st.status("🧠 Phase 2: Advanced Reasoning (Analyst)...", expanded=True) as status:
            st.write("🔢 **Thinking Step-by-Step (CoT)...**")
            
            # Streaming CoT directly in the middle of the screen
            cot_result = st.write_stream(
                analyst.execute_stream(f"Using Chain of Thought (CoT), decompose the analysis for {company_input}: {research_summary}")
            )
            
            st.write("⚖️ **Performing Self-Correction (Reflexion)...**")
            
            # Streaming final Reflexion & Verdict
            final_verdict = st.write_stream(
                analyst.execute_stream("Critique your analysis. Identify any potential errors and provide final timing and duration recommendations.")
            )
            
            status.update(label="✅ Analysis & Reflexion Complete!", state="complete", expanded=False)
            st.session_state.final_verdict = final_verdict
            st.session_state.analysis_done = True
        
        step_placeholder.markdown("### ✅ Status:\n**Investigation Concluded**")

    # --- Main Dashboard Display ---
    if st.session_state.analysis_done:
        st.divider()
        st.subheader(f"📊 Investment Intelligence Report: {company_input}")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Final Verdict", "🧠 Reasoning Detail", "🛠️ Data Observation Logs"])
        
        with tab1:
            st.success("#### Strategy Recommendation")
            st.markdown(st.session_state.final_verdict)
            st.download_button("Download Report (PDF)", data=st.session_state.final_verdict, file_name=f"{company_input}_Report.txt")

        with tab2:
            st.markdown("#### Logic Chain Summary")
            st.write(st.session_state.research_summary)

        with tab3:
            st.markdown("#### 🛠️ Data Observation Logs")
            # This shows the RAW ReAct observation data in a technical code window
            st.code(st.session_state.research_summary, language="markdown")

if __name__ == "__main__":
    main()