"""
🏦 AI Financial Fragility Detector — Streamlit App
Interactive interface for exploring bank fragility scores, RAG assessments, and LLM analysis.

Run with: streamlit run app.py
Requirements: pip install streamlit pandas plotly groq
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import re

# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="AI Financial Fragility Detector",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    .block-container { padding-top: 2rem; }
    
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; }
    
    .score-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
    }
    .score-critical { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .score-high { background: #fff7ed; color: #ea580c; border: 1px solid #fed7aa; }
    .score-elevated { background: #fefce8; color: #ca8a04; border: 1px solid #fef08a; }
    .score-moderate { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .score-low { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
    
    .failed-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    .stable-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════
@st.cache_data
def load_data():
    """Load all CSV data files."""
    data = {}
    data_dir = "data"
    
    files = {
        'fragility': 'fragility_scored.csv',
        'llm_scores': 'llm_fragility_scores.csv',
        'rag': 'rag_risk_assessments.csv',
        'enhanced_rag': 'enhanced_rag_assessments.csv',
        'drawdowns': 'stock_drawdowns.csv',
        'textual': 'edgar_textual_stress_scores.csv',
        'synthetic': 'synthetic_bank_profiles.csv',
        'raw': 'fdic_financials_raw.csv',
    }
    
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            try:
                data[key] = pd.read_csv(path)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()
    
    return data

data = load_data()

# ══════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════
def get_risk_tier(score):
    """Convert fragility score to risk tier."""
    if pd.isna(score): return "Unknown", "score-moderate"
    pct = score * 100 if score <= 1 else score
    if pct >= 75: return "Critical", "score-critical"
    elif pct >= 60: return "High", "score-high"
    elif pct >= 45: return "Elevated", "score-elevated"
    elif pct >= 25: return "Moderate", "score-moderate"
    else: return "Low", "score-low"

def get_bank_list():
    """Get sorted list of bank names."""
    if 'fragility' in data and not data['fragility'].empty:
        return sorted(data['fragility']['bank_name'].unique())
    return []

def get_latest_data(bank_name):
    """Get the most recent year's data for a bank."""
    df = data['fragility']
    bank_data = df[df['bank_name'] == bank_name].sort_values('year')
    if bank_data.empty:
        return None
    return bank_data.iloc[-1]

def get_bank_timeseries(bank_name):
    """Get all years of data for a bank."""
    df = data['fragility']
    return df[df['bank_name'] == bank_name].sort_values('year')

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Fragility Detector")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔍 Bank Analyzer", "📊 Compare Banks", "🤖 AI Assessment", "📋 Data Explorer"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("##### About")
    st.markdown(
        "AI-powered early warning system detecting structural weakness "
        "in regional banks using financial ratios, NLP, and generative AI."
    )
    st.markdown("---")
    st.markdown("##### Stats")
    if not data['fragility'].empty:
        n_banks = int(data['fragility']['bank_name'].nunique())
        yr_min = int(data['fragility']['year'].min())
        yr_max = int(data['fragility']['year'].max())
        st.metric("Banks Analyzed", f"{n_banks}")
        st.metric("Years of Data", f"{yr_min}–{yr_max}")
        st.metric("Gen AI Components", "3")

# ══════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🏦 AI Financial Fragility Detector")
    st.markdown("*Early Warning Signals in Regional Banks*")
    st.markdown("---")
    
    with st.expander("ℹ️ How to read fragility scores"):
        st.markdown("""
        - **0.0 – 0.3**: Low fragility — strong capital, healthy liquidity
        - **0.3 – 0.5**: Moderate — some structural concerns worth monitoring
        - **0.5 – 0.7**: Elevated — multiple warning signs present
        - **0.7 – 1.0**: Critical — severe structural weakness detected
        
        Scores combine 5 components: liquidity risk, leverage risk, interest coverage risk, 
        deposit flight risk, and textual stress from SEC filings. The V2 index uses 
        data-driven weights learned from actual bank failures.
        """)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if not data['fragility'].empty:
        latest = data['fragility'].sort_values('year').groupby('bank_name').last().reset_index()
        
        with col1:
            st.metric("Banks Analyzed", str(len(latest)))
        with col2:
            failed_count = int(latest['failed'].sum()) if 'failed' in latest.columns else 0
            st.metric("Failed Banks", str(failed_count))
        with col3:
            scored = int(latest['fragility_score'].notna().sum())
            st.metric("Scored (V1)", str(scored))
        with col4:
            if 'fragility_score_v2' in latest.columns:
                scored_v2 = int(latest['fragility_score_v2'].notna().sum())
                st.metric("Scored (V2)", str(scored_v2))
        with col5:
            if not data['synthetic'].empty:
                st.metric("Synthetic Profiles", str(len(data['synthetic'])))
            
            st.markdown("---")
    
    # Fragility Rankings
    st.markdown("### 📊 Fragility Rankings (Latest Year)")
    
    if not data['fragility'].empty:
        latest = data['fragility'].sort_values('year').groupby('bank_name').last().reset_index()
        
        score_col = 'fragility_score_v2' if 'fragility_score_v2' in latest.columns else 'fragility_score'
        latest_sorted = latest.sort_values(score_col, ascending=False)
        
        # Display as a styled table
        display_cols = ['bank_name', score_col, 'failed']
        if 'fragility_percentile_v2' in latest.columns:
            display_cols.insert(2, 'fragility_percentile_v2')
        
        for _, row in latest_sorted.iterrows():
            score = row[score_col]
            tier, css_class = get_risk_tier(score)
            failed = row.get('failed', False)
            
            col_a, col_b, col_c, col_d = st.columns([3, 1.5, 1, 1])
            with col_a:
                st.markdown(f"**{row['bank_name']}**")
            with col_b:
                st.markdown(f"`{score:.4f}`" if pd.notna(score) else "`N/A`")
            with col_c:
                st.markdown(f'<span class="{css_class} score-badge">{tier}</span>', unsafe_allow_html=True)
            with col_d:
                if failed:
                    st.markdown('<span class="failed-tag">FAILED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="stable-tag">Stable</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Heatmap
    st.markdown("### 🗺️ Fragility Heatmap")
    if not data['fragility'].empty:
        score_col = 'fragility_score_v2' if 'fragility_score_v2' in data['fragility'].columns else 'fragility_score'
        pivot = data['fragility'].pivot_table(values=score_col, index='bank_name', columns='year', aggfunc='mean')
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        
        fig = px.imshow(
            pivot, color_continuous_scale='RdYlGn_r',
            labels=dict(x="Year", y="Bank", color="Fragility"),
            title="Fragility Scores Across Banks and Years",
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# PAGE 2: BANK ANALYZER
# ══════════════════════════════════════════
elif page == "🔍 Bank Analyzer":
    st.markdown("# 🔍 Bank Analyzer")
    st.markdown("*Deep dive into a single bank's fragility profile*")
    st.markdown("---")
    
    banks = get_bank_list()
    if not banks:
        st.error("No bank data available. Make sure the data/ directory contains the CSV files.")
        st.stop()
    
    selected_bank = st.selectbox("Select a Bank", banks, index=0)
    
    latest = get_latest_data(selected_bank)
    timeseries = get_bank_timeseries(selected_bank)
    
    if latest is None:
        st.warning(f"No data available for {selected_bank}")
        st.stop()
    
    # Bank header
    failed = latest.get('failed', False)
    status_html = '<span class="failed-tag">⚠️ FAILED</span>' if failed else '<span class="stable-tag">✅ Stable</span>'
    st.markdown(f"## {selected_bank} {status_html}", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    score_v1 = latest.get('fragility_score', np.nan)
    score_v2 = latest.get('fragility_score_v2', np.nan)
    tier, _ = get_risk_tier(score_v2 if pd.notna(score_v2) else score_v1)
    
    with col1:
        st.metric("Fragility Score (V2)", f"{float(score_v2):.4f}" if pd.notna(score_v2) else "N/A")
    with col2:
        st.metric("Risk Tier", str(tier))
    with col3:
        roa = latest.get('ROA', np.nan)
        st.metric("ROA", f"{float(roa):.2f}%" if pd.notna(roa) else "N/A")
    with col4:
        roe = latest.get('ROE', np.nan)
        st.metric("ROE", f"{float(roe):.2f}%" if pd.notna(roe) else "N/A")
    
    st.markdown("---")
    
    # Two columns: ratios + chart
    left, right = st.columns([1, 1.5])
    
    with left:
        st.markdown("#### Financial Ratios")
        
        ratios = {
            'Liquidity Ratio': latest.get('liquidity_ratio', np.nan),
            'Debt-to-Equity': latest.get('debt_to_equity', np.nan),
            'Interest Coverage': latest.get('interest_coverage', np.nan),
            'Loan-to-Deposit': latest.get('loan_to_deposit', np.nan),
            'Uninsured Deposit %': latest.get('uninsured_deposit_ratio', np.nan),
            'Core Deposit Ratio': latest.get('core_deposit_ratio', np.nan),
            'NPA Ratio': latest.get('npa_ratio', np.nan),
        }
        
        for name, val in ratios.items():
            if pd.notna(val):
                st.markdown(f"**{name}:** `{val:.4f}`")
            else:
                st.markdown(f"**{name}:** N/A")
    
    with right:
        st.markdown("#### Fragility Score Over Time")
        if not timeseries.empty:
            score_col = 'fragility_score_v2' if 'fragility_score_v2' in timeseries.columns else 'fragility_score'
            fig = px.line(
                timeseries, x='year', y=score_col,
                markers=True,
                labels={score_col: 'Fragility Score', 'year': 'Year'},
            )
            fig.update_traces(line_color='#ef4444' if failed else '#3b82f6', line_width=3)
            fig.update_layout(height=350, margin=dict(t=10, b=30))
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Fragility components
    st.markdown("#### Fragility Score Components")
    components = ['liquidity_risk', 'leverage_risk', 'coverage_risk', 'deposit_risk', 'textual_stress']
    comp_vals = {c.replace('_', ' ').title(): latest.get(c, 0.5) for c in components}
    comp_vals = {k: v for k, v in comp_vals.items() if pd.notna(v)}
    
    if comp_vals:
        fig = go.Figure(go.Bar(
            x=list(comp_vals.values()),
            y=list(comp_vals.keys()),
            orientation='h',
            marker_color=['#ef4444', '#f59e0b', '#eab308', '#3b82f6', '#8b5cf6']
        ))
        fig.update_layout(
            height=280, margin=dict(t=10, b=20, l=0, r=0),
            xaxis_title="Risk Score (0-1)",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM Score
    st.markdown("---")
    st.markdown("#### 🤖 LLM Risk Assessment")
    
    if not data['llm_scores'].empty:
        llm_row = data['llm_scores'][data['llm_scores']['bank_name'] == selected_bank]
        if not llm_row.empty:
            llm_row = llm_row.iloc[0]
            
            lcol1, lcol2, lcol3, lcol4 = st.columns(4)
            with lcol1:
                ov = llm_row.get('overall_fragility', 'N/A')
                st.metric("Overall Score", f"{int(ov)}/10" if pd.notna(ov) and str(ov) != 'nan' else "N/A")
            with lcol2:
                lr_val = llm_row.get('liquidity_risk', 'N/A')
                st.metric("Liquidity Risk", f"{int(lr_val)}/10" if pd.notna(lr_val) and str(lr_val) != 'nan' else "N/A")
            with lcol3:
                df_val = llm_row.get('deposit_flight_risk', 'N/A')
                st.metric("Deposit Flight", f"{int(df_val)}/10" if pd.notna(df_val) and str(df_val) != 'nan' else "N/A")
            with lcol4:
                conf = str(llm_row.get('confidence', 'N/A'))
                st.metric("Confidence", conf if conf != 'nan' else "N/A")
            
            concern = llm_row.get('key_concern', '')
            if concern and str(concern) != 'nan':
                st.info(f"**Key Concern:** {concern}")
            
            reasoning = llm_row.get('reasoning', '')
            if reasoning and str(reasoning) != 'nan':
                st.markdown(f"**Reasoning:** {reasoning}")
        else:
            st.info("No LLM assessment available for this bank.")
    
    # RAG Assessment
    st.markdown("---")
    st.markdown("#### 📄 RAG Risk Assessment")
    
    rag_source = data.get('enhanced_rag', data.get('rag', pd.DataFrame()))
    if not rag_source.empty:
        rag_row = rag_source[rag_source['bank_name'] == selected_bank]
        if not rag_row.empty:
            rag_row = rag_row.iloc[0]
            assessment = rag_row.get('assessment', '')
            sources = rag_row.get('num_sources', 0)
            
            st.markdown(f"*Sources retrieved: {sources}*")
            if assessment and str(assessment) != 'nan':
                st.markdown(f"```\n{assessment}\n```")
        else:
            st.info("No RAG assessment available for this bank.")

# ══════════════════════════════════════════
# PAGE 3: COMPARE BANKS
# ══════════════════════════════════════════
elif page == "📊 Compare Banks":
    st.markdown("# 📊 Compare Banks")
    st.markdown("*Side-by-side fragility comparison*")
    st.markdown("---")
    
    banks = get_bank_list()
    
    col1, col2 = st.columns(2)
    with col1:
        bank_a = st.selectbox("Bank A", banks, index=0)
    with col2:
        default_b = min(1, len(banks) - 1)
        bank_b = st.selectbox("Bank B", banks, index=default_b)
    
    if bank_a and bank_b:
        data_a = get_latest_data(bank_a)
        data_b = get_latest_data(bank_b)
        
        if data_a is not None and data_b is not None:
            st.markdown("---")
            
            # Side by side metrics
            col1, col2 = st.columns(2)
            
            score_col = 'fragility_score_v2' if 'fragility_score_v2' in data['fragility'].columns else 'fragility_score'
            
            with col1:
                failed_a = data_a.get('failed', False)
                tag_a = '<span class="failed-tag">FAILED</span>' if failed_a else '<span class="stable-tag">Stable</span>'
                st.markdown(f"### {bank_a} {tag_a}", unsafe_allow_html=True)
                st.metric("Fragility Score", f"{data_a.get(score_col, 0):.4f}" if pd.notna(data_a.get(score_col)) else "N/A")
                st.metric("ROA", f"{data_a.get('ROA', 0):.2f}%" if pd.notna(data_a.get('ROA')) else "N/A")
                st.metric("Liquidity Ratio", f"{data_a.get('liquidity_ratio', 0):.4f}" if pd.notna(data_a.get('liquidity_ratio')) else "N/A")
                st.metric("Uninsured Deposits", f"{data_a.get('uninsured_deposit_ratio', 0):.4f}" if pd.notna(data_a.get('uninsured_deposit_ratio')) else "N/A")
            
            with col2:
                failed_b = data_b.get('failed', False)
                tag_b = '<span class="failed-tag">FAILED</span>' if failed_b else '<span class="stable-tag">Stable</span>'
                st.markdown(f"### {bank_b} {tag_b}", unsafe_allow_html=True)
                st.metric("Fragility Score", f"{data_b.get(score_col, 0):.4f}" if pd.notna(data_b.get(score_col)) else "N/A")
                st.metric("ROA", f"{data_b.get('ROA', 0):.2f}%" if pd.notna(data_b.get('ROA')) else "N/A")
                st.metric("Liquidity Ratio", f"{data_b.get('liquidity_ratio', 0):.4f}" if pd.notna(data_b.get('liquidity_ratio')) else "N/A")
                st.metric("Uninsured Deposits", f"{data_b.get('uninsured_deposit_ratio', 0):.4f}" if pd.notna(data_b.get('uninsured_deposit_ratio')) else "N/A")
            
            # Comparison chart
            st.markdown("---")
            st.markdown("#### Fragility Score Timeline Comparison")
            
            ts_a = get_bank_timeseries(bank_a)
            ts_b = get_bank_timeseries(bank_b)
            
            if not ts_a.empty and not ts_b.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_a['year'], y=ts_a[score_col],
                    name=bank_a, mode='lines+markers',
                    line=dict(color='#ef4444' if failed_a else '#3b82f6', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=ts_b['year'], y=ts_b[score_col],
                    name=bank_b, mode='lines+markers',
                    line=dict(color='#ef4444' if failed_b else '#22c55e', width=3)
                ))
                fig.update_layout(
                    height=400,
                    yaxis_title="Fragility Score",
                    xaxis_title="Year",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart comparison
            st.markdown("#### Component Comparison")
            components = ['liquidity_risk', 'leverage_risk', 'coverage_risk', 'deposit_risk', 'textual_stress']
            comp_labels = [c.replace('_', ' ').title() for c in components]
            
            vals_a = [data_a.get(c, 0.5) if pd.notna(data_a.get(c)) else 0.5 for c in components]
            vals_b = [data_b.get(c, 0.5) if pd.notna(data_b.get(c)) else 0.5 for c in components]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals_a + [vals_a[0]], theta=comp_labels + [comp_labels[0]],
                fill='toself', name=bank_a, fillcolor='rgba(59,130,246,0.15)',
                line=dict(color='#3b82f6')
            ))
            fig.add_trace(go.Scatterpolar(
                r=vals_b + [vals_b[0]], theta=comp_labels + [comp_labels[0]],
                fill='toself', name=bank_b, fillcolor='rgba(239,68,68,0.15)',
                line=dict(color='#ef4444')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# PAGE 4: AI ASSESSMENT
# ══════════════════════════════════════════
elif page == "🤖 AI Assessment":
    st.markdown("# 🤖 AI-Powered Risk Assessment")
    st.markdown("*Generate a live fragility assessment using Groq/Llama*")
    st.markdown("---")
    
    # API key input
    api_key = st.text_input("Groq API Key", type="password", 
                            help="Get a free key at https://console.groq.com")
    
    banks = get_bank_list()
    selected_bank = st.selectbox("Select a Bank to Assess", banks)
    
    if st.button("🔍 Generate Assessment", type="primary", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Groq API key.")
        else:
            latest = get_latest_data(selected_bank)
            
            if latest is None:
                st.error("No data available for this bank.")
            else:
                with st.status(f"Analyzing {selected_bank}...", expanded=True) as status:
                    try:
                        status.write("📊 Loading financial ratios...")
                        from groq import Groq
                        client = Groq(api_key=api_key)
                        
                        # Build context
                        def safe_val(val, fmt=".4f"):
                            if pd.isna(val): return "N/A"
                            try: return f"{float(val):{fmt}}"
                            except: return str(val)
                        
                        context = f"""BANK: {selected_bank}
YEAR: {safe_val(latest.get('year', 'N/A'), '.0f')}

FINANCIAL RATIOS:
  Liquidity Ratio:       {safe_val(latest.get('liquidity_ratio'))}
  Debt-to-Equity:        {safe_val(latest.get('debt_to_equity'), '.2f')}
  Interest Coverage:     {safe_val(latest.get('interest_coverage'), '.2f')}
  Loan-to-Deposit:       {safe_val(latest.get('loan_to_deposit'))}
  Uninsured Deposit %:   {safe_val(latest.get('uninsured_deposit_ratio'))}
  Core Deposit Ratio:    {safe_val(latest.get('core_deposit_ratio'))}
  ROA:                   {safe_val(latest.get('ROA'), '.2f')}%
  ROE:                   {safe_val(latest.get('ROE'), '.2f')}%

TOTAL ASSETS: ${safe_val(latest.get('ASSET'), ',.0f')} thousand"""

                        prompt = f"""You are a senior financial risk analyst. Analyze this bank and provide a structured fragility assessment.

{context}

Provide:
1. Overall fragility rating (1-10, where 10 = most fragile)
2. Top 3 risk factors based on the ratios
3. Liquidity assessment
4. One-paragraph reasoning

Be specific and cite the actual ratio values."""

                        status.write("🤖 Querying Groq/Llama 3.3-70B...")
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=1500
                        )
                        
                        result = response.choices[0].message.content
                        status.write("✅ Assessment complete")
                        status.update(label="Assessment ready", state="complete")
                        
                        st.markdown("---")
                        st.markdown(f"### Assessment: {selected_bank}")
                        st.markdown(result)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    if not api_key:
        st.info("💡 Enter your Groq API key above to generate live AI assessments. "
                "Get a free key at [console.groq.com](https://console.groq.com)")

# ══════════════════════════════════════════
# PAGE 5: DATA EXPLORER
# ══════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.markdown("# 📋 Data Explorer")
    st.markdown("*Browse the raw data behind the fragility scores*")
    st.markdown("---")
    
    dataset = st.selectbox("Select Dataset", [
        "Fragility Scores",
        "LLM Scores",
        "RAG Assessments",
        "Stock Drawdowns",
        "Textual Stress Scores",
        "Synthetic Bank Profiles",
    ])
    
    dataset_map = {
        "Fragility Scores": 'fragility',
        "LLM Scores": 'llm_scores',
        "RAG Assessments": 'enhanced_rag',
        "Stock Drawdowns": 'drawdowns',
        "Textual Stress Scores": 'textual',
        "Synthetic Bank Profiles": 'synthetic',
    }
    
    key = dataset_map[dataset]
    if key in data and not data[key].empty:
        df = data[key]
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df)}")
        with col2:
            st.metric("Columns", f"{len(df.columns)}")
        with col3:
            if 'bank_name' in df.columns:
                st.metric("Banks", f"{int(df['bank_name'].nunique())}")
        
        st.markdown("---")
        
        # Filter by bank if applicable
        if 'bank_name' in df.columns:
            banks = ["All"] + sorted(df['bank_name'].unique())
            filter_bank = st.selectbox("Filter by Bank", banks)
            if filter_bank != "All":
                df = df[df['bank_name'] == filter_bank]
        
        st.dataframe(df, use_container_width=True, height=500)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{key}_data.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No data available for {dataset}.")

# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 13px;'>"
    "🏦 AI Financial Fragility Detector — Nikhil Patwal — Northeastern University — April 2026"
    "</div>",
    unsafe_allow_html=True
)
