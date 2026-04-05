import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

def create_valuation_plot(ticker_data):
    """Plot 1: Valuation Anchor (Current Price vs Intrinsic Value)"""
    # Simplified calculation for visualization
    current = ticker_data.get('currentPrice', 0)
    intrinsic = ticker_data.get('intrinsicValue', current * 0.9)
    
    fig = go.Figure(go.Bar(
        x=['Current Market Price', 'Estimated Fair Value'],
        y=[current, intrinsic],
        marker_color=['#636EFA', '#00CC96']
    ))
    fig.update_layout(title="Is the Stock on Sale?", template="plotly_dark", height=300)
    return fig

def create_momentum_plot(df):
    """Plot 2: Momentum Speedometer (RSI)"""
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    fig = px.line(df, y='RSI', title="Market Temperature (RSI)")
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overheated")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Bargain Zone")
    fig.update_layout(template="plotly_dark", height=300)
    return fig

def create_safety_plot(ticker_data):
    """Plot 3: Safety Radar (Debt vs Equity)"""
    # Visualizing the balance sheet health
    labels = ['Equity (Ownership)', 'Debt (Borrowed)']
    values = [ticker_data.get('totalEquity', 100), ticker_data.get('totalDebt', 50)]
    
    fig = px.pie(names=labels, values=values, hole=0.4, title="Foundation Strength")
    fig.update_traces(marker=dict(colors=['#00CC96', '#EF553B']))
    fig.update_layout(template="plotly_dark", height=300)
    return fig

def create_simulation_plot(df, amount):
    """Plot 4: Growth Simulation (Historical $ Amount)"""
    initial_price = df['Close'].iloc[0]
    df['InvestmentValue'] = (df['Close'] / initial_price) * amount
    
    fig = px.area(df, y='InvestmentValue', title=f"If you had invested ${amount}...")
    fig.update_traces(line_color='#00ffcc')
    fig.update_layout(template="plotly_dark", height=300)
    return fig