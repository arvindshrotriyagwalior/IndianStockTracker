import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import base64
from PIL import Image
from io import BytesIO, StringIO

# Import utility modules
from utils.data_fetcher import get_stock_data, validate_stock_symbol, get_common_indian_stocks, get_financial_health_data
from utils.technical_indicators import add_technical_indicators, get_indicator_explanations, generate_trading_signals, get_overall_recommendation
from utils.ai_models import get_ai_recommendation
from utils.advanced_patterns import (get_detected_patterns, add_patterns_to_chart, fibonacci_retracement_levels, 
                                    add_fibonacci_to_chart, calculate_volume_profile, add_volume_profile_to_chart,
                                    get_pattern_signals, get_pattern_explanations)
from utils.news_sentiment import get_news_sentiment

# Helper functions for export/download functionality
def format_value(value):
    """Format a value based on its type for display in HTML reports"""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)
def get_table_download_link(df, filename, text):
    """Generate a download link for a DataFrame as CSV file"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

def get_json_download_link(data, filename, text):
    """Generate a download link for a dictionary as JSON file"""
    json_str = json.dumps(data, default=str, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">{text}</a>'
    return href

def get_html_download_link(fig, filename, text):
    """Generate a download link for a plotly figure as HTML"""
    buffer = StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    b64 = base64.b64encode(html_bytes).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">{text}</a>'
    return href

def get_pdf_report(symbol, hist, hist_with_indicators, info, signals, ai_recommendation, timeframe):
    """Generate a PDF report with stock analysis"""
    report_html = f"""
    <html>
    <head>
        <title>Stock Analysis Report - {symbol}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .metrics {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .metric {{ padding: 10px; background-color: #f8f9fa; border-radius: 5px; width: 30%; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ text-align: center; font-size: 12px; margin-top: 50px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Stock Analysis Report</h1>
            <h2>{symbol} - {datetime.now().strftime('%Y-%m-%d')}</h2>
            <p>Timeframe: {timeframe}</p>
        </div>

        <div class="section">
            <h2>Price Information</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Current Price</h3>
                    <p>₹{hist['Close'].iloc[-1]:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Day Range</h3>
                    <p>₹{hist['Low'].iloc[-1]:.2f} - ₹{hist['High'].iloc[-1]:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Volume</h3>
                    <p>{hist['Volume'].iloc[-1]:,.0f}</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Technical Indicators</h2>
            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>SMA (20)</td>
                    <td>₹{hist_with_indicators['SMA_20'].iloc[-1]:.2f}</td>
                </tr>
                <tr>
                    <td>EMA (20)</td>
                    <td>₹{hist_with_indicators['EMA_20'].iloc[-1]:.2f}</td>
                </tr>
                <tr>
                    <td>RSI</td>
                    <td>{hist_with_indicators['RSI'].iloc[-1]:.2f}</td>
                </tr>
                <tr>
                    <td>MACD</td>
                    <td>{hist_with_indicators['MACD'].iloc[-1]:.2f}</td>
                </tr>
                <tr>
                    <td>Signal Line</td>
                    <td>{hist_with_indicators['Signal_Line'].iloc[-1]:.2f}</td>
                </tr>
            </table>
        </div>
    """

    # Add trading signals section if available
    if signals:
        rec, conf, exp = get_overall_recommendation(signals)
        report_html += f"""
        <div class="section">
            <h2>Trading Signals</h2>
            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Signal</th>
                    <th>Strength</th>
                    <th>Value</th>
                </tr>
        """

        for indicator, data in signals.items():
            report_html += f"""
                <tr>
                    <td>{indicator}</td>
                    <td>{data['signal']}</td>
                    <td>{data['strength']}</td>
                    <td>{format_value(data['value'])}</td>
                </tr>
            """

        report_html += f"""
            </table>
            <h3>Overall Recommendation: {rec}</h3>
            <p><strong>Confidence:</strong> {conf}</p>
            <p><strong>Explanation:</strong> {exp}</p>
        </div>
        """

    # Add AI recommendations section if ai_recommendation contains the required keys
    if ai_recommendation and 'prediction' in ai_recommendation and 'confidence' in ai_recommendation and 'accuracy' in ai_recommendation:
        report_html += f"""
        <div class="section">
            <h2>AI Trading Recommendation</h2>
            <p><strong>Prediction:</strong> {ai_recommendation['prediction']}</p>
            <p><strong>Confidence:</strong> {ai_recommendation['confidence']*100:.1f}%</p>
            <p><strong>Model Accuracy:</strong> {ai_recommendation['accuracy']*100:.1f}%</p>
        </div>
        """
    else:
        report_html += """
        <div class="section">
            <h2>AI Trading Recommendation</h2>
            <p>AI model data is not available for this analysis.</p>
        </div>
        """

    # Add footer
    report_html += f"""
        <div class="footer">
            <p>This report was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Indian Stock Market Analysis & AI Recommendations Tool | Data sourced from Yahoo Finance</p>
            <p>DISCLAIMER: This is not financial advice. Always do your own research before investing.</p>
        </div>
    </body>
    </html>
    """

    # Convert HTML to PDF (placeholder - would need a PDF library)
    return report_html

# Set page configuration
st.set_page_config(
    page_title="Indian Stock Market Analysis & AI Recommendations",
    page_icon="📈",
    layout="wide"
)

# Define a function to detect mobile devices based on viewport width
def is_mobile():
    """Detect if user is on a mobile device based on viewport width"""
    # Get viewport width using custom JavaScript
    viewport_width = st.session_state.get('viewport_width', 1200)  # Default to desktop width
    return viewport_width < 768  # Standard mobile breakpoint

# Initialize session state for viewport width if not already set
if 'viewport_width' not in st.session_state:
    st.session_state.viewport_width = 1200  # Default desktop width

# Add JavaScript to detect viewport width and store it in session state
st.markdown("""
<script>
    // Function to update viewport width in session state
    function updateViewportWidth() {
        const width = window.innerWidth;
        const viewportData = {
            width: width
        };
        
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: viewportData
        }, "*");
    }
    
    // Run on page load and on resize
    updateViewportWidth();
    window.addEventListener('resize', updateViewportWidth);
</script>
""", unsafe_allow_html=True)

# Add responsive CSS styles
st.markdown("""
<style>
    /* Base styles */
    .stApp {
        max-width: 100%;
        padding: 0;
    }
    
    /* Mobile-specific styles */
    @media screen and (max-width: 768px) {
        .reportview-container .main .block-container {
            padding: 0.5rem !important;
        }
        
        /* Make tabs more touch-friendly */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
        }
        .stTabs [data-baseweb="tab"] {
            padding-top: 5px;
            padding-bottom: 5px;
            padding-left: 5px;
            padding-right: 5px;
            font-size: 0.7rem;
        }
        
        /* Adjust chart sizes */
        .chart-container {
            height: 300px !important;
        }
        
        /* Make text more readable on small screens */
        p, li, .table-container {
            font-size: 0.9rem !important;
        }
        
        /* Adjust metric display */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        
        /* Better table display on mobile */
        table {
            font-size: 0.8rem !important;
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
    }
</style>
""", unsafe_allow_html=True)

# Display title and description
st.title("Indian Stock Market Analysis & AI Recommendations")
st.markdown("""
This application provides comprehensive analysis of Indian stocks with technical indicators,
fundamental metrics, and AI-powered trading recommendations.
""")

# Sidebar configuration
st.sidebar.header("Stock Selection")

# Add example symbols for user reference
example_symbols = get_common_indian_stocks()
st.sidebar.markdown("#### Example Symbols")
st.sidebar.write(", ".join(example_symbols[:5]))
st.sidebar.write(", ".join(example_symbols[5:10]))

# User inputs
symbol = st.sidebar.text_input("Enter NSE/BSE stock symbol (e.g., RELIANCE.NS, TCS.BO)", value="RELIANCE.NS")
timeframe = st.sidebar.selectbox(
    "Select timeframe",
    ["1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y"],
    index=2
)

# Add explanation for symbol format
st.sidebar.info("""
📌 **Note**: Add '.NS' for NSE stocks and '.BO' for BSE stocks.
Example: RELIANCE.NS or TATASTEEL.BO
""")

# Add a "Load Data" button to trigger analysis
load_data = st.sidebar.button("Analyse")

# Advanced options in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Advanced Options")
show_explanations = st.sidebar.checkbox("Show indicator explanations", value=True)
show_ai_details = st.sidebar.checkbox("Show AI model details", value=False)

# Display indicator explanations
if show_explanations:
    with st.sidebar.expander("Technical Indicator Explanations"):
        explanations = get_indicator_explanations()
        for indicator, explanation in explanations.items():
            st.sidebar.markdown(f"**{indicator}**: {explanation}")

# Main content
if symbol and load_data:
    # Validate symbol
    if not validate_stock_symbol(symbol):
        st.error(f"Invalid symbol: '{symbol}'. Please check if you've entered a valid NSE/BSE symbol with the correct suffix (.NS or .BO).")
    else:
        # Display loading message
        with st.spinner(f"Loading data for {symbol}..."):
            # Fetch data
            stock, hist, info = get_stock_data(symbol, timeframe)

            if hist.empty:
                st.error(f"No data available for {symbol}. Please check if the symbol is correct and try again.")
            else:
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Price Charts", "Technical Analysis", "Fundamental Analysis", "AI Recommendations", "News & Sentiment", "Alternative Data"])

                # Add technical indicators
                hist_with_indicators = add_technical_indicators(hist)

                with tab1:
                    st.subheader(f"Stock Price Chart: {symbol}")

                    # Basic info - Adjust columns based on device
                    # Use fewer columns on mobile for better readability
                    if is_mobile():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Current Price", 
                                f"₹{hist['Close'].iloc[-1]:.2f}", 
                                f"{((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100:.2f}%" if len(hist) > 1 else None
                            )
                        with col2:
                            st.metric(
                                "Volume", 
                                f"{hist['Volume'].iloc[-1]:,.0f}" if 'Volume' in hist.columns else "N/A"
                            )
                        # Add day range in a new row for mobile
                        st.metric(
                            "Day Range", 
                            f"₹{hist['Low'].iloc[-1]:.2f} - ₹{hist['High'].iloc[-1]:.2f}"
                        )
                    else:
                        # Desktop layout with 3 columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Current Price", 
                                f"₹{hist['Close'].iloc[-1]:.2f}", 
                                f"{((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100:.2f}%" if len(hist) > 1 else None
                            )
                        with col2:
                            st.metric(
                                "Volume", 
                                f"{hist['Volume'].iloc[-1]:,.0f}" if 'Volume' in hist.columns else "N/A"
                            )
                        with col3:
                            st.metric(
                                "Day Range", 
                                f"₹{hist['Low'].iloc[-1]:.2f} - ₹{hist['High'].iloc[-1]:.2f}"
                            )

                    # Chart type selector
                    chart_type = st.radio("Select chart type:", ["Candlestick", "Line"], horizontal=True)

                    # Create chart
                    fig = go.Figure()

                    if chart_type == "Candlestick":
                        fig.add_trace(go.Candlestick(
                            x=hist.index,
                            open=hist['Open'],
                            high=hist['High'],
                            low=hist['Low'],
                            close=hist['Close'],
                            name="Price"
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name="Close Price",
                            line=dict(color='blue', width=2)
                        ))

                    # Update layout with improved styling and responsive height
                    chart_height = 350 if is_mobile() else 600
                    fig.update_layout(
                        title=f"{symbol} - {timeframe} Chart",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=chart_height,
                        xaxis_rangeslider_visible=not is_mobile(),  # Hide rangeslider on mobile to save space
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='rgba(255,255,255,1)',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="#505050"
                        ),
                        margin=dict(t=40, b=40, l=30, r=20) if is_mobile() else dict(t=50, b=50, l=50, r=50),
                        hovermode="x unified"
                    )
                    # Add better grid and axes
                    fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )
                    fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )

                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Add comparison with market index (NIFTY 50 or SENSEX)
                    st.subheader("Comparison with Market Index")

                    # Determine which index to use based on the exchange
                    index_symbol = "^NSEI" if ".NS" in symbol else "^BSESN"
                    index_name = "NIFTY 50" if ".NS" in symbol else "SENSEX"

                    _, index_hist, _ = get_stock_data(index_symbol, timeframe)

                    if not index_hist.empty:
                        # Normalize prices for comparison
                        comparison_df = pd.DataFrame({
                            'Stock': hist['Close'] / hist['Close'].iloc[0] * 100,
                            'Index': index_hist['Close'] / index_hist['Close'].iloc[0] * 100
                        })

                        # Create comparison chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=comparison_df.index,
                            y=comparison_df['Stock'],
                            mode='lines',
                            name=symbol
                        ))
                        fig.add_trace(go.Scatter(
                            x=comparison_df.index,
                            y=comparison_df['Index'],
                            mode='lines',
                            name=index_name
                        ))

                        # Adjust chart height and layout for mobile
                        comparison_height = 250 if is_mobile() else 400
                        fig.update_layout(
                            title=f"Normalized Price Comparison ({symbol} vs {index_name})",
                            xaxis_title="Date",
                            yaxis_title="Normalized Price (First day = 100)",
                            height=comparison_height,
                            plot_bgcolor='rgba(240,240,240,0.8)',
                            paper_bgcolor='rgba(255,255,255,1)',
                            font=dict(
                                family="Arial, sans-serif",
                                size=12,
                                color="#505050"
                            ),
                            margin=dict(t=30, b=30, l=30, r=20) if is_mobile() else dict(t=50, b=50, l=50, r=50),
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        # Add better grid and axes
                        fig.update_xaxes(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor='rgba(220,220,220,0.8)',
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(0,0,0,0.5)'
                        )
                        fig.update_yaxes(
                            showgrid=True, 
                            gridwidth=1, 
                            gridcolor='rgba(220,220,220,0.8)',
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(0,0,0,0.5)'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate correlation
                        correlation = comparison_df['Stock'].corr(comparison_df['Index'])

                        st.info(f"Correlation with {index_name}: {correlation:.2f}")

                        if correlation > 0.8:
                            st.write("This stock is strongly correlated with the market index.")
                        elif correlation < 0.3:
                            st.write("This stock shows low correlation with the market index, suggesting it may move independently.")
                    else:
                        st.warning(f"Unable to load {index_name} data for comparison.")

                with tab2:
                    st.subheader("Technical Analysis")
                    
                    # Add tabs for different technical analysis sections
                    ta_tab1, ta_tab2, ta_tab3, ta_tab4 = st.tabs([
                        "Basic Indicators", 
                        "Advanced Patterns", 
                        "Fibonacci Analysis",
                        "Volume Profile"
                    ])

                    with ta_tab1:
                        # Moving Averages
                        st.markdown("### Moving Averages")

                        ma_fig = go.Figure()
                        ma_fig.add_trace(go.Scatter(
                            x=hist.index, y=hist['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ))
                        ma_fig.add_trace(go.Scatter(
                            x=hist_with_indicators.index,
                            y=hist_with_indicators['SMA_20'],
                            mode='lines',
                            name='SMA (20)',
                            line=dict(color='orange')
                        ))
                        ma_fig.add_trace(go.Scatter(
                            x=hist_with_indicators.index,
                            y=hist_with_indicators['EMA_20'],
                            mode='lines',
                            name='EMA (20)',
                            line=dict(color='green')
                        ))

                    # Adjust chart height for mobile
                    ma_chart_height = 300 if is_mobile() else 400
                    ma_fig.update_layout(
                        title="Price with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=ma_chart_height,
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='rgba(255,255,255,1)',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="#505050"
                        ),
                        margin=dict(t=30, b=30, l=30, r=20) if is_mobile() else dict(t=50, b=50, l=50, r=50),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    ma_fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )
                    ma_fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )

                    st.plotly_chart(ma_fig, use_container_width=True)

                    # MACD
                    st.markdown("### MACD")

                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ))
                    macd_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['Signal_Line'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='red')
                    ))

                    # Add MACD Histogram
                    colors = ['green' if x > 0 else 'red' for x in hist_with_indicators['MACD_Histogram'].dropna()]
                    macd_fig.add_trace(go.Bar(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['MACD_Histogram'],
                        name='Histogram',
                        marker_color=colors
                    ))

                    # Adjust chart height for mobile
                    macd_chart_height = 300 if is_mobile() else 400
                    macd_fig.update_layout(
                        title="MACD Indicator",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        height=macd_chart_height,
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='rgba(255,255,255,1)',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="#505050"
                        ),
                        margin=dict(t=30, b=30, l=30, r=20) if is_mobile() else dict(t=50, b=50, l=50, r=50),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    macd_fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )
                    macd_fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )

                    st.plotly_chart(macd_fig, use_container_width=True)

                    # RSI with Momentum Trend
                    st.markdown("### RSI with Momentum Trend")

                    # Calculate RSI momentum
                    rsi_values = hist_with_indicators['RSI'].dropna()
                    rsi_momentum = rsi_values.diff(3)  # 3-period momentum

                    # Generate RSI-based signals
                    rsi_signals = []
                    signal_colors = []
                    signal_texts = []
                    signal_points_x = []
                    signal_points_y = []

                    for i in range(len(rsi_values)):
                        if i >= 3:  # Need at least 3 periods for momentum
                            current_rsi = rsi_values.iloc[i]
                            current_momentum = rsi_momentum.iloc[i]

                            if current_rsi < 30 and current_momentum > 0:
                                rsi_signals.append("BUY")
                                signal_colors.append("green")
                                signal_texts.append("Buy")
                                signal_points_x.append(rsi_values.index[i])
                                signal_points_y.append(current_rsi)
                            elif current_rsi > 70 and current_momentum < 0:
                                rsi_signals.append("SELL")
                                signal_colors.append("red")
                                signal_texts.append("Sell")
                                signal_points_x.append(rsi_values.index[i])
                                signal_points_y.append(current_rsi)

                    rsi_fig = go.Figure()

                    # Plot RSI line
                    rsi_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ))

                    # Add signal points
                    rsi_fig.add_trace(go.Scatter(
                        x=signal_points_x,
                        y=signal_points_y,
                        mode='markers+text',
                        marker=dict(color=signal_colors, size=10),
                        text=signal_texts,
                        textposition="top center",
                        name='Signals'
                    ))

                    # Add RSI levels
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")

                    # Add RSI momentum indicator
                    rsi_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=rsi_momentum,
                        mode='lines',
                        name='RSI Momentum',
                        line=dict(color='orange', dash='dot'),
                        yaxis='y2'
                    ))

                    rsi_fig.update_layout(
                        title="RSI with Momentum Trend",
                        xaxis_title="Date",
                        yaxis_title="RSI Value",
                        yaxis2=dict(
                            title="Momentum",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        ),
                        height=400,
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='rgba(255,255,255,1)',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="#505050"
                        ),
                        margin=dict(t=50, b=50, l=50, r=50),
                        hovermode="x unified"
                    )
                    rsi_fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )
                    rsi_fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )

                    st.plotly_chart(rsi_fig, use_container_width=True)

                    # Bollinger Bands
                    st.markdown("### Bollinger Bands")

                    bb_fig = go.Figure()
                    bb_fig.add_trace(go.Scatter(
                        x=hist.index, y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ))
                    bb_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['Upper_BB'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='red', dash='dash')
                    ))
                    bb_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['BB_Middle'],
                        mode='lines',
                        name='Middle Band (SMA)',
                        line=dict(color='green', dash='dash')
                    ))
                    bb_fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['Lower_BB'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='red', dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.2)'
                    ))

                    bb_fig.update_layout(
                        title="Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=400,
                        plot_bgcolor='rgba(240,240,240,0.8)',
                        paper_bgcolor='rgba(255,255,255,1)',
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color="#505050"
                        ),
                        margin=dict(t=50, b=50, l=50, r=50),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    bb_fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )
                    bb_fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(220,220,220,0.8)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0,0,0,0.5)'
                    )

                    st.plotly_chart(bb_fig, use_container_width=True)

                    # Trading signals
                    st.markdown("### Trading Signals")
                    signals = generate_trading_signals(hist_with_indicators)

                    # Display signals in a table
                    if signals:
                        signal_df = pd.DataFrame.from_dict(
                            {k: [v["signal"], v["strength"], v["value"]] for k, v in signals.items()},
                            orient='index',
                            columns=['Signal', 'Strength', 'Value']
                        )

                        # Add color highlighting
                        def highlight_signals(val):
                            if val == 'BUY':
                                return 'background-color: green; color: white'
                            elif val == 'SELL':
                                return 'background-color: red; color: white'
                            elif val == 'Strong':
                                return 'font-weight: bold'
                            return ''

                        st.dataframe(signal_df.style.map(highlight_signals, subset=['Signal', 'Strength']))

                        # Display overall recommendation
                        rec, conf, exp = get_overall_recommendation(signals)

                        rec_col1, rec_col2 = st.columns(2)
                        with rec_col1:
                            st.markdown(f"### Overall Recommendation: **{rec}**")
                            st.markdown(f"**Confidence**: {conf}")
                        with rec_col2:
                            if rec == "BUY":
                                st.success(f"Technical analysis suggests a **BUY** signal for {symbol}")
                            elif rec == "SELL":
                                st.error(f"Technical analysis suggests a **SELL** signal for {symbol}")
                            else:
                                st.info(f"Technical analysis is **NEUTRAL** for {symbol}")

                        st.markdown(f"**Explanation**: {exp}")
                    else:
                        st.warning("Insufficient data to generate trading signals.")

                    # Detailed Trend Explanations
                    st.markdown("### Detailed Trend Analysis")

                    # Get latest data point
                    latest = hist_with_indicators.iloc[-1]
                    data = hist_with_indicators #added to handle data variable

                    # Get indicator explanations with trends
                    explanations = get_indicator_explanations()

                    # Create expandable sections for each indicator
                    with st.expander("Moving Averages (SMA & EMA) Analysis"):
                        st.markdown(f"**{explanations['SMA_20']['description']}**")
                        for trend, desc in explanations['SMA_20']['trends'].items():
                            st.markdown(f"- **{trend}**: {desc}")
                        st.markdown("\n**Current Trend**: " + (
                            "Bullish" if latest['Close'] > latest['SMA_20'] and latest['SMA_20'] > data['SMA_20'].iloc[-2]
                            else "Bearish" if latest['Close'] < latest['SMA_20'] and latest['SMA_20'] < data['SMA_20'].iloc[-2]
                            else "Neutral"
                        ))

                    with st.expander("MACD Analysis"):
                        st.markdown(f"**{explanations['MACD']['description']}**")
                        for trend, desc in explanations['MACD']['trends'].items():
                            st.markdown(f"- **{trend}**: {desc}")
                        st.markdown("\n**Current Trend**: " + (
                            "Bullish" if latest['MACD'] > latest['Signal_Line']
                            else "Bearish" if latest['MACD'] < latest['Signal_Line']
                            else "Neutral"
                        ))

                    with st.expander("RSI Analysis"):
                        st.markdown(f"**{explanations['RSI']['description']}**")
                        for trend, desc in explanations['RSI']['trends'].items():
                            st.markdown(f"- **{trend}**: {desc}")
                        rsi_val = latest['RSI']
                        st.markdown("\n**Current Reading**: " + (
                            "Overbought" if rsi_val > 70                            else "Oversold" if rsi_val < 30
                            else "Neutral"
                        ) + f" (RSI: {rsi_val:.2f})")

                    with st.expander("Bollinger Bands Analysis"):
                        st.markdown(f"**{explanations['Bollinger_Bands']['description']}**")
                        for trend, desc in explanations['Bollinger_Bands']['trends'].items():
                            st.markdown(f"- **{trend}**: {desc}")
                        bb_width = (latest['Upper_BB'] - latest['Lower_BB']) / latest['BB_Middle']
                        st.markdown("\n**Current Pattern**: " + (
                            "Squeeze" if bb_width < 0.1
                            else "Expansion" if bb_width > 0.2
                            else "Normal"
                        ))

                    with st.expander("Volume Analysis (OBV)"):
                        st.markdown(f"**{explanations['OBV']['description']}**")
                        for trend, desc in explanations['OBV']['trends'].items():
                            st.markdown(f"- **{trend}**: {desc}")
                        # Check if we have enough data points for OBV trend
                        if len(data) > 1:
                            obv_trend = "Bullish" if latest['OBV'] > data['OBV'].iloc[-2] else "Bearish"
                        else:
                            obv_trend = "Neutral"
                        st.markdown(f"\n**Current Trend**: {obv_trend}")

                    with st.expander("ATR (Volatility) Analysis"):
                        st.markdown(f"**{explanations['ATR']['description']}**")
                        for trend, desc in explanations['ATR']['trends'].items():
                            st.markdown(f"- **{trend}**: {desc}")
                        # Check if we have enough data for 5-period comparison
                        if len(data) >= 5:
                            atr_change = (latest['ATR'] / data['ATR'].iloc[-5] - 1) * 100
                            st.markdown(f"\n**Volatility Change**: {atr_change:.1f}% over last 5 periods")
                        elif len(data) > 1:
                            # Use the earliest available data point for comparison if we have at least 2 data points
                            atr_change = (latest['ATR'] / data['ATR'].iloc[0] - 1) * 100
                            st.markdown(f"\n**Volatility Change**: {atr_change:.1f}% since start of period")
                        else:
                            # Not enough data points for comparison
                            st.markdown("\n**Volatility Change**: Not enough data for comparison")
                    
                    # Advanced Patterns Tab
                    with ta_tab2:
                        st.markdown("### Advanced Chart Pattern Recognition")
                        
                        # Detect patterns
                        patterns = get_detected_patterns(hist)
                        
                        # Create candlestick chart with patterns
                        pattern_fig = go.Figure()
                        
                        # Add candlestick
                        pattern_fig.add_trace(go.Candlestick(
                            x=hist.index,
                            open=hist['Open'],
                            high=hist['High'],
                            low=hist['Low'],
                            close=hist['Close'],
                            name="Price"
                        ))
                        
                        # Add detected patterns to the chart
                        pattern_fig = add_patterns_to_chart(pattern_fig, patterns, hist)
                        
                        # Update layout
                        pattern_fig.update_layout(
                            title="Chart Patterns Detection",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            height=500,
                            plot_bgcolor='rgba(240,240,240,0.8)',
                            paper_bgcolor='rgba(255,255,255,1)',
                            hovermode="x unified"
                        )
                        
                        # Display the chart
                        st.plotly_chart(pattern_fig, use_container_width=True)
                        
                        # Display detected patterns
                        if (patterns['head_and_shoulders'] or patterns['inverse_head_and_shoulders'] or 
                            patterns['cup_and_handle']):
                            st.subheader("Detected Patterns")
                            
                            # Head and Shoulders patterns
                            if patterns['head_and_shoulders']:
                                for i, pattern in enumerate(patterns['head_and_shoulders']):
                                    with st.expander(f"Head and Shoulders (Bearish) #{i+1}"):
                                        st.markdown(f"**Type:** {pattern['type']}")
                                        st.markdown(f"**Direction:** {pattern['direction']}")
                                        st.markdown(f"**Strength:** {pattern['strength']}")
                                        st.markdown(f"**Target Price:** ₹{pattern['target']:.2f}")
                                        st.markdown(f"**Formation:**")
                                        st.markdown(f"- **Left Shoulder:** {pattern['left_shoulder']}")
                                        st.markdown(f"- **Head:** {pattern['head']}")
                                        st.markdown(f"- **Right Shoulder:** {pattern['right_shoulder']}")
                                        st.markdown(f"- **Neckline:** ₹{pattern['neckline']:.2f}")
                            
                            # Inverse Head and Shoulders patterns
                            if patterns['inverse_head_and_shoulders']:
                                for i, pattern in enumerate(patterns['inverse_head_and_shoulders']):
                                    with st.expander(f"Inverse Head and Shoulders (Bullish) #{i+1}"):
                                        st.markdown(f"**Type:** {pattern['type']}")
                                        st.markdown(f"**Direction:** {pattern['direction']}")
                                        st.markdown(f"**Strength:** {pattern['strength']}")
                                        st.markdown(f"**Target Price:** ₹{pattern['target']:.2f}")
                                        st.markdown(f"**Formation:**")
                                        st.markdown(f"- **Left Shoulder:** {pattern['left_shoulder']}")
                                        st.markdown(f"- **Head:** {pattern['head']}")
                                        st.markdown(f"- **Right Shoulder:** {pattern['right_shoulder']}")
                                        st.markdown(f"- **Neckline:** ₹{pattern['neckline']:.2f}")
                            
                            # Cup and Handle patterns
                            if patterns['cup_and_handle']:
                                for i, pattern in enumerate(patterns['cup_and_handle']):
                                    with st.expander(f"Cup and Handle (Bullish) #{i+1}"):
                                        st.markdown(f"**Type:** {pattern['type']}")
                                        st.markdown(f"**Direction:** {pattern['direction']}")
                                        st.markdown(f"**Strength:** {pattern['strength']}")
                                        st.markdown(f"**Target Price:** ₹{pattern['target']:.2f}")
                                        st.markdown(f"**Formation:**")
                                        st.markdown(f"- **Left High:** {pattern['left_high']}")
                                        st.markdown(f"- **Cup Bottom:** {pattern['cup_bottom']}")
                                        st.markdown(f"- **Right High:** {pattern['right_high']}")
                                        st.markdown(f"- **Handle Low:** {pattern['handle_low']}")
                        else:
                            st.info("No significant chart patterns detected in the current timeframe. Try a different timeframe or check back later.")
                        
                        # Add pattern signals to overall signals
                        pattern_signals = get_pattern_signals(patterns)
                        if pattern_signals:
                            st.subheader("Trading Signals from Patterns")
                            pattern_signal_df = pd.DataFrame.from_dict(
                                {k: [v["signal"], v["strength"], v["value"]] for k, v in pattern_signals.items()},
                                orient='index',
                                columns=['Signal', 'Strength', 'Value']
                            )
                            
                            # Color coding for signals
                            def highlight_pattern_signals(val):
                                if val == 'BUY':
                                    return 'background-color: green; color: white'
                                elif val == 'SELL':
                                    return 'background-color: red; color: white'
                                elif val == 'Strong':
                                    return 'font-weight: bold'
                                return ''
                            
                            st.dataframe(pattern_signal_df.style.map(highlight_pattern_signals, subset=['Signal', 'Strength']))
                        
                        # Display pattern explanations
                        with st.expander("Pattern Explanations & Trading Strategies"):
                            pattern_explanations = get_pattern_explanations()
                            for pattern, explanation in pattern_explanations.items():
                                st.markdown(f"### {pattern.replace('_', ' ')}")
                                st.markdown(explanation['description'])
                                for trend, desc in explanation['trends'].items():
                                    st.markdown(f"**{trend}**: {desc}")
                                st.markdown("---")
                    
                    # Fibonacci Analysis Tab
                    with ta_tab3:
                        st.markdown("### Fibonacci Retracement Analysis")
                        
                        # Controls for Fibonacci analysis
                        fib_col1, fib_col2 = st.columns(2)
                        with fib_col1:
                            trend_type = st.radio("Select trend direction for Fibonacci analysis:",
                                                ["Uptrend", "Downtrend"])
                        with fib_col2:
                            st.markdown("#### What are Fibonacci Levels?")
                            st.markdown("""
                            Fibonacci retracement levels identify potential support/resistance levels where price 
                            trends may reverse. Key levels are 23.6%, 38.2%, 50%, 61.8%, and 78.6%.
                            """)
                        
                        # Calculate Fibonacci retracement levels
                        is_uptrend = (trend_type == "Uptrend")
                        fib_data = fibonacci_retracement_levels(hist, is_uptrend)
                        
                        if fib_data:
                            # Create chart with Fibonacci levels
                            fib_fig = go.Figure()
                            
                            # Add candlestick
                            fib_fig.add_trace(go.Candlestick(
                                x=hist.index,
                                open=hist['Open'],
                                high=hist['High'],
                                low=hist['Low'],
                                close=hist['Close'],
                                name="Price"
                            ))
                            
                            # Add Fibonacci levels to chart
                            fib_fig = add_fibonacci_to_chart(fib_fig, fib_data, hist)
                            
                            # Update layout
                            fib_fig.update_layout(
                                title="Fibonacci Retracement Levels",
                                xaxis_title="Date",
                                yaxis_title="Price (₹)",
                                height=500,
                                plot_bgcolor='rgba(240,240,240,0.8)',
                                paper_bgcolor='rgba(255,255,255,1)',
                                hovermode="x unified"
                            )
                            
                            # Display the chart
                            st.plotly_chart(fib_fig, use_container_width=True)
                            
                            # Display levels in a table
                            st.subheader("Fibonacci Retracement Levels")
                            st.markdown(f"**Trend Direction:** {fib_data['trend']}")
                            
                            # Create DataFrame for levels
                            fib_levels = {f"{float(level)*100:.1f}%": price 
                                         for level, price in fib_data['levels'].items()}
                            fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', 
                                                          columns=['Price Level'])
                            st.dataframe(fib_df)
                            
                            # Trading strategies based on Fibonacci
                            with st.expander("Fibonacci Trading Strategies"):
                                st.markdown("""
                                **Uptrend Retracement Strategy:**
                                1. Wait for a strong uptrend to establish
                                2. Look for price to retrace to key Fibonacci levels (38.2%, 50%, or 61.8%)
                                3. Watch for price action confirmation at these levels (candlestick patterns, RSI divergence)
                                4. Enter long positions when price bounces from the Fibonacci level
                                5. Set stop loss just below the retracement level
                                6. Target the previous high or extension levels (127.2%, 161.8%)
                                
                                **Downtrend Retracement Strategy:**
                                1. Wait for a strong downtrend to establish
                                2. Look for price to retrace to key Fibonacci levels (38.2%, 50%, or 61.8%)
                                3. Watch for price action confirmation at these levels
                                4. Enter short positions when price reverses from the Fibonacci level
                                5. Set stop loss just above the retracement level
                                6. Target the previous low or extension levels
                                """)
                        else:
                            st.warning("Unable to calculate Fibonacci levels. Please try a different timeframe or trend direction.")
                    
                    # Volume Profile Tab
                    with ta_tab4:
                        st.markdown("### Volume Profile Analysis")
                        
                        # Controls for Volume Profile
                        vol_col1, vol_col2 = st.columns(2)
                        with vol_col1:
                            price_bins = st.slider("Number of price levels:", min_value=5, max_value=20, value=10)
                        with vol_col2:
                            st.markdown("#### What is Volume Profile?")
                            st.markdown("""
                            Volume Profile shows the distribution of volume across price levels. 
                            It helps identify significant support/resistance zones based on trading activity.
                            The Point of Control (POC) is the price level with the highest trading volume.
                            """)
                        
                        # Calculate Volume Profile
                        volume_profile = calculate_volume_profile(hist, price_bins=price_bins)
                        
                        if volume_profile:
                            # Create chart with Volume Profile
                            vp_fig = go.Figure()
                            
                            # Add candlestick
                            vp_fig.add_trace(go.Candlestick(
                                x=hist.index,
                                open=hist['Open'],
                                high=hist['High'],
                                low=hist['Low'],
                                close=hist['Close'],
                                name="Price"
                            ))
                            
                            # Add Volume Profile to chart
                            vp_fig = add_volume_profile_to_chart(vp_fig, volume_profile, hist)
                            
                            # Update layout
                            vp_fig.update_layout(
                                title="Volume Profile Analysis",
                                xaxis_title="Date",
                                yaxis_title="Price (₹)",
                                height=500,
                                plot_bgcolor='rgba(240,240,240,0.8)',
                                paper_bgcolor='rgba(255,255,255,1)',
                                hovermode="x unified"
                            )
                            
                            # Display the chart
                            st.plotly_chart(vp_fig, use_container_width=True)
                            
                            # Display key volume levels
                            st.subheader("Key Volume Levels")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if volume_profile['poc']:
                                    st.metric("Point of Control (POC)", f"₹{volume_profile['poc'][0]:.2f}")
                                else:
                                    st.metric("Point of Control (POC)", "N/A")
                            
                            with col2:
                                if volume_profile['value_area_high']:
                                    st.metric("Value Area High", f"₹{volume_profile['value_area_high']:.2f}")
                                else:
                                    st.metric("Value Area High", "N/A")
                            
                            with col3:
                                if volume_profile['value_area_low']:
                                    st.metric("Value Area Low", f"₹{volume_profile['value_area_low']:.2f}")
                                else:
                                    st.metric("Value Area Low", "N/A")
                            
                            # Trading strategies based on Volume Profile
                            with st.expander("Volume Profile Trading Strategies"):
                                st.markdown("""
                                **Volume Profile Trading Strategies:**
                                
                                1. **Support and Resistance:** 
                                   - The Point of Control (POC) is a significant support/resistance level.
                                   - Value Area High and Low often act as resistance and support levels.
                                
                                2. **Breakout Strategy:**
                                   - Look for price breaks above Value Area High with increasing volume.
                                   - Look for price breaks below Value Area Low with increasing volume.
                                
                                3. **Value Area Reversal:**
                                   - When price reaches Value Area extremes, watch for reversal signals.
                                   - Enter counter-trend trades with tight stop losses.
                                
                                4. **Low Volume Nodes:**
                                   - Price tends to move quickly through areas with low trading volume.
                                   - These areas offer minimal support/resistance.
                                """)
                        else:
                            st.warning("Unable to calculate Volume Profile. This could be due to insufficient volume data.")

                with tab3:
                    st.subheader("Fundamental Analysis")

                    # Get latest data point
                    latest = hist_with_indicators.iloc[-1]
                    data = hist_with_indicators #added to handle data variable
                    
                    # Get comprehensive financial health data
                    with st.spinner("Fetching comprehensive financial data..."):
                        financial_data = get_financial_health_data(symbol)

                    # Check if we have fundamental data
                    if info and len(info) > 5:
                        # Display key metrics
                        st.markdown("### Key Metrics")

                        # Create metrics in columns
                        col1, col2, col3 = st.columns(3)

                        # Format market cap
                        market_cap = info.get("marketCap", 0)
                        if market_cap > 1_000_000_000_000:  # Trillion
                            market_cap_str = f"₹{market_cap/1_000_000_000_000:.2f}T"
                        elif market_cap > 1_000_000_000:  # Billion
                            market_cap_str = f"₹{market_cap/1_000_000_000:.2f}B"
                        elif market_cap > 1_000_000:  # Million
                            market_cap_str = f"₹{market_cap/1_000_000:.2f}M"
                        else:
                            market_cap_str = f"₹{market_cap:,.0f}"

                        with col1:
                            st.metric("Market Cap", market_cap_str)
                            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                            st.metric("EPS", f"₹{info.get('trailingEps', 'N/A')}")

                        with col2:
                            st.metric("52 Week High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                            st.metric("52 Week Low", f"₹{info.get('fiftyTwoWeekLow', 'N/A')}")
                            st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")

                        with col3:
                            st.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
                            st.metric("Beta", f"{info.get('beta', 'N/A')}")
                            st.metric("Volume", f"{info.get('volume', 'N/A'):,}")

                        # Display company information
                        st.markdown("### Company Information")

                        with st.expander("Company Profile"):
                            st.write(financial_data.get("longBusinessSummary", info.get("longBusinessSummary", "No company information available.")))

                            # Create two columns for company details
                            detail_col1, detail_col2 = st.columns(2)

                            with detail_col1:
                                st.write("**Sector:**", financial_data.get("sector", info.get("sector", "N/A")))
                                st.write("**Industry:**", financial_data.get("industry", info.get("industry", "N/A")))
                                st.write("**Employees:**", f"{financial_data.get('fullTimeEmployees', info.get('fullTimeEmployees', 'N/A')):,}" if financial_data.get('fullTimeEmployees', info.get('fullTimeEmployees')) else "N/A")
                                st.write("**Promoter Holding:**", f"{financial_data.get('screener_promoter_holding', 'N/A')}%")

                            with detail_col2:
                                st.write("**Website:**", financial_data.get("website", info.get("website", "N/A")))
                                st.write("**Exchange:**", financial_data.get("exchange", info.get("exchange", "N/A")))
                                st.write("**Currency:**", financial_data.get("currency", info.get("currency", "N/A")))
                                st.write("**FII Holding:**", f"{financial_data.get('screener_fii_holding', 'N/A')}%")

                        # Financial Health
                        st.markdown("### Financial Health")

                        fin_col1, fin_col2, fin_col3 = st.columns(3)

                        with fin_col1:
                            # Liquidity Ratios
                            st.subheader("Liquidity")
                            # Try to get data from multiple sources (financial_data has more sources)
                            current_ratio = financial_data.get('currentRatio', financial_data.get('screener_current_ratio', info.get('currentRatio')))
                            if current_ratio is not None and isinstance(current_ratio, (int, float)):
                                st.metric("Current Ratio", f"{current_ratio:.2f}")
                            else:
                                st.metric("Current Ratio", "N/A")

                            quick_ratio = financial_data.get('quickRatio', financial_data.get('screener_quick_ratio', info.get('quickRatio')))
                            if quick_ratio is not None and isinstance(quick_ratio, (int, float)):
                                st.metric("Quick Ratio", f"{quick_ratio:.2f}")
                            else:
                                st.metric("Quick Ratio", "N/A")

                            working_capital = financial_data.get('workingCapital', financial_data.get('screener_working_capital', info.get('workingCapital')))
                            if working_capital is not None and working_capital != 'N/A' and isinstance(working_capital, (int, float)):
                                st.metric("Working Capital", f"₹{working_capital/10000000:.2f}Cr")
                            else:
                                st.metric("Working Capital", "N/A")

                        with fin_col2:
                            # Solvency Ratios
                            st.subheader("Solvency")
                            debt_to_equity = financial_data.get('debtToEquity', financial_data.get('screener_debt_to_equity', info.get('debtToEquity')))
                            if debt_to_equity is not None and debt_to_equity != 'N/A' and isinstance(debt_to_equity, (int, float)):
                                st.metric("Debt to Equity", f"{debt_to_equity:.2f}")
                            else:
                                st.metric("Debt to Equity", "N/A")

                            total_debt = financial_data.get('totalDebt', financial_data.get('screener_total_debt', info.get('totalDebt')))
                            if total_debt is not None and total_debt != 'N/A' and isinstance(total_debt, (int, float)):
                                st.metric("Total Debt", f"₹{total_debt/10000000:.2f}Cr")
                            else:
                                st.metric("Total Debt", "N/A")

                        with fin_col3:
                            # Profitability Ratios
                            st.subheader("Profitability")
                            roa = financial_data.get('returnOnAssets', financial_data.get('screener_return_on_assets', financial_data.get('screener_roa', info.get('returnOnAssets'))))
                            if roa is not None and roa != 'N/A' and isinstance(roa, (int, float)):
                                st.metric("Return on Assets", f"{roa*100:.2f}%")
                            else:
                                st.metric("Return on Assets", "N/A")

                            operating_margin = financial_data.get('operatingMargin', financial_data.get('screener_operating_margin', info.get('operatingMargin')))
                            if operating_margin is not None and operating_margin != 'N/A' and isinstance(operating_margin, (int, float)):
                                st.metric("Operating Margin", f"{operating_margin*100:.2f}%")
                            else:
                                st.metric("Operating Margin", "N/A")

                            # Display Return on Equity
                            return_on_equity = financial_data.get('returnOnEquity', financial_data.get('screener_return_on_equity', financial_data.get('screener_roe', info.get('returnOnEquity'))))
                            if return_on_equity is not None and return_on_equity != 'N/A' and isinstance(return_on_equity, (int, float)):
                                st.metric("Return on Equity", f"{return_on_equity * 100:.2f}%")
                            else:
                                st.metric("Return on Equity", "N/A")

                            # Display Profit Margin
                            profit_margin = financial_data.get('profitMargin', financial_data.get('screener_profit_margin', financial_data.get('screener_net_profit_margin', info.get('profitMargins'))))
                            if profit_margin is not None and profit_margin != 'N/A' and isinstance(profit_margin, (int, float)):
                                st.metric("Profit Margin", f"{profit_margin * 100:.2f}%")
                            else:
                                st.metric("Profit Margin", "N/A")

                        # Section removed to avoid duplication of financial health indicators

                        # Recommendation summary
                        if 'recommendationMean' in info:
                            st.markdown("### Analyst Recommendations")

                            rec_mean = info.get('recommendationMean', 3)

                            # Create recommendation scale
                            rec_col1, rec_col2, rec_col3, rec_col4, rec_col5 = st.columns(5)

                            rec_cols = [rec_col1, rec_col2, rec_col3, rec_col4, rec_col5]
                            rec_labels = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]

                            # Highlight the current recommendation
                            for i, (col, label) in enumerate(zip(rec_cols, rec_labels)):
                                with col:
                                    if (i+1) == round(rec_mean):
                                        st.markdown(f"**{label}**")
                                        st.markdown("⬆️")
                                    else:
                                        st.markdown(f"{label}")
                                        st.markdown(" ")

                            st.markdown(f"**Analyst Recommendation Score**: {rec_mean:.2f}/5.0")
                    else:
                        st.warning("Limited or no fundamental data available for this stock. This might be due to API limitations.")

                with tab4:
                    st.subheader("AI Trading Recommendations")

                    # Get AI recommendation
                    ai_recommendation = get_ai_recommendation(hist_with_indicators)
                    
                    # Check if AI recommendation is valid
                    has_valid_recommendations = (
                        isinstance(ai_recommendation, dict) and 
                        'prediction' in ai_recommendation and 
                        'confidence' in ai_recommendation and 
                        'accuracy' in ai_recommendation
                    )

                    if has_valid_recommendations:
                        # Display recommendation
                        ai_col1, ai_col2 = st.columns(2)

                        with ai_col1:
                            st.markdown("### AI Prediction")

                            if ai_recommendation['prediction'] == "BUY":
                                st.success(f"### AI Recommends: BUY {symbol}")
                            elif ai_recommendation['prediction'] == "SELL":
                                st.error(f"### AI Recommends: SELL {symbol}")
                            else:
                                st.info(f"### AI Recommends: HOLD {symbol}")

                            st.markdown(f"**Prediction Confidence**: {ai_recommendation['confidence']*100:.1f}%")
                            st.markdown(f"**Model Accuracy**: {ai_recommendation['accuracy']*100:.1f}%")

                        with ai_col2:
                            st.markdown("### How this works")
                            st.write("""
                            The AI model analyzes the stock's technical indicators and price patterns to predict 
                            its likely movement in the near future. The model is trained on historical data from
                            this specific stock to identify patterns that precede price increases or decreases.
                            """)

                            if show_ai_details and 'features_used' in ai_recommendation:
                                st.markdown("### Model Details")
                                st.write(f"Features used: {', '.join(ai_recommendation['features_used'])}")
                                st.write("Algorithm: XGBoost Classifier")
                                st.write("The model is trained on recent data from this specific stock.")

                        # Trading plan section
                        st.markdown("### AI-Generated Trading Plan")
                        
                        # Define current price here to ensure it's available to all code paths
                        current_price = hist['Close'].iloc[-1]

                        # Calculate suggested entry, target and stop loss prices
                        if ai_recommendation['prediction'] == "BUY":
                            entry_price = current_price
                            target_price = current_price * 1.05  # 5% profit target
                            stop_loss = current_price * 0.97    # 3% stop loss

                            plan = f"""
                            **Entry Strategy**: Consider entering at current price of ₹{entry_price:.2f} or on a slight pullback.

                            **Target Price**: ₹{target_price:.2f} (5% above entry)

                            **Stop Loss**: ₹{stop_loss:.2f} (3% below entry)

                            **Risk:Reward Ratio**: 1:1.67

                            **Position Sizing**: Consider allocating no more than 2-5% of your portfolio to this trade.
                            """
                        elif ai_recommendation['prediction'] == "SELL":
                            # Set default for SELL case - the full code is handled later
                            plan = "SELL recommendation"
                    else:
                        st.warning("""
                        ### AI Model Could Not Generate Recommendations
                        
                        The AI model was unable to generate reliable recommendations for this stock with the current timeframe.
                        This could be due to:
                        
                        1. Insufficient historical data for the selected timeframe
                        2. Unusual market conditions not represented in the training data
                        3. Technical limitations in processing the indicators
                        
                        Try selecting a different timeframe or checking back later.
                        """)
                        
                        # Provide fallback recommendation based on technical indicators only
                        if 'signals' in locals():
                            st.info("""
                            ### Using Technical Indicators Instead
                            
                            Switching to traditional technical analysis based recommendations:
                            """)
                            rec, conf, exp = get_overall_recommendation(signals)
                            
                            if rec == "BUY":
                                st.success(f"### Technical Analysis Recommends: BUY {symbol}")
                            elif rec == "SELL":
                                st.error(f"### Technical Analysis Recommends: SELL {symbol}")
                            else:
                                st.info(f"### Technical Analysis Recommends: HOLD {symbol}")
                                
                            st.markdown(f"**Confidence**: {conf}")
                            st.markdown(f"**Explanation**: {exp}")
                        
                        # Set an empty dictionary to ensure the rest of the code doesn't fail
                        ai_recommendation = {'prediction': 'HOLD'}
                        
                    # Make sure current_price is defined for all code paths
                    if 'current_price' not in locals():
                        current_price = hist['Close'].iloc[-1]
                    
                    # Continue with the original code - this will only execute for SELL when has_valid_recommendations is True
                    if has_valid_recommendations and ai_recommendation['prediction'] == "SELL":
                        if 'shortable' in info and info['shortable']:
                            entry_price = current_price
                            target_price = current_price * 0.95  # 5% profit target
                            stop_loss = current_price * 1.03     # 3% stop loss

                            plan = f"""
                            **Entry Strategy**: Consider shorting at current price of ₹{entry_price:.2f} or on a slight bounce.

                            **Target Price**: ₹{target_price:.2f} (5% below entry)

                            **Stop Loss**: ₹{stop_loss:.2f} (3% above entry)

                            **Risk:Reward Ratio**: 1:1.67

                            **Position Sizing**: Consider allocating no more than 2-5% of your portfolio to this trade.
                            """
                        else:
                            plan = f"""
                            **Strategy**: If you own this stock, consider selling at the current price of ₹{current_price:.2f}.

                            If you don't own the stock, avoid buying at this time as the AI predicts a price decrease.
                            """
                    else:
                        plan = """
                        **Strategy**: The AI does not have a strong signal for either buying or selling.

                        It's recommended to wait for a clearer signal or look for other opportunities.
                        """

                    st.markdown(plan)

                    # Combined Technical and Fundamental Analysis Recommendation
                    st.markdown("### Combined Analysis Recommendation")

                    # Get technical signals
                    tech_signals = signals if 'signals' in locals() else {}
                    tech_rec, tech_conf, _ = get_overall_recommendation(tech_signals)

                    # Get fundamental signals
                    fund_signals = {}
                    if info:
                        pe_ratio = info.get('trailingPE', None)
                        if pe_ratio and pe_ratio > 0:
                            if pe_ratio < 15:
                                fund_signals['PE'] = {"signal": "BUY", "strength": "Strong", "value": f"{pe_ratio:.2f}"}
                            elif pe_ratio > 30:
                                fund_signals['PE'] = {"signal": "SELL", "strength": "Strong", "value": f"{pe_ratio:.2f}"}

                        debt_equity = info.get('debtToEquity', None)
                        if debt_equity and debt_equity > 0:
                            if debt_equity < 50:
                                fund_signals['Debt/Equity'] = {"signal": "BUY", "strength": "Strong", "value": f"{debt_equity:.2f}"}
                            elif debt_equity > 100:
                                fund_signals['Debt/Equity'] = {"signal": "SELL", "strength": "Strong", "value": f"{debt_equity:.2f}"}

                        profit_margin = info.get('profitMargins', None)
                        if profit_margin and profit_margin > 0:
                            if profit_margin > 0.2:
                                fund_signals['Profit Margin'] = {"signal": "BUY", "strength": "Strong", "value": f"{profit_margin*100:.1f}%"}
                            elif profit_margin < 0.05:
                                fund_signals['Profit Margin'] = {"signal": "SELL", "strength": "Strong", "value": f"{profit_margin*100:.1f}%"}

                    fund_rec, fund_conf, _ = get_overall_recommendation(fund_signals)

                    # Display recommendations
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Technical Analysis")
                        if tech_rec == "BUY":
                            st.success(f"Technical Analysis suggests: **BUY** ({tech_conf} confidence)")
                        elif tech_rec == "SELL":
                            st.error(f"Technical Analysis suggests: **SELL** ({tech_conf} confidence)")
                        else:
                            st.info(f"Technical Analysis suggests: **NEUTRAL** ({tech_conf} confidence)")

                    with col2:
                        st.subheader("Fundamental Analysis")
                        if fund_rec == "BUY":
                            st.success(f"Fundamental Analysis suggests: **BUY** ({fund_conf} confidence)")
                        elif fund_rec == "SELL":
                            st.error(f"Fundamental Analysis suggests: **SELL** ({fund_conf} confidence)")
                        else:
                            st.info(f"Fundamental Analysis suggests: **NEUTRAL** ({fund_conf} confidence)")

                    # Combined recommendation
                    st.markdown("#### Final Recommendation")
                    if tech_rec == fund_rec and tech_rec != "NEUTRAL":
                        if tech_rec == "BUY":
                            st.success("### STRONG BUY SIGNAL 🚀")
                            st.markdown("Both technical and fundamental analyses suggest a buying opportunity.")
                        else:
                            st.error("### STRONG SELL SIGNAL 📉")
                            st.markdown("Both technical and fundamental analyses suggest selling.")
                    elif tech_rec != "NEUTRAL" and tech_conf == "High":
                        st.info(f"### TECHNICAL {tech_rec} SIGNAL")
                        st.markdown("Strong technical signals with mixed fundamental indicators.")
                    elif fund_rec != "NEUTRAL" and fund_conf == "High":
                        st.info(f"### FUNDAMENTAL {fund_rec} SIGNAL")
                        st.markdown("Strong fundamental signals with mixed technical indicators.")
                    else:
                        st.warning("### NEUTRAL / HOLD")
                        st.markdown("Mixed signals from both analyses. Consider holding current position.")

                    # Add disclaimer
                    st.warning("""
                    **Disclaimer**: These recommendations are based on technical and fundamental analysis.
                    They should not be considered as financial advice. Always conduct your own research and consider consulting with a
                    financial advisor before making investment decisions.
                    """)

                # Add Export/Download functionality - placed after all tabs are populated
                st.markdown("---")
                st.subheader("📥 Export Analysis Reports")

                # Initialize variables for export that might not be defined in some tabs
                chart_type = chart_type if 'chart_type' in locals() else "Line"
                signals = signals if 'signals' in locals() else {}

                with tab5:
                    st.subheader("News & Market Sentiment Analysis")
                    
                    with st.spinner("Fetching and analyzing news..."):
                        # Get news and sentiment
                        news_df, sentiment_metrics = get_news_sentiment(symbol, max_articles=5)
                        
                        if news_df.empty:
                            st.error("No news articles found for this stock. Try another symbol or check your internet connection.")
                        else:
                            # Display sentiment summary
                            st.markdown("### Market Sentiment Summary")
                            
                            # Determine sentiment color
                            sentiment_color = "gray"
                            if 'positive' in sentiment_metrics['overall_sentiment']:
                                sentiment_color = "green"
                            elif 'negative' in sentiment_metrics['overall_sentiment']:
                                sentiment_color = "red"
                            
                            # Create sentiment metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; text-align: center;'>
                                    <h4 style='color: {sentiment_color}; margin: 0;'>{sentiment_metrics['overall_sentiment'].upper()}</h4>
                                    <p>Overall Sentiment</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # Create sentiment score display
                                score = sentiment_metrics['overall_score']
                                score_percentage = (score + 1) / 2 * 100  # Convert -1 to 1 scale to 0-100%
                                
                                st.markdown(f"""
                                <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; text-align: center;'>
                                    <h4 style='color: {sentiment_color}; margin: 0;'>{score:.2f}</h4>
                                    <p>Sentiment Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; text-align: center;'>
                                    <h4 style='margin: 0;'>{sentiment_metrics['total_articles']}</h4>
                                    <p>Articles Analyzed</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Create sentiment distribution chart
                            st.markdown("### Sentiment Distribution")
                            
                            sentiment_dist = {
                                'Positive': sentiment_metrics['positive_articles'],
                                'Neutral': sentiment_metrics['neutral_articles'],
                                'Negative': sentiment_metrics['negative_articles']
                            }
                            
                            # Create chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=list(sentiment_dist.keys()),
                                y=list(sentiment_dist.values()),
                                marker_color=['rgba(0, 200, 0, 0.6)', 'rgba(180, 180, 180, 0.6)', 'rgba(200, 0, 0, 0.6)']
                            ))
                            
                            fig.update_layout(
                                title="News Sentiment Distribution",
                                xaxis_title="Sentiment",
                                yaxis_title="Number of Articles",
                                height=300,
                                plot_bgcolor='rgba(240,240,240,0.8)',
                                paper_bgcolor='rgba(255,255,255,1)',
                                font=dict(family="Arial, sans-serif", size=12, color="#505050"),
                                margin=dict(t=50, b=50, l=50, r=50)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display recent news with sentiment
                            st.markdown("### Recent News Articles")
                            
                            # Check if we're using demo data
                            if 'demo' in news_df.columns and news_df['demo'].any():
                                st.warning("""
                                📌 **Demo Data**: Unable to fetch real-time news from data providers. 
                                Showing sample news for demonstration purposes. 
                                This could be due to API limitations or network connectivity issues.
                                """)
                            
                            # Format and display each news article
                            for i, news in news_df.iterrows():
                                sentiment_icon = "🔍"
                                if 'positive' in str(news['sentiment']):
                                    sentiment_icon = "📈"
                                elif 'negative' in str(news['sentiment']):
                                    sentiment_icon = "📉"
                                elif 'neutral' in str(news['sentiment']):
                                    sentiment_icon = "⚖️"
                                
                                # Format date
                                date_str = news['date'].strftime('%b %d, %Y')
                                
                                st.markdown(f"""
                                <div style='background-color: rgba(240,240,240,0.8); padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                                    <h4 style='margin-top: 0;'>{sentiment_icon} {news['title']}</h4>
                                    <p><strong>Source:</strong> {news['source']} | <strong>Date:</strong> {date_str}</p>
                                    <p><strong>Sentiment:</strong> {news['sentiment'].capitalize()} (Score: {news['score']:.2f})</p>
                                    <a href="{news['url']}" target="_blank">Read full article</a>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add sentiment impact on stock
                            st.markdown("### Sentiment Impact")
                            
                            # Provide interpretation based on sentiment
                            if 'positive' in sentiment_metrics['overall_sentiment']:
                                st.success("""
                                **Positive Market Sentiment**: The recent news coverage for this stock is predominantly positive, 
                                which may indicate favorable market perception. Positive news can potentially drive buying interest and price appreciation.
                                """)
                            elif 'negative' in sentiment_metrics['overall_sentiment']:
                                st.error("""
                                **Negative Market Sentiment**: The recent news coverage for this stock shows concerning signals, 
                                with negative sentiment dominating. This could lead to selling pressure in the near term.
                                """)
                            else:
                                st.info("""
                                **Neutral Market Sentiment**: The recent news coverage for this stock is balanced or neutral, 
                                suggesting no strong directional bias from news sentiment alone. Other factors may have more influence on price movements.
                                """)
                            
                            # Add cross-analysis with technical indicators
                            if 'signals' in locals() and 'ai_recommendation' in locals() and 'recommendation' in ai_recommendation:
                                st.markdown("### Sentiment vs. Technical Analysis")
                                
                                rec = ai_recommendation['recommendation']
                                
                                # Compare sentiment with technical indicators
                                if ('positive' in sentiment_metrics['overall_sentiment'] and rec == 'BUY') or \
                                   ('negative' in sentiment_metrics['overall_sentiment'] and rec == 'SELL'):
                                    st.success("""
                                    **Strong Confirmation**: Market sentiment aligns with technical analysis signals, 
                                    providing stronger confirmation for the trading recommendation.
                                    """)
                                elif ('positive' in sentiment_metrics['overall_sentiment'] and rec == 'SELL') or \
                                     ('negative' in sentiment_metrics['overall_sentiment'] and rec == 'BUY'):
                                    st.warning("""
                                    **Mixed Signals**: Market sentiment contradicts technical analysis signals. 
                                    This divergence suggests caution and more thorough research before taking action.
                                    """)
                                else:
                                    st.info("""
                                    **Neutral Alignment**: Market sentiment is neutral relative to technical signals. 
                                    Consider prioritizing technical factors in your decision-making.
                                    """)
                            
                            # Disclaimer about news analysis
                            st.warning("""
                            **DISCLAIMER:** News sentiment analysis is based on a basic keyword-based sentiment algorithm, 
                            not comprehensive natural language processing. Results should be considered approximations and not definitive market sentiment indicators.
                            Always perform thorough research and consider multiple factors before making investment decisions.
                            """)

                with tab6:
                    st.subheader("Alternative Data Analysis")

                    from utils.alternative_data import get_google_trends_data, get_stocktwits_sentiment

                    # Get company name without exchange suffix
                    company_name = symbol.split('.')[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Google Trends Analysis")
                        trends_data = get_google_trends_data(company_name)

                        if not trends_data.empty:
                            # Plot Google Trends data
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=trends_data.index,
                                y=trends_data[company_name],
                                mode='lines',
                                name='Search Interest'
                            ))

                            fig.update_layout(
                                title="Search Interest Over Time",
                                xaxis_title="Date",
                                yaxis_title="Search Interest",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Calculate momentum
                            current_interest = trends_data[company_name].iloc[-1]
                            avg_interest = trends_data[company_name].mean()

                            if current_interest > avg_interest * 1.2:
                                st.success("🔥 High search interest detected! This could indicate increased market attention.")
                            elif current_interest < avg_interest * 0.8:
                                st.warning("📉 Low search interest detected. The stock might be under the radar.")
                        else:
                            st.warning("No Google Trends data available.")

                    with col2:
                        st.markdown("### StockTwits Sentiment")
                        stocktwits_data = get_stocktwits_sentiment(symbol)

                        if stocktwits_data:
                            # Create sentiment gauge chart
                            sentiment_score = (stocktwits_data['bullish_ratio'] - stocktwits_data['bearish_ratio']) * 100

                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = sentiment_score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [-100, 100]},
                                    'bar': {'color': "gray"},
                                    'steps': [
                                        {'range': [-100, -30], 'color': "red"},
                                        {'range': [-30, 30], 'color': "yellow"},
                                        {'range': [30, 100], 'color': "green"}
                                    ]
                                },
                                title = {'text': "Sentiment Score"}
                            ))

                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                            # Display message volume
                            st.metric("Message Volume", stocktwits_data['message_volume'])

                            # Interpret sentiment
                            if sentiment_score > 30:
                                st.success("👍 Bullish sentiment dominates on StockTwits")
                            elif sentiment_score < -30:
                                st.error("👎 Bearish sentiment dominates on StockTwits")
                            else:
                                st.info("🤝 Mixed sentiment on StockTwits")
                        else:
                            st.warning("No StockTwits data available.")

                    # Combined insights
                    st.markdown("### Combined Alternative Data Insights")
                    if not trends_data.empty and stocktwits_data:
                        trends_momentum = (trends_data[company_name].iloc[-1] / trends_data[company_name].mean() - 1) * 100
                        sentiment_score = (stocktwits_data['bullish_ratio'] - stocktwits_data['bearish_ratio']) * 100

                        if trends_momentum > 20 and sentiment_score > 30:
                            st.success("🌟 Strong positive signals from both search trends and social sentiment!")
                        elif trends_momentum < -20 and sentiment_score < -30:
                            st.error("⚠️ Negative signals from both search trends and social sentiment.")
                        else:
                            st.info("📊 Mixed signals from alternative data sources. Consider other factors.")


                # Define default values for recommendation variables
                rec, conf, exp = "NEUTRAL", "Medium", "Not enough data for a strong recommendation."
                if signals:
                    rec, conf, exp = get_overall_recommendation(signals)

                export_col1, export_col2, export_col3 = st.columns(3)

                # Price data export options
                with export_col1:
                    st.markdown("### Price Data")
                    st.markdown(get_table_download_link(hist, f"{symbol}_price_data", "📊 Download Price History (CSV)"), unsafe_allow_html=True)

                    # Create a download link for the chart as HTML
                    price_fig = go.Figure()
                    if chart_type == "Candlestick":
                        price_fig.add_trace(go.Candlestick(
                            x=hist.index, open=hist['Open'], high=hist['High'],
                            low=hist['Low'], close=hist['Close'], name="Price"
                        ))
                    else:
                        price_fig.add_trace(go.Scatter(
                            x=hist.index, y=hist['Close'], mode='lines',
                            name="Close Price", line=dict(color='blue', width=2)
                        ))
                    price_fig.update_layout(
                        title=f"{symbol} - {timeframe} Chart",
                        xaxis_title="Date", yaxis_title="Price (₹)",
                        height=400, width=600
                    )

                    st.markdown(get_html_download_link(price_fig, f"{symbol}_price_chart", "📈 Download Price Chart (HTML)"), unsafe_allow_html=True)

                # Technical indicators export
                with export_col2:
                    st.markdown("### Technical Analysis")
                    st.markdown(get_table_download_link(hist_with_indicators, f"{symbol}_technical_indicators", "📈 Download Technical Indicators (CSV)"), unsafe_allow_html=True)

                    # Generate signal data for download
                    if signals:
                        signal_df = pd.DataFrame.from_dict(
                            {k: [v["signal"], v["strength"], v["value"]] for k, v in signals.items()},
                            orient='index',
                            columns=['Signal', 'Strength', 'Value']
                        )
                        st.markdown(get_table_download_link(signal_df, f"{symbol}_trading_signals", "🎯 Download Trading Signals (CSV)"), unsafe_allow_html=True)

                    # Create and offer summary report as JSON
                    tech_data = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "current_price": float(hist['Close'].iloc[-1]),
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "technical_indicators": {
                            "sma_20": float(hist_with_indicators['SMA_20'].iloc[-1]),
                            "ema_20": float(hist_with_indicators['EMA_20'].iloc[-1]),
                            "rsi": float(hist_with_indicators['RSI'].iloc[-1]),
                            "macd": float(hist_with_indicators['MACD'].iloc[-1]),
                            "signal_line": float(hist_with_indicators['Signal_Line'].iloc[-1]),
                        }
                    }

                    if signals:
                        tech_data["recommendation"] = {
                            "overall": rec,
                            "confidence": conf,
                            "explanation": exp
                        }

                    st.markdown(get_json_download_link(tech_data, f"{symbol}_technical_summary", "📑 Download Technical Summary (JSON)"), unsafe_allow_html=True)

                # Complete report export
                with export_col3:
                    st.markdown("### Complete Analysis")

                    # Create full report data
                    full_report_data = {
                        "symbol": symbol,
                        "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                        "timeframe": timeframe,
                        "price_data": {
                            "current": float(hist['Close'].iloc[-1]),
                            "open": float(hist['Open'].iloc[-1]),
                            "high": float(hist['High'].iloc[-1]),
                            "low": float(hist['Low'].iloc[-1]),
                            "volume": float(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                        },
                        "technical_analysis": tech_data["technical_indicators"]
                    }
                    
                    # Check if ai_recommendation exists and has required keys
                    if 'ai_recommendation' in locals() and isinstance(ai_recommendation, dict):
                        ai_rec_data = {}
                        
                        if 'prediction' in ai_recommendation:
                            ai_rec_data["prediction"] = ai_recommendation['prediction']
                            
                        if 'confidence' in ai_recommendation:
                            ai_rec_data["confidence"] = float(ai_recommendation['confidence'])
                            
                        if 'accuracy' in ai_recommendation:
                            ai_rec_data["accuracy"] = float(ai_recommendation['accuracy'])
                        
                        if ai_rec_data:  # Only add if we have some data
                            full_report_data["ai_recommendation"] = ai_rec_data

                    if signals:
                        full_report_data["trading_signals"] = {k: {"signal": v["signal"], "strength": v["strength"]} for k, v in signals.items()}
                        full_report_data["overall_recommendation"] = {"signal": rec, "confidence": conf}

                    # Generate report links
                    st.markdown(get_json_download_link(full_report_data, f"{symbol}_complete_analysis", "📊 Download Full Analysis (JSON)"), unsafe_allow_html=True)

                    # Offer HTML report - with safe handling of ai_recommendation
                    report_html = get_pdf_report(symbol, hist, hist_with_indicators, info, signals, 
                                                ai_recommendation if 'ai_recommendation' in locals() else {}, 
                                                timeframe)
                    b64 = base64.b64encode(report_html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{symbol}_analysis_report.html">📑 Download Analysis Report (HTML)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.info("The HTML report includes all charts, indicators, signals and recommendations in a printer-friendly format.")
else:
    # Display welcome message and guide
    st.markdown("""
    ## Welcome to the Indian Stock Market Analysis Tool

    This application allows you to analyze Indian stocks with technical indicators, fundamental metrics, and AI-powered recommendations.

    ### How to use:
    1. Enter a valid Indian stock symbol in the sidebar (add .NS for NSE or .BO for BSE)
    2. Select a timeframe for analysis
    3. Click "Analyse" to see the analysis

    ### Features:
    - Interactive price charts
    - Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
    - Fundamental analysis metrics
    - AI-powered buy/sell recommendations
    - Trading signals and plans

    ### Example stocks you can analyze:
    - Reliance Industries: RELIANCE.NS
    - Tata Consultancy Services: TCS.NS
    - HDFC Bank: HDFCBANK.NS
    - Infosys: INFY.NS
    - State Bank of India: SBIN.NS
    """)

    # Add an image or icon
    st.markdown("""
    <div style="text-align:center">
        <svg width="200" height="200" viewBox="0 0 200 200">
            <rect width="200" height="200" fill="#f0f2f6" rx="10"/>
            <path d="M40,160 L40,80 L80,40 L120,80 L160,40 L160,160" stroke="#1f77b4" fill="none" stroke-width="8"/>
            <circle cx="40" cy="120" r="5" fill="#ff9f1c"/>
            <circle cx="70" cy="90" r="5" fill="#ff9f1c"/>
            <circle cx="100" cy="110" r="5" fill="#ff9f1c"/>
            <circle cx="130" cy="70" r="5" fill="#ff9f1c"/>
            <circle cx="160" cy="100" r="5" fill="#ff9f1c"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Enter a stock symbol in the sidebar and click "Analyse" to begin your analysis.
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Indian Stock Market Analysis & AI Recommendations Tool | Data sourced from Yahoo Finance
</div>
""", unsafe_allow_html=True)