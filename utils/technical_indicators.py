import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Parameters:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy and handle missing values
    data = df.copy()
    data = data.ffill().bfill()
    
    # Simple Moving Average (SMA)
    data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    
    # Exponential Moving Average (EMA)
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False, min_periods=1).mean()
    
    # MACD
    data['MACD'] = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean() - \
                   data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # RSI with proper handling of edge cases
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['BB_Std'] = data['Close'].rolling(window=20, min_periods=1).std()
    data['Upper_BB'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['Lower_BB'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    # Average True Range (ATR)
    tr = pd.DataFrame()
    tr['h-l'] = data['High'] - data['Low']
    tr['h-pc'] = abs(data['High'] - data['Close'].shift())
    tr['l-pc'] = abs(data['Low'] - data['Close'].shift())
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    data['ATR'] = tr['tr'].rolling(window=14, min_periods=1).mean()
    
    # Stochastic Oscillator
    low_min = data['Low'].rolling(window=14, min_periods=1).min()
    high_max = data['High'].rolling(window=14, min_periods=1).max()
    data['%K'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    data['%D'] = data['%K'].rolling(window=3, min_periods=1).mean()
    
    # Volume-based indicators
    # On-Balance Volume (OBV)
    data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
    
    # VWAP (Volume Weighted Average Price)
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Price momentum
    data['Price_Momentum'] = data['Close'].pct_change(periods=10)
    
    # Volatility
    data['Volatility'] = data['Close'].rolling(window=20, min_periods=1).std()
    
    return data
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['Upper_BB'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['Lower_BB'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Average Directional Index (ADX)
    data['TR'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(
            abs(data['High'] - data['Close'].shift(1)),
            abs(data['Low'] - data['Close'].shift(1))
        )
    )
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # Additional features for AI model
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=5).std()
    data['Price_to_SMA'] = data['Close'] / data['SMA_20']
    
    return data

def get_indicator_explanations():
    """
    Get explanations for various technical indicators with trend interpretations
    
    Returns:
        dict: Dictionary of indicator explanations
    """
    return {
        "SMA_20": {
            "description": "Simple Moving Average (20-day): The average price over the last 20 days.",
            "trends": {
                "Bullish": "Price above SMA with increasing SMA slope",
                "Bearish": "Price below SMA with decreasing SMA slope",
                "Neutral": "Price crossing SMA frequently with flat slope"
            }
        },
        "EMA_20": {
            "description": "Exponential Moving Average (20-day): Weighted average favoring recent prices.",
            "trends": {
                "Bullish": "Price and EMA rising, with price above EMA",
                "Bearish": "Price and EMA falling, with price below EMA",
                "Neutral": "Price oscillating around EMA"
            }
        },
        "MACD": {
            "description": "Moving Average Convergence Divergence: Momentum indicator showing relationship between two EMAs.",
            "trends": {
                "Bullish": "MACD line crosses above signal line, both rising",
                "Bearish": "MACD line crosses below signal line, both falling",
                "Neutral": "MACD and signal lines moving sideways"
            }
        },
        "RSI": {
            "description": "Relative Strength Index: Momentum oscillator measuring speed and change of price movements.",
            "trends": {
                "Overbought": "RSI above 70 - potential reversal or correction",
                "Oversold": "RSI below 30 - potential buying opportunity",
                "Neutral": "RSI between 30-70 - ranging market"
            }
        },
        "Bollinger_Bands": {
            "description": "Price channels that are 2 standard deviations from SMA.",
            "trends": {
                "Squeeze": "Bands narrowing - potential breakout incoming",
                "Expansion": "Bands widening - high volatility period",
                "Trend": "Price riding upper/lower band indicates strong trend"
            }
        },
        "ATR": {
            "description": "Average True Range: Measures market volatility.",
            "trends": {
                "High": "ATR increasing - higher volatility, wider stops needed",
                "Low": "ATR decreasing - lower volatility, tighter stops possible",
                "Stable": "ATR flat - consistent volatility"
            }
        },
        "OBV": {
            "description": "On-Balance Volume: Cumulative indicator of volume flow.",
            "trends": {
                "Bullish": "OBV increasing while price rises - strong uptrend",
                "Bearish": "OBV decreasing while price falls - strong downtrend",
                "Divergence": "OBV and price moving in opposite directions - potential reversal"
            }
        },
        "Stochastic": {
            "description": "Stochastic Oscillator: Momentum indicator comparing close to price range.",
            "trends": {
                "Overbought": "%K and %D above 80 - potential selling opportunity",
                "Oversold": "%K and %D below 20 - potential buying opportunity",
                "Crossover": "%K crossing %D - potential trend change"
            }
        }
    }

def generate_trading_signals(data):
    """
    Generate trading signals based on technical indicators
    
    Parameters:
        data (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: Dictionary of trading signals and strengths
    """
    signals = {}
    
    # Latest values
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else None
    
    # RSI signals
    rsi_value = latest.get('RSI', None)
    if rsi_value is not None and not np.isnan(rsi_value):
        if rsi_value > 70:
            signals['RSI'] = {"signal": "SELL", "strength": "Strong", "value": f"{rsi_value:.2f}"}
        elif rsi_value < 30:
            signals['RSI'] = {"signal": "BUY", "strength": "Strong", "value": f"{rsi_value:.2f}"}
        elif rsi_value > 60:
            signals['RSI'] = {"signal": "SELL", "strength": "Weak", "value": f"{rsi_value:.2f}"}
        elif rsi_value < 40:
            signals['RSI'] = {"signal": "BUY", "strength": "Weak", "value": f"{rsi_value:.2f}"}
        else:
            signals['RSI'] = {"signal": "NEUTRAL", "strength": "Neutral", "value": f"{rsi_value:.2f}"}
    
    # MACD signals
    macd = latest.get('MACD', None)
    signal = latest.get('Signal_Line', None)
    prev_macd = previous.get('MACD', None) if previous is not None else None
    prev_signal = previous.get('Signal_Line', None) if previous is not None else None
    
    if None not in [macd, signal, prev_macd, prev_signal] and not np.isnan(macd) and not np.isnan(signal):
        # MACD crossover
        if macd > signal and prev_macd <= prev_signal:
            signals['MACD'] = {"signal": "BUY", "strength": "Strong", "value": f"MACD: {macd:.2f}, Signal: {signal:.2f}"}
        elif macd < signal and prev_macd >= prev_signal:
            signals['MACD'] = {"signal": "SELL", "strength": "Strong", "value": f"MACD: {macd:.2f}, Signal: {signal:.2f}"}
        elif macd > signal:
            signals['MACD'] = {"signal": "BUY", "strength": "Weak", "value": f"MACD: {macd:.2f}, Signal: {signal:.2f}"}
        elif macd < signal:
            signals['MACD'] = {"signal": "SELL", "strength": "Weak", "value": f"MACD: {macd:.2f}, Signal: {signal:.2f}"}
        else:
            signals['MACD'] = {"signal": "NEUTRAL", "strength": "Neutral", "value": f"MACD: {macd:.2f}, Signal: {signal:.2f}"}
    
    # Bollinger Bands signals
    close = latest.get('Close', None)
    upper_bb = latest.get('Upper_BB', None)
    lower_bb = latest.get('Lower_BB', None)
    
    if None not in [close, upper_bb, lower_bb] and not np.isnan(close) and not np.isnan(upper_bb) and not np.isnan(lower_bb):
        bb_percent = (close - lower_bb) / (upper_bb - lower_bb) if (upper_bb - lower_bb) != 0 else 0.5
        
        if bb_percent > 0.95:
            signals['BB'] = {"signal": "SELL", "strength": "Strong", "value": f"Price: {close:.2f}, %B: {bb_percent:.2f}"}
        elif bb_percent < 0.05:
            signals['BB'] = {"signal": "BUY", "strength": "Strong", "value": f"Price: {close:.2f}, %B: {bb_percent:.2f}"}
        elif bb_percent > 0.8:
            signals['BB'] = {"signal": "SELL", "strength": "Weak", "value": f"Price: {close:.2f}, %B: {bb_percent:.2f}"}
        elif bb_percent < 0.2:
            signals['BB'] = {"signal": "BUY", "strength": "Weak", "value": f"Price: {close:.2f}, %B: {bb_percent:.2f}"}
        else:
            signals['BB'] = {"signal": "NEUTRAL", "strength": "Neutral", "value": f"Price: {close:.2f}, %B: {bb_percent:.2f}"}
    
    # Moving Average signals
    sma = latest.get('SMA_20', None)
    ema = latest.get('EMA_20', None)
    
    if None not in [close, sma, ema] and not np.isnan(close) and not np.isnan(sma) and not np.isnan(ema):
        # Price vs SMA
        if close > sma * 1.05:
            signals['MA'] = {"signal": "BUY", "strength": "Strong", "value": f"Price/SMA: {close/sma:.2f}"}
        elif close < sma * 0.95:
            signals['MA'] = {"signal": "SELL", "strength": "Strong", "value": f"Price/SMA: {close/sma:.2f}"}
        elif close > sma:
            signals['MA'] = {"signal": "BUY", "strength": "Weak", "value": f"Price/SMA: {close/sma:.2f}"}
        elif close < sma:
            signals['MA'] = {"signal": "SELL", "strength": "Weak", "value": f"Price/SMA: {close/sma:.2f}"}
        else:
            signals['MA'] = {"signal": "NEUTRAL", "strength": "Neutral", "value": f"Price/SMA: {close/sma:.2f}"}
    
    return signals

def get_overall_recommendation(signals):
    """
    Generate an overall recommendation based on all signals
    
    Parameters:
        signals (dict): Dictionary of trading signals
        
    Returns:
        tuple: (recommendation, confidence, explanation)
    """
    if not signals:
        return "NEUTRAL", "Low", "Insufficient data for recommendation"
    
    # Count types of signals
    buy_count = sum(1 for s in signals.values() if s["signal"] == "BUY")
    sell_count = sum(1 for s in signals.values() if s["signal"] == "SELL")
    strong_buy = sum(1 for s in signals.values() if s["signal"] == "BUY" and s["strength"] == "Strong")
    strong_sell = sum(1 for s in signals.values() if s["signal"] == "SELL" and s["strength"] == "Strong")
    
    # Calculate confidence level
    confidence = "Low"
    if (buy_count >= 2 or sell_count >= 2) and (strong_buy >= 1 or strong_sell >= 1):
        confidence = "High"
    elif (buy_count >= 2 or sell_count >= 2):
        confidence = "Medium"
    
    # Generate recommendation
    if buy_count > sell_count + 1:
        return "BUY", confidence, f"{buy_count} buy signals vs {sell_count} sell signals"
    elif sell_count > buy_count + 1:
        return "SELL", confidence, f"{sell_count} sell signals vs {buy_count} buy signals"
    elif strong_buy > strong_sell:
        return "BUY", confidence, f"{strong_buy} strong buy signals vs {strong_sell} strong sell signals"
    elif strong_sell > strong_buy:
        return "SELL", confidence, f"{strong_sell} strong sell signals vs {strong_buy} strong buy signals"
    else:
        return "NEUTRAL", confidence, "Mixed signals, no clear direction"
