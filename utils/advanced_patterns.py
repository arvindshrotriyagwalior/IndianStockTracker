"""
Advanced Technical Pattern Recognition Module

This module provides functionality to identify complex chart patterns such as:
- Head and Shoulders (Bullish/Bearish)
- Cup and Handle
- Double Top/Bottom
- Triple Top/Bottom
- Wedges
- Rectangles
- Triangles
- Fibonacci Retracements/Extensions
- Volume Profile Analysis
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import timedelta


def identify_swing_highs_lows(df, window=5, price_col='Close'):
    """
    Identify swing highs and lows in the price data
    
    Parameters:
        df (pd.DataFrame): OHLC price data
        window (int): Window size to identify local maxima/minima
        price_col (str): Column name for price data
        
    Returns:
        tuple: (swing_highs_df, swing_lows_df)
    """
    # Create a copy of the dataframe
    price_data = df.copy()
    
    # Find local maxima
    price_data['is_max'] = price_data[price_col].rolling(window=window, center=True).apply(
        lambda x: x[window//2] == max(x), raw=True)
    
    # Find local minima
    price_data['is_min'] = price_data[price_col].rolling(window=window, center=True).apply(
        lambda x: x[window//2] == min(x), raw=True)
    
    # Extract swing highs and lows
    swing_highs = price_data[price_data['is_max'] == True].copy()
    swing_lows = price_data[price_data['is_min'] == True].copy()
    
    return swing_highs, swing_lows


def detect_head_and_shoulders(df, price_col='Close', window=5, threshold=0.03):
    """
    Detect Head and Shoulders and Inverse Head and Shoulders patterns
    
    Parameters:
        df (pd.DataFrame): OHLC price data
        price_col (str): Column name for price data
        window (int): Window size for detecting swing highs/lows
        threshold (float): Percentage threshold for pattern validity
        
    Returns:
        dict: Dictionary with pattern information
    """
    swing_highs, swing_lows = identify_swing_highs_lows(df, window, price_col)
    
    # Pattern containers
    head_shoulders = []
    inv_head_shoulders = []
    
    # Minimum number of points required
    min_points = 5
    
    # Regular Head and Shoulders (bearish pattern)
    if len(swing_highs) >= min_points:
        # Iterate through potential patterns
        for i in range(len(swing_highs) - min_points + 1):
            points = swing_highs.iloc[i:i+min_points]
            prices = points[price_col].values
            
            # Check for head and shoulders pattern
            # Left shoulder, head, right shoulder pattern where head is higher
            if (prices[0] < prices[2] > prices[4] and 
                abs(prices[0] - prices[4]) / prices[0] < threshold and 
                prices[2] > prices[0] * (1 + threshold) and
                prices[1] < prices[0] and prices[1] < prices[2] and
                prices[3] < prices[2] and prices[3] < prices[4]):
                
                pattern_data = {
                    'type': 'Head and Shoulders',
                    'direction': 'Bearish',
                    'left_shoulder': points.index[0],
                    'head': points.index[2],
                    'right_shoulder': points.index[4],
                    'neckline': min(points.loc[points.index[1], price_col], points.loc[points.index[3], price_col]),
                    'strength': 'Strong' if abs(prices[0] - prices[4]) / prices[0] < 0.01 else 'Moderate',
                    'target': prices[2] - (prices[0] - min(prices[1], prices[3])) # Head height from neckline
                }
                head_shoulders.append(pattern_data)
    
    # Inverse Head and Shoulders (bullish pattern)
    if len(swing_lows) >= min_points:
        # Iterate through potential patterns
        for i in range(len(swing_lows) - min_points + 1):
            points = swing_lows.iloc[i:i+min_points]
            prices = points[price_col].values
            
            # Check for inverse head and shoulders pattern
            # Left shoulder, head, right shoulder pattern where head is lower
            if (prices[0] > prices[2] < prices[4] and 
                abs(prices[0] - prices[4]) / prices[0] < threshold and 
                prices[2] < prices[0] * (1 - threshold) and
                prices[1] > prices[0] and prices[1] > prices[2] and
                prices[3] > prices[2] and prices[3] > prices[4]):
                
                pattern_data = {
                    'type': 'Inverse Head and Shoulders',
                    'direction': 'Bullish',
                    'left_shoulder': points.index[0],
                    'head': points.index[2],
                    'right_shoulder': points.index[4],
                    'neckline': max(points.loc[points.index[1], price_col], points.loc[points.index[3], price_col]),
                    'strength': 'Strong' if abs(prices[0] - prices[4]) / prices[0] < 0.01 else 'Moderate',
                    'target': prices[2] + (max(prices[1], prices[3]) - prices[0]) # Head height from neckline
                }
                inv_head_shoulders.append(pattern_data)
    
    # Return all detected patterns
    return {
        'head_and_shoulders': head_shoulders,
        'inverse_head_and_shoulders': inv_head_shoulders
    }


def detect_cup_and_handle(df, price_col='Close', window=5, min_cup_duration=20):
    """
    Detect Cup and Handle patterns
    
    Parameters:
        df (pd.DataFrame): OHLC price data
        price_col (str): Column name for price data
        window (int): Window size for detecting swing highs/lows
        min_cup_duration (int): Minimum number of periods for cup formation
        
    Returns:
        list: List of detected cup and handle patterns
    """
    swing_highs, swing_lows = identify_swing_highs_lows(df, window, price_col)
    
    # Cup and Handle patterns
    cup_handle_patterns = []
    
    # Need at least two swing highs and one swing low in between
    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1):
            left_high = swing_highs.iloc[i]
            right_high = swing_highs.iloc[i+1]
            
            # Find potential cup bottom (lowest point between the two highs)
            mask = (df.index > left_high.name) & (df.index < right_high.name)
            if mask.sum() < min_cup_duration:
                continue  # Cup is too small
                
            cup_section = df.loc[mask]
            if len(cup_section) == 0:
                continue
                
            cup_bottom_idx = cup_section[price_col].idxmin()
            cup_bottom = df.loc[cup_bottom_idx]
            
            # Check if it's a cup pattern: U-shaped with similar heights
            # Check height similarity
            left_height = left_high[price_col]
            right_height = right_high[price_col]
            height_diff_pct = abs(left_height - right_height) / left_height
            
            # Filter based on cup criteria
            if height_diff_pct < 0.05:  # Relatively flat rim
                # Check for handle after right high
                handle_section_mask = (df.index > right_high.name)
                handle_section = df.loc[handle_section_mask]
                
                if len(handle_section) >= 5:  # Minimum handle duration
                    # Handle should be a small pullback
                    handle_low = handle_section[price_col].min()
                    handle_low_idx = handle_section[price_col].idxmin()
                    handle_depth = (right_high[price_col] - handle_low) / right_high[price_col]
                    
                    # Handle should not be too deep and should be smaller than the cup
                    if 0.03 < handle_depth < 0.15:
                        pattern_data = {
                            'type': 'Cup and Handle',
                            'direction': 'Bullish',
                            'left_high': left_high.name,
                            'cup_bottom': cup_bottom_idx,
                            'right_high': right_high.name,
                            'handle_low': handle_low_idx,
                            'strength': 'Strong' if height_diff_pct < 0.03 else 'Moderate',
                            'target': right_high[price_col] + (right_high[price_col] - cup_bottom[price_col])
                        }
                        cup_handle_patterns.append(pattern_data)
    
    return cup_handle_patterns


def fibonacci_retracement_levels(df, is_uptrend=True):
    """
    Calculate Fibonacci retracement levels
    
    Parameters:
        df (pd.DataFrame): OHLC price data
        is_uptrend (bool): True for uptrend retracement, False for downtrend
        
    Returns:
        dict: Dictionary with Fibonacci levels
    """
    # Common Fibonacci ratios
    fib_ratios = {
        '0.0': 0.0,
        '0.236': 0.236,
        '0.382': 0.382,
        '0.5': 0.5,
        '0.618': 0.618,
        '0.786': 0.786,
        '1.0': 1.0
    }
    
    if is_uptrend:
        # For uptrend, retracement is from low to high
        low_price = df['Low'].min()
        low_idx = df['Low'].idxmin()
        
        # Find the high after the low
        high_section = df.loc[df.index > low_idx]
        if len(high_section) == 0:
            return None
            
        high_price = high_section['High'].max()
        high_idx = high_section['High'].idxmax()
        
        price_range = high_price - low_price
        
        # Calculate retracement levels
        levels = {
            'trend': 'Uptrend',
            'swing_low': (low_idx, low_price),
            'swing_high': (high_idx, high_price),
            'levels': {}
        }
        
        for ratio_name, ratio in fib_ratios.items():
            retracement_price = high_price - (price_range * ratio)
            levels['levels'][ratio_name] = retracement_price
            
    else:
        # For downtrend, retracement is from high to low
        high_price = df['High'].max()
        high_idx = df['High'].idxmax()
        
        # Find the low after the high
        low_section = df.loc[df.index > high_idx]
        if len(low_section) == 0:
            return None
            
        low_price = low_section['Low'].min()
        low_idx = low_section['Low'].idxmin()
        
        price_range = high_price - low_price
        
        # Calculate retracement levels
        levels = {
            'trend': 'Downtrend',
            'swing_high': (high_idx, high_price),
            'swing_low': (low_idx, low_price),
            'levels': {}
        }
        
        for ratio_name, ratio in fib_ratios.items():
            retracement_price = low_price + (price_range * ratio)
            levels['levels'][ratio_name] = retracement_price
            
    return levels


def calculate_volume_profile(df, price_bins=10, volume_col='Volume'):
    """
    Calculate volume profile (volume by price)
    
    Parameters:
        df (pd.DataFrame): OHLC price data with volume
        price_bins (int): Number of price bins
        volume_col (str): Column name for volume data
        
    Returns:
        tuple: (price_levels, volume_by_price)
    """
    # Check if volume data is available
    if volume_col not in df.columns or df[volume_col].isnull().all():
        return None
    
    # Calculate price range
    min_price = df['Low'].min()
    max_price = df['High'].max()
    price_range = max_price - min_price
    
    # Create price bins
    bin_size = price_range / price_bins
    price_bins_array = [min_price + (i * bin_size) for i in range(price_bins + 1)]
    
    # Create a midpoint price for each candle (approximation)
    df['price_point'] = (df['High'] + df['Low']) / 2
    
    # Count volume in each price bin
    volume_by_price = {}
    for i in range(len(price_bins_array) - 1):
        lower_bound = price_bins_array[i]
        upper_bound = price_bins_array[i + 1]
        
        # Find candles where price point falls within this bin
        mask = (df['price_point'] >= lower_bound) & (df['price_point'] < upper_bound)
        volume_in_bin = df.loc[mask, volume_col].sum()
        
        # Store midpoint of bin and volume
        bin_midpoint = (lower_bound + upper_bound) / 2
        volume_by_price[bin_midpoint] = volume_in_bin
    
    # Calculate POC (Point of Control) - price level with highest volume
    if volume_by_price:
        poc = max(volume_by_price.items(), key=lambda x: x[1])
    else:
        poc = None
    
    # Calculate Value Area (70% of volume)
    total_volume = sum(volume_by_price.values())
    value_area_target = total_volume * 0.7
    
    # Sort bins by volume in descending order
    sorted_bins = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
    
    value_area = []
    current_vol_sum = 0
    
    for price, volume in sorted_bins:
        value_area.append((price, volume))
        current_vol_sum += volume
        if current_vol_sum >= value_area_target:
            break
    
    # Get the min and max price in value area
    if value_area:
        value_area_prices = [price for price, _ in value_area]
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
    else:
        value_area_high = None
        value_area_low = None
    
    return {
        'price_bins': price_bins_array,
        'volume_by_price': volume_by_price,
        'poc': poc,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low
    }


def add_patterns_to_chart(fig, patterns, df):
    """
    Add pattern visualization to a plotly chart
    
    Parameters:
        fig (go.Figure): Plotly figure object
        patterns (dict): Dictionary with pattern data
        df (pd.DataFrame): Original price dataframe
        
    Returns:
        go.Figure: Updated figure
    """
    # Add Head and Shoulders patterns
    for pattern in patterns.get('head_and_shoulders', []):
        # Get dates and prices
        left_shoulder_date = pattern['left_shoulder']
        head_date = pattern['head']
        right_shoulder_date = pattern['right_shoulder']
        
        left_shoulder_price = df.loc[left_shoulder_date, 'Close']
        head_price = df.loc[head_date, 'Close']
        right_shoulder_price = df.loc[right_shoulder_date, 'Close']
        neckline = pattern['neckline']
        
        # Add points for the pattern
        fig.add_trace(go.Scatter(
            x=[left_shoulder_date, head_date, right_shoulder_date],
            y=[left_shoulder_price, head_price, right_shoulder_price],
            mode='markers+lines',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=10, symbol='circle', color='red'),
            name='H&S (Bearish)',
            legendgroup='patterns',
            hoverinfo='text',
            hovertext=f"Head & Shoulders<br>Target: {pattern['target']:.2f}"
        ))
        
        # Add neckline
        fig.add_shape(
            type='line',
            x0=left_shoulder_date,
            y0=neckline,
            x1=right_shoulder_date,
            y1=neckline,
            line=dict(color='red', width=2, dash='dot'),
            opacity=0.7
        )
        
    # Add Inverse Head and Shoulders patterns
    for pattern in patterns.get('inverse_head_and_shoulders', []):
        # Get dates and prices
        left_shoulder_date = pattern['left_shoulder']
        head_date = pattern['head']
        right_shoulder_date = pattern['right_shoulder']
        
        left_shoulder_price = df.loc[left_shoulder_date, 'Close']
        head_price = df.loc[head_date, 'Close']
        right_shoulder_price = df.loc[right_shoulder_date, 'Close']
        neckline = pattern['neckline']
        
        # Add points for the pattern
        fig.add_trace(go.Scatter(
            x=[left_shoulder_date, head_date, right_shoulder_date],
            y=[left_shoulder_price, head_price, right_shoulder_price],
            mode='markers+lines',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=10, symbol='circle', color='green'),
            name='Inv H&S (Bullish)',
            legendgroup='patterns',
            hoverinfo='text',
            hovertext=f"Inv. Head & Shoulders<br>Target: {pattern['target']:.2f}"
        ))
        
        # Add neckline
        fig.add_shape(
            type='line',
            x0=left_shoulder_date,
            y0=neckline,
            x1=right_shoulder_date,
            y1=neckline,
            line=dict(color='green', width=2, dash='dot'),
            opacity=0.7
        )
    
    # Add Cup and Handle patterns
    for pattern in patterns.get('cup_and_handle', []):
        left_high_date = pattern['left_high']
        cup_bottom_date = pattern['cup_bottom']
        right_high_date = pattern['right_high']
        handle_low_date = pattern['handle_low']
        
        left_high_price = df.loc[left_high_date, 'Close']
        cup_bottom_price = df.loc[cup_bottom_date, 'Close']
        right_high_price = df.loc[right_high_date, 'Close']
        handle_low_price = df.loc[handle_low_date, 'Close']
        
        # Add cup points
        fig.add_trace(go.Scatter(
            x=[left_high_date, cup_bottom_date, right_high_date, handle_low_date],
            y=[left_high_price, cup_bottom_price, right_high_price, handle_low_price],
            mode='markers+lines',
            line=dict(color='blue', width=2, dash='dash'),
            marker=dict(size=10, symbol='circle', color='blue'),
            name='Cup & Handle',
            legendgroup='patterns',
            hoverinfo='text',
            hovertext=f"Cup & Handle<br>Target: {pattern['target']:.2f}"
        ))
    
    return fig


def add_fibonacci_to_chart(fig, fib_data, df):
    """
    Add Fibonacci retracement levels to a plotly chart
    
    Parameters:
        fig (go.Figure): Plotly figure object
        fib_data (dict): Dictionary with Fibonacci levels
        df (pd.DataFrame): Original price dataframe
        
    Returns:
        go.Figure: Updated figure
    """
    if not fib_data:
        return fig
        
    # Colors for Fibonacci levels
    fib_colors = {
        '0.0': 'rgba(0, 255, 0, 0.3)',    # Green
        '0.236': 'rgba(65, 185, 255, 0.3)',  # Light blue
        '0.382': 'rgba(30, 144, 255, 0.3)',  # Dodger blue
        '0.5': 'rgba(255, 165, 0, 0.3)',   # Orange
        '0.618': 'rgba(255, 69, 0, 0.3)',   # Red-orange
        '0.786': 'rgba(255, 0, 0, 0.3)',    # Red
        '1.0': 'rgba(128, 0, 128, 0.3)'    # Purple
    }
    
    # Get swing points
    if fib_data['trend'] == 'Uptrend':
        swing_low_date, swing_low_price = fib_data['swing_low']
        swing_high_date, swing_high_price = fib_data['swing_high']
    else:
        swing_high_date, swing_high_price = fib_data['swing_high']
        swing_low_date, swing_low_price = fib_data['swing_low']
    
    # Add swing points
    fig.add_trace(go.Scatter(
        x=[swing_low_date, swing_high_date],
        y=[swing_low_price, swing_high_price],
        mode='markers+lines',
        line=dict(color='purple', width=2, dash='dash'),
        marker=dict(size=10, symbol='triangle-up', color='purple'),
        name=f'Fibonacci {fib_data["trend"]}',
        legendgroup='fibonacci'
    ))
    
    # Calculate x-axis dates for extending lines
    start_date = min(df.index)
    end_date = max(df.index)
    extend_right = True
    
    if extend_right:
        x_start = swing_low_date if fib_data['trend'] == 'Uptrend' else swing_high_date
        x_end = end_date
    else:
        x_start = swing_low_date if fib_data['trend'] == 'Uptrend' else swing_high_date
        x_end = swing_high_date if fib_data['trend'] == 'Uptrend' else swing_low_date
    
    # Add Fibonacci levels
    for level_name, price in fib_data['levels'].items():
        # Add horizontal line for the level
        fig.add_shape(
            type='line',
            x0=x_start,
            y0=price,
            x1=x_end,
            y1=price,
            line=dict(color=fib_colors[level_name], width=2),
            opacity=0.7
        )
        
        # Add annotation for the level
        fig.add_annotation(
            x=x_end,
            y=price,
            text=f"{level_name} ({price:.2f})",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            bgcolor=fib_colors[level_name].replace('0.3', '0.7'),
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        )
    
    return fig


def add_volume_profile_to_chart(fig, volume_profile, df):
    """
    Add volume profile to a plotly chart
    
    Parameters:
        fig (go.Figure): Plotly figure object
        volume_profile (dict): Dictionary with volume profile data
        df (pd.DataFrame): Original price dataframe
        
    Returns:
        go.Figure: Updated figure
    """
    if not volume_profile:
        return fig
    
    # Extract volume profile data
    volume_by_price = volume_profile['volume_by_price']
    poc = volume_profile['poc']
    value_area_high = volume_profile['value_area_high']
    value_area_low = volume_profile['value_area_low']
    
    # Normalize volume values to width for display
    max_volume = max(volume_by_price.values())
    norm_factor = max(df.index) - min(df.index)
    norm_factor = norm_factor.total_seconds() / (20 * 86400)  # 5% of the chart width
    
    x_end_date = max(df.index)
    x_base_date = max(df.index) + pd.Timedelta(days=1)  # Slightly offset for better visibility
    
    # Add volume profile bars (horizontal bars)
    for price, volume in volume_by_price.items():
        # Normalize volume for display width
        width = (volume / max_volume) * norm_factor
        
        # Create horizontal bar
        bar_color = 'rgba(0, 128, 255, 0.5)'  # Default blue
        
        # Check if this price is in the value area
        if value_area_low and value_area_high:
            if price == poc[0]:
                bar_color = 'rgba(255, 0, 0, 0.7)'  # POC is red
            elif value_area_low <= price <= value_area_high:
                bar_color = 'rgba(0, 255, 0, 0.5)'  # Value area is green
        
        # Add shape for the volume bar
        fig.add_shape(
            type='rect',
            x0=x_base_date,
            y0=price - (price * 0.001),  # Small offset for height
            x1=x_base_date + pd.Timedelta(days=width),
            y1=price + (price * 0.001),
            fillcolor=bar_color,
            line=dict(color='rgba(0,0,0,0.5)', width=1),
            opacity=0.7,
            layer='below'
        )
    
    # Add POC line (Point of Control)
    if poc:
        fig.add_shape(
            type='line',
            x0=min(df.index),
            y0=poc[0],
            x1=x_end_date,
            y1=poc[0],
            line=dict(color='red', width=2, dash='dash'),
            opacity=0.7
        )
        
        fig.add_annotation(
            x=x_base_date,
            y=poc[0],
            text=f"POC: {poc[0]:.2f}",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color='red')
        )
    
    # Add Value Area High and Low lines
    if value_area_high and value_area_low:
        # Value Area High
        fig.add_shape(
            type='line',
            x0=min(df.index),
            y0=value_area_high,
            x1=x_end_date,
            y1=value_area_high,
            line=dict(color='green', width=1, dash='dot'),
            opacity=0.7
        )
        
        fig.add_annotation(
            x=x_base_date,
            y=value_area_high,
            text=f"VAH: {value_area_high:.2f}",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color='green')
        )
        
        # Value Area Low
        fig.add_shape(
            type='line',
            x0=min(df.index),
            y0=value_area_low,
            x1=x_end_date,
            y1=value_area_low,
            line=dict(color='green', width=1, dash='dot'),
            opacity=0.7
        )
        
        fig.add_annotation(
            x=x_base_date,
            y=value_area_low,
            text=f"VAL: {value_area_low:.2f}",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color='green')
        )
    
    return fig


def get_detected_patterns(df):
    """
    Get all detected patterns from the dataframe
    
    Parameters:
        df (pd.DataFrame): OHLC price data
        
    Returns:
        dict: All detected patterns
    """
    # Detect patterns
    hs_patterns = detect_head_and_shoulders(df)
    cup_handle = detect_cup_and_handle(df)
    
    # Combine all patterns
    all_patterns = {
        'head_and_shoulders': hs_patterns['head_and_shoulders'],
        'inverse_head_and_shoulders': hs_patterns['inverse_head_and_shoulders'],
        'cup_and_handle': cup_handle
    }
    
    return all_patterns


def get_pattern_signals(patterns):
    """
    Generate trading signals from detected patterns
    
    Parameters:
        patterns (dict): Dictionary of detected patterns
        
    Returns:
        dict: Trading signals and their strengths
    """
    signals = {}
    
    # Head and Shoulders (Bearish)
    if patterns['head_and_shoulders']:
        pattern = patterns['head_and_shoulders'][0]  # Use the most recent pattern
        signals['Head_and_Shoulders'] = {
            'signal': 'SELL',
            'strength': pattern['strength'],
            'value': f"Target: {pattern['target']:.2f}"
        }
    
    # Inverse Head and Shoulders (Bullish)
    if patterns['inverse_head_and_shoulders']:
        pattern = patterns['inverse_head_and_shoulders'][0]  # Use the most recent pattern
        signals['Inverse_Head_and_Shoulders'] = {
            'signal': 'BUY',
            'strength': pattern['strength'],
            'value': f"Target: {pattern['target']:.2f}"
        }
    
    # Cup and Handle (Bullish)
    if patterns['cup_and_handle']:
        pattern = patterns['cup_and_handle'][0]  # Use the most recent pattern
        signals['Cup_and_Handle'] = {
            'signal': 'BUY',
            'strength': pattern['strength'],
            'value': f"Target: {pattern['target']:.2f}"
        }
    
    return signals


def get_pattern_explanations():
    """
    Get explanations for various chart patterns with trend interpretations
    
    Returns:
        dict: Dictionary of pattern explanations
    """
    return {
        'Head_and_Shoulders': {
            'description': 'The Head and Shoulders pattern is a reversal pattern that signals a trend change from bullish to bearish.',
            'trends': {
                'Bearish': 'Forms at the end of an uptrend with three peaks (left shoulder, head, right shoulder). The head is the highest peak with shoulders at similar heights. A breakdown below the neckline confirms the pattern.',
                'Volume Pattern': 'Volume typically decreases at each peak and increases on the neckline breakdown.',
                'Target': 'The target is calculated by measuring the height from the head to the neckline and projecting that distance down from the neckline breakout point.'
            }
        },
        'Inverse_Head_and_Shoulders': {
            'description': 'The Inverse Head and Shoulders pattern is a reversal pattern that signals a trend change from bearish to bullish.',
            'trends': {
                'Bullish': 'Forms at the end of a downtrend with three troughs (left shoulder, head, right shoulder). The head is the lowest trough with shoulders at similar heights. A breakout above the neckline confirms the pattern.',
                'Volume Pattern': 'Volume typically decreases at each trough and increases on the neckline breakout.',
                'Target': 'The target is calculated by measuring the height from the head to the neckline and projecting that distance up from the neckline breakout point.'
            }
        },
        'Cup_and_Handle': {
            'description': 'The Cup and Handle pattern is a bullish continuation pattern that signals a pause in an uptrend followed by its continuation.',
            'trends': {
                'Bullish': 'Forms during an uptrend with a U-shaped cup followed by a slight downward drift (handle). The pattern is confirmed when price breaks above the cup\'s rim.',
                'Volume Pattern': 'Volume typically decreases during the cup formation and increases during the handle formation and breakout.',
                'Target': 'The target is calculated by measuring the depth of the cup and projecting that distance up from the cup\'s rim at the breakout point.'
            }
        },
        'Fibonacci_Retracement': {
            'description': 'Fibonacci Retracement levels identify potential support/resistance levels where a price trend might reverse.',
            'trends': {
                'Uptrend': 'In an uptrend, retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) act as potential support zones where price might bounce back to resume the uptrend.',
                'Downtrend': 'In a downtrend, retracement levels act as potential resistance zones where price might reverse to continue the downtrend.',
                'Key Levels': '38.2% and 61.8% are considered the most significant Fibonacci levels. The 50% level, while not a Fibonacci ratio, is also important psychologically.'
            }
        },
        'Volume_Profile': {
            'description': 'Volume Profile shows the distribution of volume across price levels, highlighting where the most trading activity occurred.',
            'trends': {
                'Point of Control (POC)': 'The price level with the highest trading volume, often acting as a strong support/resistance level.',
                'Value Area': 'The price range containing 70% of the total volume, indicating where most of the trading activity occurred.',
                'Value Area High/Low': 'The upper and lower boundaries of the value area, often serving as support/resistance levels.',
                'Low Volume Nodes': 'Price areas with minimal trading activity where price may move quickly through with less resistance.'
            }
        }
    }