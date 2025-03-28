
import requests
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import pandas as pd

def get_google_trends_data(keyword, timeframe='today 3-m'):
    """Fetch Google Trends data for a stock/company"""
    try:
        pytrends = TrendReq(hl='en-US', tz=330)  # Indian timezone
        kw_list = [keyword]
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='IN')
        interest_over_time_df = pytrends.interest_over_time()
        
        if not interest_over_time_df.empty:
            # Normalize data
            interest_over_time_df = interest_over_time_df.drop(columns=['isPartial'])
            return interest_over_time_df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching Google Trends data: {e}")
        return pd.DataFrame()

def get_stocktwits_sentiment(symbol):
    """Fetch StockTwits sentiment data"""
    try:
        # Remove exchange suffix for StockTwits
        base_symbol = symbol.split('.')[0]
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{base_symbol}.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            messages = data.get('messages', [])
            
            # Process sentiment
            sentiment_counts = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0}
            total_messages = len(messages)
            
            for message in messages:
                if 'entities' in message and 'sentiment' in message['entities']:
                    sentiment = message['entities']['sentiment']['basic']
                    if sentiment:
                        sentiment_counts[sentiment] += 1
                else:
                    sentiment_counts['Neutral'] += 1
            
            # Calculate sentiment metrics
            sentiment_metrics = {
                'bullish_ratio': sentiment_counts['Bullish'] / total_messages if total_messages > 0 else 0,
                'bearish_ratio': sentiment_counts['Bearish'] / total_messages if total_messages > 0 else 0,
                'message_volume': total_messages,
                'sentiment_counts': sentiment_counts
            }
            
            return sentiment_metrics
        return None
    except Exception as e:
        print(f"Error fetching StockTwits data: {e}")
        return None
