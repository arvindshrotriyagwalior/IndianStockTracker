import yfinance as yf
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json

def get_financial_health_data(symbol):
    """Get comprehensive financial health data from multiple sources"""
    try:
        # Initialize data dictionary
        info = {}
        
        # 1. YFinance data
        stock = yf.Ticker(symbol)
        info.update(stock.info)
        
        # 2. Screener.in data
        try:
            # Convert symbol format (e.g., RELIANCE.NS -> RELIANCE)
            screener_symbol = symbol.split('.')[0]
            screener_url = f"https://www.screener.in/company/{screener_symbol}/consolidated/"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(screener_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract ratios from the top card
                ratio_cards = soup.find_all('div', class_='flex flex-wrap text-small')
                for ratio_card in ratio_cards:
                    try:
                        ratios = ratio_card.find_all('div', class_='flex')
                        for ratio in ratios:
                            label_elem = ratio.find('span', class_='name')
                            value_elem = ratio.find('span', class_='number')
                            if label_elem and value_elem:
                                label = label_elem.text.strip()
                                value_text = value_elem.text.strip()
                                
                                # Try to convert to number (removing ₹, %, Cr, etc.)
                                value = value_text
                                try:
                                    # Remove currency symbols, commas, and percentage signs
                                    clean_value = re.sub(r'[₹,%]', '', value_text)
                                    # Convert crores to actual value
                                    if 'Cr' in value_text:
                                        clean_value = clean_value.replace('Cr', '')
                                        clean_value = float(clean_value) * 10000000  # 1 Cr = 10^7
                                    else:
                                        clean_value = float(clean_value)
                                    value = clean_value
                                except:
                                    pass
                                
                                # Normalize the label
                                norm_label = label.lower().replace(' ', '_').replace('-', '_')
                                info[f'screener_{norm_label}'] = value
                    except Exception as inner_e:
                        print(f"Error processing ratio card: {str(inner_e)}")
                        continue
                
                # Try to extract data from script tags (JSON data)
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string and "window.App" in script.string:
                        try:
                            # Extract JSON data from the script
                            json_str = re.search(r'window\.App=(.*?);', script.string, re.DOTALL)
                            if json_str:
                                json_data = json.loads(json_str.group(1))
                                if 'company' in json_data and 'ratios' in json_data['company']:
                                    ratios_data = json_data['company']['ratios']
                                    for ratio_data in ratios_data:
                                        if 'name' in ratio_data and 'value' in ratio_data:
                                            name = ratio_data['name'].lower().replace(' ', '_')
                                            value = ratio_data['value']
                                            # Store as 'screener_ratio_name'
                                            info[f'screener_ratio_{name}'] = value
                        except Exception as json_e:
                            print(f"Error extracting JSON data: {str(json_e)}")
                
                # Extract key financial ratios
                ratio_tables = soup.find_all('table', class_='data-table')
                for table in ratio_tables:
                    table_header = table.find_previous('h2')
                    if table_header and 'Ratios' in table_header.text:
                        rows = table.find_all('tr')
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) >= 2:
                                ratio_name = cols[0].text.strip().lower().replace(' ', '_')
                                ratio_value = cols[1].text.strip()
                                
                                # Try to convert numbers
                                try:
                                    if '%' in ratio_value:
                                        ratio_value = float(ratio_value.replace('%', '')) / 100
                                    elif ratio_value and ratio_value != '-':
                                        ratio_value = float(ratio_value)
                                except:
                                    pass
                                
                                info[f'screener_ratio_{ratio_name}'] = ratio_value
                
                # Extract quarterly results
                quarters_table = soup.find('table', class_='quarters-table')
                if quarters_table:
                    rows = quarters_table.find_all('tr')
                    if len(rows) > 1:  # Ensure we have data rows
                        header_row = rows[0].find_all('th')
                        data_row = rows[1].find_all('td')
                        
                        if len(header_row) > 0 and len(data_row) > 0:
                            # Get the headers
                            headers = [th.text.strip() for th in header_row]
                            # Get the data from the first data row
                            values = [td.text.strip() for td in data_row]
                            
                            # Combine headers and values
                            for i in range(min(len(headers), len(values))):
                                if i > 0:  # Skip the first column (which is often just a label)
                                    info[f'screener_quarterly_{headers[i].lower().replace(" ", "_")}'] = values[i]
                
                # Map the screener data to our financial health keys for redundancy
                # Current Ratio
                if 'screener_current_ratio' in info:
                    try:
                        current_ratio_val = info['screener_current_ratio']
                        if isinstance(current_ratio_val, str):
                            current_ratio_val = float(current_ratio_val.replace('x', '').strip())
                        info['currentRatio'] = current_ratio_val
                    except:
                        pass
                
                # Quick Ratio
                if 'screener_quick_ratio' in info:
                    try:
                        quick_ratio_val = info['screener_quick_ratio']
                        if isinstance(quick_ratio_val, str):
                            quick_ratio_val = float(quick_ratio_val.replace('x', '').strip())
                        info['quickRatio'] = quick_ratio_val
                    except:
                        pass
                
                # Debt to Equity
                if 'screener_debt_to_equity' in info:
                    try:
                        debt_equity_val = info['screener_debt_to_equity']
                        if isinstance(debt_equity_val, str):
                            debt_equity_val = float(debt_equity_val.replace('x', '').strip())
                        info['debtToEquity'] = debt_equity_val
                    except:
                        pass
                
                # Return on Equity
                if 'screener_return_on_equity' in info or 'screener_roe' in info:
                    try:
                        roe_val = info.get('screener_return_on_equity', info.get('screener_roe'))
                        if isinstance(roe_val, str):
                            roe_val = float(roe_val.replace('%', '').strip()) / 100
                        info['returnOnEquity'] = roe_val
                    except:
                        pass
                
                # Return on Assets
                if 'screener_return_on_assets' in info or 'screener_roa' in info:
                    try:
                        roa_val = info.get('screener_return_on_assets', info.get('screener_roa'))
                        if isinstance(roa_val, str):
                            roa_val = float(roa_val.replace('%', '').strip()) / 100
                        info['returnOnAssets'] = roa_val
                    except:
                        pass
                
                # Operating Margin
                if 'screener_operating_margin' in info:
                    try:
                        op_margin_val = info['screener_operating_margin']
                        if isinstance(op_margin_val, str):
                            op_margin_val = float(op_margin_val.replace('%', '').strip()) / 100
                        info['operatingMargin'] = op_margin_val
                    except:
                        pass
                
                # Profit Margin
                if 'screener_profit_margin' in info or 'screener_net_profit_margin' in info:
                    try:
                        profit_margin_val = info.get('screener_profit_margin', info.get('screener_net_profit_margin'))
                        if isinstance(profit_margin_val, str):
                            profit_margin_val = float(profit_margin_val.replace('%', '').strip()) / 100
                        info['profitMargin'] = profit_margin_val
                    except:
                        pass
                
                # Working Capital
                if 'screener_working_capital' in info:
                    try:
                        working_capital_val = info['screener_working_capital']
                        if isinstance(working_capital_val, str):
                            # Handle crore formatting
                            if 'Cr' in working_capital_val:
                                working_capital_val = float(working_capital_val.replace('Cr', '').replace(',', '').strip()) * 10000000
                            else:
                                working_capital_val = float(working_capital_val.replace(',', '').strip())
                        info['workingCapital'] = working_capital_val
                    except:
                        pass
                            
        except Exception as e:
            print(f"Error fetching Screener.in data: {str(e)}")
            
        # 3. NSE India website data
        try:
            nse_symbol = symbol.replace('.NS', '')
            nse_url = f"https://www.nseindia.com/get-quotes/equity?symbol={nse_symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            session = requests.Session()
            response = session.get(nse_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract relevant data
                market_data = soup.find('div', {'id': 'equity_marketDeptOrder'})
                if market_data:
                    info['nse_delivery_quantity'] = market_data.get('deliveryQuantity', 'N/A')
                    info['nse_delivery_percent'] = market_data.get('deliveryToTradedQuantity', 'N/A')
                    
        except Exception as e:
            print(f"Error fetching NSE data: {str(e)}")
            
        # 4. BSE India website data (for BSE listed stocks)
        if '.BO' in symbol:
            try:
                bse_symbol = symbol.replace('.BO', '')
                bse_url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={bse_symbol}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(bse_url, headers=headers)
                
                if response.status_code == 200:
                    bse_data = response.json()
                    info['bse_market_cap'] = bse_data.get('Mktcap', 'N/A')
                    info['bse_face_value'] = bse_data.get('FaceValue', 'N/A')
                    
            except Exception as e:
                print(f"Error fetching BSE data: {str(e)}")
        
        # Get balance sheet data
        balance_sheet = stock.balance_sheet
        
        # Get income statement
        income_stmt = stock.income_stmt
        
        # Get cash flow statement
        cash_flow = stock.cashflow
        
        # Calculate additional financial ratios with safe type checking
        financial_data = {}
        
        def safe_divide(numerator, denominator):
            try:
                if numerator is not None and denominator is not None and denominator != 0:
                    return float(numerator) / float(denominator)
                return None
            except (ValueError, TypeError):
                return None

        # Current Ratio
        total_current_assets = info.get('totalCurrentAssets', info.get('currentAssets'))
        total_current_liabilities = info.get('totalCurrentLiabilities', info.get('currentLiabilities'))
        financial_data['currentRatio'] = safe_divide(total_current_assets, total_current_liabilities)

        # Quick Ratio
        inventory = info.get('inventory', info.get('totalInventory', 0))
        if total_current_assets and total_current_liabilities:
            financial_data['quickRatio'] = safe_divide(
                (float(total_current_assets) - float(inventory)), 
                float(total_current_liabilities)
            )

        # Debt to Equity
        total_debt = info.get('totalDebt', info.get('debt'))
        stockholder_equity = info.get('totalStockholderEquity', info.get('stockholderEquity'))
        financial_data['debtToEquity'] = safe_divide(total_debt, stockholder_equity)

        # Return on Assets (ROA)
        net_income = info.get('netIncome', info.get('netIncomeToCommon'))
        total_assets = info.get('totalAssets', info.get('assets'))
        financial_data['returnOnAssets'] = safe_divide(net_income, total_assets)

        # Additional ratios
        financial_data['returnOnEquity'] = info.get('returnOnEquity', safe_divide(net_income, stockholder_equity))
        financial_data['profitMargin'] = info.get('profitMargin', info.get('profitMargins'))
        financial_data['operatingMarginTTM'] = info.get('operatingMarginTTM', info.get('operatingMargins'))
            
        # Additional metrics from financial statements
        if not balance_sheet.empty:
            latest_bs = balance_sheet.iloc[:, 0]
            # Fix for NoneType subtraction issue
            total_current_assets = latest_bs.get('Total Current Assets', None)
            total_current_liabilities = latest_bs.get('Total Current Liabilities', None)
            if total_current_assets is not None and total_current_liabilities is not None:
                financial_data['workingCapital'] = total_current_assets - total_current_liabilities
            
        if not income_stmt.empty:
            latest_is = income_stmt.iloc[:, 0]
            # Add similar protection for division by zero
            operating_income = latest_is.get('Operating Income', None)
            total_revenue = latest_is.get('Total Revenue', None)
            if operating_income is not None and total_revenue is not None and total_revenue != 0:
                financial_data['operatingMargin'] = operating_income / total_revenue
            
        # Combine all data
        info.update(financial_data)
        
        return info
        
    except Exception as e:
        st.error(f"Error fetching financial health data: {str(e)}")
        return {}

def get_stock_data(symbol, timeframe="1d"):
    """
    Fetch stock data for the given symbol and timeframe
    
    Parameters:
        symbol (str): Stock symbol (e.g., RELIANCE.NS)
        timeframe (str): Time period for data (e.g., 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        tuple: (stock object, historical data DataFrame, info dict)
    """
    try:
        # Cache only the historical data and info, not the Ticker object
        @st.cache_data(ttl=60*5)  # Cache for 5 minutes
        def fetch_data_cacheable(sym, tf):
            stock = yf.Ticker(sym)
            hist = stock.history(period=tf)
            info = stock.info
            return hist, info
        
        # Get the cacheable data
        hist, info = fetch_data_cacheable(symbol, timeframe)
        
        # Create a fresh Ticker object (not cached)
        stock = yf.Ticker(symbol)
        
        # Check if data is valid
        if hist.empty:
            return None, pd.DataFrame(), {}
            
        return stock, hist, info
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, pd.DataFrame(), {}

def validate_stock_symbol(symbol):
    """
    Validate if the given symbol is a valid stock symbol
    
    Parameters:
        symbol (str): Stock symbol to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not symbol:
        return False
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Check if info contains essential fields
        if 'symbol' in info and info['symbol'] == symbol:
            return True
            
        # For Indian stocks, the symbol in the info may not match exactly
        # So we check if we got any price data
        hist = stock.history(period="1d")
        if not hist.empty:
            return True
            
        return False
    
    except:
        return False

def get_common_indian_stocks():
    """
    Return a list of common Indian stock symbols for suggestions
    
    Returns:
        list: List of common Indian stock symbols with exchanges
    """
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS", "BHARTIARTL.NS",
        "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "TATASTEEL.BO", "WIPRO.BO", "SUNPHARMA.BO", "ONGC.BO", "NTPC.BO"
    ]
