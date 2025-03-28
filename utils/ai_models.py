import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
import streamlit as st

def prepare_features(data):
    """
    Prepare features for the ML model
    
    Parameters:
        data (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        tuple: (X, y, X_latest) - features, target, and latest features
    """
    try:
        # Ensure data is cleaned and we have a copy to avoid modifying the original
        data = data.copy().dropna()
        
        if len(data) <= 20:  # Need sufficient data
            return None, None, None
            
        # Define features - ordered by importance
        feature_columns = [
            'Close', 'SMA_20', 'EMA_20', 'MACD', 'RSI', 
            'Volatility', 'Price_to_SMA', 'Upper_BB', 'Lower_BB', 
            'ATR', 'Volume'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        if len(available_features) < 3:  # Need sufficient features
            st.warning(f"Not enough features available for AI recommendation. Found: {available_features}")
            return None, None, None
            
        # Create features and target
        X = data[available_features].copy()
        
        # Check for infinite or NaN values and replace them
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(method='ffill', inplace=True)  # Forward fill remaining NaNs
        X.fillna(0, inplace=True)  # Fill any remaining NaNs with 0
        
        # Future returns (shifted by -1 to get next day's return)
        data['future_return'] = data['Close'].pct_change(-1)
        y = (data['future_return'] > 0).astype(int)  # 1 if price goes up, 0 if down
        
        # Keep the latest data point for prediction
        X_latest = X.iloc[-1:].copy()
        
        # Remove the last row from training data as we don't have target for it
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        # Final check for data validity
        if len(X) < 10 or len(y) < 10:
            st.warning(f"Not enough valid data points for training: {len(X)} features, {len(y)} targets")
            return None, None, None
            
        # Check that we don't have all the same target values
        if len(y.unique()) < 2:
            st.warning("Target data lacks variation (all same values), can't train a meaningful model")
            return None, None, None
            
        return X, y, X_latest
        
    except Exception as prep_error:
        st.error(f"Error preparing features: {str(prep_error)}")
        return None, None, None

@st.cache_resource(ttl=60*30)  # Cache the model for 30 minutes
def train_ml_model(X, y):
    """
    Train an XGBoost model with time series cross-validation
    
    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        
    Returns:
        tuple: (model, scaler, accuracy)
    """
    if X is None or y is None or len(X) < 10:
        return None, None, 0
        
    try:
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3
        }
        
        # Train with cross-validation
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            # For newer versions of XGBoost, early_stopping_rounds is passed to the constructor
            # or we need to use callbacks instead
            try:
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            except Exception as fit_error:
                # Fallback approach without eval_set if there's an error
                model.fit(X_train, y_train)
            
            accuracies.append(model.score(X_val, y_val))
        
        # Final model on full dataset
        final_model = xgb.XGBClassifier(**params)
        try:
            # Try to fit the model without early stopping or eval set
            final_model.fit(X_scaled, y)
        except Exception as final_fit_error:
            st.warning(f"Alternative model training approach used due to: {str(final_fit_error)}")
            # Simplify parameters if needed
            simple_params = {
                'objective': 'binary:logistic',
                'max_depth': 3,
                'n_estimators': 100
            }
            final_model = xgb.XGBClassifier(**simple_params)
            final_model.fit(X_scaled, y)
        
        return final_model, scaler, np.mean(accuracies)
        
    except Exception as e:
        st.error(f"Error training ML model: {str(e)}")
        return None, None, 0

def get_ai_recommendation(data):
    """
    Get AI-based recommendation for the stock
    
    Parameters:
        data (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: AI recommendation details
    """
    result = {
        'prediction': 'NEUTRAL',
        'confidence': 0.0,
        'accuracy': 0.0,
        'features_used': []
    }
    
    # Prepare data
    X, y, X_latest = prepare_features(data)
    
    if X is None or len(X) < 10:
        return result
        
    # Record features used
    result['features_used'] = list(X.columns)
    
    # Train model
    model, scaler, accuracy = train_ml_model(X, y)
    
    if model is None:
        return result
        
    # Make prediction with error handling
    try:
        X_latest_scaled = scaler.transform(X_latest)
        prediction_prob = model.predict_proba(X_latest_scaled)[0]
        prediction = model.predict(X_latest_scaled)[0]
        
        # Convert to recommendation
        result['accuracy'] = accuracy
        result['confidence'] = max(prediction_prob)
        result['prediction'] = "BUY" if prediction == 1 else "SELL"
    except Exception as pred_error:
        st.error(f"Error during prediction: {str(pred_error)}")
        # Provide a fallback prediction
        result['accuracy'] = accuracy
        result['confidence'] = 0.5
        result['prediction'] = "NEUTRAL"
    
    return result
