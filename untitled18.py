#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 01:32:21 2025

@author: alexandru-cristiancalin
"""

import wrds
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. Data Acquisition
def get_stock_data(ticker_symbol='AAPL', start_date='2018-01-01', end_date='2022-12-31'):
    try:
        conn = wrds.Connection()
        permno_query = f"""
            SELECT permno 
            FROM crsp.dsenames 
            WHERE ticker = '{ticker_symbol}' 
            AND namedt <= '{end_date}'
            AND nameendt >= '{start_date}'
            ORDER BY namedt DESC 
            LIMIT 1
        """
        permno_result = conn.raw_sql(permno_query)
        if permno_result.empty:
            if ticker_symbol == 'AAPL':
                permno = 14593
            else:
                return None
        else:
            permno = permno_result.iloc[0]['permno']
        
        query = f"""
            SELECT a.permno, a.date, a.prc, a.ret, a.vol
            FROM crsp.dsf a
            WHERE a.permno = {permno}
              AND a.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY a.date ASC
        """
        stock_data = conn.raw_sql(query, date_cols=['date'])
        if not stock_data.empty:
            market_query = f"""
                SELECT date, vwretd as market_return
                FROM crsp.dsi
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date ASC
            """
            market_data = conn.raw_sql(market_query, date_cols=['date'])
            stock_data = stock_data.merge(market_data, on='date', how='left')
        
        conn.close()
        if stock_data.empty:
            return None
        
        stock_data['prc'] = stock_data['prc'].abs()
        stock_data['close'] = stock_data['prc']
        stock_data['return'] = stock_data['ret']
        stock_data.set_index('date', inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return None

def create_sample_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(date_range)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, n)
    price = 100 * (1 + returns).cumprod()
    return pd.DataFrame({
        'permno': [14593] * n,
        'prc': price,
        'close': price,
        'ret': returns,
        'return': returns,
        'vol': np.random.lognormal(15, 1, n),
        'market_return': np.random.normal(0.0004, 0.01, n)
    }, index=date_range)

# 2. Feature Engineering
def engineer_features(df):
    data = df.copy()
    
    # Existing features
    data['price_change'] = data['close'].pct_change()
    data['price_5d_mean'] = data['close'].rolling(window=5).mean()
    data['price_20d_mean'] = data['close'].rolling(window=20).mean()
    data['price_5d_std'] = data['close'].rolling(window=5).std()
    data['price_20d_std'] = data['close'].rolling(window=20).std()
    data['return_5d_mean'] = data['return'].rolling(window=5).mean()
    data['return_20d_mean'] = data['return'].rolling(window=20).mean()
    data['return_5d_std'] = data['return'].rolling(window=5).std()
    data['return_20d_std'] = data['return'].rolling(window=20).std()
    data['vol_change'] = data['vol'].pct_change()
    data['vol_5d_mean'] = data['vol'].rolling(window=5).mean()
    data['vol_20d_mean'] = data['vol'].rolling(window=20).mean()
    data['sma_ratio'] = data['price_5d_mean'] / data['price_20d_mean']
    if 'market_return' in data.columns:
        data['excess_return'] = data['return'] - data['market_return']
        data['excess_return_5d_mean'] = data['excess_return'].rolling(window=5).mean()
        data['market_return_5d_mean'] = data['market_return'].rolling(window=5).mean()
        # Calculate beta
        window = 252  # 1-year window
        betas = []
        for i in range(len(data)):
            if i >= window:
                stock_ret = data['return'].iloc[i-window:i]
                mkt_ret = data['market_return'].iloc[i-window:i]
                beta = linregress(mkt_ret, stock_ret).slope
                betas.append(beta)
            else:
                betas.append(np.nan)
        data['beta'] = betas
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # Lagged returns
    for lag in range(1, 6):
        data[f'return_lag{lag}'] = data['return'].shift(lag)
    
    # New features
    # Bollinger Bands
    data['bb_upper'] = data['price_20d_mean'] + (2 * data['price_20d_std'])
    data['bb_lower'] = data['price_20d_mean'] - (2 * data['price_20d_std'])
    data['bb_distance'] = (data['close'] - data['price_20d_mean']) / data['price_20d_std']
    
    # EMAs
    data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['ema_30'] = data['close'].ewm(span=30, adjust=False).mean()
    
    # Momentum
    data['momentum_10'] = data['close'] - data['close'].shift(10)
    
    # Target
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    data = data.dropna()
    
    feature_columns = [
        'price_change', 'price_5d_mean', 'price_20d_mean', 'price_5d_std', 'price_20d_std',
        'return_5d_mean', 'return_20d_mean', 'return_5d_std', 'return_20d_std',
        'vol_change', 'vol_5d_mean', 'vol_20d_mean', 'sma_ratio', 'rsi_14',
        'macd', 'macd_signal', 'macd_histogram',
        'return_lag1', 'return_lag2', 'return_lag3', 'return_lag4', 'return_lag5',
        'bb_distance', 'ema_10', 'ema_30', 'momentum_10'
    ]
    if 'excess_return' in data.columns:
        feature_columns.extend(['excess_return', 'excess_return_5d_mean', 'market_return_5d_mean', 'beta'])
    
    return data, feature_columns

# 3. Train and Evaluate Model with Tuning
def train_evaluate_model(train_data, test_data, features):
    X_train = train_data[features]
    y_train = train_data['target']
    X_test = test_data[features]
    y_test = test_data['target']
    
    # Grid search for XGBoost
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    base_model = XGBClassifier(random_state=42, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, metrics, feature_importance, y_pred, y_prob

# 4. Implement Trading Strategy with Threshold Optimization
def implement_strategy(test_data, predictions, probabilities):
    strategy_data = test_data.copy()
    strategy_data['prediction'] = predictions
    strategy_data['probability'] = probabilities
    
    thresholds = [0.5, 0.55, 0.6]
    best_sharpe = -np.inf
    best_strategy_data = None
    best_performance = None
    
    for threshold in thresholds:
        temp_data = strategy_data.copy()
        temp_data['signal'] = (temp_data['probability'] > threshold).astype(int)
        temp_data['position'] = temp_data['signal'].shift(1).fillna(0)
        temp_data['strategy_return'] = temp_data['return'] * temp_data['position']
        temp_data['position_change'] = temp_data['position'].diff().fillna(0)
        temp_data['transaction_cost'] = abs(temp_data['position_change']) * 0.001
        temp_data['strategy_return'] -= temp_data['transaction_cost']
        temp_data['strategy_cumulative'] = (1 + temp_data['strategy_return']).cumprod()
        temp_data['buy_hold_cumulative'] = (1 + temp_data['return']).cumprod()
        
        total_return = temp_data['strategy_cumulative'].iloc[-1] - 1
        annualized_return = ((1 + total_return) ** (252 / len(temp_data)) - 1)
        sharpe_ratio = temp_data['strategy_return'].mean() / temp_data['strategy_return'].std() * np.sqrt(252)
        bh_return = temp_data['buy_hold_cumulative'].iloc[-1] - 1
        bh_annual = ((1 + bh_return) ** (252 / len(temp_data)) - 1)
        bh_sharpe = temp_data['return'].mean() / temp_data['return'].std() * np.sqrt(252)
        trades = temp_data[temp_data['position'] != 0]
        win_rate = (trades['strategy_return'] > 0).mean() if not trades.empty else 0
        
        performance = {
            'Threshold': threshold,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Win Rate': win_rate,
            'Buy & Hold Return': bh_return,
            'Buy & Hold Annual': bh_annual,
            'Buy & Hold Sharpe': bh_sharpe
        }
        
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_strategy_data = temp_data
            best_performance = performance
    
    return best_strategy_data, best_performance

# 5. Walk-Forward Validation (Optional)
def walk_forward_validation(data, features, test_size=0.1):
    results = []
    start = int(len(data) * 0.7)
    end = len(data)
    step = int(len(data) * test_size)
    
    while start + step < end:
        train_data = data.iloc[:start]
        test_data = data.iloc[start:start + step]
        model, metrics, _, y_pred, y_prob = train_evaluate_model(train_data, test_data, features)
        strategy_data, performance = implement_strategy(test_data, y_pred, y_prob)
        results.append({'metrics': metrics, 'performance': performance})
        start += step
    
    return results

# 6. Run Full Analysis
def run_analysis(ticker='AAPL', start_date='2018-01-01', end_date='2022-12-31', test_split=0.7, use_walk_forward=False):
    print(f"Running analysis for {ticker} from {start_date} to {end_date}")
    
    data = get_stock_data(ticker, start_date, end_date)
    if data is None:
        print("Using sample data instead")
        data = create_sample_data(start_date, end_date)
    
    data, features = engineer_features(data)
    print(f"Data shape after feature engineering: {data.shape}")
    print(f"Features: {features}")
    
    if use_walk_forward:
        results = walk_forward_validation(data, features)
        print("\nWalk-Forward Validation Results:")
        for i, result in enumerate(results):
            print(f"\nFold {i+1}:")
            for metric, value in result['metrics'].items():
                print(f"{metric}: {value:.4f}")
            for metric, value in result['performance'].items():
                print(f"{metric}: {value:.4f}")
        return results
    
    split_idx = int(len(data) * test_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    print(f"Train data: {train_data.shape}, Test data: {test_data.shape}")
    
    model, metrics, importance, predictions, probabilities = train_evaluate_model(train_data, test_data, features)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTop 5 Important Features:")
    print(importance.head(5))
    
    strategy_data, performance = implement_strategy(test_data, predictions, probabilities)
    
    print("\nTrading Strategy Performance (Best Threshold):")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(strategy_data['strategy_cumulative'], label='Strategy')
    plt.plot(strategy_data['buy_hold_cumulative'], label='Buy & Hold')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    top_features = importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title('Top 10 Feature Importances')
    
    plt.subplot(2, 2, 3)
    plt.plot(strategy_data['position'])
    plt.title('Trading Positions (1=Long, 0=Cash)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.scatter(strategy_data.index, strategy_data['target'], alpha=0.5, label='Actual', c='blue')
    plt.scatter(strategy_data.index, strategy_data['prediction'], alpha=0.5, label='Predicted', c='red')
    plt.title('Model Predictions vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'data': data,
        'model': model,
        'metrics': metrics,
        'importance': importance,
        'strategy_data': strategy_data,
        'performance': performance
    }

if __name__ == "__main__":
    # Run standard analysis with XGBoost
    results = run_analysis('AAPL', '2018-01-01', '2022-12-31', use_walk_forward=False)
    # Optional: Run with walk-forward validation
    # results_wf = run_analysis('AAPL', '2018-01-01', '2022-12-31', use_walk_forward=True), 