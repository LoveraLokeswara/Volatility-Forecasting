# ============================================================================
# SECTION 4.5 & 4.6: LONG-SHORT TRADING STRATEGY AND PERFORMANCE METRICS
# ADD THIS CODE TO THE END OF YOUR JUPYTER NOTEBOOK
# ============================================================================

# ============================================================================
# SECTION 4.5: LONG-SHORT TRADING STRATEGY IMPLEMENTATION
# ============================================================================

def generate_trading_signals(predictions, transaction_costs=0.005/100):
    """
    Generate Long-Short trading signals based on predicted returns.
    
    Signal generation rule (Equation 21 from paper):
        Signal = 1    if predicted_return > transaction_costs
        Signal = -1   if predicted_return < -transaction_costs
        Signal = 0    if |predicted_return| <= transaction_costs
    """
    signals = np.zeros(len(predictions))
    
    for i in range(len(predictions)):
        if predictions[i] > transaction_costs:
            signals[i] = 1      # Long position
        elif predictions[i] < -transaction_costs:
            signals[i] = -1     # Short position
        else:
            signals[i] = 0      # No position
    
    return signals


def generate_trading_signals_long_only(predictions, transaction_costs=0.005/100):
    """
    Generate Long Only trading signals based on predicted returns.
    
    In Long Only strategy:
        Signal = 1    if predicted_return > transaction_costs (enter long)
        Signal = 0    if predicted_return < 0 and in position (exit)
        Signal = 1    if already in position (hold)
        Signal = 0    otherwise (no position)
    """
    signals = np.zeros(len(predictions))
    in_position = False
    
    for i in range(len(predictions)):
        if predictions[i] > transaction_costs and not in_position:
            signals[i] = 1
            in_position = True
        elif predictions[i] < 0 and in_position:
            signals[i] = 0
            in_position = False
        elif in_position:
            signals[i] = 1
        else:
            signals[i] = 0
    
    return signals


def backtest_strategy(prices, signals, transaction_costs=0.005/100, trading_days=252):
    """
    Perform backtest of a trading strategy using normalized Wealth Index.
    
    Starts with wealth_index = 1.0 (no arbitrary initial capital).
    Returns the growth factor and daily returns.
    
    Parameters:
    -----------
    prices : pandas.Series or ndarray
        Asset prices during the test period
    signals : ndarray
        Trading signals (1: long, -1: short, 0: no position)
    transaction_costs : float
        Transaction costs as percentage (e.g., 0.005/100 for 0.005%)
    trading_days : int
        Trading days per year (252 for stocks, 365 for crypto)
    
    Returns:
    --------
    wealth_index : ndarray
        Portfolio value growth (starting from 1.0)
    strategy_returns : ndarray
        Daily returns from the strategy
    """
    # Convert to numpy if pandas Series
    if isinstance(prices, pd.Series):
        prices_array = prices.values
    else:
        prices_array = prices
    
    # Calculate log returns
    log_returns = np.log(prices_array[1:] / prices_array[:-1])
    
    # Initialize Wealth Index at 1.0 (normalized, no dollar amount)
    wealth_index = np.zeros(len(signals))
    wealth_index[0] = 1.0
    
    strategy_returns = np.zeros(len(signals) - 1)
    
    # Backtest loop
    for t in range(len(signals) - 1):
        position = signals[t]
        daily_return = position * log_returns[t]
        
        # Subtract transaction costs when position changes
        if t > 0 and signals[t] != signals[t-1]:
            daily_return -= abs(signals[t] - signals[t-1]) * transaction_costs
        
        strategy_returns[t] = daily_return
        wealth_index[t + 1] = wealth_index[t] * np.exp(daily_return)
    
    return wealth_index, strategy_returns


# ============================================================================
# SECTION 4.6: TRADING PERFORMANCE METRICS CALCULATION
# ============================================================================

def calculate_annualized_return(wealth_index, trading_days=252):
    """
    Calculate annualized rate of return (ARC).
    
    Formula (Equation 22):
        ARC = (Final_Wealth / Initial_Wealth)^(1/Years) - 1
        
    With normalized wealth index starting at 1.0:
        ARC = (Final_Wealth)^(1/Years) - 1
    """
    years = len(wealth_index) / trading_days
    if years <= 0:
        return 0
    arc = (wealth_index[-1]) ** (1 / years) - 1
    return arc


def calculate_annualized_std(returns, trading_days=252):
    """
    Calculate annualized standard deviation (ASD).
    
    Formula (Equation 23):
        ASD = sqrt(T) * std(returns)
    where T = number of trading days per year
    """
    asd = np.sqrt(trading_days) * np.std(returns)
    return asd


def calculate_maximum_drawdown(wealth_index):
    """
    Calculate maximum drawdown (MD).
    
    Formula (Equation 24):
        MD = max_{x<=y} (Peak_x - Value_y) / Peak_x
    
    Represents the largest percentage decline from peak to trough.
    """
    cumulative_max = np.maximum.accumulate(wealth_index)
    drawdown = (cumulative_max - wealth_index) / cumulative_max
    md = np.max(drawdown)
    return md


def calculate_information_ratio(returns, trading_days=252):
    """
    Calculate Information Ratio (IR).
    
    Formula (Equation 25):
        IR = ARC / ASD
    
    Measures risk-adjusted returns. Higher is better.
    """
    arc = np.mean(returns) * trading_days
    asd = calculate_annualized_std(returns, trading_days)
    
    if asd == 0:
        return 0
    
    return arc / asd


def calculate_adjusted_information_ratio(returns, wealth_index, trading_days=252):
    """
    Calculate Adjusted Information Ratio (IR*).
    
    Formula (Equation 26):
        IR* = ARC^2 * sign(ARC) / (ASD * MD)
    
    Extended version considering maximum drawdown.
    Higher is better.
    """
    arc = calculate_annualized_return(wealth_index, trading_days)
    asd = calculate_annualized_std(returns, trading_days)
    md = calculate_maximum_drawdown(wealth_index)
    
    if asd == 0 or md == 0:
        return 0
    
    ir_star = (arc ** 2 * np.sign(arc)) / (asd * md)
    return ir_star


def calculate_sortino_ratio(returns, wealth_index, trading_days=252):
    """
    Calculate Sortino Ratio (SR).
    
    Formula (Equation 27):
        SR = ARC / ASD_downside
    
    where ASD_downside = annualized std of negative returns only
    
    Only penalizes downside volatility (better for risk-averse investors).
    Higher is better.
    """
    arc = calculate_annualized_return(wealth_index, trading_days)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        asd_downside = 1  # Avoid division by zero
    else:
        asd_downside = np.sqrt(trading_days) * np.std(downside_returns)
    
    if asd_downside == 0:
        return 0
    
    sr = arc / asd_downside
    return sr


def calculate_buy_hold_benchmark(prices, trading_days=252):
    """
    Calculate Buy & Hold benchmark metrics using normalized Wealth Index.
    
    Parameters:
    -----------
    prices : pandas.Series or ndarray
        Asset prices during the period
    trading_days : int
        Trading days per year
    
    Returns:
    --------
    metrics : dict
        Dictionary with keys: ARC, ASD, MD, IR, IR*, SR
    """
    # Convert to numpy if pandas Series
    if isinstance(prices, pd.Series):
        prices_array = prices.values
    else:
        prices_array = prices
    
    # Calculate log returns
    log_returns = np.log(prices_array[1:] / prices_array[:-1])
    
    # Build wealth index (buy & hold, starting at 1.0)
    wealth_index = np.zeros(len(prices_array))
    wealth_index[0] = 1.0
    wealth_index[1:] = np.exp(np.cumsum(log_returns))
    
    # Calculate all metrics
    arc = calculate_annualized_return(wealth_index, trading_days)
    asd = calculate_annualized_std(log_returns, trading_days)
    md = calculate_maximum_drawdown(wealth_index)
    ir = calculate_information_ratio(log_returns, trading_days)
    ir_star = calculate_adjusted_information_ratio(log_returns, wealth_index, trading_days)
    sr = calculate_sortino_ratio(log_returns, wealth_index, trading_days)
    
    return {
        'ARC': arc,
        'ASD': asd,
        'MD': md,
        'IR': ir,
        'IR*': ir_star,
        'SR': sr
    }


def create_performance_table(method_name, returns, wealth_index, trading_days=252):
    """
    Create a dictionary of performance metrics for a single trading method.
    
    Parameters:
    -----------
    method_name : str
        Name of the trading method (e.g., 'ARIMA', 'SVM-LSTM (1)')
    returns : ndarray
        Daily strategy returns
    wealth_index : ndarray
        Portfolio growth over time
    trading_days : int
        Trading days per year
    
    Returns:
    --------
    metrics_dict : dict
        Dictionary with keys: Method, ARC, ASD, MD, IR, IR*, SR
    """
    metrics_dict = {
        'Method': method_name,
        'ARC': calculate_annualized_return(wealth_index, trading_days),
        'ASD': calculate_annualized_std(returns, trading_days),
        'MD': calculate_maximum_drawdown(wealth_index),
        'IR': calculate_information_ratio(returns, trading_days),
        'IR*': calculate_adjusted_information_ratio(returns, wealth_index, trading_days),
        'SR': calculate_sortino_ratio(returns, wealth_index, trading_days)
    }
    return metrics_dict


def build_results_table(test_prices, predictions_dict, asset_name='Asset',
                       transaction_costs=0.005/100, trading_days=252,
                       strategy_type='long_short'):
    """
    Build complete results table for all models.
    
    Parameters:
    -----------
    test_prices : pandas.Series
        Prices for the test period
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
    asset_name : str
        Name of the asset (e.g., 'S&P 500', 'Bitcoin')
    transaction_costs : float
        Transaction costs as percentage
    trading_days : int
        Trading days per year
    strategy_type : str
        'long_short' or 'long_only'
    
    Returns:
    --------
    results_df : pandas.DataFrame
        Results table with all metrics
    """
    results = []
    
    # Add Buy & Hold benchmark
    bh_metrics = calculate_buy_hold_benchmark(test_prices, trading_days)
    bh_metrics['Method'] = f'Buy&Hold {asset_name}'
    results.append(bh_metrics)
    
    # Add each model
    for model_name, predictions in predictions_dict.items():
        # Generate signals
        if strategy_type == 'long_short':
            signals = generate_trading_signals(predictions, transaction_costs)
        else:
            signals = generate_trading_signals_long_only(predictions, transaction_costs)
        
        # Backtest
        wealth_index, returns = backtest_strategy(
            prices=test_prices,
            signals=signals,
            transaction_costs=transaction_costs,
            trading_days=trading_days
        )
        
        # Calculate metrics
        metrics = create_performance_table(model_name, returns, wealth_index, trading_days)
        results.append(metrics)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def format_results_table(results_df):
    """
    Format results table for display (matching paper format).
    
    Converts metrics to percentages and decimal places for pretty printing.
    """
    display_df = results_df.copy()
    
    # Format as percentages
    display_df['ARC'] = display_df['ARC'].apply(lambda x: f"{x:.2%}")
    display_df['ASD'] = display_df['ASD'].apply(lambda x: f"{x:.2%}")
    display_df['MD'] = display_df['MD'].apply(lambda x: f"{x:.2%}")
    
    # Format as decimal with 2 places
    display_df['IR'] = display_df['IR'].apply(lambda x: f"{x:.2f}")
    display_df['IR*'] = display_df['IR*'].apply(lambda x: f"{x:.2f}")
    display_df['SR'] = display_df['SR'].apply(lambda x: f"{x:.2f}")
    
    return display_df


print("=" * 80)
print("All trading strategy and performance metrics functions loaded successfully!")
print("=" * 80)
print("\nReady to generate Table 2 and Table 4 from the paper.")
print("\nUsage Example:")
print("-" * 80)
print("""
# S&P 500 Long-Short (Table 2)
sp500_results = build_results_table(
    test_prices=sp500_test_prices,
    predictions_dict=sp500_test_predictions,
    asset_name='S&P 500',
    transaction_costs=0.005/100,
    trading_days=252,
    strategy_type='long_short'
)

# Display results
print("Table 2: S&P 500 Long-Short Strategy Performance")
print(format_results_table(sp500_results).to_string(index=False))

# Bitcoin Long-Short (Table 4)
bitcoin_results = build_results_table(
    test_prices=bitcoin_test_prices,
    predictions_dict=bitcoin_test_predictions,
    asset_name='Bitcoin',
    transaction_costs=0.01/100,
    trading_days=365,
    strategy_type='long_short'
)

# Display results
print("Table 4: Bitcoin Long-Short Strategy Performance")
print(format_results_table(bitcoin_results).to_string(index=False))
""")
