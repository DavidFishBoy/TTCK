"""
Simple test script to verify analysis modules are working correctly.
Run this script to test the new analysis modules with real data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis import (
    load_all_coins_data,
    get_all_metrics,
    calculate_market_breadth,
    identify_market_regime,
    equal_weight_portfolio,
    backtest_portfolio,
    calculate_portfolio_metrics,
    create_factor_dataframe,
    cluster_by_factors
)


def test_financial_metrics():
    """Test financial metrics calculation."""
    print("\n" + "="*60)
    print("Testing Financial Metrics Module")
    print("="*60)
    
    # Load data
    data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        print("❌ No data loaded. Make sure data/raw/train contains CSV files.")
        return False
    
    print(f"✅ Loaded {len(data_dict)} coins")
    
    # Calculate metrics for first coin
    first_coin = list(data_dict.keys())[0]
    df = data_dict[first_coin]
    
    metrics = get_all_metrics(df['close'], coin_name=first_coin)
    
    print(f"\n📊 Metrics for {first_coin}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return True


def test_market_analyzer():
    """Test market analyzer module."""
    print("\n" + "="*60)
    print("Testing Market Analyzer Module")
    print("="*60)
    
    # Load data
    data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        print("❌ No data loaded")
        return False
    
    # Market breadth
    breadth = calculate_market_breadth(data_dict)
    print("\n📈 Market Breadth:")
    print(breadth)
    
    # Market regime
    regime = identify_market_regime(data_dict)
    print("\n🌍 Market Regime:")
    for key, value in regime.items():
        print(f"  {key}: {value}")
    
    return True


def test_portfolio_engine():
    """Test portfolio engine module."""
    print("\n" + "="*60)
    print("Testing Portfolio Engine Module")
    print("="*60)
    
    # Load data
    data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        print("❌ No data loaded")
        return False
    
    # Create equal weight portfolio
    weights = {coin: 1.0 / len(data_dict) for coin in data_dict.keys()}
    
    print(f"\n💼 Backtesting equal-weight portfolio with {len(weights)} coins...")
    
    portfolio = backtest_portfolio(data_dict, weights, initial_capital=10000)
    
    if portfolio.empty:
        print("❌ Portfolio backtest failed")
        return False
    
    print(f"✅ Portfolio backtest complete: {len(portfolio)} days")
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(portfolio)
    
    print("\n📊 Portfolio Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return True


def test_factor_analyzer():
    """Test factor analyzer module."""
    print("\n" + "="*60)
    print("Testing Factor Analyzer Module")
    print("="*60)
    
    # Load data
    data_dict = load_all_coins_data(data_dir="data/raw/train")
    
    if not data_dict:
        print("❌ No data loaded")
        return False
    
    # Create factor dataframe
    factor_df = create_factor_dataframe(data_dict)
    
    print("\n🧩 Factor Analysis:")
    print(factor_df[['coin', 'momentum_30d', 'volatility', 'size']].head())
    
    # Cluster coins
    clustered = cluster_by_factors(factor_df, n_clusters=3)
    
    print("\n🔍 Coin Clusters:")
    for cluster_id in sorted(clustered['cluster'].unique()):
        coins_in_cluster = clustered[clustered['cluster'] == cluster_id]['coin'].tolist()
        print(f"  Cluster {cluster_id}: {', '.join(coins_in_cluster)}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 Testing Analysis Modules")
    print("="*60)
    
    tests = [
        ("Financial Metrics", test_financial_metrics),
        ("Market Analyzer", test_market_analyzer),
        ("Portfolio Engine", test_portfolio_engine),
        ("Factor Analyzer", test_factor_analyzer),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Analysis modules are ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
