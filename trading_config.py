"""
Configuration file for BTM Live Trading Strategy
Modify these settings to customize the strategy behavior
"""

from btm_utils import TradingConfig

# Default configuration for paper trading
DEFAULT_CONFIG = TradingConfig(
    symbol="SPY",                    # Base symbol for analysis
    session="paper",                 # "paper" or "live"
    leverage_cap=3.0,               # Maximum leverage
    target_daily_volatility=0.02,   # Target daily volatility (2%)
    lookback_days=14,               # Days for noise band calculation
    volatility_multiplier=1.0,      # Multiplier for noise bands
    use_gap_adjustment=True,        # Use gap adjustment for bands
    start_trading_time="10:00",     # First possible decision time
    close_time="16:00"              # Market close time
)

# Conservative configuration (lower leverage, higher volatility target)
CONSERVATIVE_CONFIG = TradingConfig(
    symbol="SPY",
    session="paper",
    leverage_cap=2.0,               # Lower maximum leverage
    target_daily_volatility=0.015,  # Lower target volatility (1.5%)
    lookback_days=21,               # Longer lookback period
    volatility_multiplier=0.8,      # Smaller noise bands
    use_gap_adjustment=True,
    start_trading_time="10:00",
    close_time="16:00"
)

# Aggressive configuration (higher leverage, lower volatility target)
AGGRESSIVE_CONFIG = TradingConfig(
    symbol="SPY",
    session="paper",
    leverage_cap=3.0,               # Maximum leverage
    target_daily_volatility=0.025,  # Higher target volatility (2.5%)
    lookback_days=10,               # Shorter lookback period
    volatility_multiplier=1.2,      # Larger noise bands
    use_gap_adjustment=True,
    start_trading_time="10:00",
    close_time="16:00"
)

# Live trading configuration (use with caution!)
LIVE_CONFIG = TradingConfig(
    symbol="SPY",
    session="live",                 # LIVE TRADING - USE WITH CAUTION!
    leverage_cap=2.0,               # Conservative leverage for live trading
    target_daily_volatility=0.015,  # Conservative volatility target
    lookback_days=14,
    volatility_multiplier=1.0,
    use_gap_adjustment=True,
    start_trading_time="10:00",
    close_time="16:00"
)

# Configuration presets
CONFIG_PRESETS = {
    "default": DEFAULT_CONFIG,
    "conservative": CONSERVATIVE_CONFIG,
    "aggressive": AGGRESSIVE_CONFIG,
    "live": LIVE_CONFIG
}

def get_config(preset_name: str = "default") -> TradingConfig:
    """
    Get a configuration preset by name.
    
    Args:
        preset_name: Name of the preset ("default", "conservative", "aggressive", "live")
    
    Returns:
        TradingConfig object
    """
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(CONFIG_PRESETS.keys())}")
    
    return CONFIG_PRESETS[preset_name]

def print_config_summary(config: TradingConfig) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: TradingConfig object
    """
    print("=" * 50)
    print("BTM TRADING CONFIGURATION")
    print("=" * 50)
    print(f"Session: {config.session}")
    print(f"Base Symbol: {config.symbol}")
    print(f"Leverage Cap: {config.leverage_cap}x")
    print(f"Target Daily Volatility: {config.target_daily_volatility*100:.1f}%")
    print(f"Lookback Days: {config.lookback_days}")
    print(f"Volatility Multiplier: {config.volatility_multiplier}")
    print(f"Gap Adjustment: {config.use_gap_adjustment}")
    print(f"Start Trading Time: {config.start_trading_time}")
    print(f"Close Time: {config.close_time}")
    print("=" * 50)

if __name__ == "__main__":
    # Print all available configurations
    for preset_name, config in CONFIG_PRESETS.items():
        print(f"\n{preset_name.upper()} CONFIGURATION:")
        print_config_summary(config)
