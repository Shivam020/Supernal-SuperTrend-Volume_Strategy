import os
import re
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Strategy Parameters - USER CONFIGURABLE
ATR_LENGTH = 10
MULT = 3.0
FLAT_LOOKBACK = 10
VOLUME_LOOKBACK = 15
DELTA_LOOKBACK = 15
TRAIL_PCT = 1  # 1%

# Date Range Configuration - USER CONFIGURABLE
START_DATE = "2023-01-01"  # Change these dates as needed
END_DATE = "2023-12-31"    # Change these dates as needed

# Timeframe Configuration - USER CONFIGURABLE
CANDLE_TIMEFRAME = "1T"   # Options: "1T", "3T", "5T", "15T" (1m, 3m, 5m, 15m)

# Strategy Sensitivity - U
VOLUME_MULTIPLIER = 3.0   
DELTA_MULTIPLIER = 3.0    
FLAT_TOLERANCE = 0.0000001     

# -----------------------------
# Filename parser
# -----------------------------
FILENAME_RE = re.compile(r'(?P<symbol>[A-Z]+)(?P<expiry>\d{6})(?P<strike>\d+)(?P<opt>CE|PE)\.csv$', re.IGNORECASE)

def parse_filename(filename: str):
    base = os.path.basename(filename)
    m = FILENAME_RE.search(base)
    if not m:
        raise ValueError(f"Filename doesn't match pattern: {base}")
    d = m.groupdict()
    expiry = f"20{d['expiry']}"  # YYMMDD -> YYYYMMDD
    return {
        'symbol': d['symbol'].upper(),
        'expiry': expiry,
        'strike': int(d['strike']),
        'type': d['opt'].upper()
    }

# -----------------------------
# Safe datetime parser
# -----------------------------
def safe_to_datetime(date_series, time_series):
    """Try multiple parsing strategies until success"""
    dt = None
    try:
        # Try YYYYMMDD + HH:MM:SS
        dt = pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str),
                            format="%Y%m%d %H:%M:%S", errors="coerce")
    except Exception:
        pass
    
    if dt is None or dt.isna().all():
        try:
            # Try DD-MM-YYYY + HH:MM:SS
            dt = pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str),
                                format="%d-%m-%Y %H:%M:%S", errors="coerce")
        except Exception:
            pass
    
    # fallback generic parser
    if dt is None or dt.isna().all():
        dt = pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str), errors="coerce")

    if dt.isna().any():
        bad_rows = dt[dt.isna()]
        print(f"Warning: {len(bad_rows)} rows could not be parsed as datetime")
    return dt

# -----------------------------
# Enhanced Spot Loader with Timeframe Options
# -----------------------------
class SpotLoader:
    def __init__(self, path, timeframe="15T"):
        self.path = path
        self.timeframe = timeframe
        
        # Timeframe mapping for user-friendly display
        self.timeframe_display = {
            "1T": "1 minute",
            "3T": "3 minutes", 
            "5T": "5 minutes",
            "15T": "15 minutes"
        }

    def load(self):
        """Load spot data and create OHLC candles with specified timeframe"""
        print(f"Loading spot data with {self.timeframe_display.get(self.timeframe, self.timeframe)} candles...")
        
        df = pd.read_csv(self.path, header=None, usecols=[0, 1, 2])
        df.columns = ["Date", "Time", "Price"]
        df["datetime"] = safe_to_datetime(df["Date"], df["Time"])
        df = df.dropna(subset=["datetime"])
        
        if df.empty:
            raise ValueError("No valid datetime records found after parsing")
        
        df = df.set_index("datetime").sort_index()
        
        # Create OHLC candles with specified timeframe
        ohlc = df["Price"].resample(self.timeframe).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last'
        })
        
        # Add volume proxy (using price range as volume indicator)
        ohlc['volume'] = (ohlc['high'] - ohlc['low']) * 1000  # Synthetic volume
        
        # Forward fill missing values
        ohlc = ohlc.ffill().dropna()
        
        print(f"Created {len(ohlc)} {self.timeframe_display.get(self.timeframe)} candles")
        return ohlc

    def load_tick_data(self):
        """Load raw tick data (unchanged)"""
        df = pd.read_csv(self.path, header=None, usecols=[0, 1, 2])
        df.columns = ["Date", "Time", "Price"]
        df["datetime"] = safe_to_datetime(df["Date"], df["Time"])
        df = df.dropna(subset=["datetime"])
        
        if df.empty:
            raise ValueError("No valid datetime records found after parsing")
        
        df = df.set_index("datetime")
        return df[["Price"]].sort_index()

# -----------------------------
# Options Manager (unchanged)
# -----------------------------
class OptionsDataManager:
    def __init__(self, folder: str):
        self.folder = folder
        self._file_registry = {}
        self._cache = {}
        self.max_cache_size = 20  # Increased cache size
        
        self._build_registry()

    def _build_registry(self):
        """Build registry of available files with metadata"""
        paths = glob.glob(os.path.join(self.folder, "*.csv"))
        print(f"Scanning folder: {self.folder}")
        print(f"Found {len(paths)} CSV files")
        
        for path in paths:
            filename = os.path.basename(path)
            try:
                meta = parse_filename(filename)
                size = os.path.getsize(path)
                
                self._file_registry[filename] = {
                    'path': path,
                    'meta': meta,
                    'size': size
                }
                
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
                continue
        
        print(f"Registered {len(self._file_registry)} valid option files")

    def list_files(self) -> List[str]:
        return list(self._file_registry.keys())

    def get_file_info(self, filename: str) -> Dict:
        if filename not in self._file_registry:
            raise FileNotFoundError(f"File {filename} not found in registry")
        return self._file_registry[filename].copy()

    def load_file(self, filename: str) -> pd.DataFrame:
        if filename not in self._file_registry:
            raise FileNotFoundError(f"File {filename} not found in registry")
        
        if filename in self._cache:
            return self._cache[filename]
        
        file_info = self._file_registry[filename]
        path = file_info['path']
        
        try:
            df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3])
            df.columns = ["Date", "Time", "Price", "Volume"]
            df["datetime"] = safe_to_datetime(df["Date"], df["Time"])
            df = df.dropna(subset=["datetime"])
            
            if df.empty:
                raise ValueError("No valid datetime records found after parsing")
            
            df = df.set_index("datetime")
            df = df[["Price", "Volume"]].sort_index()
            
            self._add_to_cache(filename, df)
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading {filename}: {e}")

    def _add_to_cache(self, filename: str, data: pd.DataFrame):
        if len(self._cache) >= self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[filename] = data

    def clear_cache(self):
        self._cache.clear()
        print("Cache cleared")

    def filter_files(self, symbol: str = None, expiry: str = None, 
                    strike: int = None, option_type: str = None) -> List[str]:
        matching_files = []
        
        for filename, info in self._file_registry.items():
            meta = info['meta']
            
            if symbol and meta['symbol'] != symbol.upper():
                continue
            if expiry and meta['expiry'] != expiry:
                continue
            if strike and meta['strike'] != strike:
                continue
            if option_type and meta['type'] != option_type.upper():
                continue
                
            matching_files.append(filename)
        
        return matching_files

    def load_filtered_data(self, symbol: str = None, expiry: str = None, 
                          strike: int = None, option_type: str = None) -> Dict[str, pd.DataFrame]:
        matching_files = self.filter_files(symbol, expiry, strike, option_type)
        
        result = {}
        for filename in matching_files:
            try:
                result[filename] = self.load_file(filename)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        
        return result

    def get_summary(self) -> Dict:
        if not self._file_registry:
            return {"total_files": 0}
        
        symbols = set()
        expiries = set()
        strikes = set()
        option_types = set()
        total_size = 0
        
        for info in self._file_registry.values():
            meta = info['meta']
            symbols.add(meta['symbol'])
            expiries.add(meta['expiry'])
            strikes.add(meta['strike'])
            option_types.add(meta['type'])
            total_size += info['size']
        
        return {
            "total_files": len(self._file_registry),
            "unique_symbols": len(symbols),
            "unique_expiries": len(expiries),
            "unique_strikes": len(strikes),
            "option_types": list(option_types),
            "total_size_mb": round(total_size / (1024*1024), 2),
            "cached_files": len(self._cache)
        }

# -----------------------------
# Enhanced Technical Indicators (From Reference Code)
# -----------------------------
def calculate_atr(high, low, close, length=ATR_LENGTH):
    """Calculate Average True Range (Enhanced)"""
    hl = high - low
    hc = np.abs(high - close.shift())
    lc = np.abs(low - close.shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calculate_supertrend_enhanced(high, low, close, atr_length=ATR_LENGTH, multiplier=MULT):
    """Enhanced Supertrend calculation (from reference code)"""
    atr = calculate_atr(high, low, close, atr_length)
    
    upperBand = (high + low) / 2 + multiplier * atr
    lowerBand = (high + low) / 2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = lowerBand.iloc[i] if close.iloc[i] > upperBand.iloc[i] else upperBand.iloc[i]
        else:
            if supertrend.iloc[i-1] == upperBand.iloc[i-1]:
                supertrend.iloc[i] = lowerBand.iloc[i] if close.iloc[i] > upperBand.iloc[i] else min(upperBand.iloc[i], supertrend.iloc[i-1])
            else:
                supertrend.iloc[i] = upperBand.iloc[i] if close.iloc[i] < lowerBand.iloc[i] else max(lowerBand.iloc[i], supertrend.iloc[i-1])
    
    return supertrend

def is_supertrend_flat_relaxed(supertrend, lookback=FLAT_LOOKBACK, tolerance=FLAT_TOLERANCE):
    """Relaxed flat condition - allows small variations"""
    if len(supertrend) < lookback:
        return False
    
    recent_values = supertrend.tail(lookback)
    variation = recent_values.std() / recent_values.mean()  # Coefficient of variation
    return variation < tolerance  # More relaxed condition

# -----------------------------
# Enhanced Options Strategy Backtester
# -----------------------------
class OptionsStrategyBacktester:
    def __init__(self, spot_loader, options_manager, start_date=START_DATE, end_date=END_DATE):
        self.spot_loader = spot_loader
        self.options_manager = options_manager
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Strategy parameters
        self.atr_length = ATR_LENGTH
        self.multiplier = MULT
        self.flat_lookback = FLAT_LOOKBACK
        self.volume_lookback = VOLUME_LOOKBACK
        self.delta_lookback = DELTA_LOOKBACK
        self.trail_pct = TRAIL_PCT
        
        # Relaxed conditions
        self.volume_multiplier = VOLUME_MULTIPLIER
        self.delta_multiplier = DELTA_MULTIPLIER
        self.flat_tolerance = FLAT_TOLERANCE
        
        # Results storage
        self.trades = []
        self.spot_data = None
        self.results = {}
        
    def load_and_prepare_spot_data(self):
        """Load and prepare spot data with enhanced indicators"""
        print("Loading and preparing spot data...")
        
        # Load spot data
        self.spot_data = self.spot_loader.load()
        
        # Filter by date range
        self.spot_data = self.spot_data[
            (self.spot_data.index >= self.start_date) & 
            (self.spot_data.index <= self.end_date)
        ]
        
        if self.spot_data.empty:
            raise ValueError(f"No spot data found in date range {self.start_date} to {self.end_date}")
        
        # Calculate enhanced supertrend
        self.spot_data['supertrend'] = calculate_supertrend_enhanced(
            self.spot_data['high'], 
            self.spot_data['low'], 
            self.spot_data['close'], 
            self.atr_length, 
            self.multiplier
        )
        
        # Enhanced volume condition (relaxed)
        self.spot_data['volume_avg'] = self.spot_data['volume'].rolling(self.volume_lookback).mean()
        self.spot_data['volume_condition'] = self.spot_data['volume'] > (self.volume_multiplier * self.spot_data['volume_avg'])
        
        # Enhanced delta condition (using close-open delta with volume)
        self.spot_data['delta'] = (self.spot_data['close'] - self.spot_data['open']) * self.spot_data['volume']
        self.spot_data['delta_avg'] = self.spot_data['delta'].rolling(self.delta_lookback).mean()
        self.spot_data['delta_condition'] = self.spot_data['delta'] > (self.delta_multiplier * self.spot_data['delta_avg'])
        
        # Enhanced flat supertrend condition (relaxed)
        self.spot_data['st_flat'] = self.spot_data['supertrend'].rolling(self.flat_lookback).apply(
            lambda x: is_supertrend_flat_relaxed(x, self.flat_lookback, self.flat_tolerance), raw=False
        ).fillna(False).astype(bool)
        
        print(f"Prepared spot data: {len(self.spot_data)} candles from {self.spot_data.index[0]} to {self.spot_data.index[-1]}")
        
        # Debug: Show condition statistics
        flat_count = self.spot_data['st_flat'].sum()
        vol_count = self.spot_data['volume_condition'].sum()
        delta_count = self.spot_data['delta_condition'].sum()
        
        print(f"Condition Statistics:")
        print(f"   • Flat Supertrend: {flat_count} candles ({flat_count/len(self.spot_data)*100:.1f}%)")
        print(f"   • Volume Condition: {vol_count} candles ({vol_count/len(self.spot_data)*100:.1f}%)")
        print(f"   • Delta Condition: {delta_count} candles ({delta_count/len(self.spot_data)*100:.1f}%)")
        
    def generate_signals(self):
        """Generate buy/sell signals with relaxed conditions"""
        print("Generating trading signals...")
        
        signals = []
        
        # Create combined signal condition
        signal_condition = (
            self.spot_data['st_flat'] & 
            self.spot_data['volume_condition'] & 
            self.spot_data['delta_condition']
        )
        
        # Find signal points
        signal_points = self.spot_data[signal_condition]
        
        for timestamp, row in signal_points.iterrows():
            # Determine signal direction based on price vs supertrend
            signal_type = 'BUY' if row['close'] > row['supertrend'] else 'SELL'
            
            signals.append({
                'timestamp': timestamp,
                'spot_price': row['close'],
                'signal': signal_type,
                'supertrend': row['supertrend'],
                'volume': row['volume'],
                'delta': row['delta']
            })
        
        print(f"Generated {len(signals)} signals")
        return signals
        
    def find_nearest_option_file(self, spot_price, timestamp, option_type='CE'):
        """Find nearest strike option file for given spot price and timestamp"""
        # Round spot price to nearest 50 (typical NIFTY strike interval)
        strike_interval = 50
        nearest_strike = round(spot_price / strike_interval) * strike_interval
        
        # Try multiple strikes around the nearest one (wider range)
        strikes_to_try = [
            nearest_strike,
            nearest_strike + strike_interval,
            nearest_strike - strike_interval,
            nearest_strike + 2*strike_interval,
            nearest_strike - 2*strike_interval,
            nearest_strike + 100,  # Try 100 point intervals too
            nearest_strike - 100
        ]
        
        # Try YYMMDD format first (based on your files like NIFTY220603...)
        date_str_6 = timestamp.strftime("%y%m%d")  # 6 digits: 231228
        date_str_8 = timestamp.strftime("%Y%m%d")  # 8 digits: 20231228
        
        # Try both date formats
        for date_str in [date_str_6, date_str_8]:
            for strike in strikes_to_try:
                # Construct expected filename pattern
                filename_pattern = f"NIFTY{date_str}{int(strike)}{option_type}.csv"
                
                # Check if file exists in registry
                matching_files = [f for f in self.options_manager.list_files() if f == filename_pattern]
                
                if matching_files:
                    return matching_files[0], strike
        
        # If exact date not found, try nearby expiries (common for weekly/monthly options)
        for days_offset in [-1, -2, -3, -4, -5, -6, -7, 1, 2, 3, 4, 5, 6, 7, 14, 21]:
            try_date = timestamp + timedelta(days=days_offset)
            
            for date_format in ["%y%m%d", "%Y%m%d"]:
                date_str = try_date.strftime(date_format)
                
                for strike in strikes_to_try:
                    filename_pattern = f"NIFTY{date_str}{int(strike)}{option_type}.csv"
                    matching_files = [f for f in self.options_manager.list_files() if f == filename_pattern]
                    
                    if matching_files:
                        return matching_files[0], strike
        
        # Last resort: Find any file with similar strike price
        all_files = self.options_manager.list_files()
        
        for strike in strikes_to_try:
            # Look for any file containing this strike and option type
            pattern = f"{int(strike)}{option_type}.csv"
            matching_files = [f for f in all_files if pattern in f]
            
            if matching_files:
                # Sort by filename to get most recent or relevant
                matching_files.sort()
                return matching_files[0], strike
        
        return None, None
    
    def get_option_price_at_time(self, option_file, timestamp):
        """Get option price at specific timestamp - MISSING METHOD ADDED"""
        try:
            option_data = self.options_manager.load_file(option_file)
            
            # Find exact time or nearest time
            if timestamp in option_data.index:
                return option_data.loc[timestamp, 'Price']
            
            # Find nearest timestamp
            if len(option_data) == 0:
                return None
            
            # Get the closest timestamp (forward fill approach)
            nearest_data = option_data[option_data.index <= timestamp]
            
            if len(nearest_data) > 0:
                return nearest_data.iloc[-1]['Price']
            
            # If no data before timestamp, use first available price
            return option_data.iloc[0]['Price']
            
        except Exception as e:
            print(f"Error getting option price for {option_file} at {timestamp}: {e}")
            return None
    
    def execute_backtest(self):
        """Execute the complete backtesting process"""
        print("Starting backtest execution...")
        
        # Load and prepare data
        self.load_and_prepare_spot_data()
        
        # Generate signals
        signals = self.generate_signals()
        
        if not signals:
            print("No trading signals generated")
            print("Try relaxing the strategy parameters:")
            print(f"   • Current VOLUME_MULTIPLIER: {self.volume_multiplier} (try 1.2)")
            print(f"   • Current DELTA_MULTIPLIER: {self.delta_multiplier} (try 1.2)")
            print(f"   • Current FLAT_TOLERANCE: {self.flat_tolerance} (try 0.01)")
            return
        
        # Execute trades
        print(f"Executing {len(signals)} trades...")
        
        for i, signal in enumerate(signals, 1):
            print(f"   Processing trade {i}/{len(signals)}: {signal['signal']} at {signal['timestamp']}")
            self.execute_trade(signal)
        
        # Calculate results
        self.calculate_results()
        
        print("Backtest completed successfully!")
    
    def execute_trade(self, signal):
        """Execute a single trade based on signal"""
        timestamp = signal['timestamp']
        spot_price = signal['spot_price']
        signal_type = signal['signal']
        
        # Determine option type
        option_type = 'PE' if signal_type == 'BUY' else 'CE'
        
        # Find nearest option file
        option_file, strike = self.find_nearest_option_file(spot_price, timestamp, option_type)
        
        if not option_file:
            print(f"No option file found for {timestamp}, spot: {spot_price}")
            return
        
        # Get entry price
        entry_price = self.get_option_price_at_time(option_file, timestamp)
        
        if entry_price is None:
            print(f"No option price found for {option_file} at {timestamp}")
            return
        
        # Create trade record
        trade = {
            'entry_time': timestamp,
            'spot_price': spot_price,
            'signal_type': signal_type,
            'option_type': option_type,
            'strike': strike,
            'option_file': option_file,
            'entry_price': entry_price,
            'exit_time': None,
            'exit_price': None,
            'pnl': 0,
            'max_profit': 0,
            'max_loss': 0,
            'status': 'OPEN'
        }
        
        # Manage trade until exit
        self.manage_trade(trade)
        
        # Add to trades list
        self.trades.append(trade)
    
    def manage_trade(self, trade):
        """Manage an open trade until exit condition is met"""
        try:
            option_data = self.options_manager.load_file(trade['option_file'])
            
            # Get data after entry time
            future_data = option_data[option_data.index > trade['entry_time']]
            
            if future_data.empty:
                # No future data, close at entry price
                trade['exit_time'] = trade['entry_time']
                trade['exit_price'] = trade['entry_price']
                trade['pnl'] = 0
                trade['status'] = 'CLOSED'
                return
            
            max_price = trade['entry_price']
            
            for timestamp, row in future_data.iterrows():
                current_price = row['Price']
                
                # Update max profit/loss tracking
                if current_price > max_price:
                    max_price = current_price
                    trade['max_profit'] = max_price - trade['entry_price']
                
                current_loss = trade['entry_price'] - current_price
                if current_loss > trade['max_loss']:
                    trade['max_loss'] = current_loss
                
                # Check trailing stop-loss condition
                trailing_stop_price = max_price * (1 - self.trail_pct / 100)
                
                if current_price <= trailing_stop_price:
                    # Exit trade
                    trade['exit_time'] = timestamp
                    trade['exit_price'] = current_price
                    trade['pnl'] = current_price - trade['entry_price']
                    trade['status'] = 'CLOSED'
                    return
                
                # Check if we've reached end of trading day
                if timestamp.time() >= pd.Timestamp('15:30').time():
                    # Close at market close
                    trade['exit_time'] = timestamp
                    trade['exit_price'] = current_price
                    trade['pnl'] = current_price - trade['entry_price']
                    trade['status'] = 'CLOSED'
                    return
            
            # If no exit condition met, close at last available price
            if not future_data.empty:
                last_row = future_data.iloc[-1]
                trade['exit_time'] = future_data.index[-1]
                trade['exit_price'] = last_row['Price']
                trade['pnl'] = last_row['Price'] - trade['entry_price']
                trade['status'] = 'CLOSED'
                
        except Exception as e:
            print(f"Error managing trade: {e}")
            # Close trade at entry price if error occurs
            trade['exit_time'] = trade['entry_time']
            trade['exit_price'] = trade['entry_price']
            trade['pnl'] = 0
            trade['status'] = 'CLOSED'
    
    def calculate_results(self):
        """Calculate comprehensive backtest results and metrics"""
        if not self.trades:
            print("No trades to analyze")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # P&L statistics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_profit = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate equity curve for drawdown analysis
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        equity_curve = trades_df['cumulative_pnl'].values
        
        # Maximum drawdown
        if len(equity_curve) > 0:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / np.maximum(peak, 1) * 100
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Average holding period
        trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # hours
        avg_holding_period = trades_df['holding_period'].mean()
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'avg_holding_period_hours': round(avg_holding_period, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2)
        }
        
        self.trades_df = trades_df
    
    def print_results(self):
        """Print formatted backtest results"""
        if not self.results:
            print("No results to display. Run backtest first.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']}%")
        print("-"*60)
        print(f"Total P&L: ₹{self.results['total_pnl']}")
        print(f"Average P&L per Trade: ₹{self.results['avg_pnl']}")
        print(f"Maximum Profit: ₹{self.results['max_profit']}")
        print(f"Maximum Loss: ₹{self.results['max_loss']}")
        print("-"*60)
        print(f"Profit Factor: {self.results['profit_factor']}")
        print(f"Maximum Drawdown: {self.results['max_drawdown']}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']}")
        print(f"Average Holding Period: {self.results['avg_holding_period_hours']} hours")
        print("-"*60)
        print(f"Gross Profit: ₹{self.results['gross_profit']}")
        print(f"Gross Loss: ₹{self.results['gross_loss']}")
        print("="*60)
    
    def save_results_to_excel(self, filename='backtest_results.xlsx'):
        """Save detailed results to Excel file"""
        if not self.trades:
            print("No trades to save")
            return
        
        print(f"Saving results to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Trade details
            self.trades_df.to_excel(writer, sheet_name='Trade_Details', index=False)
            
            # Summary metrics
            summary_df = pd.DataFrame([self.results]).T
            summary_df.columns = ['Value']
            summary_df.to_excel(writer, sheet_name='Summary_Metrics')
            
            # Equity curve
            equity_df = pd.DataFrame({
                'Trade_Number': range(1, len(self.trades_df) + 1),
                'Cumulative_PnL': self.trades_df['cumulative_pnl'].values
            })
            equity_df.to_excel(writer, sheet_name='Equity_Curve', index=False)
            
            # Monthly performance (if we have enough data)
            if len(self.trades_df) > 0:
                monthly_pnl = self.trades_df.groupby(self.trades_df['entry_time'].dt.to_period('M'))['pnl'].sum()
                monthly_df = pd.DataFrame({
                    'Month': monthly_pnl.index.astype(str),
                    'PnL': monthly_pnl.values
                })
                monthly_df.to_excel(writer, sheet_name='Monthly_Performance', index=False)
            
            # Strategy parameters sheet
            params_df = pd.DataFrame({
                'Parameter': ['ATR_LENGTH', 'MULT', 'FLAT_LOOKBACK', 'VOLUME_LOOKBACK', 
                             'DELTA_LOOKBACK', 'TRAIL_PCT', 'VOLUME_MULTIPLIER', 'DELTA_MULTIPLIER', 
                             'FLAT_TOLERANCE', 'CANDLE_TIMEFRAME', 'START_DATE', 'END_DATE'],
                'Value': [ATR_LENGTH, MULT, FLAT_LOOKBACK, VOLUME_LOOKBACK, 
                         DELTA_LOOKBACK, TRAIL_PCT, VOLUME_MULTIPLIER, DELTA_MULTIPLIER, 
                         FLAT_TOLERANCE, CANDLE_TIMEFRAME, START_DATE, END_DATE]
            })
            params_df.to_excel(writer, sheet_name='Strategy_Parameters', index=False)
        
        print(f"Results saved to {filename}")

# -----------------------------
# Complete Example Usage with Configuration Options
# -----------------------------
if __name__ == "__main__":
    
    print("NIFTY OPTIONS SUPERTREND STRATEGY BACKTESTER")
    print("="*70)
    
    timeframe_display = {
        "1T": "1 min", 
        "3T": "3 min", 
        "5T": "5 min", 
        "15T": "15 min"
    }
    
    print(f"Current Configuration:")
    print(f"   • Candle Timeframe: {CANDLE_TIMEFRAME} ({timeframe_display.get(CANDLE_TIMEFRAME, CANDLE_TIMEFRAME)})")
    print(f"   • Date Range: {START_DATE} to {END_DATE}")
    print(f"   • Volume Multiplier: {VOLUME_MULTIPLIER}x (relaxed from 3.0x)")
    print(f"   • Delta Multiplier: {DELTA_MULTIPLIER}x (relaxed from 3.0x)")
    print(f"   • Flat Tolerance: {FLAT_TOLERANCE} ({FLAT_TOLERANCE*100}%)")
    
    # Initialize data loaders (UPDATE THESE PATHS TO YOUR DATA FOLDERS)
    print("\nInitializing data loaders...")
    
    try:
        spot_loader = SpotLoader(r"data\NIFTY_SPOT\NIFTY.csv", timeframe=CANDLE_TIMEFRAME)
        opt_manager = OptionsDataManager(r"data\NSE Options\NIFTY_Options")
        
        # Show options data summary
        print("\nOptions Data Overview:")
        summary = opt_manager.get_summary()
        for key, value in summary.items():
            print(f"   • {key}: {value}")
        
        # List some sample files
        all_files = opt_manager.list_files()
        if all_files:
            print(f"\nSample option files (first 5):")
            for file in all_files[:5]:
                info = opt_manager.get_file_info(file)
                print(f"   • {file} - Strike: {info['meta']['strike']}, Type: {info['meta']['type']}")
        
        # Create and run backtester
        print("\nCreating backtester...")
        backtester = OptionsStrategyBacktester(
            spot_loader=spot_loader,
            options_manager=opt_manager,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Display strategy parameters
        print(f"\nStrategy Parameters:")
        print(f"   • ATR Length: {ATR_LENGTH}")
        print(f"   • Multiplier: {MULT}")
        print(f"   • Flat Lookback: {FLAT_LOOKBACK}")
        print(f"   • Volume Lookback: {VOLUME_LOOKBACK}")
        print(f"   • Delta Lookback: {DELTA_LOOKBACK}")
        print(f"   • Trail Stop %: {TRAIL_PCT}%")
        print(f"   • Volume Multiplier: {VOLUME_MULTIPLIER}x (relaxed)")
        print(f"   • Delta Multiplier: {DELTA_MULTIPLIER}x (relaxed)")
        print(f"   • Flat Tolerance: {FLAT_TOLERANCE} (relaxed)")
        
        # Run backtest
        print("\n" + "="*70)
        backtester.execute_backtest()
        
        # Print results
        backtester.print_results()
        
        # Save to Excel with timestamp
        result_filename = f'nifty_supertrend_{CANDLE_TIMEFRAME}_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        backtester.save_results_to_excel(result_filename)
        
        print(f"\nBacktesting completed successfully!")
        print(f"Check {result_filename} for detailed results")
        
        # Suggestions for improvement
        if len(backtester.trades) == 0:
            print(f"\nNo trades generated. Try these adjustments:")
            print(f"   • Reduce VOLUME_MULTIPLIER to 1.2")
            print(f"   • Reduce DELTA_MULTIPLIER to 1.2") 
            print(f"   • Increase FLAT_TOLERANCE to 0.01")
            print(f"   • Try different CANDLE_TIMEFRAME ('5T' or '1T')")
        elif len(backtester.trades) < 10:
            print(f"\nFew trades generated ({len(backtester.trades)}). Consider:")
            print(f"   • Slightly relaxing conditions further")
            print(f"   • Using shorter timeframes")
            print(f"   • Extending date range")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nCommon fixes:")
        print(f"   • Check file paths are correct")
        print(f"   • Ensure data files exist")
        print(f"   • Try with a smaller date range first")
        print(f"   • Check data format matches expected structure")
        
    finally:
        # Optional: Clear cache to free memory
        try:
            opt_manager.clear_cache()
            print("\nMemory cleaned up")
        except:
            pass

