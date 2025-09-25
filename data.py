import os
import re
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
        print(f"⚠️ Warning: {len(bad_rows)} rows could not be parsed as datetime")
    return dt

# -----------------------------
# Spot Loader (unchanged)
# -----------------------------
class SpotLoader:
    def __init__(self, path, resample="1min"):
        self.path = path
        self.resample = resample

    def load(self):
        df = pd.read_csv(self.path, header=None, usecols=[0, 1, 2])
        df.columns = ["Date", "Time", "Price"]
        df["datetime"] = safe_to_datetime(df["Date"], df["Time"])
        df = df.dropna(subset=["datetime"])
        
        if df.empty:
            raise ValueError("No valid datetime records found after parsing")
        
        df = df.set_index("datetime")
        ohlc = df["Price"].resample(self.resample).ohlc()
        ohlc = ohlc.ffill()
        return ohlc

    def load_tick_data(self):
        df = pd.read_csv(self.path, header=None, usecols=[0, 1, 2])
        df.columns = ["Date", "Time", "Price"]
        df["datetime"] = safe_to_datetime(df["Date"], df["Time"])
        df = df.dropna(subset=["datetime"])
        
        if df.empty:
            raise ValueError("No valid datetime records found after parsing")
        
        df = df.set_index("datetime")
        return df[["Price"]].sort_index()

# -----------------------------
# Optimized Lazy Options Manager
# -----------------------------
class OptionsDataManager:
    def __init__(self, folder: str):
        self.folder = folder
        self._file_registry = {}  # {filename: {'path': str, 'meta': dict, 'size': int}}
        self._cache = {}  # Simple cache for recently loaded data
        self.max_cache_size = 10  # Keep max 10 files in memory
        
        # Build file registry on initialization
        self._build_registry()

    def _build_registry(self):
        """Build registry of available files with metadata"""
        paths = glob.glob(os.path.join(self.folder, "*.csv"))
        print(f"🔍 Scanning folder: {self.folder}")
        print(f"📁 Found {len(paths)} CSV files")
        
        for path in paths:
            filename = os.path.basename(path)
            try:
                # Parse metadata from filename
                meta = parse_filename(filename)
                
                # Get file size for reference
                size = os.path.getsize(path)
                
                self._file_registry[filename] = {
                    'path': path,
                    'meta': meta,
                    'size': size
                }
                
            except ValueError as e:
                print(f"⚠️ Skipping {filename}: {e}")
                continue
        
        print(f"✅ Registered {len(self._file_registry)} valid option files")

    def list_files(self) -> List[str]:
        """Get list of all available option files"""
        return list(self._file_registry.keys())

    def get_file_info(self, filename: str) -> Dict:
        """Get metadata and info about a specific file"""
        if filename not in self._file_registry:
            raise FileNotFoundError(f"File {filename} not found in registry")
        return self._file_registry[filename].copy()

    def load_file(self, filename: str) -> pd.DataFrame:
        """Load a specific options file by filename"""
        if filename not in self._file_registry:
            raise FileNotFoundError(f"File {filename} not found in registry")
        
        # Check cache first
        if filename in self._cache:
            print(f"📋 Loading {filename} from cache")
            return self._cache[filename]
        
        # Load from disk
        file_info = self._file_registry[filename]
        path = file_info['path']
        
        print(f"💾 Loading {filename} from disk...")
        
        try:
            # Read only first 4 columns (A, B, C, D) with no headers
            df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3])
            
            # Assign column names for A, B, C, D
            df.columns = ["Date", "Time", "Price", "Volume"]
            
            # Convert Date and Time to datetime
            df["datetime"] = safe_to_datetime(df["Date"], df["Time"])
            
            # Remove rows with invalid datetime
            df = df.dropna(subset=["datetime"])
            
            if df.empty:
                raise ValueError("No valid datetime records found after parsing")
            
            # Set datetime as index and keep Price, Volume
            df = df.set_index("datetime")
            df = df[["Price", "Volume"]].sort_index()
            
            # Add to cache (with size management)
            self._add_to_cache(filename, df)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading {filename}: {e}")

    def _add_to_cache(self, filename: str, data: pd.DataFrame):
        """Add data to cache with size management"""
        # Remove oldest entries if cache is full
        if len(self._cache) >= self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            print(f"🗑️ Removed {oldest_key} from cache")
        
        self._cache[filename] = data
        print(f"💾 Added {filename} to cache ({len(data)} rows)")

    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        print("🗑️ Cache cleared")

    def filter_files(self, symbol: str = None, expiry: str = None, 
                    strike: int = None, option_type: str = None) -> List[str]:
        """Filter files by criteria and return matching filenames"""
        matching_files = []
        
        for filename, info in self._file_registry.items():
            meta = info['meta']
            
            # Apply filters
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
        """Load multiple files matching the filter criteria"""
        matching_files = self.filter_files(symbol, expiry, strike, option_type)
        
        print(f"📊 Loading {len(matching_files)} files matching criteria...")
        
        result = {}
        for filename in matching_files:
            try:
                result[filename] = self.load_file(filename)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        
        return result

    def get_summary(self) -> Dict:
        """Get summary statistics of all registered files"""
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


# Example usage
if __name__ == "__main__":
    
    # Initialize the options data manager
    opt_manager = OptionsDataManager(r"data\NSE Options\NIFTY_Options")
    
    # Get summary
    summary = opt_manager.get_summary()
    print(f"\n📊 Options Data Summary:")
    for key, value in summary.items():
        print(f"   • {key}: {value}")
    
    # List some files
    all_files = opt_manager.list_files()
    print(f"\n📄 Available files (first 5):")
    for file in all_files[:5]:
        info = opt_manager.get_file_info(file)
        print(f"   • {file} - {info['meta']} ({info['size']/1024:.1f} KB)")
    
    # Filter and load specific options
    ce_files = opt_manager.filter_files(symbol="NIFTY", option_type="CE")
    print(f"\n🔍 Found {len(ce_files)} NIFTY CE files")
    
    # Load a specific file on demand
    if ce_files:
        sample_file = ce_files[0]
        print(f"\n💾 Loading sample file: {sample_file}")
        data = opt_manager.load_file(sample_file)
        print(f"Data shape: {data.shape}")
        print("Data sample:")
        print(data.head())
    
    # Load spot data
    print(f"\n📈 Loading spot data...")
    spot_loader = SpotLoader(r"data\NIFTY_SPOT\NIFTY.csv")
    spot = spot_loader.load()
    print("Spot Data shape:", spot.shape)
    print("Spot data sample:")
    print(spot.head())
