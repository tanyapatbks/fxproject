"""
Multi-Currency Bagging AI Trading System
Data Processor with Anti-Data Leakage Protection

This module handles all data processing while maintaining strict separation
between training, validation, and test sets to prevent data leakage.

Key Design Principles:
1. Test set (2022) is completely isolated and only accessed in final evaluation
2. Only 2018-2021 data is used during model development
3. Clear temporal boundaries prevent future information leakage
4. Comprehensive logging tracks all data operations
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import warnings

# Technical Analysis Libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using custom implementations.")

import ta  # Alternative technical analysis library
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import our adaptive exit strategy
from adaptive_exit_strategy import AdaptiveExitStrategy, create_adaptive_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DataLeakageProtection:
    """
    A security class that enforces strict data separation to prevent leakage.
    Think of this as a security guard that ensures the test set remains untouched
    during the entire development process.
    """
    
    def __init__(self):
        # Define strict temporal boundaries
        self.TRAIN_START = '2018-01-01'
        self.TRAIN_END = '2020-12-31'
        self.VALIDATION_START = '2021-01-01'
        self.VALIDATION_END = '2021-12-31'
        self.TEST_START = '2022-01-01'  # This will be protected!
        self.TEST_END = '2022-12-31'
        
        # Flag to track if test set has been accessed
        self.test_set_accessed = False
        self.development_phase = True
        
        logging.info("Data Leakage Protection System Initialized")
        logging.info(f"Training Period: {self.TRAIN_START} to {self.TRAIN_END}")
        logging.info(f"Validation Period: {self.VALIDATION_START} to {self.VALIDATION_END}")
        logging.info(f"Test Period: {self.TEST_START} to {self.TEST_END} [PROTECTED]")
    
    def validate_development_access(self, start_date: str, end_date: str) -> bool:
        """
        Validates that no test set data is being accessed during development.
        This is like a checkpoint that ensures we don't accidentally peek at the future.
        """
        if self.development_phase:
            if start_date >= self.TEST_START or end_date >= self.TEST_START:
                raise ValueError(
                    f"🚨 DATA LEAKAGE DETECTED! 🚨\n"
                    f"Attempted to access test set data during development phase.\n"
                    f"Requested: {start_date} to {end_date}\n"
                    f"Test set starts: {self.TEST_START}\n"
                    f"This would compromise model integrity!"
                )
        return True
    
    def unlock_test_set(self, final_evaluation: bool = False):
        """
        Unlocks access to test set for final evaluation only.
        This is like opening a sealed envelope - can only be done once!
        """
        if final_evaluation and not self.test_set_accessed:
            self.development_phase = False
            self.test_set_accessed = True
            logging.warning("🔓 TEST SET UNLOCKED FOR FINAL EVALUATION")
            logging.warning("⚠️  This should only happen ONCE in the entire project!")
        elif self.test_set_accessed:
            logging.error("❌ Test set has already been accessed. Cannot access again!")
            raise RuntimeError("Test set contamination: Already accessed once!")


class ForexDataProcessor:
    """
    Main data processing class that handles loading, cleaning, and preparing
    forex data while maintaining strict temporal boundaries.
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.protection = DataLeakageProtection()
        
        # Currency pairs and timeframes to process
        self.currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
        self.timeframes = ['1H', '4H']
        
        # Spread assumptions (in pips)
        self.spreads = {
            'EURUSD': 2,
            'GBPUSD': 2, 
            'USDJPY': 2
        }
        
        # Initialize analysis engines
        self.indicator_engine = TechnicalIndicatorEngine()
        self.cross_currency_analyzer = CrossCurrencyAnalyzer(self.currency_pairs)
        self.adaptive_exit_strategy = AdaptiveExitStrategy(self.currency_pairs)
        
        # Storage for processed data
        self.raw_data = {}
        self.processed_data = {}
        self.feature_data = {}
        self.labeled_data = {}
        
        logging.info(f"ForexDataProcessor initialized for {len(self.currency_pairs)} pairs")
        logging.info(f"Timeframes: {self.timeframes}")
        logging.info(f"Spread assumptions: {self.spreads}")
        logging.info("Technical Indicator Engine, Cross-Currency Analyzer, and Adaptive Exit Strategy ready")
    
    def load_csv_file(self, filename: str) -> pd.DataFrame:
        """
        Loads a single CSV file with error handling and validation.
        This function reads the raw market data and performs initial quality checks.
        """
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            logging.info(f"Loaded {filename}: {len(df)} rows")
            
            # Validate column structure
            expected_columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in expected_columns):
                raise ValueError(f"Missing required columns in {filename}")
            
            # Convert Local time to datetime
            df['Local time'] = pd.to_datetime(df['Local time'])
            df.set_index('Local time', inplace=True)
            
            # Sort by time to ensure chronological order
            df.sort_index(inplace=True)
            
            # Basic data quality checks
            self._validate_ohlc_data(df, filename)
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def _validate_ohlc_data(self, df: pd.DataFrame, filename: str):
        """
        Validates OHLC data integrity.
        This ensures our price data makes logical sense (High >= Low, etc.)
        """
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (df[col] <= 0).any():
                logging.warning(f"Found non-positive prices in {filename}, column {col}")
        
        # Validate OHLC relationships
        invalid_high = (df['High'] < df['Low']).sum()
        invalid_open = ((df['Open'] > df['High']) | (df['Open'] < df['Low'])).sum()
        invalid_close = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
        
        if invalid_high > 0:
            logging.warning(f"{filename}: {invalid_high} rows where High < Low")
        if invalid_open > 0:
            logging.warning(f"{filename}: {invalid_open} rows where Open outside High-Low range")
        if invalid_close > 0:
            logging.warning(f"{filename}: {invalid_close} rows where Close outside High-Low range")
        
        logging.info(f"Data validation completed for {filename}")
    
    def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Loads all forex data files and organizes them by currency pair and timeframe.
        This creates our main data repository while checking for temporal consistency.
        """
        self.raw_data = {}
        
        for pair in self.currency_pairs:
            self.raw_data[pair] = {}
            
            for timeframe in self.timeframes:
                filename = f"{pair}_{timeframe}.csv"
                df = self.load_csv_file(filename)
                
                # Apply data leakage protection
                start_date = df.index.min().strftime('%Y-%m-%d')
                end_date = df.index.max().strftime('%Y-%m-%d')
                
                logging.info(f"{pair} {timeframe} data spans: {start_date} to {end_date}")
                
                self.raw_data[pair][timeframe] = df
        
        logging.info("All data files loaded successfully")
        return self.raw_data
    
    def split_data_safely(self) -> Tuple[Dict, Dict, Dict]:
        """
        Splits data into train, validation, and test sets with leakage protection.
        This is the most critical function for maintaining research integrity.
        """
        if not self.raw_data:
            raise ValueError("No data loaded. Please run load_all_data() first.")
        
        train_data = {}
        validation_data = {}
        test_data = {}
        
        for pair in self.currency_pairs:
            train_data[pair] = {}
            validation_data[pair] = {}
            test_data[pair] = {}
            
            for timeframe in self.timeframes:
                df = self.raw_data[pair][timeframe]
                
                # Apply temporal splits with protection
                train_mask = (df.index >= self.protection.TRAIN_START) & (df.index <= self.protection.TRAIN_END)
                val_mask = (df.index >= self.protection.VALIDATION_START) & (df.index <= self.protection.VALIDATION_END)
                test_mask = (df.index >= self.protection.TEST_START) & (df.index <= self.protection.TEST_END)
                
                train_data[pair][timeframe] = df[train_mask].copy()
                validation_data[pair][timeframe] = df[val_mask].copy()
                test_data[pair][timeframe] = df[test_mask].copy()
                
                # Log the split statistics
                logging.info(f"{pair} {timeframe} split - Train: {len(train_data[pair][timeframe])}, "
                           f"Val: {len(validation_data[pair][timeframe])}, "
                           f"Test: {len(test_data[pair][timeframe])}")
        
        # Validate splits don't overlap
        self._validate_temporal_splits(train_data, validation_data, test_data)
        
        logging.info("Data splitting completed with leakage protection")
        return train_data, validation_data, test_data
    
    def _validate_temporal_splits(self, train_data: Dict, validation_data: Dict, test_data: Dict):
        """
        Validates that temporal splits don't overlap and maintain chronological order.
        This double-checks our data leakage protection.
        """
        for pair in self.currency_pairs:
            for timeframe in self.timeframes:
                train_max = train_data[pair][timeframe].index.max()
                val_min = validation_data[pair][timeframe].index.min()
                val_max = validation_data[pair][timeframe].index.max()
                test_min = test_data[pair][timeframe].index.min()
                
                # Ensure no temporal overlap
                if train_max >= val_min:
                    raise ValueError(f"Temporal overlap detected: Train-Val in {pair} {timeframe}")
                if val_max >= test_min:
                    raise ValueError(f"Temporal overlap detected: Val-Test in {pair} {timeframe}")
                
                logging.debug(f"{pair} {timeframe} temporal validation passed")
    
    def get_development_data(self) -> Tuple[Dict, Dict]:
        """
        Returns only training and validation data for model development.
        This function ensures test set remains protected during development.
        """
        train_data, validation_data, _ = self.split_data_safely()
        
        # Double-check we're in development phase
        if not self.protection.development_phase:
            raise RuntimeError("Cannot access development data - system is in evaluation mode")
        
        logging.info("Development data retrieved (Train + Validation only)")
        return train_data, validation_data
    
    def get_test_data_for_final_evaluation(self) -> Dict:
        """
        🚨 CRITICAL FUNCTION 🚨
        Provides access to test set for final evaluation ONLY.
        This should be called exactly ONCE in the entire project lifecycle.
        """
        # Unlock test set for final evaluation
        self.protection.unlock_test_set(final_evaluation=True)
        
        _, _, test_data = self.split_data_safely()
        
        logging.critical("🔓 TEST SET ACCESSED FOR FINAL EVALUATION")
        logging.critical("⚠️  This is the final assessment - no more development allowed!")
        
        return test_data
    
    def create_comprehensive_features(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create comprehensive feature sets by combining technical indicators,
        cross-currency analysis, and adaptive exit indicators.
        
        This is the central feature engineering pipeline that transforms raw OHLCV data
        into rich, multi-dimensional features that our AI models can learn from.
        Think of this as a master chef who takes simple ingredients and transforms
        them into a sophisticated dish with multiple layers of flavor.
        
        Args:
            data_dict: Raw OHLCV data organized by {pair: {timeframe: dataframe}}
        
        Returns:
            Dictionary with comprehensive features for each pair and timeframe
        """
        logging.info("Starting comprehensive feature engineering pipeline")
        feature_dict = {}
        
        # Step 1: Create basic technical indicators for each pair and timeframe
        logging.info("Step 1: Creating basic technical indicators")
        for pair in self.currency_pairs:
            if pair not in data_dict:
                logging.warning(f"No data available for {pair}")
                continue
                
            feature_dict[pair] = {}
            
            for timeframe in self.timeframes:
                if timeframe not in data_dict[pair]:
                    logging.warning(f"No {timeframe} data for {pair}")
                    continue
                
                # Start with raw data
                df = data_dict[pair][timeframe].copy()
                
                # Add basic technical indicators
                df = self.indicator_engine.create_basic_indicators(df, timeframe)
                
                # Add adaptive exit indicators
                df = self.indicator_engine.create_adaptive_exit_indicators(df, timeframe)
                
                feature_dict[pair][timeframe] = df
                
                logging.info(f"Created features for {pair} {timeframe}: {len(df.columns)} total columns")
        
        # Step 2: Calculate cross-currency features
        logging.info("Step 2: Creating cross-currency features")
        
        # Currency strength indices
        strength_data = self.cross_currency_analyzer.calculate_currency_strength_indices(feature_dict)
        
        # Correlation features
        correlation_data = self.cross_currency_analyzer.calculate_correlation_features(feature_dict)
        
        # Cross-timeframe features
        cross_tf_data = self.cross_currency_analyzer.calculate_cross_timeframe_features(feature_dict)
        
        # Synthetic indicators
        synthetic_data = self.cross_currency_analyzer.create_synthetic_indicators(feature_dict, strength_data)
        
        # Step 3: Integrate cross-currency features back into main feature sets
        logging.info("Step 3: Integrating cross-currency features")
        for pair in self.currency_pairs:
            if pair not in feature_dict:
                continue
                
            for timeframe in self.timeframes:
                if timeframe not in feature_dict[pair]:
                    continue
                
                base_df = feature_dict[pair][timeframe]
                
                # Add currency strength features
                if timeframe in strength_data:
                    strength_df = strength_data[timeframe].reindex(base_df.index, method='ffill')
                    base_df = pd.concat([base_df, strength_df], axis=1)
                
                # Add correlation features
                if timeframe in correlation_data:
                    corr_df = correlation_data[timeframe].reindex(base_df.index, method='ffill')
                    base_df = pd.concat([base_df, corr_df], axis=1)
                
                # Add cross-timeframe features (only for this specific pair)
                if pair in cross_tf_data:
                    ctf_df = cross_tf_data[pair].reindex(base_df.index, method='ffill')
                    # Only add non-OHLCV columns to avoid duplication
                    ctf_features = ctf_df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')
                    base_df = pd.concat([base_df, ctf_features], axis=1)
                
                # Add synthetic indicators
                if timeframe in synthetic_data:
                    synthetic_df = synthetic_data[timeframe].reindex(base_df.index, method='ffill')
                    base_df = pd.concat([base_df, synthetic_df], axis=1)
                
                feature_dict[pair][timeframe] = base_df
                
                logging.info(f"Enhanced {pair} {timeframe}: {len(base_df.columns)} total features")
        
        # Step 4: Final feature engineering and cleaning
        logging.info("Step 4: Final feature processing and validation")
        feature_dict = self._finalize_features(feature_dict)
        
        logging.info("Comprehensive feature engineering completed successfully")
        return feature_dict
    
    def _finalize_features(self, feature_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Finalize feature engineering by handling missing values, creating interaction features,
        and ensuring all features are ready for machine learning models.
        
        This is like the quality control and packaging stage in manufacturing -
        we make sure everything is clean, properly formatted, and ready for use.
        """
        processed_features = {}
        
        for pair in feature_dict:
            processed_features[pair] = {}
            
            for timeframe in feature_dict[pair]:
                df = feature_dict[pair][timeframe].copy()
                
                # Handle missing values intelligently
                df = self._handle_missing_values(df)
                
                # Create interaction features
                df = self._create_interaction_features(df, pair, timeframe)
                
                # Feature scaling and normalization indicators
                df = self._add_normalization_features(df)
                
                # Remove highly correlated features to prevent multicollinearity
                df = self._remove_highly_correlated_features(df, threshold=0.95)
                
                # Ensure no infinite or extremely large values
                df = self._clean_extreme_values(df)
                
                processed_features[pair][timeframe] = df
                
                logging.info(f"Finalized {pair} {timeframe}: {len(df.columns)} features after processing")
        
        return processed_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values intelligently based on the type of feature.
        Technical indicators may have NaN values at the beginning due to rolling calculations.
        """
        # For price-based indicators, forward fill is usually appropriate
        price_indicators = [col for col in df.columns if any(x in col.lower() for x in ['ema', 'sma', 'price', 'close', 'open', 'high', 'low'])]
        for col in price_indicators:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # For oscillators (RSI, Stochastic), use median fill
        oscillators = [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'stoch', 'cci'])]
        for col in oscillators:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # For correlation and strength indicators, interpolate
        correlation_cols = [col for col in df.columns if any(x in col.lower() for x in ['corr', 'strength', 'divergence'])]
        for col in correlation_cols:
            if col in df.columns:
                df[col] = df[col].interpolate().fillna(0)
        
        # For binary indicators (session, regime), use mode
        binary_cols = [col for col in df.columns if df[col].nunique() <= 3]
        for col in binary_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_val)
        
        # For any remaining missing values, use forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Create interaction features that capture relationships between different indicators.
        These interactions often contain valuable information that individual indicators miss.
        """
        # RSI and Bollinger Band interactions
        if all(col in df.columns for col in ['rsi', 'bb_position']):
            df['rsi_bb_interaction'] = df['rsi'] * df['bb_position']
            df['rsi_overbought_in_upper_bb'] = ((df['rsi'] > 70) & (df['bb_position'] > 0.8)).astype(int)
            df['rsi_oversold_in_lower_bb'] = ((df['rsi'] < 30) & (df['bb_position'] < 0.2)).astype(int)
        
        # MACD and trend strength interactions
        if all(col in df.columns for col in ['macd_histogram', 'trend_strength']):
            df['macd_trend_alignment'] = df['macd_histogram'] * df['trend_strength']
        
        # Volatility and momentum interactions
        if all(col in df.columns for col in ['atr', 'momentum_persistence']):
            df['volatility_momentum_interaction'] = (df['atr'] / df['Close']) * df['momentum_persistence']
        
        # Currency strength and price action
        base_currency = pair[:3]
        quote_currency = pair[3:]
        base_strength_col = f'{base_currency}_strength'
        quote_strength_col = f'{quote_currency}_strength'
        
        if all(col in df.columns for col in [base_strength_col, quote_strength_col]):
            df['currency_strength_differential'] = df[base_strength_col] - df[quote_strength_col]
        
        # Session and volatility interactions
        session_cols = [col for col in df.columns if 'session' in col.lower()]
        if session_cols and 'atr' in df.columns:
            for session_col in session_cols:
                df[f'{session_col}_volatility'] = df[session_col] * (df['atr'] / df['Close'])
        
        return df
    
    def _add_normalization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that help models understand the relative scale and distribution of indicators.
        This helps the model understand not just the raw values but their significance.
        """
        # Percentile ranks for key indicators (helps models understand relative positioning)
        key_indicators = ['rsi', 'cci', 'bb_position']
        for indicator in key_indicators:
            if indicator in df.columns:
                df[f'{indicator}_percentile_rank'] = df[indicator].rolling(100).rank(pct=True)
        
        # Z-scores for price-based indicators
        price_indicators = [col for col in df.columns if any(x in col for x in ['ema', 'Close', 'price'])]
        for indicator in price_indicators[:3]:  # Limit to prevent too many features
            if indicator in df.columns:
                rolling_mean = df[indicator].rolling(50).mean()
                rolling_std = df[indicator].rolling(50).std()
                df[f'{indicator}_zscore'] = (df[indicator] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    def _remove_highly_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove features that are highly correlated to prevent multicollinearity.
        We keep the feature that has higher correlation with price movement.
        """
        # Only check non-price columns for correlation removal
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in price_cols]
        
        if len(feature_cols) < 2:
            return df
        
        # Calculate correlation matrix for features only
        correlation_matrix = df[feature_cols].corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        # For each pair, remove the one with lower correlation to price change
        price_change = df['Close'].pct_change()
        to_remove = set()
        
        for col1, col2 in high_corr_pairs:
            if col1 in to_remove or col2 in to_remove:
                continue
                
            corr1 = abs(df[col1].corr(price_change))
            corr2 = abs(df[col2].corr(price_change))
            
            if corr1 < corr2:
                to_remove.add(col1)
            else:
                to_remove.add(col2)
        
        if to_remove:
            df = df.drop(columns=list(to_remove))
            logging.info(f"Removed {len(to_remove)} highly correlated features")
        
        return df
    
    def _clean_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean extreme values that could destabilize model training.
        Use winsorization to cap extreme values at reasonable levels.
        """
        # Define columns that shouldn't be winsorized (binary, categorical)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + \
                      [col for col in df.columns if df[col].nunique() <= 3]
        
        winsorize_cols = [col for col in df.columns if col not in exclude_cols]
        
        for col in winsorize_cols:
            if df[col].dtype in ['float64', 'int64']:
                # Cap at 1st and 99th percentiles
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Replace any remaining infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def create_ml_ready_datasets(self) -> Tuple[Dict, Dict]:
        """
        Create machine learning ready datasets with proper feature engineering
        and data leakage protection.
        
        This function orchestrates the entire feature engineering pipeline and
        returns clean, comprehensive datasets ready for model training.
        
        Returns:
            Tuple of (train_features, validation_features) dictionaries
        """
        logging.info("Creating ML-ready datasets with comprehensive features")
        
        # Get development data (train + validation only)
        train_data, validation_data = self.get_development_data()
        
        # Create comprehensive features for training data
        logging.info("Processing training data features...")
        train_features = self.create_comprehensive_features(train_data)
        
        # Create comprehensive features for validation data
        logging.info("Processing validation data features...")
        validation_features = self.create_comprehensive_features(validation_data)
        
        # Store for future use
        self.feature_data['train'] = train_features
        self.feature_data['validation'] = validation_features
        
        # Log feature summary
        self._log_feature_summary(train_features, 'training')
        self._log_feature_summary(validation_features, 'validation')
        
        logging.info("ML-ready datasets created successfully")
        return train_features, validation_features
    
    def create_labeled_training_datasets(self) -> Tuple[Dict, Dict]:
        """
        Create complete labeled training datasets that include both comprehensive features
        and adaptive exit strategy labels.
        
        This is the culmination of our data preparation pipeline, producing datasets
        that are ready for training AI models with embedded adaptive exit strategy.
        
        Think of this as the final assembly line where we take all our engineered
        components (features) and add the sophisticated control system (adaptive labels)
        that will guide our AI's decision-making process.
        
        Returns:
            Tuple of (labeled_train_data, labeled_validation_data) with both features and labels
        """
        logging.info("Creating complete labeled training datasets")
        logging.info("This combines: Features + Adaptive Exit Labels + Risk Management")
        
        # First, create comprehensive features if not already done
        if not hasattr(self, 'train_features') or not hasattr(self, 'validation_features'):
            self.train_features, self.validation_features = self.create_ml_ready_datasets()
        
        # Create adaptive exit labels for training data
        logging.info("Creating adaptive exit labels for training data...")
        labeled_train_data = self.adaptive_exit_strategy.create_adaptive_labels(self.train_features)
        
        # Create adaptive exit labels for validation data
        logging.info("Creating adaptive exit labels for validation data...")
        labeled_validation_data = self.adaptive_exit_strategy.create_adaptive_labels(self.validation_features)
        
        # Store labeled data
        self.labeled_data['train'] = labeled_train_data
        self.labeled_data['validation'] = labeled_validation_data
        
        # Log comprehensive statistics
        self._log_labeled_dataset_summary(labeled_train_data, 'training')
        self._log_labeled_dataset_summary(labeled_validation_data, 'validation')
        
        logging.info("✅ Complete labeled training datasets ready for model training")
        return labeled_train_data, labeled_validation_data
    
    def _log_labeled_dataset_summary(self, labeled_data: Dict[str, Dict[str, pd.DataFrame]], dataset_name: str):
        """
        Log comprehensive summary of the labeled datasets including both features and labels.
        """
        logging.info(f"=== {dataset_name.upper()} LABELED DATASET SUMMARY ===")
        
        total_features = 0
        total_labels = 0
        
        for pair in labeled_data:
            for timeframe in labeled_data[pair]:
                df = labeled_data[pair][timeframe]
                
                # Count features vs labels
                feature_cols = [col for col in df.columns if not any(x in col.lower() for x in 
                               ['trade_direction', 'prob_profit', 'optimal_exit', 'confidence', 'expected_profit'])]
                label_cols = [col for col in df.columns if any(x in col.lower() for x in 
                             ['trade_direction', 'prob_profit', 'optimal_exit', 'confidence', 'expected_profit'])]
                
                n_features = len(feature_cols)
                n_labels = len(label_cols)
                n_samples = len(df)
                
                total_features += n_features
                total_labels += n_labels
                
                logging.info(f"{pair} {timeframe}: {n_samples} samples")
                logging.info(f"  - Features: {n_features}")
                logging.info(f"  - Labels: {n_labels}")
                
                # Log label categories
                direction_labels = len([col for col in label_cols if 'direction' in col])
                probability_labels = len([col for col in label_cols if 'prob_' in col])
                timing_labels = len([col for col in label_cols if 'optimal_exit' in col])
                confidence_labels = len([col for col in label_cols if 'confidence' in col])
                
                logging.info(f"  - Direction labels: {direction_labels}")
                logging.info(f"  - Probability labels: {probability_labels}")
                logging.info(f"  - Timing labels: {timing_labels}")
                logging.info(f"  - Confidence labels: {confidence_labels}")
                
                # Sample some key statistics
                if 'trade_direction' in df.columns:
                    direction_dist = df['trade_direction'].value_counts()
                    total_signals = len(df)
                    logging.info(f"  - Trade signals: Long={direction_dist.get(1,0)} "
                               f"({direction_dist.get(1,0)/total_signals*100:.1f}%), "
                               f"Short={direction_dist.get(-1,0)} "
                               f"({direction_dist.get(-1,0)/total_signals*100:.1f}%), "
                               f"None={direction_dist.get(0,0)} "
                               f"({direction_dist.get(0,0)/total_signals*100:.1f}%)")
        
        logging.info(f"Total across all pairs/timeframes:")
        logging.info(f"  - Features: {total_features}")
        logging.info(f"  - Labels: {total_labels}")
        logging.info(f"  - Feature/Label ratio: {total_features/max(total_labels,1):.1f}:1")
        logging.info("=" * 60)
    
    def _log_feature_summary(self, feature_dict: Dict[str, Dict[str, pd.DataFrame]], dataset_name: str):
        """
        Log a comprehensive summary of features created for analysis and debugging.
        """
        logging.info(f"=== {dataset_name.upper()} DATASET FEATURE SUMMARY ===")
        
        total_features = 0
        for pair in feature_dict:
            for timeframe in feature_dict[pair]:
                df = feature_dict[pair][timeframe]
                n_features = len(df.columns)
                n_samples = len(df)
                total_features += n_features
                
                logging.info(f"{pair} {timeframe}: {n_samples} samples, {n_features} features")
                
                # Log feature categories
                technical_features = len([col for col in df.columns if any(x in col.lower() for x in ['ema', 'rsi', 'macd', 'bb', 'atr'])])
                cross_currency_features = len([col for col in df.columns if any(x in col.lower() for x in ['strength', 'corr', 'divergence'])])
                adaptive_features = len([col for col in df.columns if any(x in col.lower() for x in ['momentum', 'trend', 'exit', 'profit'])])
                session_features = len([col for col in df.columns if 'session' in col.lower()])
                
                logging.info(f"  - Technical indicators: {technical_features}")
                logging.info(f"  - Cross-currency features: {cross_currency_features}")
                logging.info(f"  - Adaptive exit features: {adaptive_features}")
                logging.info(f"  - Session features: {session_features}")
        
        logging.info(f"Total features across all pairs/timeframes: {total_features}")
        logging.info("=" * 50)


class TechnicalIndicatorEngine:
    """
    A comprehensive engine for creating technical indicators tailored to our
    Multi-Currency Bagging approach with Adaptive Exit Strategy.
    
    This class is like a Swiss Army knife for technical analysis - it contains
    all the tools needed to transform raw price data into meaningful signals
    that our AI models can understand and learn from.
    
    The indicators are specifically designed to support:
    1. Fast response for 1H timeframe (for t+1 decisions)
    2. Trend confirmation for 4H timeframe (for t+2, t+3 context)
    3. Cross-currency relationship detection
    4. Market regime identification
    """
    
    def __init__(self):
        """
        Initialize the technical indicator engine with optimized parameters
        for forex trading and adaptive exit strategies.
        """
        # Define optimized periods for different timeframes
        self.fast_periods = {
            '1H': {
                'ema_fast': 9,
                'ema_medium': 12, 
                'ema_slow': 26,
                'rsi': 14,
                'bollinger': 20,
                'stoch': 14
            }
        }
        
        self.slow_periods = {
            '4H': {
                'ema_fast': 21,
                'ema_medium': 50,
                'ema_slow': 100,
                'ema_very_slow': 200,
                'rsi': 14,
                'bollinger': 20,
                'stoch': 14
            }
        }
        
        logging.info("Technical Indicator Engine initialized")
        logging.info(f"TA-Lib available: {TALIB_AVAILABLE}")
    
    def create_basic_indicators(self, df: pd.DataFrame, timeframe: str = '1H') -> pd.DataFrame:
        """
        Creates fundamental technical indicators that form the backbone of our analysis.
        
        Think of these indicators as the vital signs of the market - just like a doctor
        checks pulse, blood pressure, and temperature to understand a patient's health,
        we use these indicators to understand market health and direction.
        
        Args:
            df: OHLCV DataFrame with datetime index
            timeframe: Either '1H' for fast-response indicators or '4H' for trend indicators
        
        Returns:
            DataFrame with original data plus technical indicators
        """
        result_df = df.copy()
        periods = self.fast_periods['1H'] if timeframe == '1H' else self.slow_periods['4H']
        
        logging.info(f"Creating basic indicators for {timeframe} timeframe")
        
        # === MOVING AVERAGES ===
        # Exponential Moving Averages - these respond faster to recent price changes
        # Like a person who pays more attention to recent events than old ones
        result_df[f'ema_{periods["ema_fast"]}'] = self._calculate_ema(df['Close'], periods['ema_fast'])
        result_df[f'ema_{periods["ema_medium"]}'] = self._calculate_ema(df['Close'], periods['ema_medium'])
        result_df[f'ema_{periods["ema_slow"]}'] = self._calculate_ema(df['Close'], periods['ema_slow'])
        
        if timeframe == '4H':
            # Additional long-term EMA for 4H timeframe
            result_df[f'ema_{periods["ema_very_slow"]}'] = self._calculate_ema(df['Close'], periods['ema_very_slow'])
        
        # === MOMENTUM INDICATORS ===
        # RSI (Relative Strength Index) - measures if currency is overbought or oversold
        # Like checking if someone has been running too fast and needs to rest
        result_df['rsi'] = self._calculate_rsi(df['Close'], periods['rsi'])
        
        # MACD - shows relationship between two moving averages
        # Reveals the momentum behind price movements
        macd_line, macd_signal, macd_histogram = self._calculate_macd(df['Close'])
        result_df['macd_line'] = macd_line
        result_df['macd_signal'] = macd_signal
        result_df['macd_histogram'] = macd_histogram
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands - show price volatility and potential support/resistance
        # Like elastic bands that stretch when market gets excited
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'], periods['bollinger'])
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle  # Normalized width
        result_df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)  # Position within bands
        
        # === VOLUME INDICATORS ===
        # Volume-based indicators help confirm price movements
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            # On-Balance Volume - accumulates volume based on price direction
            result_df['obv'] = self._calculate_obv(df['Close'], df['Volume'])
            
            # Volume Rate of Change
            result_df['volume_roc'] = df['Volume'].pct_change(periods=5)
        else:
            # For forex data without meaningful volume, create placeholder
            result_df['obv'] = 0
            result_df['volume_roc'] = 0
        
        # === PRICE ACTION INDICATORS ===
        # True Range and ATR - measure volatility in price movement
        result_df['true_range'] = self._calculate_true_range(df)
        result_df['atr'] = result_df['true_range'].rolling(window=14).mean()
        
        # Price Rate of Change - momentum indicator
        result_df['price_roc'] = df['Close'].pct_change(periods=10)
        
        # Commodity Channel Index - identifies cyclical turns
        result_df['cci'] = self._calculate_cci(df)
        
        # Stochastic Oscillator - momentum indicator comparing closing price to price range
        stoch_k, stoch_d = self._calculate_stochastic(df, periods['stoch'])
        result_df['stoch_k'] = stoch_k
        result_df['stoch_d'] = stoch_d
        
        logging.info(f"Created {len([col for col in result_df.columns if col not in df.columns])} basic indicators")
        return result_df
    
    def create_adaptive_exit_indicators(self, df: pd.DataFrame, timeframe: str = '1H') -> pd.DataFrame:
        """
        Creates specialized indicators designed specifically for our Adaptive Exit Strategy.
        
        These indicators are tuned to detect early profit opportunities (for t+1 exits)
        and momentum continuation patterns (for t+2, t+3 decisions). Think of them as
        specialized sensors that help determine the best timing for exiting trades.
        
        Args:
            df: DataFrame with basic indicators already calculated
            timeframe: Either '1H' or '4H'
        
        Returns:
            DataFrame with additional adaptive exit indicators
        """
        result_df = df.copy()
        
        logging.info(f"Creating adaptive exit indicators for {timeframe}")
        
        # === QUICK PROFIT SIGNALS (for t+1 exits) ===
        if timeframe == '1H':
            # Fast momentum reversal detection
            result_df['momentum_reversal'] = self._detect_momentum_reversal(df)
            
            # Quick profit threshold indicators
            # These help identify when a small profit is likely to be the maximum we'll see
            result_df['quick_profit_signal'] = self._calculate_quick_profit_signal(df)
            
            # Micro-trend exhaustion
            result_df['micro_trend_exhaustion'] = self._calculate_micro_trend_exhaustion(df)
        
        # === CONTINUATION PATTERNS (for t+2, t+3 decisions) ===
        # Trend strength indicator - helps decide whether to hold longer
        result_df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Momentum persistence score
        result_df['momentum_persistence'] = self._calculate_momentum_persistence(df)
        
        # Support/Resistance proximity
        result_df['sr_proximity'] = self._calculate_support_resistance_proximity(df)
        
        # === ADAPTIVE THRESHOLDS ===
        # Dynamic profit targets based on market volatility
        result_df['dynamic_profit_target'] = self._calculate_dynamic_profit_target(df)
        
        # Risk-adjusted exit signals
        result_df['risk_adjusted_exit'] = self._calculate_risk_adjusted_exit(df)
        
        logging.info(f"Created adaptive exit indicators for {timeframe}")
        return result_df
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average using pandas for consistency."""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        RSI measures the speed and magnitude of price changes, ranging from 0 to 100.
        """
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)
        else:
            # Custom RSI calculation
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        MACD shows the relationship between two moving averages of a security's price.
        """
        if TALIB_AVAILABLE:
            macd_line, macd_signal, macd_hist = talib.MACD(series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return (pd.Series(macd_line, index=series.index),
                    pd.Series(macd_signal, index=series.index),
                    pd.Series(macd_hist, index=series.index))
        else:
            # Custom MACD calculation
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        These bands expand and contract based on market volatility.
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.
        OBV adds volume on up days and subtracts volume on down days.
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range - the greatest of three price differences.
        True Range captures the full extent of price movement including gaps.
        """
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return pd.Series(true_range, index=df.index)
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index.
        CCI identifies cyclical turns in commodities and forex.
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        ma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (typical_price - ma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        Shows the location of the close relative to the high-low range over a set period.
        """
        lowest_low = df['Low'].rolling(window=period).min()
        highest_high = df['High'].rolling(window=period).max()
        
        stoch_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(window=3).mean()  # 3-period SMA of %K
        
        return stoch_k, stoch_d
    
    def _detect_momentum_reversal(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect early momentum reversals for quick profit-taking opportunities.
        This indicator is specifically designed for t+1 exit decisions.
        """
        # Combine multiple momentum signals
        rsi_reversal = ((df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)) | ((df['rsi'] < 30) & (df['rsi'].shift(1) >= 30))
        macd_reversal = (df['macd_line'] > df['macd_signal']) != (df['macd_line'].shift(1) > df['macd_signal'].shift(1))
        stoch_reversal = ((df['stoch_k'] > 80) & (df['stoch_k'].shift(1) <= 80)) | ((df['stoch_k'] < 20) & (df['stoch_k'].shift(1) >= 20))
        
        # Combine signals with weights
        reversal_score = (rsi_reversal.astype(int) * 0.4 + 
                         macd_reversal.astype(int) * 0.3 + 
                         stoch_reversal.astype(int) * 0.3)
        
        return reversal_score
    
    def _calculate_quick_profit_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signals that indicate quick profit opportunities should be taken.
        Based on the philosophy that small profits taken quickly are better than waiting for larger profits that may not materialize.
        """
        # Price position relative to recent range
        price_position = (df['Close'] - df['Low'].rolling(10).min()) / (df['High'].rolling(10).max() - df['Low'].rolling(10).min())
        
        # Volatility-adjusted momentum
        momentum = df['Close'].pct_change(3)
        volatility = df['atr'] / df['Close']
        adjusted_momentum = momentum / volatility
        
        # Quick profit signal combines position and momentum
        quick_profit_signal = (price_position * 0.6 + adjusted_momentum.rank(pct=True) * 0.4)
        
        return quick_profit_signal
    
    def _calculate_micro_trend_exhaustion(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect when micro-trends are showing signs of exhaustion.
        This helps identify optimal exit points before momentum fades.
        """
        # Short-term trend strength
        short_trend = (df['Close'] / df['Close'].shift(5) - 1) * 100
        
        # Decreasing momentum
        momentum_change = short_trend - short_trend.shift(1)
        
        # Volume confirmation (if available)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            volume_trend = df['Volume'].rolling(3).mean() / df['Volume'].rolling(10).mean()
        else:
            volume_trend = pd.Series(1, index=df.index)  # Neutral if no volume data
        
        # Exhaustion signal
        exhaustion = ((momentum_change < 0) & (short_trend > 0) & (volume_trend < 1)).astype(int)
        
        return exhaustion
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate overall trend strength to help with t+2 and t+3 exit decisions.
        Strong trends suggest holding positions longer.
        """
        # Multiple timeframe trend alignment
        short_ema = df[f'ema_9'] if f'ema_9' in df.columns else df['Close'].ewm(span=9).mean()
        medium_ema = df[f'ema_21'] if f'ema_21' in df.columns else df['Close'].ewm(span=21).mean()
        long_ema = df[f'ema_50'] if f'ema_50' in df.columns else df['Close'].ewm(span=50).mean()
        
        # Trend alignment score
        trend_alignment = ((short_ema > medium_ema) & (medium_ema > long_ema)).astype(int) - ((short_ema < medium_ema) & (medium_ema < long_ema)).astype(int)
        
        # ADX-like trend strength calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        directional_movement = np.where(df['High'].diff() > -df['Low'].diff(), 
                                       np.maximum(df['High'].diff(), 0), 0)
        
        trend_strength = (directional_movement.rolling(14).mean() / true_range.rolling(14).mean()) * trend_alignment
        
        return pd.Series(trend_strength, index=df.index)
    
    def _calculate_momentum_persistence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate how likely momentum is to persist, helping with hold vs exit decisions.
        """
        # Price momentum over multiple periods
        momentum_3 = df['Close'].pct_change(3)
        momentum_5 = df['Close'].pct_change(5)
        momentum_10 = df['Close'].pct_change(10)
        
        # Momentum consistency
        momentum_consistency = ((momentum_3 > 0) == (momentum_5 > 0)) & ((momentum_5 > 0) == (momentum_10 > 0))
        
        # Accelerating momentum
        momentum_acceleration = (momentum_3.abs() > momentum_5.abs()) & (momentum_5.abs() > momentum_10.abs())
        
        # Combine factors
        persistence_score = (momentum_consistency.astype(int) * 0.6 + momentum_acceleration.astype(int) * 0.4)
        
        return persistence_score
    
    def _calculate_support_resistance_proximity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate proximity to potential support/resistance levels.
        Being close to these levels might suggest taking profits early.
        """
        # Recent highs and lows as support/resistance
        resistance_20 = df['High'].rolling(20).max()
        support_20 = df['Low'].rolling(20).min()
        
        # Distance to support/resistance as percentage of ATR
        dist_to_resistance = (resistance_20 - df['Close']) / df['atr']
        dist_to_support = (df['Close'] - support_20) / df['atr']
        
        # Proximity score (higher when close to S/R)
        proximity_score = 1 / (1 + np.minimum(dist_to_resistance.abs(), dist_to_support.abs()))
        
        return proximity_score
    
    def _calculate_dynamic_profit_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate dynamic profit targets based on current market volatility.
        In volatile markets, we might want to take profits sooner.
        """
        # Base profit target (e.g., 2 pips to cover spread)
        base_target = 0.0002  # 2 pips for major pairs
        
        # Volatility adjustment
        volatility_multiplier = df['atr'] / df['Close'] / 0.001  # Normalize to typical forex volatility
        
        # Trend strength adjustment
        trend_multiplier = 1 + (df.get('trend_strength', 0) * 0.5)
        
        # Dynamic target
        dynamic_target = base_target * volatility_multiplier * trend_multiplier
        
        return dynamic_target
    
    def _calculate_risk_adjusted_exit(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate risk-adjusted exit signals that consider both profit potential and risk.
        """
        # Current profit as percentage of ATR
        if 'entry_price' in df.columns:
            current_profit = (df['Close'] - df['entry_price']) / df['atr']
        else:
            # Proxy using short-term price change
            current_profit = df['Close'].pct_change(1) / (df['atr'] / df['Close'])
        
        # Risk level (how much we could lose)
        risk_level = df['atr'] / df['Close']
        
        # Risk-adjusted signal (take profit when risk-adjusted return is sufficient)
        risk_adjusted_exit = (current_profit / risk_level) > 1.5  # Take profit when return > 1.5x risk
        
        return risk_adjusted_exit.astype(int)


class CrossCurrencyAnalyzer:
    """
    Advanced Cross-Currency Analysis Engine for Multi-Currency Bagging
    
    This is the crown jewel of our system - the component that gives us the edge
    over single-currency trading approaches. It analyzes relationships between
    EURUSD, GBPUSD, and USDJPY to extract insights that wouldn't be visible
    when looking at each pair in isolation.
    
    Think of this as having multiple weather stations across a region. While
    each station tells you local conditions, looking at all stations together
    reveals weather patterns, storm movements, and regional trends that no
    single station could detect.
    """
    
    def __init__(self, currency_pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY']):
        """
        Initialize the cross-currency analyzer with the currency pairs we'll analyze.
        
        Args:
            currency_pairs: List of currency pairs to analyze relationships between
        """
        self.currency_pairs = currency_pairs
        self.base_currencies = ['EUR', 'GBP', 'USD']
        self.quote_currencies = ['USD', 'USD', 'JPY']
        
        # Rolling windows for different types of analysis
        self.correlation_windows = [24, 72, 168]  # 1 day, 3 days, 1 week (in hours)
        self.strength_windows = [12, 24, 48]      # Short, medium, long-term strength
        
        logging.info(f"Cross-Currency Analyzer initialized for pairs: {currency_pairs}")
    
    def calculate_currency_strength_indices(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate individual currency strength indices by analyzing how each currency
        performs against others. This helps identify which currencies are inherently
        strong or weak at any given time.
        
        Think of this like calculating batting averages in baseball - we look at how
        each currency performs across all its "games" (trading pairs) to get a true
        measure of its strength.
        
        Args:
            data_dict: Dictionary with structure {pair: {timeframe: dataframe}}
        
        Returns:
            Dictionary with currency strength data for each timeframe
        """
        strength_data = {}
        
        for timeframe in ['1H', '4H']:
            logging.info(f"Calculating currency strength indices for {timeframe}")
            
            # Get price data for all pairs
            price_data = {}
            for pair in self.currency_pairs:
                if pair in data_dict and timeframe in data_dict[pair]:
                    price_data[pair] = data_dict[pair][timeframe]['Close']
            
            if len(price_data) < len(self.currency_pairs):
                logging.warning(f"Missing data for some pairs in {timeframe}")
                continue
            
            # Normalize all prices to start at 100 for comparison
            normalized_prices = {}
            for pair, prices in price_data.items():
                normalized_prices[pair] = (prices / prices.iloc[0]) * 100
            
            # Calculate individual currency strength
            strength_df = pd.DataFrame(index=normalized_prices[self.currency_pairs[0]].index)
            
            # USD strength (appears in all our pairs)
            # Strong USD = EURUSD down, GBPUSD down, USDJPY up
            usd_strength = (
                (100 - normalized_prices['EURUSD']) * 0.4 +  # USD strength vs EUR
                (100 - normalized_prices['GBPUSD']) * 0.4 +  # USD strength vs GBP
                (normalized_prices['USDJPY'] - 100) * 0.2    # USD strength vs JPY
            )
            strength_df['USD_strength'] = usd_strength
            
            # EUR strength (only in EURUSD)
            # Strong EUR = EURUSD up
            eur_strength = normalized_prices['EURUSD'] - 100
            strength_df['EUR_strength'] = eur_strength
            
            # GBP strength (only in GBPUSD)
            # Strong GBP = GBPUSD up
            gbp_strength = normalized_prices['GBPUSD'] - 100
            strength_df['GBP_strength'] = gbp_strength
            
            # JPY strength (only in USDJPY)
            # Strong JPY = USDJPY down
            jpy_strength = 100 - normalized_prices['USDJPY']
            strength_df['JPY_strength'] = jpy_strength
            
            # Calculate relative strength rankings
            currency_strengths = strength_df[['USD_strength', 'EUR_strength', 'GBP_strength', 'JPY_strength']]
            for window in self.strength_windows:
                # Rolling rank of each currency's strength
                strength_ranks = currency_strengths.rolling(window=window).rank(axis=1, pct=True)
                strength_ranks.columns = [f"{col}_rank_{window}h" for col in strength_ranks.columns]
                strength_df = pd.concat([strength_df, strength_ranks], axis=1)
            
            strength_data[timeframe] = strength_df
        
        return strength_data
    
    def calculate_correlation_features(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate dynamic correlation features between currency pairs over multiple timeframes.
        
        Correlations in forex change over time based on economic conditions, market sentiment,
        and global events. By tracking these changing relationships, we can identify
        opportunities when normal correlations break down or strengthen.
        
        Args:
            data_dict: Dictionary with structure {pair: {timeframe: dataframe}}
        
        Returns:
            Dictionary with correlation features for each timeframe
        """
        correlation_data = {}
        
        for timeframe in ['1H', '4H']:
            logging.info(f"Calculating correlation features for {timeframe}")
            
            # Get return data for all pairs
            returns_data = {}
            for pair in self.currency_pairs:
                if pair in data_dict and timeframe in data_dict[pair]:
                    returns_data[pair] = data_dict[pair][timeframe]['Close'].pct_change()
            
            if len(returns_data) < len(self.currency_pairs):
                logging.warning(f"Missing data for correlation analysis in {timeframe}")
                continue
            
            # Create correlation features DataFrame
            corr_df = pd.DataFrame(index=returns_data[self.currency_pairs[0]].index)
            
            # Calculate rolling correlations between all pair combinations
            pair_combinations = [
                ('EURUSD', 'GBPUSD'),
                ('EURUSD', 'USDJPY'),
                ('GBPUSD', 'USDJPY')
            ]
            
            for pair1, pair2 in pair_combinations:
                if pair1 in returns_data and pair2 in returns_data:
                    for window in self.correlation_windows:
                        # Rolling correlation
                        rolling_corr = returns_data[pair1].rolling(window=window).corr(returns_data[pair2])
                        corr_df[f'{pair1}_{pair2}_corr_{window}h'] = rolling_corr
                        
                        # Correlation strength (absolute value)
                        corr_df[f'{pair1}_{pair2}_corr_strength_{window}h'] = rolling_corr.abs()
                        
                        # Correlation stability (rolling standard deviation of correlation)
                        corr_stability = rolling_corr.rolling(window=window//2).std()
                        corr_df[f'{pair1}_{pair2}_corr_stability_{window}h'] = corr_stability
            
            # Cross-correlation lead-lag relationships
            for pair1, pair2 in pair_combinations:
                if pair1 in returns_data and pair2 in returns_data:
                    # Check if one pair leads the other (correlation with lag)
                    corr_lead_1 = returns_data[pair1].rolling(window=24).corr(returns_data[pair2].shift(1))
                    corr_lag_1 = returns_data[pair1].shift(1).rolling(window=24).corr(returns_data[pair2])
                    
                    corr_df[f'{pair1}_leads_{pair2}'] = corr_lead_1
                    corr_df[f'{pair1}_lags_{pair2}'] = corr_lag_1
            
            # Divergence detection
            corr_df = self._calculate_divergence_indicators(corr_df, returns_data)
            
            correlation_data[timeframe] = corr_df
        
        return correlation_data
    
    def calculate_cross_timeframe_features(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate features that combine insights from both 1H and 4H timeframes.
        
        This creates a multi-dimensional view of the market, like having both
        a microscope (1H) and a telescope (4H) view of the same phenomenon.
        The combination often reveals patterns invisible to either timeframe alone.
        
        Args:
            data_dict: Dictionary with structure {pair: {timeframe: dataframe}}
        
        Returns:
            Dictionary with cross-timeframe features
        """
        cross_tf_data = {}
        
        for pair in self.currency_pairs:
            if pair not in data_dict or '1H' not in data_dict[pair] or '4H' not in data_dict[pair]:
                logging.warning(f"Missing timeframe data for {pair}")
                continue
            
            logging.info(f"Calculating cross-timeframe features for {pair}")
            
            # Get 1H data as base (more frequent)
            df_1h = data_dict[pair]['1H'].copy()
            df_4h = data_dict[pair]['4H'].copy()
            
            # Align 4H data with 1H timestamps
            df_4h_aligned = df_4h.reindex(df_1h.index, method='ffill')
            
            cross_tf_df = df_1h[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Timeframe divergence indicators
            # Compare short-term (1H) momentum with longer-term (4H) trend
            if 'rsi' in df_1h.columns and 'rsi' in df_4h_aligned.columns:
                cross_tf_df['rsi_1h_4h_div'] = df_1h['rsi'] - df_4h_aligned['rsi']
                cross_tf_df['rsi_timeframe_alignment'] = (
                    ((df_1h['rsi'] > 50) == (df_4h_aligned['rsi'] > 50)).astype(int)
                )
            
            # MACD timeframe analysis
            if all(col in df_1h.columns for col in ['macd_line', 'macd_signal']) and \
               all(col in df_4h_aligned.columns for col in ['macd_line', 'macd_signal']):
                
                macd_1h_signal = (df_1h['macd_line'] > df_1h['macd_signal']).astype(int)
                macd_4h_signal = (df_4h_aligned['macd_line'] > df_4h_aligned['macd_signal']).astype(int)
                
                cross_tf_df['macd_timeframe_alignment'] = (macd_1h_signal == macd_4h_signal).astype(int)
                cross_tf_df['macd_1h_4h_strength'] = (
                    df_1h['macd_histogram'].abs() / df_4h_aligned['macd_histogram'].abs().replace(0, np.nan)
                )
            
            # Trend alignment across timeframes
            # Calculate simple trend for each timeframe
            trend_1h = (df_1h['Close'] > df_1h['Close'].shift(4)).astype(int) - (df_1h['Close'] < df_1h['Close'].shift(4)).astype(int)
            trend_4h = (df_4h_aligned['Close'] > df_4h_aligned['Close'].shift(1)).astype(int) - (df_4h_aligned['Close'] < df_4h_aligned['Close'].shift(1)).astype(int)
            
            cross_tf_df['trend_alignment'] = (trend_1h == trend_4h).astype(int)
            cross_tf_df['trend_strength_ratio'] = trend_1h.abs() / (trend_4h.abs() + 0.01)
            
            # Volatility regime detection
            volatility_1h = df_1h['Close'].rolling(24).std()
            volatility_4h = df_4h_aligned['Close'].rolling(6).std()
            
            cross_tf_df['volatility_regime'] = (volatility_1h / volatility_4h).rolling(12).rank(pct=True)
            
            cross_tf_data[pair] = cross_tf_df
        
        return cross_tf_data
    
    def create_synthetic_indicators(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                  strength_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create synthetic indicators that don't exist for individual pairs but emerge
        from cross-currency analysis. These are unique insights only possible with
        our multi-currency approach.
        
        Think of this like creating new instruments by combining different musical
        instruments - the result has properties that no single instrument could produce.
        
        Args:
            data_dict: Raw price data
            strength_data: Currency strength indices from previous calculations
        
        Returns:
            Dictionary with synthetic indicators for each timeframe
        """
        synthetic_data = {}
        
        for timeframe in ['1H', '4H']:
            if timeframe not in strength_data:
                continue
                
            logging.info(f"Creating synthetic indicators for {timeframe}")
            
            # Use EURUSD as base index (most liquid pair)
            base_pair = 'EURUSD'
            if base_pair not in data_dict or timeframe not in data_dict[base_pair]:
                continue
            
            synthetic_df = pd.DataFrame(index=data_dict[base_pair][timeframe].index)
            strength_df = strength_data[timeframe]
            
            # Currency Basket Indicators
            # Create synthetic currency baskets based on strength
            if all(col in strength_df.columns for col in ['USD_strength', 'EUR_strength', 'GBP_strength', 'JPY_strength']):
                # Major currency basket (weighted average of strengths)
                synthetic_df['major_currency_basket'] = (
                    strength_df['USD_strength'] * 0.4 +
                    strength_df['EUR_strength'] * 0.3 +
                    strength_df['GBP_strength'] * 0.2 +
                    strength_df['JPY_strength'] * 0.1
                )
                
                # Risk-on vs Risk-off sentiment
                # EUR and GBP typically strengthen in risk-on environments
                # JPY and USD strengthen in risk-off environments
                synthetic_df['risk_sentiment'] = (
                    (strength_df['EUR_strength'] + strength_df['GBP_strength']) -
                    (strength_df['JPY_strength'] + strength_df['USD_strength'] * 0.5)
                )
            
            # Cross-Pair Momentum Indicators
            momentum_features = []
            for pair in self.currency_pairs:
                if pair in data_dict and timeframe in data_dict[pair]:
                    pair_momentum = data_dict[pair][timeframe]['Close'].pct_change(5)
                    momentum_features.append(pair_momentum)
            
            if momentum_features:
                # Combined momentum across all pairs
                combined_momentum = pd.concat(momentum_features, axis=1)
                combined_momentum.columns = [f'{pair}_momentum' for pair in self.currency_pairs[:len(momentum_features)]]
                
                synthetic_df['cross_pair_momentum_avg'] = combined_momentum.mean(axis=1)
                synthetic_df['cross_pair_momentum_std'] = combined_momentum.std(axis=1)
                synthetic_df['momentum_dispersion'] = synthetic_df['cross_pair_momentum_std'] / synthetic_df['cross_pair_momentum_avg'].abs()
            
            # Market Regime Detection
            synthetic_df = self._detect_market_regimes(synthetic_df, data_dict, timeframe)
            
            # Session-based Effects
            synthetic_df = self._calculate_session_effects(synthetic_df, timeframe)
            
            synthetic_data[timeframe] = synthetic_df
        
        return synthetic_data
    
    def _calculate_divergence_indicators(self, corr_df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate indicators that detect when currency pairs diverge from their normal relationships.
        These divergences often signal trading opportunities.
        """
        # Expected relationships based on currency components
        # EURUSD and GBPUSD should generally move together (both vs USD)
        # USDJPY should move opposite to EURUSD and GBPUSD
        
        if all(pair in returns_data for pair in ['EURUSD', 'GBPUSD', 'USDJPY']):
            # Normal relationship: EUR/USD and GBP/USD positively correlated
            expected_eur_gbp_corr = 0.7
            actual_eur_gbp_corr = corr_df.get('EURUSD_GBPUSD_corr_24h', pd.Series(expected_eur_gbp_corr, index=corr_df.index))
            corr_df['eur_gbp_divergence'] = actual_eur_gbp_corr - expected_eur_gbp_corr
            
            # Normal relationship: EUR/USD and USD/JPY negatively correlated
            expected_eur_jpy_corr = -0.3
            actual_eur_jpy_corr = corr_df.get('EURUSD_USDJPY_corr_24h', pd.Series(expected_eur_jpy_corr, index=corr_df.index))
            corr_df['eur_jpy_divergence'] = actual_eur_jpy_corr - expected_eur_jpy_corr
            
            # Divergence strength (how far from normal)
            corr_df['total_divergence_strength'] = (
                corr_df['eur_gbp_divergence'].abs() + corr_df['eur_jpy_divergence'].abs()
            )
        
        return corr_df
    
    def _detect_market_regimes(self, synthetic_df: pd.DataFrame, data_dict: Dict[str, Dict[str, pd.DataFrame]], timeframe: str) -> pd.DataFrame:
        """
        Detect different market regimes: trending, ranging, or volatile.
        Different regimes require different trading strategies.
        """
        # Collect volatility data from all pairs
        volatilities = []
        trends = []
        
        for pair in self.currency_pairs:
            if pair in data_dict and timeframe in data_dict[pair]:
                df = data_dict[pair][timeframe]
                
                # Volatility measurement
                vol = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
                volatilities.append(vol)
                
                # Trend measurement (slope of linear regression)
                close_prices = df['Close'].rolling(20)
                trend_slope = close_prices.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan,
                    raw=True
                )
                trends.append(trend_slope.abs())
        
        if volatilities and trends:
            avg_volatility = pd.concat(volatilities, axis=1).mean(axis=1)
            avg_trend_strength = pd.concat(trends, axis=1).mean(axis=1)
            
            # Define regime thresholds based on percentiles
            vol_high_threshold = avg_volatility.rolling(200).quantile(0.75)
            vol_low_threshold = avg_volatility.rolling(200).quantile(0.25)
            trend_high_threshold = avg_trend_strength.rolling(200).quantile(0.75)
            
            # Classify regimes
            synthetic_df['market_regime'] = 'ranging'  # Default
            
            # High volatility = volatile regime
            synthetic_df.loc[avg_volatility > vol_high_threshold, 'market_regime'] = 'volatile'
            
            # High trend + low volatility = trending regime
            trending_mask = (avg_trend_strength > trend_high_threshold) & (avg_volatility < vol_low_threshold)
            synthetic_df.loc[trending_mask, 'market_regime'] = 'trending'
            
            # Encode regimes as numbers for ML models
            regime_encoding = {'ranging': 0, 'trending': 1, 'volatile': 2}
            synthetic_df['market_regime_encoded'] = synthetic_df['market_regime'].map(regime_encoding)
        
        return synthetic_df
    
    def _calculate_session_effects(self, synthetic_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculate trading session effects (Asian, European, US sessions).
        Different sessions have different volatility and correlation patterns.
        """
        # Define session hours in GMT+7 (Thai time as per the data)
        # Adjusted for typical forex session times
        asian_hours = list(range(6, 14))      # 6 AM - 2 PM
        european_hours = list(range(14, 22))   # 2 PM - 10 PM  
        us_hours = list(range(22, 24)) + list(range(0, 6))  # 10 PM - 6 AM
        
        synthetic_df['hour'] = synthetic_df.index.hour
        
        # Session indicators
        synthetic_df['is_asian_session'] = synthetic_df['hour'].isin(asian_hours).astype(int)
        synthetic_df['is_european_session'] = synthetic_df['hour'].isin(european_hours).astype(int)
        synthetic_df['is_us_session'] = synthetic_df['hour'].isin(us_hours).astype(int)
        
        # Session overlap indicators (higher volatility periods)
        synthetic_df['is_asian_european_overlap'] = (synthetic_df['hour'] == 14).astype(int)  # 2 PM
        synthetic_df['is_european_us_overlap'] = (synthetic_df['hour'] == 22).astype(int)    # 10 PM
        
        # Remove temporary hour column
        synthetic_df = synthetic_df.drop('hour', axis=1)
        
        return synthetic_df


# Demo and validation functions
def run_complete_pipeline_demo():
    """
    Runs a comprehensive demonstration of the complete data pipeline including
    feature engineering and adaptive exit strategy labeling.
    
    This demo showcases the full power of our Multi-Currency Bagging approach
    from raw data to training-ready labeled datasets.
    """
    processor = ForexDataProcessor()
    
    logging.info("🚀 Starting COMPLETE PIPELINE demonstration")
    logging.info("=" * 80)
    
    # Phase 1: Load and validate data
    logging.info("Phase 1: Loading and validating raw data...")
    raw_data = processor.load_all_data()
    train_data, validation_data = processor.get_development_data()
    
    # Phase 2: Feature engineering
    logging.info("Phase 2: Creating comprehensive features...")
    train_features, validation_features = processor.create_ml_ready_datasets()
    
    # Phase 3: Adaptive exit labeling
    logging.info("Phase 3: Creating adaptive exit strategy labels...")
    labeled_train_data, labeled_validation_data = processor.create_labeled_training_datasets()
    
    # Phase 4: Analysis and validation
    logging.info("Phase 4: Analyzing complete pipeline results...")
    
    # Show comprehensive statistics
    for pair in processor.currency_pairs:
        if pair in labeled_train_data:
            for timeframe in processor.timeframes:
                if timeframe in labeled_train_data[pair]:
                    df = labeled_train_data[pair][timeframe]
                    
                    logging.info(f"\n📊 {pair} {timeframe} Complete Dataset Analysis:")
                    logging.info(f"   • Total columns: {len(df.columns)}")
                    logging.info(f"   • Data range: {df.index.min()} to {df.index.max()}")
                    logging.info(f"   • Total samples: {len(df)}")
                    
                    # Categorize columns
                    column_categories = {
                        'Raw Price Data': ['Open', 'High', 'Low', 'Close', 'Volume'],
                        'Technical Indicators': [col for col in df.columns if any(x in col.lower() for x in ['ema', 'rsi', 'macd', 'bb', 'atr'])],
                        'Cross-Currency Features': [col for col in df.columns if any(x in col.lower() for x in ['strength', 'corr', 'divergence', 'sentiment'])],
                        'Adaptive Exit Labels': [col for col in df.columns if any(x in col.lower() for x in ['trade_direction', 'prob_profit', 'optimal_exit'])],
                        'Risk Management': [col for col in df.columns if any(x in col.lower() for x in ['risk_adjusted', 'confidence', 'kelly'])],
                        'Market Regime': [col for col in df.columns if any(x in col.lower() for x in ['regime', 'session', 'volatility'])]
                    }
                    
                    for category, columns in column_categories.items():
                        matching_cols = [col for col in columns if col in df.columns]
                        if matching_cols:
                            logging.info(f"   • {category}: {len(matching_cols)} features")
                            if category == 'Adaptive Exit Labels':
                                # Show sample label values
                                if 'trade_direction' in df.columns:
                                    recent_signals = df['trade_direction'].tail(5)
                                    logging.info(f"     Recent signals: {recent_signals.tolist()}")
    
    # Quality validation
    logging.info("\nPhase 5: Data quality validation...")
    quality_checks_passed = 0
    total_checks = 0
    
    for pair in labeled_train_data:
        for timeframe in labeled_train_data[pair]:
            df = labeled_train_data[pair][timeframe]
            total_checks += 1
            
            # Check for missing values in labels
            label_cols = [col for col in df.columns if any(x in col.lower() for x in 
                         ['trade_direction', 'prob_profit', 'optimal_exit'])]
            missing_labels = df[label_cols].isnull().sum().sum()
            
            # Check for infinite values
            inf_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            
            # Check label distributions
            if 'trade_direction' in df.columns:
                signal_distribution = df['trade_direction'].value_counts()
                total_signals = len(df)
                # Check if we have reasonable signal distribution (not all one class)
                max_class_pct = signal_distribution.max() / total_signals
                
                if missing_labels == 0 and inf_values == 0 and max_class_pct < 0.95:
                    quality_checks_passed += 1
                    logging.info(f"   ✅ {pair} {timeframe}: Quality check passed")
                else:
                    logging.warning(f"   ⚠️ {pair} {timeframe}: Quality issues detected")
                    if missing_labels > 0:
                        logging.warning(f"      - Missing labels: {missing_labels}")
                    if inf_values > 0:
                        logging.warning(f"      - Infinite values: {inf_values}")
                    if max_class_pct >= 0.95:
                        logging.warning(f"      - Label imbalance: {max_class_pct*100:.1f}% single class")
    
    logging.info(f"\nQuality Summary: {quality_checks_passed}/{total_checks} datasets passed quality checks")
    
    # Show adaptive exit strategy insights
    logging.info("\nPhase 6: Adaptive Exit Strategy Analysis...")
    if 'EURUSD' in labeled_train_data and '1H' in labeled_train_data['EURUSD']:
        sample_df = labeled_train_data['EURUSD']['1H']
        
        # Analyze exit timing patterns
        if 'optimal_exit_time_long' in sample_df.columns:
            exit_timing = sample_df['optimal_exit_time_long'].value_counts()
            total_trades = len(sample_df.dropna(subset=['optimal_exit_time_long']))
            logging.info("   Optimal Exit Timing Distribution (Long positions):")
            for time_horizon, count in exit_timing.items():
                logging.info(f"     t+{int(time_horizon)}: {count} trades ({count/total_trades*100:.1f}%)")
        
        # Analyze profit probabilities
        prob_cols = [col for col in sample_df.columns if 'prob_profit' in col and 'long' in col]
        if prob_cols:
            logging.info("   Average Profit Probabilities (Long positions):")
            for col in prob_cols[:3]:  # Show first 3
                avg_prob = sample_df[col].mean()
                logging.info(f"     {col}: {avg_prob:.3f}")
        
        # Show confidence levels
        if 'prediction_confidence' in sample_df.columns:
            conf_stats = sample_df['prediction_confidence'].describe()
            logging.info(f"   Prediction Confidence: mean={conf_stats['mean']:.3f}, "
                        f"std={conf_stats['std']:.3f}, "
                        f"range=[{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
    
    logging.info("\n🎯 COMPLETE PIPELINE DEMONSTRATION FINISHED")
    logging.info("✅ All systems operational - Ready for model training!")
    logging.info("📈 Datasets include: Features + Adaptive Labels + Risk Management")
    logging.info("🔒 Data leakage protection maintained throughout")
    logging.info("=" * 80)
    
    return True


def run_comprehensive_feature_demo():
    """
    Runs a comprehensive demonstration of our advanced feature engineering capabilities.
    
    This demo showcases the full power of our Multi-Currency Bagging approach by
    creating technical indicators, cross-currency features, and adaptive exit indicators.
    Think of this as a test drive of our newly built sports car - we want to see
    how all the components work together under real conditions.
    """
    processor = ForexDataProcessor()
    
    logging.info("🚀 Starting comprehensive feature engineering demonstration")
    logging.info("=" * 70)
    
    # Load all data
    logging.info("Phase 1: Loading and validating data...")
    raw_data = processor.load_all_data()
    
    # Get development data (protecting test set)
    train_data, validation_data = processor.get_development_data()
    
    # Test feature engineering on a smaller sample first
    logging.info("Phase 2: Testing feature engineering on training data sample...")
    
    # Create features for training data
    train_features = processor.create_comprehensive_features(train_data)
    
    # Display sample of created features
    logging.info("Phase 3: Analyzing created features...")
    
    for pair in processor.currency_pairs:
        if pair in train_features:
            for timeframe in processor.timeframes:
                if timeframe in train_features[pair]:
                    df = train_features[pair][timeframe]
                    
                    logging.info(f"\n📊 {pair} {timeframe} Features Analysis:")
                    logging.info(f"   • Total features: {len(df.columns)}")
                    logging.info(f"   • Data range: {df.index.min()} to {df.index.max()}")
                    logging.info(f"   • Total samples: {len(df)}")
                    
                    # Show sample feature categories
                    feature_categories = {
                        'Technical Indicators': [col for col in df.columns if any(x in col.lower() for x in ['ema', 'rsi', 'macd', 'bb'])],
                        'Cross-Currency': [col for col in df.columns if any(x in col.lower() for x in ['strength', 'corr', 'divergence'])],
                        'Adaptive Exit': [col for col in df.columns if any(x in col.lower() for x in ['momentum', 'trend', 'exit', 'profit'])],
                        'Market Regime': [col for col in df.columns if any(x in col.lower() for x in ['regime', 'session', 'sentiment'])]
                    }
                    
                    for category, features in feature_categories.items():
                        if features:
                            logging.info(f"   • {category}: {len(features)} features")
                            # Show first few feature names as examples
                            examples = features[:3]
                            logging.info(f"     Examples: {', '.join(examples)}")
    
    logging.info("\nPhase 4: Validating feature quality...")
    
    # Check for any data quality issues
    quality_issues = 0
    for pair in train_features:
        for timeframe in train_features[pair]:
            df = train_features[pair][timeframe]
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                logging.warning(f"   ⚠️ {pair} {timeframe}: {missing_count} missing values found")
                quality_issues += 1
            
            # Check for infinite values
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                logging.warning(f"   ⚠️ {pair} {timeframe}: {inf_count} infinite values found")
                quality_issues += 1
            
            # Check feature variance (constant features are problematic)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            zero_variance_features = [col for col in numeric_cols if df[col].std() == 0]
            if zero_variance_features:
                logging.warning(f"   ⚠️ {pair} {timeframe}: {len(zero_variance_features)} zero-variance features")
                quality_issues += 1
    
    if quality_issues == 0:
        logging.info("   ✅ All feature quality checks passed!")
    else:
        logging.warning(f"   ⚠️ Found {quality_issues} quality issues to address")
    
    logging.info("\nPhase 5: Testing cross-currency insights...")
    
    # Demonstrate cross-currency insights
    if 'EURUSD' in train_features and '1H' in train_features['EURUSD']:
        sample_df = train_features['EURUSD']['1H']
        
        # Show currency strength example
        strength_cols = [col for col in sample_df.columns if 'strength' in col.lower()]
        if strength_cols:
            logging.info(f"   • Currency strength features available: {len(strength_cols)}")
            
            # Show latest strength values as example
            latest_strength = sample_df[strength_cols].iloc[-1]
            logging.info("   • Latest currency strength readings:")
            for currency_strength in strength_cols[:4]:  # Show first 4
                if currency_strength in latest_strength:
                    value = latest_strength[currency_strength]
                    logging.info(f"     {currency_strength}: {value:.2f}")
        
        # Show correlation insights
        corr_cols = [col for col in sample_df.columns if 'corr' in col.lower()]
        if corr_cols:
            logging.info(f"   • Cross-currency correlation features: {len(corr_cols)}")
        
        # Show adaptive exit features
        exit_cols = [col for col in sample_df.columns if any(x in col.lower() for x in ['exit', 'profit', 'momentum_reversal'])]
        if exit_cols:
            logging.info(f"   • Adaptive exit strategy features: {len(exit_cols)}")
    
    logging.info("\n🎯 Comprehensive feature engineering demonstration completed successfully!")
    logging.info("✅ All systems ready for model training phase")
    logging.info("=" * 70)
    
    return True


def run_data_integrity_check():
    """
    Runs comprehensive data integrity checks to ensure our data is ready for modeling.
    This is like a health checkup for our data.
    """
    processor = ForexDataProcessor()
    
    logging.info("Starting comprehensive data integrity check...")
    
    # Load all data
    raw_data = processor.load_all_data()
    
    # Check data availability and consistency
    for pair in processor.currency_pairs:
        for timeframe in processor.timeframes:
            df = raw_data[pair][timeframe]
            
            # Check for missing data
            missing_data = df.isnull().sum().sum()
            if missing_data > 0:
                logging.warning(f"{pair} {timeframe}: {missing_data} missing values detected")
            
            # Check data range
            date_range = f"{df.index.min()} to {df.index.max()}"
            logging.info(f"{pair} {timeframe}: {len(df)} records from {date_range}")
            
            # Check for gaps in time series
            if timeframe == '1H':
                expected_freq = '1H'
            else:
                expected_freq = '4H'
            
            # This is a simplified gap check - full implementation would be more sophisticated
            time_diffs = df.index.to_series().diff().dropna()
            irregular_intervals = len(time_diffs[time_diffs != pd.Timedelta(expected_freq)])
            
            if irregular_intervals > 0:
                logging.info(f"{pair} {timeframe}: {irregular_intervals} irregular time intervals")
    
    # Test the data splitting mechanism
    train_data, validation_data = processor.get_development_data()
    
    logging.info("Data integrity check completed successfully!")
    logging.info("✅ All systems ready for model development")


if __name__ == "__main__":
    # Run the complete pipeline demo when this file is executed directly
    run_complete_pipeline_demo()