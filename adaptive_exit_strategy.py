"""
Adaptive Exit Strategy and Multi-Horizon Labeling System

This module implements the core philosophy of our trading system:
"Take profits quickly when opportunities arise, but be patient when results aren't immediate"

The Adaptive Exit Strategy is NOT just a rule applied after trading decisions.
It's embedded into the very learning process of our AI models, teaching them to:
- Recognize when small profits should be taken immediately (t+1)
- Identify when patience might yield better results (t+2) 
- Know when to cut losses regardless of outcome (t+3)

This represents a fundamental shift from traditional ML approaches where
exit strategies are afterthoughts. Here, the exit strategy IS the strategy.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logger = logging.getLogger('AdaptiveExitStrategy')


class ExitDecision(Enum):
    """
    Enumeration of possible exit decisions at each time horizon.
    These represent the core actions our system can take.
    """
    HOLD = 0          # Continue holding the position
    EXIT_PROFIT = 1   # Exit with profit
    EXIT_LOSS = 2     # Exit with loss
    EXIT_NEUTRAL = 3  # Exit at breakeven


@dataclass
class TradeOutcome:
    """
    Data class to represent the outcome of a trade at different time horizons.
    This encapsulates all the information needed to evaluate trade performance.
    """
    entry_price: float
    exit_price_t1: float
    exit_price_t2: float
    exit_price_t3: float
    direction: int  # 1 for long, -1 for short
    gross_profit_t1: float
    gross_profit_t2: float
    gross_profit_t3: float
    net_profit_t1: float  # After spread
    net_profit_t2: float
    net_profit_t3: float
    spread_cost: float
    optimal_exit_time: int  # 1, 2, or 3
    actual_exit_time: int
    exit_decision_t1: ExitDecision
    exit_decision_t2: ExitDecision
    exit_decision_t3: ExitDecision


class AdaptiveExitStrategy:
    """
    Core implementation of the Adaptive Exit Strategy.
    
    This class embodies our trading philosophy by creating multi-horizon labels
    that teach models when to exit trades at t+1, t+2, or t+3. Unlike traditional
    approaches where exit strategies are separate from prediction models, we
    integrate exit logic directly into the learning objectives.
    
    Think of this as creating a sophisticated reward system that teaches an AI
    to value immediate small gains over uncertain large future gains, while
    still recognizing when patience is warranted.
    """
    
    def __init__(self, currency_pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY']):
        """
        Initialize the Adaptive Exit Strategy system.
        
        Args:
            currency_pairs: List of currency pairs to create exit strategies for
        """
        self.currency_pairs = currency_pairs
        
        # Spread assumptions (in pips) - critical for realistic profit calculations
        self.spreads = {
            'EURUSD': 2,
            'GBPUSD': 2,
            'USDJPY': 2
        }
        
        # Pip values for different currency pairs
        self.pip_values = {
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDJPY': 0.01
        }
        
        # Minimum profit thresholds (in pips) after spread
        # These represent the minimum profit needed to justify a trade
        self.min_profit_thresholds = {
            'EURUSD': 3,  # 3 pips gross = 1 pip net after 2 pip spread
            'GBPUSD': 3,
            'USDJPY': 3
        }
        
        # Adaptive exit parameters
        self.exit_params = {
            'quick_profit_threshold': 1.0,    # Take profit when >= 1 pip net profit
            'patience_threshold': 0.5,        # Be patient when profit < 0.5 pips
            'max_holding_periods': 3,         # Never hold longer than 3 periods
            'stop_loss_pips': -10,           # Stop loss at -10 pips
            'volatility_adjustment': True,    # Adjust thresholds based on volatility
            'trend_adjustment': True          # Adjust based on trend strength
        }
        
        logger.info("Adaptive Exit Strategy initialized")
        logger.info(f"Currency pairs: {currency_pairs}")
        logger.info(f"Spread assumptions: {self.spreads}")
        logger.info(f"Minimum profit thresholds: {self.min_profit_thresholds}")
    
    def create_adaptive_labels(self, feature_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create multi-horizon labels that embed the adaptive exit strategy into the learning process.
        
        This is the heart of our approach - instead of predicting just price direction,
        we predict optimal exit timing and probability of profit at different horizons.
        
        The labels created include:
        - Direction predictions (long/short/neutral)
        - Profit probability at t+1, t+2, t+3
        - Optimal exit recommendations
        - Risk-adjusted profit expectations
        - Confidence scores for each time horizon
        
        Args:
            feature_data: Dictionary with structure {pair: {timeframe: dataframe_with_features}}
        
        Returns:
            Dictionary with same structure but including adaptive exit labels
        """
        logger.info("Creating adaptive exit labels for all currency pairs")
        
        labeled_data = {}
        
        for pair in self.currency_pairs:
            if pair not in feature_data:
                logger.warning(f"No feature data available for {pair}")
                continue
            
            labeled_data[pair] = {}
            
            for timeframe in ['1H', '4H']:
                if timeframe not in feature_data[pair]:
                    logger.warning(f"No {timeframe} data for {pair}")
                    continue
                
                logger.info(f"Creating labels for {pair} {timeframe}")
                
                df = feature_data[pair][timeframe].copy()
                
                # Create comprehensive labels for this pair/timeframe
                df = self._create_trade_direction_labels(df, pair, timeframe)
                df = self._create_profit_probability_labels(df, pair, timeframe)
                df = self._create_exit_timing_labels(df, pair, timeframe)
                df = self._create_risk_adjusted_labels(df, pair, timeframe)
                df = self._create_confidence_labels(df, pair, timeframe)
                
                labeled_data[pair][timeframe] = df
                
                # Log label statistics
                self._log_label_statistics(df, pair, timeframe)
        
        logger.info("Adaptive exit labels created successfully for all pairs")
        return labeled_data
    
    def _create_trade_direction_labels(self, df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Create trade direction labels that incorporate adaptive exit philosophy.
        
        Unlike simple buy/sell/hold labels, these consider the probability of
        profitable exits at different time horizons. A "buy" signal is only
        generated if there's reasonable probability of profit within our
        3-period maximum holding time.
        """
        logger.debug(f"Creating direction labels for {pair} {timeframe}")
        
        # Calculate future price movements
        close_prices = df['Close']
        
        # Price changes at t+1, t+2, t+3
        price_change_t1 = close_prices.shift(-1) - close_prices
        price_change_t2 = close_prices.shift(-2) - close_prices  
        price_change_t3 = close_prices.shift(-3) - close_prices
        
        # Convert to pips
        pip_value = self.pip_values[pair]
        pips_t1 = price_change_t1 / pip_value
        pips_t2 = price_change_t2 / pip_value
        pips_t3 = price_change_t3 / pip_value
        
        # Account for spread in profit calculations
        spread_pips = self.spreads[pair]
        net_pips_t1_long = pips_t1 - spread_pips
        net_pips_t2_long = pips_t2 - spread_pips
        net_pips_t3_long = pips_t3 - spread_pips
        
        net_pips_t1_short = -pips_t1 - spread_pips
        net_pips_t2_short = -pips_t2 - spread_pips
        net_pips_t3_short = -pips_t3 - spread_pips
        
        # Determine if any horizon offers profitable exit for long positions
        long_profitable_t1 = net_pips_t1_long >= self.exit_params['quick_profit_threshold']
        long_profitable_t2 = net_pips_t2_long >= self.exit_params['quick_profit_threshold']
        long_profitable_t3 = net_pips_t3_long >= self.exit_params['quick_profit_threshold']
        long_profitable_any = long_profitable_t1 | long_profitable_t2 | long_profitable_t3
        
        # Determine if any horizon offers profitable exit for short positions
        short_profitable_t1 = net_pips_t1_short >= self.exit_params['quick_profit_threshold']
        short_profitable_t2 = net_pips_t2_short >= self.exit_params['quick_profit_threshold']
        short_profitable_t3 = net_pips_t3_short >= self.exit_params['quick_profit_threshold']
        short_profitable_any = short_profitable_t1 | short_profitable_t2 | short_profitable_t3
        
        # Create directional labels with adaptive exit consideration
        # Only signal trades that have reasonable chance of profit within 3 periods
        df['trade_direction'] = 0  # Default: no trade
        
        # Long signals: profitable within 3 periods AND not contradicted by short profitability
        df.loc[long_profitable_any & ~short_profitable_any, 'trade_direction'] = 1
        
        # Short signals: profitable within 3 periods AND not contradicted by long profitability  
        df.loc[short_profitable_any & ~long_profitable_any, 'trade_direction'] = -1
        
        # Add probability scores for direction confidence
        df['long_direction_confidence'] = (
            long_profitable_t1.astype(int) * 0.5 +
            long_profitable_t2.astype(int) * 0.3 +
            long_profitable_t3.astype(int) * 0.2
        )
        
        df['short_direction_confidence'] = (
            short_profitable_t1.astype(int) * 0.5 +
            short_profitable_t2.astype(int) * 0.3 +
            short_profitable_t3.astype(int) * 0.2
        )
        
        # Store raw pip movements for analysis
        df['future_pips_t1'] = pips_t1
        df['future_pips_t2'] = pips_t2
        df['future_pips_t3'] = pips_t3
        
        return df
    
    def _create_profit_probability_labels(self, df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Create labels that predict the probability of profit at each time horizon.
        
        These labels teach the model to estimate not just whether a trade will be
        profitable, but the likelihood of profit at specific exit points. This
        enables the model to make risk-weighted decisions.
        """
        logger.debug(f"Creating profit probability labels for {pair} {timeframe}")
        
        pip_value = self.pip_values[pair]
        spread_pips = self.spreads[pair]
        min_profit = self.exit_params['quick_profit_threshold']
        
        # For each time horizon, calculate probability of profit
        # We use a rolling window to estimate local probability distributions
        window_size = 50  # 50 periods for probability estimation
        
        for t in [1, 2, 3]:
            # Calculate future returns
            future_prices = df['Close'].shift(-t)
            future_returns = (future_prices - df['Close']) / df['Close']
            future_pips = future_returns * df['Close'] / pip_value
            
            # Net profit after spread for both directions
            net_profit_long = future_pips - spread_pips
            net_profit_short = -future_pips - spread_pips
            
            # Rolling probability calculations
            # Probability of profitable long position
            long_profitable = (net_profit_long >= min_profit).astype(int)
            df[f'prob_profit_long_t{t}'] = long_profitable.rolling(window=window_size, min_periods=10).mean()
            
            # Probability of profitable short position
            short_profitable = (net_profit_short >= min_profit).astype(int)
            df[f'prob_profit_short_t{t}'] = short_profitable.rolling(window=window_size, min_periods=10).mean()
            
            # Expected profit (risk-weighted)
            df[f'expected_profit_long_t{t}'] = (
                net_profit_long.rolling(window=window_size, min_periods=10).mean()
            )
            df[f'expected_profit_short_t{t}'] = (
                net_profit_short.rolling(window=window_size, min_periods=10).mean()
            )
            
            # Profit variance (risk measure)
            df[f'profit_variance_long_t{t}'] = (
                net_profit_long.rolling(window=window_size, min_periods=10).var()
            )
            df[f'profit_variance_short_t{t}'] = (
                net_profit_short.rolling(window=window_size, min_periods=10).var()
            )
        
        return df
    
    def _create_exit_timing_labels(self, df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Create labels that specify the optimal exit timing based on adaptive strategy.
        
        These labels encode our core philosophy: take profits early when available,
        but be patient when immediate profits aren't visible. The labels teach
        the model WHEN to exit, not just WHETHER to exit.
        """
        logger.debug(f"Creating exit timing labels for {pair} {timeframe}")
        
        pip_value = self.pip_values[pair]
        spread_pips = self.spreads[pair]
        profit_threshold = self.exit_params['quick_profit_threshold']
        
        # Calculate net profits at each horizon
        for direction in ['long', 'short']:
            direction_multiplier = 1 if direction == 'long' else -1
            
            # Net profits after spread
            net_profit_t1 = (df['future_pips_t1'] * direction_multiplier) - spread_pips
            net_profit_t2 = (df['future_pips_t2'] * direction_multiplier) - spread_pips
            net_profit_t3 = (df['future_pips_t3'] * direction_multiplier) - spread_pips
            
            # Optimal exit timing based on adaptive strategy
            optimal_exit = pd.Series(3, index=df.index)  # Default: exit at t+3
            
            # If profitable at t+1, exit early (adaptive strategy priority)
            optimal_exit = np.where(net_profit_t1 >= profit_threshold, 1, optimal_exit)
            
            # If not profitable at t+1 but profitable at t+2, exit at t+2
            optimal_exit = np.where(
                (net_profit_t1 < profit_threshold) & (net_profit_t2 >= profit_threshold),
                2, optimal_exit
            )
            
            # If profitable at neither t+1 nor t+2, forced exit at t+3
            # (This implements the "cut losses" part of our strategy)
            
            df[f'optimal_exit_time_{direction}'] = optimal_exit
            
            # Exit decision probabilities (for soft labels)
            # Probability of exiting at each time point
            df[f'prob_exit_t1_{direction}'] = (net_profit_t1 >= profit_threshold).astype(float)
            df[f'prob_exit_t2_{direction}'] = (
                (net_profit_t1 < profit_threshold) & (net_profit_t2 >= profit_threshold)
            ).astype(float)
            df[f'prob_exit_t3_{direction}'] = (
                (net_profit_t1 < profit_threshold) & (net_profit_t2 < profit_threshold)
            ).astype(float)
            
            # Expected profit at optimal exit time
            profits = [net_profit_t1, net_profit_t2, net_profit_t3]
            df[f'expected_profit_optimal_{direction}'] = pd.Series([
                profits[int(t)-1].iloc[i] for i, t in enumerate(optimal_exit)
            ], index=df.index)
        
        return df
    
    def _create_risk_adjusted_labels(self, df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Create risk-adjusted labels that consider both profit potential and downside risk.
        
        These labels help the model understand that a small certain profit is often
        better than a large uncertain profit, especially in the context of our
        adaptive exit strategy.
        """
        logger.debug(f"Creating risk-adjusted labels for {pair} {timeframe}")
        
        # Calculate risk-adjusted return expectations
        for t in [1, 2, 3]:
            for direction in ['long', 'short']:
                profit_col = f'expected_profit_{direction}_t{t}'
                variance_col = f'profit_variance_{direction}_t{t}'
                
                if profit_col in df.columns and variance_col in df.columns:
                    # Sharpe-like ratio: expected profit / risk
                    risk = np.sqrt(df[variance_col])
                    df[f'risk_adjusted_return_{direction}_t{t}'] = df[profit_col] / (risk + 0.01)
                    
                    # Probability of loss (downside risk)
                    df[f'prob_loss_{direction}_t{t}'] = (df[profit_col] < 0).astype(float)
                    
                    # Kelly criterion-inspired position sizing
                    # f = (bp - q) / b, where b = odds, p = win prob, q = loss prob
                    win_prob = df[f'prob_profit_{direction}_t{t}']
                    avg_win = df[profit_col].where(df[profit_col] > 0).rolling(50).mean()
                    avg_loss = df[profit_col].where(df[profit_col] < 0).rolling(50).mean().abs()
                    
                    # Simplified Kelly fraction
                    df[f'kelly_fraction_{direction}_t{t}'] = np.clip(
                        (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win,
                        0, 1
                    )
        
        # Overall risk-adjusted score combining all horizons
        for direction in ['long', 'short']:
            risk_scores = []
            for t in [1, 2, 3]:
                col = f'risk_adjusted_return_{direction}_t{t}'
                if col in df.columns:
                    # Weight earlier horizons more heavily (adaptive strategy preference)
                    weight = [0.5, 0.3, 0.2][t-1]
                    risk_scores.append(df[col] * weight)
            
            if risk_scores:
                df[f'overall_risk_score_{direction}'] = sum(risk_scores)
        
        return df
    
    def _create_confidence_labels(self, df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Create confidence labels that indicate how certain the model should be
        about its predictions at each time horizon.
        
        High confidence situations include:
        - Strong directional momentum with low volatility
        - Clear technical patterns with historical reliability
        - Favorable cross-currency correlations
        
        Low confidence situations include:
        - High volatility periods
        - Conflicting signals across timeframes
        - Unusual correlation breakdowns
        """
        logger.debug(f"Creating confidence labels for {pair} {timeframe}")
        
        # Technical confidence factors
        confidence_factors = []
        
        # 1. Trend clarity (aligned moving averages increase confidence)
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            # Trend alignment score
            uptrend_aligned = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
            downtrend_aligned = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
            trend_clarity = (uptrend_aligned | downtrend_aligned).astype(float)
            confidence_factors.append(trend_clarity * 0.3)
        
        # 2. Volatility confidence (lower volatility = higher confidence for direction)
        if 'atr' in df.columns:
            volatility_percentile = df['atr'].rolling(100).rank(pct=True)
            volatility_confidence = 1 - volatility_percentile  # Lower volatility = higher confidence
            confidence_factors.append(volatility_confidence * 0.2)
        
        # 3. Cross-currency confirmation
        if 'EUR_strength' in df.columns and 'USD_strength' in df.columns:
            # For EURUSD, EUR strength and USD weakness should align for high confidence
            if pair == 'EURUSD':
                currency_alignment = np.abs(df['EUR_strength'] + df['USD_strength'])
                currency_confidence = 1 / (1 + currency_alignment)  # Higher alignment = higher confidence
                confidence_factors.append(currency_confidence * 0.2)
        
        # 4. Volume confirmation (if available)
        if 'volume_roc' in df.columns:
            volume_confirmation = np.abs(df['volume_roc']).rolling(20).rank(pct=True)
            confidence_factors.append(volume_confirmation * 0.1)
        
        # 5. Historical success rate at this market condition
        if 'market_regime_encoded' in df.columns:
            # This would require backtesting data, simplified for now
            regime_confidence = pd.Series(0.5, index=df.index)  # Neutral confidence
            confidence_factors.append(regime_confidence * 0.2)
        
        # Combine confidence factors
        if confidence_factors:
            df['prediction_confidence'] = sum(confidence_factors)
            df['prediction_confidence'] = df['prediction_confidence'].clip(0, 1)
        else:
            df['prediction_confidence'] = 0.5  # Neutral confidence if no factors available
        
        # Time-horizon specific confidence
        # Generally, shorter horizons should have higher confidence in trending markets
        for t in [1, 2, 3]:
            horizon_weight = [0.6, 0.3, 0.1][t-1]  # Decreasing confidence with time
            df[f'confidence_t{t}'] = df['prediction_confidence'] * horizon_weight
        
        return df
    
    def _log_label_statistics(self, df: pd.DataFrame, pair: str, timeframe: str):
        """
        Log comprehensive statistics about the labels created for analysis and validation.
        """
        logger.info(f"=== LABEL STATISTICS FOR {pair} {timeframe} ===")
        
        # Direction distribution
        if 'trade_direction' in df.columns:
            direction_counts = df['trade_direction'].value_counts()
            total_signals = len(df)
            logger.info(f"Trade Direction Distribution:")
            logger.info(f"  Long signals: {direction_counts.get(1, 0)} ({direction_counts.get(1, 0)/total_signals*100:.1f}%)")
            logger.info(f"  Short signals: {direction_counts.get(-1, 0)} ({direction_counts.get(-1, 0)/total_signals*100:.1f}%)")
            logger.info(f"  No trade: {direction_counts.get(0, 0)} ({direction_counts.get(0, 0)/total_signals*100:.1f}%)")
        
        # Profit probability statistics
        profit_prob_cols = [col for col in df.columns if 'prob_profit' in col]
        if profit_prob_cols:
            logger.info(f"Average Profit Probabilities:")
            for col in profit_prob_cols[:6]:  # Show first 6 to avoid clutter
                avg_prob = df[col].mean()
                logger.info(f"  {col}: {avg_prob:.3f}")
        
        # Exit timing distribution
        exit_timing_cols = [col for col in df.columns if 'optimal_exit_time' in col]
        if exit_timing_cols:
            logger.info(f"Optimal Exit Timing Distribution:")
            for col in exit_timing_cols:
                timing_counts = df[col].value_counts()
                total = len(df.dropna(subset=[col]))
                logger.info(f"  {col}:")
                for time, count in timing_counts.items():
                    logger.info(f"    t+{time}: {count} ({count/total*100:.1f}%)")
        
        # Confidence statistics
        if 'prediction_confidence' in df.columns:
            conf_mean = df['prediction_confidence'].mean()
            conf_std = df['prediction_confidence'].std()
            logger.info(f"Prediction Confidence: mean={conf_mean:.3f}, std={conf_std:.3f}")
        
        logger.info("=" * 50)


class AdaptiveLossFunction:
    """
    Custom loss function that embeds the adaptive exit strategy into model training.
    
    Traditional loss functions optimize for prediction accuracy. Our custom loss
    function optimizes for trading performance according to our adaptive exit philosophy.
    
    The loss function rewards:
    - Early profit-taking when profits are available
    - Patient holding when immediate profits aren't visible
    - Quick loss-cutting when positions turn against us
    
    This ensures that the model learns to trade according to our philosophy,
    not just predict price movements accurately.
    """
    
    def __init__(self, spread_costs: Dict[str, float], time_decay_factor: float = 0.1):
        """
        Initialize the adaptive loss function.
        
        Args:
            spread_costs: Dictionary of spread costs for each currency pair
            time_decay_factor: How much to penalize longer holding periods
        """
        self.spread_costs = spread_costs
        self.time_decay_factor = time_decay_factor
        
        logger.info("Adaptive Loss Function initialized")
        logger.info(f"Spread costs: {spread_costs}")
        logger.info(f"Time decay factor: {time_decay_factor}")
    
    def calculate_adaptive_loss(self, predictions: Dict[str, np.ndarray], 
                              actual_outcomes: Dict[str, np.ndarray],
                              pair: str) -> float:
        """
        Calculate loss that rewards adaptive exit strategy adherence.
        
        Args:
            predictions: Dictionary with model predictions for different horizons
            actual_outcomes: Dictionary with actual profit outcomes
            pair: Currency pair being evaluated
            
        Returns:
            Adaptive loss value that guides model training
        """
        # Implementation would be model-framework specific
        # This is a conceptual framework that would be adapted for TensorFlow/PyTorch
        
        total_loss = 0.0
        
        # Reward structure based on adaptive exit philosophy
        for t in [1, 2, 3]:
            pred_key = f'profit_t{t}'
            actual_key = f'actual_profit_t{t}'
            
            if pred_key in predictions and actual_key in actual_outcomes:
                predicted_profit = predictions[pred_key]
                actual_profit = actual_outcomes[actual_key]
                
                # Base prediction loss
                prediction_error = np.mean((predicted_profit - actual_profit) ** 2)
                
                # Adaptive adjustments
                # 1. Heavily reward accurate early exit predictions
                early_exit_weight = [3.0, 2.0, 1.0][t-1]
                
                # 2. Penalize missed opportunities (predicted profit but didn't exit early)
                missed_opportunity_penalty = np.mean(
                    np.maximum(0, actual_profit - predicted_profit) * 2.0
                )
                
                # 3. Reward conservative predictions that avoid losses
                loss_avoidance_reward = np.mean(
                    np.maximum(0, predicted_profit - actual_profit) * 0.5
                )
                
                # 4. Time decay penalty (encourage earlier exits)
                time_penalty = self.time_decay_factor * (t - 1)
                
                horizon_loss = (
                    prediction_error * early_exit_weight +
                    missed_opportunity_penalty +
                    time_penalty -
                    loss_avoidance_reward
                )
                
                total_loss += horizon_loss
        
        return total_loss


def create_adaptive_training_data(feature_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Main function to create comprehensive training data with adaptive exit labels.
    
    This function orchestrates the entire labeling process, creating rich,
    multi-dimensional labels that embed our trading philosophy into the
    learning process.
    
    Args:
        feature_data: Dictionary with engineered features
        
    Returns:
        Dictionary with features and adaptive exit labels ready for model training
    """
    logger.info("🚀 Creating comprehensive adaptive training data")
    
    # Initialize the adaptive exit strategy
    adaptive_strategy = AdaptiveExitStrategy()
    
    # Create adaptive labels
    labeled_data = adaptive_strategy.create_adaptive_labels(feature_data)
    
    logger.info("✅ Adaptive training data creation completed")
    
    return labeled_data


if __name__ == "__main__":
    # Demo and testing code
    logger.info("Adaptive Exit Strategy Module - Ready for integration")