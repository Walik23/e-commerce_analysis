import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector: 
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info(f"Initialized AnomalyDetector with {len(df)} events")
    
    def statistical_anomalies(self, metric: str, 
                             threshold: float = 3.0) -> pd.DataFrame:
        logger.info(f"Searching for statistical anomalies in '{metric}' (threshold: {threshold} σ)")
        
        mean = self.df[metric].mean()
        std = self.df[metric].std()
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        anomalies = self.df[
            (self.df[metric] < lower_bound) | 
            (self.df[metric] > upper_bound)
        ].copy()
        
        anomalies['anomaly_type'] = anomalies[metric].apply(
            lambda x: 'too_low' if x < lower_bound else 'too_high'
        )
        anomalies['deviation'] = abs(anomalies[metric] - mean) / std
        
        logger.info(f"Found {len(anomalies)} anomalies ({len(anomalies)/len(self.df)*100:.2f}%)")
        
        return anomalies
    
    def detect_fraud_patterns(self, sample_size: int = 50000, user_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        logger.info("Detection of suspicious behavior patterns")
        
        unique_users = self.df['user_id'].unique()
        total_users = len(unique_users)
        
        if user_features is not None:
            logger.info(f"Using pre-computed user features for {len(user_features):,} users")
            max_events_per_hour_data = user_features.copy()

            if len(user_features) > sample_size:
                logger.info(f"Sampling {sample_size:,} users from {len(user_features):,}")
                sampled_users = np.random.choice(
                    user_features['user_id'].unique(), 
                    sample_size, 
                    replace=False
                )
                max_events_per_hour_data = user_features[user_features['user_id'].isin(sampled_users)].copy()
                df_work = self.df[self.df['user_id'].isin(sampled_users)].copy()
            else:
                df_work = self.df.copy()
        
        else:
            if total_users > sample_size:
                logger.info(f"Sampling {sample_size:,} users from {total_users:,}")
                sampled_users = np.random.choice(unique_users, sample_size, replace=False)
                df_work = self.df[self.df['user_id'].isin(sampled_users)].copy()
            else:
                df_work = self.df.copy()
                logger.info(f"Analyzing all {total_users:,} users")

            max_events_per_hour_data = df_work.groupby('user_id').size().reset_index(name='total_events')
            max_events_per_hour_data.columns = ['user_id', 'total_events']
        
        logger.info("Computing fraud indicators...")

        total_events = max_events_per_hour_data.set_index('user_id')['total_events'] \
            if 'total_events' in max_events_per_hour_data.columns \
            else df_work.groupby('user_id').size()

        event_diversity = df_work.groupby('user_id')['event_type'].nunique()

        if 'hour' in df_work.columns:
            night_mask = df_work['hour'].isin([0, 1, 2, 3, 4, 5])
            night_events = df_work[night_mask].groupby('user_id').size()
            night_ratio = (night_events / total_events).fillna(0)
        else:
            night_ratio = pd.Series(0, index=total_events.index)

        high_value_users = set()
        if 'price' in df_work.columns:
            price_data = df_work['price'].dropna()
            if len(price_data) > 0:
                avg_price = price_data.mean()
                std_price = price_data.std()
                threshold = avg_price + 4 * std_price
                
                max_price_per_user = df_work.groupby('user_id')['price'].max()
                high_value_users = set(max_price_per_user[max_price_per_user > threshold].index)

        logger.info("Building results...")
        
        suspicious_list = []
        
        for user_id in total_events.index:
            flags = []

            if total_events[user_id] > 100:
                flags.append('high_frequency')

            if event_diversity.get(user_id, 0) == 1 and total_events[user_id] > 20:
                flags.append('low_diversity')

            if night_ratio.get(user_id, 0) > 0.8:
                flags.append('night_activity')

            if user_id in high_value_users:
                flags.append('high_value_transaction')
            
            if flags:
                suspicious_list.append({
                    'user_id': user_id,
                    'flags': ', '.join(flags),
                    'total_events': int(total_events[user_id]),
                    'risk_score': len(flags)
                })
        
        result = pd.DataFrame(suspicious_list)
        
        if len(result) > 0:
            result = result.sort_values('risk_score', ascending=False)
            logger.warning(f"{len(result)} suspicious users detected")
        else:
            logger.info("No suspicious users detected")

        return result
    
    def isolation_forest_anomalies(self, 
                                   features: list,
                                   contamination: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
        logger.info(f"Isolation Forest analysis based on features: {features}")

        user_features = self.df.groupby('user_id').agg({
            'event_type': 'count',
            'session_id': 'nunique',
        }).reset_index()
        
        user_features.columns = ['user_id', 'total_events', 'total_sessions']
        user_features['events_per_session'] = (
            user_features['total_events'] / user_features['total_sessions']
        )

        if 'price' in self.df.columns:
            user_spending = self.df.groupby('user_id')['price'].agg(['sum', 'mean', 'max'])
            user_features = user_features.merge(
                user_spending, 
                left_on='user_id', 
                right_index=True
            )
            user_features.columns = list(user_features.columns[:4]) + [
                'total_spending', 'avg_spending', 'max_spending'
            ]

        available_features = [f for f in features if f in user_features.columns]
        
        if not available_features:
            logger.error(f"None of the {features} were found in the data")
            return pd.DataFrame(), {}
        
        X = user_features[available_features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)

        user_features['anomaly'] = predictions
        user_features['anomaly_score'] = scores

        anomalies = user_features[user_features['anomaly'] == -1].copy()
        anomalies = anomalies.sort_values('anomaly_score')
        
        metrics = {
            'total_users': len(user_features),
            'anomalies_found': len(anomalies),
            'anomaly_rate': round(len(anomalies) / len(user_features) * 100, 2),
            'features_used': available_features
        }
        
        logger.info(
            f"{metrics['anomalies_found']} anomalies detected "
            f"({metrics['anomaly_rate']}%)"
        )
        
        return anomalies, metrics
    
    def time_series_anomalies(self, metric: str = 'events_count',
                             window: str = 'D') -> pd.DataFrame:
        logger.info(f"Analysis of time series '{metric}' with window '{window}'")

        time_series = self.df.set_index('timestamp').resample(window).size()
        time_series = time_series.to_frame(name='count')

        time_series['rolling_mean'] = time_series['count'].rolling(window=7).mean()
        time_series['rolling_std'] = time_series['count'].rolling(window=7).std()

        time_series['lower_bound'] = (
            time_series['rolling_mean'] - 3 * time_series['rolling_std']
        )
        time_series['upper_bound'] = (
            time_series['rolling_mean'] + 3 * time_series['rolling_std']
        )

        time_series['is_anomaly'] = (
            (time_series['count'] < time_series['lower_bound']) |
            (time_series['count'] > time_series['upper_bound'])
        )
        
        anomalies = time_series[time_series['is_anomaly']].copy()
        anomalies['deviation'] = (
            (anomalies['count'] - anomalies['rolling_mean']) / 
            anomalies['rolling_std']
        )
        
        logger.info(f"Found {len(anomalies)} anomalous periods")
        
        return anomalies