import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        logger.info("Initialized DataPreprocessor")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Start cleaning data")
        initial_rows = len(df)

        df = df.drop_duplicates()

        df = df.dropna(subset=['user_id', 'event_type', 'timestamp'])

        if 'price' in df.columns:
            df = df[df['price'] >= 0]
        
        if 'quantity' in df.columns:
            df = df[df['quantity'] > 0]
        
        logger.info(f"Cleaning completed: {initial_rows} -> {len(df)} records")
        return df
    
    def parse_timestamps(self, df: pd.DataFrame, 
                        timestamp_column: str = 'timestamp') -> pd.DataFrame:
        logger.info("Timestamps parsing")
        
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df['date'] = df[timestamp_column].dt.date
        df['hour'] = df[timestamp_column].dt.hour
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['week'] = df[timestamp_column].dt.isocalendar().week
        df['month'] = df[timestamp_column].dt.month
        
        return df
    
    def create_session_id(self, df: pd.DataFrame, 
                         time_threshold: int = 30,
                         time_unit: str = 'minutes') -> pd.DataFrame:
        logger.info(f"Creating a session with a threshold {time_threshold} minutes")
        
        df = df.sort_values(['user_id', 'timestamp']).copy()

        if time_unit == 'minutes':
            threshold_td = pd.Timedelta(minutes=time_threshold)
        elif time_unit == 'seconds':
            threshold_td = pd.Timedelta(seconds=time_threshold)
        elif time_unit == 'hours':
            threshold_td = pd.Timedelta(hours=time_threshold)
        else:
            raise ValueError(f"Unsupported time_unit: {time_unit}")

        df['time_diff'] = df.groupby('user_id')['timestamp'].diff()

        df['new_session'] = (df['time_diff'] > threshold_td) | df['time_diff'].isna()

        df['session_num'] = df.groupby('user_id')['new_session'].cumsum()
        df['session_id'] = df['user_id'].astype(str) + '_' + df['session_num'].astype(str)

        df = df.drop(['time_diff', 'new_session', 'session_num'], axis=1)
        
        return df