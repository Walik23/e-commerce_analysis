import pytest
import pandas as pd
from src.data_processing.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'event_type': ['view', 'cart', 'view', 'purchase', 'view'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'price': [100.0, 200.0, 150.0, 300.0, 120.0],
            'quantity': [1, 2, 1, 3, 1]
        })
    
    def test_clean_removes_duplicates(self, sample_data):
        df = pd.concat([sample_data, sample_data.iloc[0:2]], ignore_index=True)
        assert len(df) == 7
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        
        assert len(df_clean) == 5
    
    def test_clean_removes_missing_fields(self, sample_data):
        df = sample_data.copy()
        df.loc[1, 'user_id'] = None
        df.loc[2, 'event_type'] = None
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        
        assert df_clean['user_id'].isna().sum() == 0
        assert df_clean['event_type'].isna().sum() == 0
    
    def test_clean_removes_negative_price(self, sample_data):
        df = sample_data.copy()
        df.loc[1, 'price'] = -50.0
        df.loc[3, 'price'] = -100.0
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        
        assert (df_clean['price'] >= 0).all()
        assert len(df_clean) == 3
    
    def test_clean_removes_zero_quantity(self, sample_data):
        df = sample_data.copy()
        df.loc[1, 'quantity'] = 0
        df.loc[2, 'quantity'] = -1
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        
        assert (df_clean['quantity'] > 0).all()
    
    def test_parse_timestamps_converts_to_datetime(self):
      df = pd.DataFrame({
          'user_id': [1, 2, 3],
          'event_type': ['view', 'cart', 'purchase'],
          'timestamp': ['2024-01-01 10:30:00', '2024-01-01 11:45:00', '2024-01-01 12:00:00']
      })
      
      preprocessor = DataPreprocessor()
      df_parsed = preprocessor.parse_timestamps(df)

      assert pd.api.types.is_datetime64_any_dtype(df_parsed['timestamp'])
    
    def test_parse_timestamps_creates_columns(self):
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'event_type': ['view', 'cart', 'purchase'],
            'timestamp': pd.date_range('2024-01-01 10:30:00', periods=3, freq='h')
        })
        
        preprocessor = DataPreprocessor()
        df_parsed = preprocessor.parse_timestamps(df)
        
        assert 'date' in df_parsed.columns
        assert 'hour' in df_parsed.columns
        assert 'day_of_week' in df_parsed.columns
        assert 'month' in df_parsed.columns
    
    def test_create_session_id_single_user(self):
      df = pd.DataFrame({
          'user_id': [1, 1, 1, 1],
          'event_type': ['view', 'cart', 'view', 'purchase'],
          'timestamp': pd.date_range('2024-01-01 10:00', periods=4, freq='h')
      })
      
      preprocessor = DataPreprocessor()
      df_sessions = preprocessor.create_session_id(df, time_threshold=30)
      
      assert 'session_id' in df_sessions.columns
      assert df_sessions['session_id'].nunique() == 4
    
    def test_create_session_id_multiple_users(self):
      df = pd.DataFrame({
          'user_id': [1, 1, 2, 2],
          'event_type': ['view', 'cart', 'view', 'purchase'],
          'timestamp': pd.date_range('2024-01-01 10:00', periods=4, freq='h')
      })
      
      preprocessor = DataPreprocessor()
      df_sessions = preprocessor.create_session_id(df, time_threshold=30)
      
      user1_sessions = df_sessions[df_sessions['user_id'] == 1]['session_id'].unique()
      user2_sessions = df_sessions[df_sessions['user_id'] == 2]['session_id'].unique()

      assert len(user1_sessions) == 2
      assert len(user2_sessions) == 2
    
    def test_create_session_id_with_timeout(self):
      timestamps = [
          '2024-01-01 10:00:00',
          '2024-01-01 10:15:00',
          '2024-01-01 10:46:00',
          '2024-01-01 11:00:00'
      ]
      
      df = pd.DataFrame({
          'user_id': [1, 1, 1, 1],
          'event_type': ['view', 'cart', 'view', 'purchase'],
          'timestamp': pd.to_datetime(timestamps)
      })
      
      preprocessor = DataPreprocessor()
      df_sessions = preprocessor.create_session_id(df, time_threshold=30)

      assert df_sessions['session_id'].nunique() == 2
    
    def test_full_pipeline(self):
        df = pd.DataFrame({
            'user_id': [1, 2, 1, 2, 1],
            'event_type': ['view', 'cart', 'view', 'purchase', 'view'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'price': [100, 200, 150, 300, 120]
        })
        
        preprocessor = DataPreprocessor()
        df = preprocessor.clean_data(df)
        df = preprocessor.parse_timestamps(df)
        df = preprocessor.create_session_id(df)
        
        assert 'session_id' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        assert len(df) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])