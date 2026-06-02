import pytest
import pandas as pd
import numpy as np
from src.analysis.cohort_analyzer import CohortAnalyzer


class TestCohortAnalyzer:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        return pd.DataFrame({
            'user_id': np.repeat(range(1, 11), 6),
            'event_type': ['view', 'cart', 'purchase'] * 20,
            'timestamp': np.tile(dates[:6], 10),
            'session_id': np.repeat(range(1, 61), 1)
        })
    
    def test_create_cohorts_basic(self, sample_data):
        analyzer = CohortAnalyzer(sample_data)
        cohorts = analyzer.create_cohorts()
        
        assert 'user_id' in cohorts.columns
        assert 'cohort_date' in cohorts.columns
        assert 'cohort_period' in cohorts.columns
    
    def test_create_cohorts_unique_per_user(self, sample_data):
        analyzer = CohortAnalyzer(sample_data)
        cohorts = analyzer.create_cohorts()
        
        assert len(cohorts) == 10
        assert cohorts.groupby('user_id').size().max() == 1
    
    def test_create_cohorts_period_assignment(self, sample_data):
        analyzer = CohortAnalyzer(sample_data)
        cohorts = analyzer.create_cohorts()
        
        assert cohorts['cohort_period'].notna().all()
    
    def test_create_cohorts_multiple_months(self):
      data = []

      data.append({'user_id': 1, 'timestamp': '2024-01-15', 'event_type': 'view'})
      data.append({'user_id': 1, 'timestamp': '2024-01-20', 'event_type': 'cart'})

      data.append({'user_id': 2, 'timestamp': '2024-01-10', 'event_type': 'view'})

      data.append({'user_id': 3, 'timestamp': '2024-02-01', 'event_type': 'view'})
      data.append({'user_id': 3, 'timestamp': '2024-02-05', 'event_type': 'purchase'})

      data.append({'user_id': 4, 'timestamp': '2024-02-10', 'event_type': 'view'})

      data.append({'user_id': 5, 'timestamp': '2024-03-01', 'event_type': 'view'})
      
      df = pd.DataFrame(data)
      df['timestamp'] = pd.to_datetime(df['timestamp'])
      df['session_id'] = range(1, len(df) + 1)
      
      analyzer = CohortAnalyzer(df)
      cohorts = analyzer.create_cohorts()

      unique_cohorts = cohorts['cohort_period'].unique()
      assert len(unique_cohorts) == 3, f"Expected 3 cohorts, got {len(unique_cohorts)}"
      assert set(unique_cohorts.astype(str)) == {'2024-01', '2024-02', '2024-03'}
    
    def test_calculate_retention_basic(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'event_type': ['view', 'view', 'view', 'view', 'view', 'view', 'view', 'view', 'view'],
            'timestamp': pd.to_datetime([
                '2024-01-01', '2024-02-01', '2024-03-01',
                '2024-01-01', '2024-02-01', '2024-03-01',
                '2024-01-01', '2024-02-01', '2024-03-01'
            ]),
            'session_id': ['1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '3_1', '3_2', '3_3']
        })
        
        analyzer = CohortAnalyzer(df)
        cohorts = analyzer.create_cohorts()
        retention = analyzer.calculate_retention(cohorts, periods=3)
        
        assert retention is not None
        assert not retention.empty
    
    def test_calculate_retention_percentage_values(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'event_type': ['view', 'view', 'view', 'view', 'view'],
            'timestamp': pd.to_datetime([
                '2024-01-01', '2024-02-01',
                '2024-01-01', '2024-02-01',
                '2024-01-01'
            ]),
            'session_id': ['1_1', '1_2', '2_1', '2_2', '3_1']
        })
        
        analyzer = CohortAnalyzer(df)
        cohorts = analyzer.create_cohorts()
        retention = analyzer.calculate_retention(cohorts, periods=2)
        
        assert (retention >= 0).all().all()
        assert (retention <= 100).all().all()
    
    def test_analyze_cohort_behavior(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'event_type': ['view', 'cart', 'view', 'purchase'],
            'timestamp': pd.to_datetime([
                '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'
            ]),
            'session_id': ['1_1', '1_2', '2_1', '2_2']
        })
        
        analyzer = CohortAnalyzer(df)
        cohorts = analyzer.create_cohorts()
        behavior = analyzer.analyze_cohort_behavior(cohorts)
        
        assert not behavior.empty
        assert 'total_users' in behavior.columns
    
    def test_compare_cohorts_basic(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'event_type': ['view', 'view', 'view', 'view'],
            'timestamp': pd.to_datetime([
                '2024-01-01', '2024-02-01', '2024-01-01', '2024-02-01'
            ]),
            'session_id': ['1_1', '1_2', '2_1', '2_2']
        })
        
        analyzer = CohortAnalyzer(df)
        cohorts = analyzer.create_cohorts()
        retention = analyzer.calculate_retention(cohorts, periods=2)
        comparison = analyzer.compare_cohorts(retention)
        
        assert 'best_cohort' in comparison
        assert 'worst_cohort' in comparison
    
    def test_cohort_with_single_user(self):
        df = pd.DataFrame({
            'user_id': [1],
            'event_type': ['view'],
            'timestamp': pd.to_datetime(['2024-01-01']),
            'session_id': ['1_1']
        })
        
        analyzer = CohortAnalyzer(df)
        cohorts = analyzer.create_cohorts()
        
        assert len(cohorts) == 1
        assert cohorts.iloc[0]['user_id'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])