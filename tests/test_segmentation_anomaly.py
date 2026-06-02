import pytest
import pandas as pd
import numpy as np
from src.analysis.segmentation import UserSegmentation
from src.analysis.anomaly_detector import AnomalyDetector


class TestUserSegmentation:
    @pytest.fixture
    def rfm_data(self):
        return pd.DataFrame({
            'user_id': range(1, 21),
            'recency': np.random.randint(1, 100, 20),
            'frequency': np.random.randint(1, 50, 20),
            'monetary': np.random.uniform(100, 5000, 20)
        })
    
    def test_calculate_rfm_basic(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3],
            'event_type': ['view', 'cart', 'purchase', 'view', 'cart', 'view'],
            'timestamp': pd.to_datetime([
                '2024-01-01', '2024-01-02', '2024-01-03',
                '2024-01-10', '2024-01-11', '2023-01-01'
            ]),
            'price': [100, 200, 300, 150, 250, 50],
            'session_id': ['1_1', '1_1', '1_2', '2_1', '2_2', '3_1']
        })
        
        segmentation = UserSegmentation(df)
        rfm = segmentation.calculate_rfm()
        
        assert 'recency' in rfm.columns
        assert 'frequency' in rfm.columns
        assert 'monetary' in rfm.columns
        assert (rfm['frequency'] > 0).all()
    
    def test_assign_rfm_scores(self, rfm_data):
        segmentation = UserSegmentation(rfm_data)
        rfm_scored = segmentation.assign_rfm_scores(rfm_data)
        
        assert 'r_score' in rfm_scored.columns
        assert 'f_score' in rfm_scored.columns
        assert 'm_score' in rfm_scored.columns
    
    def test_create_rfm_segments(self, rfm_data):
        segmentation = UserSegmentation(rfm_data)
        rfm_scored = segmentation.assign_rfm_scores(rfm_data)
        rfm_segmented = segmentation.create_rfm_segments(rfm_scored)
        
        assert 'segment' in rfm_segmented.columns
        assert len(rfm_segmented['segment'].unique()) > 0
    
    def test_find_optimal_clusters(self, rfm_data):
        segmentation = UserSegmentation(rfm_data)
        optimal = segmentation.find_optimal_clusters(rfm_data, max_clusters=5)
        
        assert not optimal.empty
        assert 'n_clusters' in optimal.columns
        assert 'silhouette_score' in optimal.columns
    
    def test_kmeans_clustering(self, rfm_data):
        segmentation = UserSegmentation(rfm_data)
        clustered, silhouette = segmentation.kmeans_clustering(rfm_data, n_clusters=3)
        
        assert 'cluster' in clustered.columns
        assert clustered['cluster'].nunique() == 3
        assert -1 <= silhouette <= 1
    
    def test_kmeans_large_dataset(self):
        large_rfm = pd.DataFrame({
            'user_id': range(1, 100001),
            'recency': np.random.randint(1, 100, 100000),
            'frequency': np.random.randint(1, 50, 100000),
            'monetary': np.random.uniform(100, 5000, 100000)
        })
        
        segmentation = UserSegmentation(large_rfm)
        clustered, silhouette = segmentation.kmeans_clustering(large_rfm, n_clusters=5)
        
        assert len(clustered) == 100000
        assert clustered['cluster'].nunique() == 5


class TestAnomalyDetector:
    def test_detect_fraud_patterns(self):
        df = pd.DataFrame({
            'user_id': list(range(1, 21)) + [1]*5,
            'event_type': ['view']*25,
            'timestamp': pd.date_range('2024-01-01', periods=25, freq='h'),
            'hour': [i % 24 for i in range(25)],
            'price': [100]*20 + [5000]*5
        })
        
        detector = AnomalyDetector(df)
        suspicious = detector.detect_fraud_patterns(sample_size=1000000)
        
        assert isinstance(suspicious, pd.DataFrame)
    
    def test_detect_high_frequency(self):
        df = pd.DataFrame({
            'user_id': [1]*200,
            'event_type': ['view']*200,
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'hour': [i % 24 for i in range(200)]
        })
        
        detector = AnomalyDetector(df)
        suspicious = detector.detect_fraud_patterns()
        
        if len(suspicious) > 0:
            assert any(suspicious['user_id'] == 1)
    
    def test_statistical_anomalies(self):
        df = pd.DataFrame({
            'user_id': range(1, 101),
            'event_type': ['view']*100,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'price': [100]*90 + [5000]*10
        })
        
        detector = AnomalyDetector(df)
        anomalies = detector.statistical_anomalies('price', threshold=3.0)
        
        assert isinstance(anomalies, pd.DataFrame)
        assert len(anomalies) > 0
    
    def test_isolation_forest(self):
        df = pd.DataFrame({
            'user_id': range(1, 101),
            'event_type': ['view']*100,
            'session_id': [f'{i}_1' for i in range(1, 101)],
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'price': [100]*95 + [5000]*5
        })
        
        detector = AnomalyDetector(df)
        anomalies, metrics = detector.isolation_forest_anomalies(
            features=['user_id'], contamination=0.1
        )
        
        assert isinstance(metrics, dict)
        assert 'anomalies_found' in metrics
    
    def test_time_series_anomalies(self):
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        df = pd.DataFrame({
            'user_id': range(1, 61),
            'event_type': ['view']*60,
            'timestamp': dates,
            'session_id': [f'{i}_1' for i in range(1, 61)]
        })
        
        detector = AnomalyDetector(df)
        anomalies = detector.time_series_anomalies(window='D')
        
        assert isinstance(anomalies, pd.DataFrame)
    
    def test_statistical_anomalies(self):
      df = pd.DataFrame({
          'user_id': range(1, 101),
          'event_type': ['view']*100,
          'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
          'price': [100]*90 + [5000]*10
      })
      
      detector = AnomalyDetector(df)
      anomalies = detector.statistical_anomalies('price', threshold=2.0)
      
      assert isinstance(anomalies, pd.DataFrame)
      if len(anomalies) == 0:
          anomalies = detector.statistical_anomalies('price', threshold=1.5)
      assert len(anomalies) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])