import pytest
import pandas as pd
from src.analysis.funnel_analyzer import FunnelAnalyzer


class TestFunnelAnalyzer:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5],
            'event_type': ['view', 'cart', 'view', 'cart', 'view', 'cart',
                          'view', 'cart', 'view'],
            'timestamp': pd.date_range('2024-01-01', periods=9, freq='h'),
            'session_id': ['1_1', '1_1', '2_1', '2_1', '3_1', '3_1', '4_1', '4_1', '5_1']
        })
    
    def test_define_funnel_basic(self, sample_data):
        analyzer = FunnelAnalyzer(sample_data)
        funnel_result = analyzer.define_funnel(['view', 'cart'])
        
        assert len(funnel_result) == 2
        assert funnel_result.loc[0, 'step'] == 'view'
        assert funnel_result.loc[1, 'step'] == 'cart'
    
    def test_define_funnel_first_step_100_percent(self, sample_data):
        analyzer = FunnelAnalyzer(sample_data)
        funnel_result = analyzer.define_funnel(['view', 'cart', 'purchase'])
        
        assert funnel_result.loc[0, 'conversion_rate'] == 100.0
        assert funnel_result.loc[0, 'drop_off_rate'] == 0.0
    
    def test_define_funnel_user_counts(self, sample_data):
        analyzer = FunnelAnalyzer(sample_data)
        funnel_result = analyzer.define_funnel(['view', 'cart'])
        
        assert funnel_result.loc[0, 'users'] == 5
        assert funnel_result.loc[1, 'users'] == 4
    
    def test_define_funnel_conversion_rates(self, sample_data):
        analyzer = FunnelAnalyzer(sample_data)
        funnel_result = analyzer.define_funnel(['view', 'cart'])
        
        expected_conversion = (4 / 5) * 100
        assert abs(funnel_result.loc[1, 'conversion_rate'] - expected_conversion) < 0.01
    
    def test_define_funnel_drop_off_rates(self, sample_data):
        analyzer = FunnelAnalyzer(sample_data)
        funnel_result = analyzer.define_funnel(['view', 'cart'])
        
        assert abs(funnel_result.loc[1, 'drop_off_rate'] - 20.0) < 0.01
    
    def test_define_funnel_multiple_steps(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            'event_type': ['view', 'view', 'cart', 'purchase',
                          'view', 'view', 'cart', 'remove_from_cart',
                          'view', 'view', 'cart'],
            'timestamp': pd.date_range('2024-01-01', periods=11, freq='h')
        })
        
        analyzer = FunnelAnalyzer(df)
        funnel_result = analyzer.define_funnel(['view', 'cart', 'purchase', 'remove_from_cart'])
        
        assert len(funnel_result) == 4
        assert funnel_result.loc[2, 'step'] == 'purchase'
    
    def test_define_funnel_no_conversions(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'event_type': ['view', 'cart', 'view', 'cart'],
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='h')
        })
        
        analyzer = FunnelAnalyzer(df)
        funnel_result = analyzer.define_funnel(['view', 'cart', 'purchase'])
        
        assert funnel_result.loc[2, 'users'] == 0
        assert funnel_result.loc[2, 'conversion_rate'] == 0.0
    
    def test_analyze_drop_off_points(self):
        df = pd.DataFrame({
            'user_id': list(range(1, 11)) + list(range(1, 5)),
            'event_type': ['view']*10 + ['cart']*4,
            'timestamp': pd.date_range('2024-01-01', periods=14, freq='h')
        })
        
        analyzer = FunnelAnalyzer(df)
        funnel_result = analyzer.define_funnel(['view', 'cart', 'purchase'])
        critical = analyzer.analyze_drop_off_points(funnel_result, threshold=20)
        
        assert len(critical) > 0
        assert any(cp['step'] == 'cart' for cp in critical)
    
    def test_funnel_single_user(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 1],
            'event_type': ['view', 'cart', 'purchase'],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='h')
        })
        
        analyzer = FunnelAnalyzer(df)
        funnel_result = analyzer.define_funnel(['view', 'cart', 'purchase'])
        
        assert len(funnel_result) == 3
        assert funnel_result.loc[0, 'users'] == 1
    
    def test_funnel_empty_step(self):
        df = pd.DataFrame({
            'user_id': [1, 1, 2],
            'event_type': ['view', 'cart', 'view'],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='h')
        })
        
        analyzer = FunnelAnalyzer(df)
        funnel_result = analyzer.define_funnel(['view', 'cart', 'purchase'])
        
        assert funnel_result.loc[2, 'users'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])