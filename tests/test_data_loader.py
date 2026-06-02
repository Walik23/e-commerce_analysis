import pytest
import pandas as pd
from pathlib import Path
import tempfile
from src.data_collection.data_loader import DataLoader


class TestDataLoaderBasic: 
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'event_type': ['view', 'cart', 'view', 'purchase', 'view'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'price': [100.0, 200.0, 150.0, 300.0, 120.0]
        })
    
    def test_load_single_csv_file(self, sample_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'test.csv'
            sample_data.to_csv(file_path, index=False)
            
            loader = DataLoader(str(file_path))
            df = loader.load_csv()
            
            assert df is not None
            assert len(df) == 5
            assert 'timestamp' in df.columns
    
    def test_load_multiple_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df1 = pd.DataFrame({
                'user_id': [1, 2], 'event_type': ['view', 'cart'],
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
                'price': [100, 200]
            })
            df2 = pd.DataFrame({
                'user_id': [3, 4], 'event_type': ['view', 'purchase'],
                'timestamp': pd.date_range('2024-01-01 02:00', periods=2, freq='h'),
                'price': [150, 300]
            })
            
            Path(tmpdir, 'file1.csv').write_text(df1.to_csv(index=False))
            Path(tmpdir, 'file2.csv').write_text(df2.to_csv(index=False))
            
            loader = DataLoader(tmpdir)
            df = loader.load_csv()
            
            assert df is not None
            assert len(df) == 4
            assert df['user_id'].nunique() == 4
    
    def test_validate_schema(self, sample_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'test.csv'
            sample_data.to_csv(file_path, index=False)
            
            loader = DataLoader(str(file_path))
            df = loader.load_csv()
            
            required = ['user_id', 'event_type', 'timestamp']
            assert loader.validate_schema(df, required) is True
    
    def test_nonexistent_file(self):
        loader = DataLoader('/nonexistent/path/file.csv')
        df = loader.load_csv()
        assert df is None
    
    def test_csv_with_missing_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = pd.DataFrame({
                'user_id': [1, 2, None, 4],
                'event_type': ['view', None, 'cart', 'purchase'],
                'timestamp': pd.date_range('2024-01-01', periods=4, freq='h'),
            })
            
            file_path = Path(tmpdir) / 'test_missing.csv'
            data.to_csv(file_path, index=False)
            
            loader = DataLoader(str(file_path))
            df = loader.load_csv()
            
            assert df is not None
    
    def test_csv_with_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = pd.DataFrame({
                'user_id': [1, 2, 3],
                'event_type': ['перегляд', 'кошик', 'покупка'],
                'timestamp': pd.date_range('2024-01-01', periods=3, freq='h'),
            })
            
            file_path = Path(tmpdir) / 'test_unicode.csv'
            data.to_csv(file_path, index=False, encoding='utf-8')
            
            loader = DataLoader(str(file_path))
            df = loader.load_csv(encoding='utf-8')
            
            assert df is not None
            assert 'перегляд' in df['event_type'].values
    
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            df = loader.load_csv()
            assert df is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])