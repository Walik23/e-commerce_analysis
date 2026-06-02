import pandas as pd
import logging
from typing import Optional, Dict
from pathlib import Path
import glob
import kagglehub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader: 
    COLUMN_MAPPING = {
        'event_time': 'timestamp',
        'user_session': 'session_id',
    }

    def __init__(self, data_path: Optional[str] = None,
                 column_mapping: Optional[Dict] = None,
                 kaggle_dataset: Optional[str] = None):
        
        if kaggle_dataset:
            data_path = self._download_from_kaggle(kaggle_dataset)
            
        self.data_path = Path(data_path) if data_path else None
        
        self.column_mapping = self.COLUMN_MAPPING.copy()
        if column_mapping:
            self.column_mapping.update(column_mapping)

        logger.info(f"DataLoader initialized for {self.data_path}")

    def _download_from_kaggle(self, dataset: str) -> Optional[Path]:
        logger.info(f"Downloading dataset from Kaggle (this may take a while): {dataset}")
        
        try:
            path = kagglehub.dataset_download(dataset, output_dir="./data/raw/kaggle")
            logger.info(f"Dataset downloaded to: {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}")
            raise
    
    def load_csv(self, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
        try:
            if self.data_path.is_file():
                logger.info("Detected: single file path")
                return self._load_single_file(self.data_path, encoding)

            elif self.data_path.is_dir():
                csv_files = glob.glob(str(self.data_path / "*.csv"))
                
                if not csv_files:
                    csv_files = glob.glob(str(self.data_path / "**/*.csv"), recursive=True)
                if not csv_files:
                    logger.error(f"No CSV files found in {self.data_path}")
                    return None
                if len(csv_files) == 1:
                    logger.info(f"Detected: directory with single CSV file")
                    return self._load_single_file(Path(csv_files[0]), encoding)
                else:
                    logger.info(f"Detected: directory with {len(csv_files)} CSV files - merging...")
                    return self._load_multiple_files(csv_files, encoding)
            else:
                logger.error(f"Path does not exist: {self.data_path}")
                return None
                
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return None
        
    def _load_single_file(self, file_path: Path, encoding: str) -> Optional[pd.DataFrame]:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Successfully loaded {len(df)} records")
        
        df = self._normalize_column_names(df)
        df = self._parse_timestamp_column(df)
        
        return df

    def _load_multiple_files(self, csv_files: list, encoding: str) -> Optional[pd.DataFrame]:
        logger.info(f"Files to merge:")
        for f in csv_files:
            file_size = Path(f).stat().st_size / (1024 * 1024)
            logger.info(f"  - {Path(f).name} ({file_size:.2f} MB)")

        dataframes = []
        total_records = 0
        
        for file_path in csv_files:
            logger.info(f"Loading {Path(file_path).name}...")
            df_temp = pd.read_csv(file_path, encoding=encoding)
            rows = len(df_temp)
            total_records += rows
            logger.info(f"  => {rows:,} records")

            df_temp = self._normalize_column_names(df_temp)
            dataframes.append(df_temp)

        logger.info(f"Merging {len(dataframes)} files...")
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

        merged_df = self._parse_timestamp_column(merged_df)
        
        logger.info(f"  Successfully merged: {len(merged_df):,} total records")
        
        return merged_df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        lower_columns = {col.lower(): col for col in df.columns}
        
        rename_dict = {}
        for old_name_lower, actual_name in lower_columns.items():
            if old_name_lower in self.column_mapping:
                new_name = self.column_mapping[old_name_lower]
                rename_dict[actual_name] = new_name
                logger.info(f"Renaming the column: '{actual_name}' -> '{new_name}'")
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        return df
    
    def _parse_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' not in df.columns:
            logger.warning("Column 'timestamp' not found, skipping parsing")
            return df
        
        logger.info("Parsing the timestamp column...")
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            logger.info(f"Timestamp parsed. Range: {df['timestamp'].min()} - {df['timestamp'].max()}")
            
        except Exception as e:
            logger.error(f"Timestamp parsing error: {e}")
            logger.info("Attempting alternative parsing methods...")

            try:
                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], 
                    format='%Y-%m-%d %H:%M:%S UTC',
                    errors='coerce'
                )
                logger.info("Timestamp is parsed with explicit format")
            except:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                logger.warning("Automatic format detection is used")

        nat_count = df['timestamp'].isna().sum()
        if nat_count > 0:
            logger.warning(f"Found {nat_count} invalid timestamp values")
        
        return df
    
    def validate_schema(self, df: pd.DataFrame, 
                       required_columns: list) -> bool:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
            return False
        logger.info("Data scheme is valid")
        return True