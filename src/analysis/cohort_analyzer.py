import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CohortAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info(f"Initialized CohortAnalyzer with {len(df)} events")
    
    def create_cohorts(self, cohort_type: str = 'registration') -> pd.DataFrame:
        logger.info(f"Creating cohort of type '{cohort_type}'")

        user_cohorts = self.df.groupby('user_id')['timestamp'].min().reset_index()
        user_cohorts.columns = ['user_id', 'cohort_date']

        user_cohorts['cohort_period'] = user_cohorts['cohort_date'].dt.to_period('M')
        
        logger.info(f"Created {user_cohorts['cohort_period'].nunique()} of unique cohorts")
        
        return user_cohorts
    
    def calculate_retention(self, cohorts_df: pd.DataFrame, 
                           periods: int = 12) -> pd.DataFrame:
        logger.info(f"Calculation retention for {periods} periods")

        df_with_cohorts = self.df.merge(cohorts_df, on='user_id', how='left')

        df_with_cohorts['event_period'] = df_with_cohorts['timestamp'].dt.to_period('M')

        df_with_cohorts['period_number'] = (
            (df_with_cohorts['event_period'] - df_with_cohorts['cohort_period']).apply(lambda x: x.n)
        )

        cohort_data = df_with_cohorts.groupby(['cohort_period', 'period_number'])['user_id'].nunique().reset_index()
        cohort_data.columns = ['cohort_period', 'period_number', 'users']

        cohort_sizes = cohort_data[cohort_data['period_number'] == 0].set_index('cohort_period')['users']

        cohort_data['cohort_size'] = cohort_data['cohort_period'].map(cohort_sizes)
        cohort_data['retention_rate'] = (cohort_data['users'] / cohort_data['cohort_size'] * 100).round(2)

        retention_matrix = cohort_data.pivot_table(
            index='cohort_period',
            columns='period_number',
            values='retention_rate',
            fill_value=0
        )
        
        logger.info(f"The retention matrix is created: {retention_matrix.shape}")
        
        return retention_matrix
    
    def analyze_cohort_behavior(self, cohorts_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analysis of cohort behavior")
        
        df_with_cohorts = self.df.merge(cohorts_df, on='user_id', how='left')
        
        metrics = []
        
        for cohort in df_with_cohorts['cohort_period'].unique():
            cohort_data = df_with_cohorts[df_with_cohorts['cohort_period'] == cohort]
            
            metric = {
                'cohort': str(cohort),
                'total_users': cohort_data['user_id'].nunique(),
                'total_events': len(cohort_data),
                'avg_events_per_user': len(cohort_data) / cohort_data['user_id'].nunique(),
                'avg_session_length': cohort_data.groupby('session_id').size().mean()
            }

            if 'revenue' in cohort_data.columns:
                metric['total_revenue'] = cohort_data['revenue'].sum()
                metric['avg_revenue_per_user'] = cohort_data.groupby('user_id')['revenue'].sum().mean()
            
            metrics.append(metric)
        
        return pd.DataFrame(metrics)
    
    def compare_cohorts(self, retention_matrix: pd.DataFrame) -> Dict:
        logger.info("Comperative analysis of cohort")
        
        comparison = {
            'best_cohort': {
                'period': str(retention_matrix[1].idxmax()),
                'retention_month_1': retention_matrix[1].max()
            },
            'worst_cohort': {
                'period': str(retention_matrix[1].idxmin()),
                'retention_month_1': retention_matrix[1].min()
            },
            'average_retention': {
                f'month_{i}': retention_matrix[i].mean() 
                for i in range(min(6, len(retention_matrix.columns)))
            }
        }
        
        logger.info(f"The best cohort: {comparison['best_cohort']['period']}")
        logger.info(f"The worst cohort: {comparison['worst_cohort']['period']}")
        
        return comparison