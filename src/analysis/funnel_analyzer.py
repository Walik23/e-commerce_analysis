import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FunnelAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info(f"Initialized FunnelAnalyzer with {len(df)} events")
    
    def define_funnel(self, steps: List[str]) -> pd.DataFrame:
        logger.info(f"Funnel analysis: {' -> '.join(steps)}")
        
        funnel_stats = []
        
        for i, step in enumerate(steps):
            if i == 0:
                users_at_step = set(self.df[self.df['event_type'] == step]['user_id'])
                total_users = len(users_at_step)
                
                funnel_stats.append({
                    'step': step,
                    'step_number': i + 1,
                    'users': total_users,
                    'conversion_rate': 100.0,
                    'drop_off_rate': 0.0
                })
            else:
                prev_step = steps[i-1]

                prev_users = set(self.df[self.df['event_type'] == prev_step]['user_id'])
                
                current_users = set(
                    self.df[
                        (self.df['event_type'] == step) & 
                        (self.df['user_id'].isin(prev_users))
                    ]['user_id']
                )
                
                users_count = len(current_users)
                prev_count = len(prev_users)
                
                conversion = (users_count / prev_count * 100) if prev_count > 0 else 0
                drop_off = 100 - conversion
                
                funnel_stats.append({
                    'step': step,
                    'step_number': i + 1,
                    'users': users_count,
                    'conversion_rate': round(conversion, 2),
                    'drop_off_rate': round(drop_off, 2)
                })
                
                logger.info(f"  {step}: {users_count} users ({conversion:.2f}% conversion)")
        
        return pd.DataFrame(funnel_stats)
    
    def analyze_drop_off_points(self, funnel_df: pd.DataFrame, 
                                threshold: float = 20.0) -> List[Dict]:
        logger.info(f"Searching for critical drop-off points (threshold: {threshold}%)")
        
        critical_points = []
        
        for _, row in funnel_df.iterrows():
            if row['drop_off_rate'] >= threshold:
                critical_points.append({
                    'step': row['step'],
                    'step_number': row['step_number'],
                    'drop_off_rate': row['drop_off_rate'],
                    'severity': 'High' if row['drop_off_rate'] >= 40 else 'Medium'
                })
                
                logger.warning(
                    f"Critical point on step '{row['step']}': "
                    f"{row['drop_off_rate']:.2f}% drop-off"
                )
        
        return critical_points
    
    def segment_funnel_analysis(self, steps: List[str], 
                                segment_column: str) -> pd.DataFrame:
        logger.info(f"Segmented funnel analysis by '{segment_column}'")
        
        segments = self.df[segment_column].unique()
        all_results = []
        
        for segment in segments:
            segment_df = self.df[self.df[segment_column] == segment]
            segment_analyzer = FunnelAnalyzer(segment_df)
            
            funnel_result = segment_analyzer.define_funnel(steps)
            funnel_result['segment'] = segment
            
            all_results.append(funnel_result)
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        return combined_results
    
    def calculate_time_to_conversion(self, steps: List[str]) -> pd.DataFrame:
        logger.info("Calculation of time to conversion")
        
        final_step = steps[-1]
        converted_users = set(self.df[self.df['event_type'] == final_step]['user_id'])
        
        conversion_times = []
        
        for user_id in converted_users:
            user_events = self.df[self.df['user_id'] == user_id].sort_values('timestamp')

            first_step_time = user_events[user_events['event_type'] == steps[0]]['timestamp'].min()
            last_step_time = user_events[user_events['event_type'] == final_step]['timestamp'].max()
            
            if pd.notna(first_step_time) and pd.notna(last_step_time):
                time_diff = (last_step_time - first_step_time).total_seconds() / 3600
                conversion_times.append({
                    'user_id': user_id,
                    'time_to_conversion_hours': time_diff
                })
        
        time_df = pd.DataFrame(conversion_times)
        
        if len(time_df) > 0:
            stats = {
                'mean_hours': time_df['time_to_conversion_hours'].mean(),
                'median_hours': time_df['time_to_conversion_hours'].median(),
                'min_hours': time_df['time_to_conversion_hours'].min(),
                'max_hours': time_df['time_to_conversion_hours'].max(),
                'std_hours': time_df['time_to_conversion_hours'].std()
            }
            
            logger.info(f"Average time to conversion: {stats['mean_hours']:.2f} hours")
            return pd.DataFrame([stats])
        
        return pd.DataFrame()