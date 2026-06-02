import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UserSegmentation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info(f"Initialized UserSegmentation with {len(df)} events")
    
    def calculate_rfm(self, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        logger.info("Calculation of RFM metrics")
        
        if reference_date is None:
            reference_date = self.df['timestamp'].max()

        recency = self.df.groupby('user_id')['timestamp'].max().reset_index()
        recency['recency'] = (reference_date - recency['timestamp']).dt.days
        recency = recency[['user_id', 'recency']]

        frequency = self.df.groupby('user_id').size().reset_index()
        frequency.columns = ['user_id', 'frequency']

        if 'revenue' in self.df.columns:
            monetary = self.df.groupby('user_id')['revenue'].sum().reset_index()
            monetary.columns = ['user_id', 'monetary']
        elif 'price' in self.df.columns:
            monetary = self.df.groupby('user_id')['price'].sum().reset_index()
            monetary.columns = ['user_id', 'monetary']
        else:
            purchase_events = self.df[self.df['event_type'] == 'purchase']
            monetary = purchase_events.groupby('user_id').size().reset_index()
            monetary.columns = ['user_id', 'monetary']

        rfm = recency.merge(frequency, on='user_id').merge(monetary, on='user_id')
        
        logger.info(f"RFM metrics are calculated for {len(rfm)} users")
        
        return rfm
    
    def assign_rfm_scores(self, rfm_df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:

        logger.info(f"Assigning RFM scores ({quantiles} levels)")
        
        rfm_scored = rfm_df.copy()

        def safe_qcut(series, q, labels, reverse=False):
            try:
                return pd.qcut(series, q=q, labels=labels, duplicates='drop')
            except ValueError as e:
                logger.warning(f"qcut failed for column, using alternative method: {e}")

                ranks = series.rank(method='dense', ascending=not reverse)

                min_rank = ranks.min()
                max_rank = ranks.max()
                
                if max_rank == min_rank:
                    return pd.Series([labels[len(labels)//2]] * len(series), index=series.index)

                normalized = ((ranks - min_rank) / (max_rank - min_rank) * (len(labels) - 1)).round()

                return normalized.apply(lambda x: labels[int(x)])

        rfm_scored['r_score'] = safe_qcut(
            rfm_scored['recency'], 
            q=quantiles, 
            labels=list(range(quantiles, 0, -1)),
            reverse=True
        )

        rfm_scored['f_score'] = safe_qcut(
            rfm_scored['frequency'], 
            q=quantiles, 
            labels=list(range(1, quantiles + 1)),
            reverse=False
        )

        rfm_scored['m_score'] = safe_qcut(
            rfm_scored['monetary'], 
            q=quantiles, 
            labels=list(range(1, quantiles + 1)),
            reverse=False
        )

        rfm_scored['rfm_score'] = (
            rfm_scored['r_score'].astype(str) + 
            rfm_scored['f_score'].astype(str) + 
            rfm_scored['m_score'].astype(str)
        )
        
        return rfm_scored
    
    def create_rfm_segments(self, rfm_scored: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating RFM segments")
        
        rfm_segmented = rfm_scored.copy()
        
        def assign_segment(row):
            r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])

            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'

            elif f >= 4:
                return 'Loyal Customers'

            elif r >= 3 and f >= 3:
                return 'Potential Loyalists'

            elif r >= 4 and f <= 2:
                return 'New Customers'

            elif r <= 2 and f >= 3:
                return 'At Risk'

            elif r <= 2 and m >= 4:
                return "Can't Lose Them"

            elif r <= 2:
                return 'Hibernating'

            elif r == 3 and f <= 2:
                return 'About to Sleep'
            
            else:
                return 'Others'
        
        rfm_segmented['segment'] = rfm_segmented.apply(assign_segment, axis=1)
        
        return rfm_segmented
    
    def kmeans_clustering(self, rfm_df: pd.DataFrame, 
                         n_clusters: int = 5) -> Tuple[pd.DataFrame, float]:
        logger.info(f"K-means clustering with {n_clusters} clusters on {len(rfm_df):,} users")

        features = rfm_df[['recency', 'frequency', 'monetary']].copy()

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        self.scaler = scaler

        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300,
            algorithm='elkan'
        )
        clusters = kmeans.fit_predict(features_scaled)

        self.kmeans_model = kmeans

        if len(rfm_df) <= 100000:
            silhouette_avg = silhouette_score(features_scaled, clusters)
        else:
            logger.info("Calculation of silhouette score on the sample...")
            sample_size = min(10000, len(features_scaled))
            sample_indices = np.random.choice(len(features_scaled), sample_size, replace=False)
            silhouette_avg = silhouette_score(
                features_scaled[sample_indices], 
                clusters[sample_indices]
            )
        
        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")

        result_df = rfm_df.copy()
        result_df['cluster'] = clusters

        cluster_stats = result_df.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'user_id': 'count'
        }).round(2)
        cluster_stats.columns = ['avg_recency', 'avg_frequency', 'avg_monetary', 'user_count']
        
        logger.info(f"Clusters statistic:\n{cluster_stats}")
        
        return result_df, silhouette_avg
    
    def predict_clusters_for_all(self, rfm_all: pd.DataFrame, 
                                 rfm_sample_clustered: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Cluster assignment for {len(rfm_all):,} users...")
        
        if not hasattr(self, 'kmeans_model') or not hasattr(self, 'scaler'):
            logger.error("The model is not trained! First call kmeans_clustering()")
            return rfm_all

        features = rfm_all[['recency', 'frequency', 'monetary']].copy()
        features_scaled = self.scaler.transform(features)

        clusters = self.kmeans_model.predict(features_scaled)

        result_df = rfm_all.copy()
        result_df['cluster'] = clusters
        
        logger.info(f"Clusters are assigned to all users")
        
        return result_df
    
    def find_optimal_clusters(self, rfm_df: pd.DataFrame, 
                            max_clusters: int = 10) -> pd.DataFrame:
        logger.info(f"Finding the optimal number of clusters (up to {max_clusters})")
        
        features = rfm_df[['recency', 'frequency', 'monetary']].copy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        results = []

        calculate_silhouette = len(rfm_df) <= 50000
        
        for n in range(3, max_clusters + 1):
            logger.info(f"  Testing {n} clusters...")
            
            kmeans = KMeans(
                n_clusters=n, 
                random_state=42, 
                n_init=10,
                max_iter=100,
                algorithm='elkan'
            )
            clusters = kmeans.fit_predict(features_scaled)
            
            if calculate_silhouette:
                silhouette_avg = silhouette_score(features_scaled, clusters)
            else:
                sample_size = min(10000, len(features_scaled))
                sample_indices = np.random.choice(len(features_scaled), sample_size, replace=False)
                silhouette_avg = silhouette_score(
                    features_scaled[sample_indices], 
                    clusters[sample_indices]
                )
            
            inertia = kmeans.inertia_
            
            results.append({
                'n_clusters': n,
                'silhouette_score': round(silhouette_avg, 3),
                'inertia': round(inertia, 2)
            })
            
            logger.info(f"    {n} clusters: silhouette={silhouette_avg:.3f}, inertia={inertia:.2f}")
        
        return pd.DataFrame(results)