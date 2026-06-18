import logging
import json
from pathlib import Path
from datetime import datetime

from src.data_collection.data_loader import DataLoader
from src.data_processing.preprocessor import DataPreprocessor
from src.analysis.funnel_analyzer import FunnelAnalyzer
from src.analysis.cohort_analyzer import CohortAnalyzer
from src.analysis.segmentation import UserSegmentation
from src.analysis.anomaly_detector import AnomalyDetector
from src.visualization.visualizer import AnalyticsVisualizer
from src.validation.validation import validate_rfm_sample_representativeness
from src.validation.validation import validate_anomaly_detection_sample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("BEGINNING OF USER ACTIVITY MONITORING SYSTEM")
    logger.info("=" * 80)
    
    start_time = datetime.now()

    logger.info("\n[STEP 1] Loading data...")
    data_loader = DataLoader(kaggle_dataset="mkechinov/ecommerce-events-history-in-cosmetics-shop")
    df = data_loader.load_csv()
    
    if df is None:
        logger.error("Failed to load data. Completion of work.")
        return

    required_columns = ['user_id', 'event_type', 'timestamp']
    if not data_loader.validate_schema(df, required_columns):
        logger.error("The data schema is invalid. Completion of work.")
        return
    
    logger.info(f"{len(df)} of records loaded, {df['user_id'].nunique()} users")

    logger.info("\n[STEP 2] Data processing and cleaning...")
    preprocessor = DataPreprocessor()
    
    df = preprocessor.clean_data(df)
    df = preprocessor.parse_timestamps(df)
    df = preprocessor.create_session_id(df)
    
    logger.info(f"Data processed: {len(df)} records, {df['session_id'].nunique()} sessions")

    processed_path = 'data/processed/processed_events.csv'
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved: {processed_path}")

    logger.info("\n[STEP 3] Conversion funnel analysis…")
    funnel_analyzer = FunnelAnalyzer(df)

    funnel_steps = ['view', 'cart', 'purchase']
    
    funnel_result = funnel_analyzer.define_funnel(funnel_steps)
    print("\n--- Funnel analysis results ---")
    print(funnel_result.to_string(index=False))

    critical_points = funnel_analyzer.analyze_drop_off_points(funnel_result)
    if critical_points:
        print("\nCritical drop-off points:")
        for point in critical_points:
            print(f"  • {point['step']}: {point['drop_off_rate']:.1f}% (level: {point['severity']})")

    funnel_result.to_csv('data/results/funnel_analysis.csv', index=False)

    logger.info("\n[STEP 4] Cohort analysis of retention...")
    cohort_analyzer = CohortAnalyzer(df)
    
    cohorts = cohort_analyzer.create_cohorts()
    retention_matrix = cohort_analyzer.calculate_retention(cohorts, periods=12)
    
    print("\n--- Retention matrix (first 6 months) ---")
    print(retention_matrix.iloc[:, :6].round(1))

    cohort_comparison = cohort_analyzer.compare_cohorts(retention_matrix)
    print(f"\nBest cohort: {cohort_comparison['best_cohort']['period']} "
          f"({cohort_comparison['best_cohort']['retention_month_1']:.1f}% retention)")

    retention_matrix.to_csv('data/results/retention_matrix.csv')

    logger.info("\n[STEP 5] RFM user segmentation...")
    segmentation = UserSegmentation(df)
    
    rfm = segmentation.calculate_rfm()
    rfm_scored = segmentation.assign_rfm_scores(rfm)
    rfm_segmented = segmentation.create_rfm_segments(rfm_scored)
    
    print("\n--- Distribution of users by RFM segments ---")
    segment_distribution = rfm_segmented['segment'].value_counts()
    print(segment_distribution)

    rfm_segmented.to_csv('data/results/rfm_segments.csv', index=False)

    logger.info("\n[STEP 6] K-means clasterization...")

    max_users_for_clustering = 1000000

    if len(rfm) > max_users_for_clustering:
        logger.warning(f"The dataset contains {len(rfm):,} users")
        logger.info(f"Using sampling up to {max_users_for_clustering:,} users for clustering")

        rfm_sample = rfm.sample(n=max_users_for_clustering, random_state=42)

        logger.info("\nValidating K-means sample representativeness...")

        kmeans_validation = validate_rfm_sample_representativeness(rfm, rfm_sample)

        logger.info(f"Sample size: {kmeans_validation['sample_size']:,} / {kmeans_validation['full_size']:,} "
                    f"({kmeans_validation['sample_rate_%']:.1f}%)")

        print("\n--- K-means Sample Validation (KS-test) ---")
        all_kmeans_representative = True
        for metric, result in kmeans_validation['ks_tests'].items():
            print(f"{metric}: {result['interpretation']}")
            if not result['is_representative']:
                all_kmeans_representative = False

        print("\n--- Descriptive Statistics Comparison ---")
        for metric, stats in kmeans_validation['descriptive_stats'].items():
            print(f"{metric}: mean diff = {stats['mean_diff_%']}%, median diff = {stats['median_diff_%']}%")

        if not all_kmeans_representative:
            logger.warning(" Some KS-tests failed, but proceeding with analysis")
            logger.warning(" (Sample size may still be sufficient for k-means)")

        optimal_clusters_df = segmentation.find_optimal_clusters(rfm_sample, max_clusters=8)
    else:
        rfm_sample = rfm
        optimal_clusters_df = segmentation.find_optimal_clusters(rfm, max_clusters=8)
    
    print("\n--- Metrics for different number of clusters ---")
    print(optimal_clusters_df.to_string(index=False))

    best_n_clusters = optimal_clusters_df.loc[
        optimal_clusters_df['silhouette_score'].idxmax(), 
        'n_clusters'
    ]

    clustered_sample, silhouette = segmentation.kmeans_clustering(
        rfm_sample, 
        n_clusters=int(best_n_clusters)
    )
    print(f"\nClasterization is completed on a sample of {len(rfm_sample):,} users: "
          f"{int(best_n_clusters)} clusters, Silhouette Score = {silhouette:.3f}")

    if len(rfm) > max_users_for_clustering:
        logger.info("Model-based assignment of clusters to all users...")
        clustered_df = segmentation.predict_clusters_for_all(rfm, clustered_sample)
    else:
        clustered_df = clustered_sample

    clustered_df.to_csv('data/results/user_clusters.csv', index=False)
    logger.info(f"Clusters are assigned to {len(clustered_df):,} users")

    logger.info("\n[STEP 7.1] Validating anomaly detection sample...")

    df_user_features = df.groupby('user_id').agg({
        'event_type': 'count',
        'session_id': 'nunique',
        'timestamp': 'count'
    }).reset_index()
    df_user_features.columns = ['user_id', 'total_events', 'total_sessions', 'dummy']
    df_user_features['events_per_session'] = (
        df_user_features['total_events'] / df_user_features['total_sessions']
    )
    df_user_features = df_user_features.drop('dummy', axis=1)

    df_sample_features = df_user_features.sample(
        n=min(1000000, len(df_user_features)), 
        random_state=42
    )

    anomaly_validation = validate_anomaly_detection_sample(df_user_features, df_sample_features)

    print("\n--- Anomaly Detection Sample Validation (KS-test) ---")
    for feature, result in anomaly_validation['anomaly_features'].items():
        print(f"{feature}: {result['interpretation']}")

    logger.info("\n[STEP 7.2] Anomaly detection...")
    anomaly_detector = AnomalyDetector(df)

    suspicious_users = anomaly_detector.detect_fraud_patterns(
        sample_size=1000000,
        user_features=df_user_features
    )
    if len(suspicious_users) > 0:
        print(f"\n{len(suspicious_users)} suspicious users detected")
        print("\n--- Top 5 suspicious users ---")
        print(suspicious_users.head().to_string(index=False))
        suspicious_users.to_csv('data/results/suspicious_users.csv', index=False)

    if 'price' in df.columns:
        price_anomalies = anomaly_detector.statistical_anomalies('price', threshold=3.0)
        
        if len(price_anomalies) > 0:
            print(f"\nDetected {len(price_anomalies)} anomalous transactions by price")
            print("\n--- Top 5 anomalous transactions ---")
            print(price_anomalies.nlargest(5, 'deviation')[
                ['user_id', 'price', 'anomaly_type', 'deviation']
            ].to_string(index=False))
            
            price_anomalies.to_csv('data/results/price_anomalies.csv', index=False)

    features_for_anomaly = ['total_events', 'total_sessions', 'events_per_session']
    anomalies_iso, iso_metrics = anomaly_detector.isolation_forest_anomalies(
        features=features_for_anomaly,
        contamination=0.05
    )
    
    print(f"\nIsolation Forest: detected {iso_metrics['anomalies_found']} anomalies "
          f"({iso_metrics['anomaly_rate']}%)")
    
    if len(anomalies_iso) > 0:
        anomalies_iso.to_csv('data/results/anomalies_isolation_forest.csv', index=False)

    time_anomalies = anomaly_detector.time_series_anomalies(window='D')
    if len(time_anomalies) > 0:
        print("\n--- Top 5 most anomalous days ---")
        top_anomalies = time_anomalies.nlargest(5, 'deviation')[
            ['count', 'rolling_mean', 'deviation']
        ]
        print(top_anomalies.to_string())
        
        time_anomalies.to_csv('data/results/time_series_anomalies.csv')

    logger.info("\n[STEP 8] Creating visualizations…")
    visualizer = AnalyticsVisualizer()

    visualizer.plot_funnel(funnel_result)

    visualizer.plot_retention_heatmap(retention_matrix)
    visualizer.plot_retention_curves(retention_matrix)

    visualizer.plot_rfm_segments(rfm_segmented)

    visualizer.plot_clusters_3d(clustered_df)

    if len(time_anomalies) > 0:
        visualizer.plot_anomaly_timeline(time_anomalies)

    if len(funnel_result) >= 3:
        total_conversion = (
            funnel_result.iloc[-1]['users'] / 
            funnel_result.iloc[0]['users'] * 100
        )
    else:
        total_conversion = 0

    suspicious_user_ids = set(suspicious_users['user_id'].unique())
    anomaly_user_ids = set(anomalies_iso['user_id'].unique())
    all_anomaly_users = suspicious_user_ids.union(anomaly_user_ids)

    summary_metrics = {
        'total_users': df['user_id'].nunique(),
        'total_events': len(df),
        'conversion_rate': total_conversion,
        'avg_session_length': df.groupby('session_id').size().mean(),
        'retention_30d': retention_matrix[1].mean() if 1 in retention_matrix.columns else 0,
        'anomalies_detected': len(all_anomaly_users),
        'key_insights': [
            f"The main drop-off point: {critical_points[0]['step']} ({critical_points[0]['drop_off_rate']:.1f}%)" if critical_points else "No critical points were found",
            f"The largest segment: {segment_distribution.index[0]} ({segment_distribution.values[0]} of users)",
            f"Average retention after a month: {retention_matrix[1].mean():.1f}%" if 1 in retention_matrix.columns else "Retention data not available",
            f"Detected {len(all_anomaly_users)} users with potential anomalies"
        ]
    }
    
    visualizer.create_summary_dashboard(summary_metrics)
    
    logger.info("All visualizations have been created")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info(f"Execution time: {duration:.2f} seconds")
    logger.info(f"The results are saved in the directory: data/results/")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)