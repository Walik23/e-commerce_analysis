from scipy.stats import ks_2samp

def validate_rfm_sample_representativeness(rfm_full, rfm_sample):
    validation_results = {
        'sample_size': len(rfm_sample),
        'full_size': len(rfm_full),
        'sample_rate_%': round((len(rfm_sample) / len(rfm_full) * 100), 2),
        'ks_tests': {},
        'descriptive_stats': {}
    }

    for metric in ['recency', 'frequency', 'monetary']:
        full_data = rfm_full[metric].dropna().values
        sample_data = rfm_sample[metric].dropna().values

        if len(full_data) == 0 or len(sample_data) == 0:
            validation_results['ks_tests'][metric] = {
                'ks_statistic': 0.0,
                'p_value': 1.0,
                'is_representative': True,
                'interpretation': "Insufficient data"
            }
            continue
        
        statistic, p_value = ks_2samp(full_data, sample_data)
        
        validation_results['ks_tests'][metric] = {
            'ks_statistic': float(round(statistic, 6)),
            'p_value': float(round(p_value, 6)),
            'is_representative': bool(p_value > 0.05),
            'interpretation': f"KS={statistic:.6f}, p={p_value:.6f} → {'REPRESENTATIVE' if p_value > 0.05 else 'NOT REPRESENTATIVE'}"
        }

    for metric in ['recency', 'frequency', 'monetary']:
        full_mean = float(rfm_full[metric].mean())
        sample_mean = float(rfm_sample[metric].mean())
        full_median = float(rfm_full[metric].median())
        sample_median = float(rfm_sample[metric].median())
        
        mean_diff = abs(sample_mean - full_mean) / full_mean * 100 if full_mean != 0 else 0
        median_diff = abs(sample_median - full_median) / full_median * 100 if full_median != 0 else 0
        
        validation_results['descriptive_stats'][metric] = {
            'mean_diff_%': float(round(mean_diff, 2)),
            'median_diff_%': float(round(median_diff, 2)),
            'full_mean': float(round(full_mean, 2)),
            'sample_mean': float(round(sample_mean, 2)),
            'full_median': float(round(full_median, 2)),
            'sample_median': float(round(sample_median, 2))
        }
    
    return validation_results


def validate_anomaly_detection_sample(df_full, df_sample):
    validation_results = {
        'sample_size': len(df_sample),
        'full_size': len(df_full),
        'anomaly_features': {}
    }

    features_to_check = ['total_events', 'total_sessions', 'events_per_session']
    
    for feature in features_to_check:
        if feature in df_full.columns and feature in df_sample.columns:
            statistic, p_value = ks_2samp(
                df_full[feature].dropna(),
                df_sample[feature].dropna()
            )
            
            validation_results['anomaly_features'][feature] = {
                'ks_statistic': round(statistic, 6),
                'p_value': round(p_value, 6),
                'is_representative': p_value > 0.05,
                'interpretation': f"p={p_value:.4f} → {'REPRESENTATIVE' if p_value > 0.05 else 'NOT REPRESENTATIVE'}"
            }
    
    return validation_results