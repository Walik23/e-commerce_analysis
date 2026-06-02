import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AnalyticsVisualizer:
    def __init__(self, output_dir: str = 'data/results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        logger.info(f"Visualizer initialized, results in {output_dir}")
    
    def plot_funnel(self, funnel_df: pd.DataFrame, 
                   save_name: str = 'funnel_analysis.png'):
        logger.info("Creating a visualization of the funnel")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.bar(funnel_df['step'], funnel_df['users'], color='steelblue', alpha=0.7)
        ax1.set_xlabel('Крок воронки', fontsize=12)
        ax1.set_ylabel('Кількість користувачів', fontsize=12)
        ax1.set_title('Воронка конверсії: Абсолютні значення', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        for i, v in enumerate(funnel_df['users']):
            ax1.text(i, v + funnel_df['users'].max() * 0.02, str(v), 
                    ha='center', va='bottom', fontweight='bold')

        x = range(len(funnel_df))
        ax2.plot(x, funnel_df['conversion_rate'], marker='o', 
                color='green', linewidth=2, markersize=8, label='Конверсія')
        ax2.plot(x, funnel_df['drop_off_rate'], marker='s', 
                color='red', linewidth=2, markersize=8, label='Відпадання')
        
        ax2.set_xlabel('Крок воронки', fontsize=12)
        ax2.set_ylabel('Відсоток (%)', fontsize=12)
        ax2.set_title('Конверсія та відпадання по кроках', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(funnel_df['step'], rotation=45)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"The schedule is saved: {save_path}")
        plt.close()
    
    def plot_retention_heatmap(self, retention_matrix: pd.DataFrame,
                              save_name: str = 'retention_heatmap.png'):
        logger.info("Creating retention heatmap")
        
        plt.figure(figsize=(14, 8))

        matrix_to_plot = retention_matrix.iloc[:, :12] if retention_matrix.shape[1] > 12 else retention_matrix
        
        sns.heatmap(
            matrix_to_plot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=50,
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Утримання (%)'},
            linewidths=0.5
        )
        
        plt.title('Когортний аналіз утримання користувачів', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Період (місяці від початку)', fontsize=12)
        plt.ylabel('Когорта', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"The schedule is saved: {save_path}")
        plt.close()
    
    def plot_retention_curves(self, retention_matrix: pd.DataFrame,
                             save_name: str = 'retention_curves.png'):
        logger.info("Creating retention curves")
        
        plt.figure(figsize=(14, 7))

        cohorts_to_plot = min(8, len(retention_matrix))
        
        for idx in range(cohorts_to_plot):
            cohort = retention_matrix.index[idx]
            periods = range(len(retention_matrix.columns))
            values = retention_matrix.iloc[idx].values
            
            plt.plot(periods, values, marker='o', linewidth=2, 
                    label=str(cohort), alpha=0.7)
        
        plt.xlabel('Період (місяці від початку)', fontsize=12)
        plt.ylabel('Утримання (%)', fontsize=12)
        plt.title('Криві утримання по когортах', fontsize=16, fontweight='bold')
        plt.legend(title='Когорта', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"The schedule is saved: {save_path}")
        plt.close()
    
    def plot_rfm_segments(self, rfm_segmented: pd.DataFrame,
                         save_name: str = 'rfm_segments.png'):
        logger.info("Creating RFM segments visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        segment_counts = rfm_segmented['segment'].value_counts()
        axes[0, 0].barh(segment_counts.index, segment_counts.values, color='skyblue')
        axes[0, 0].set_xlabel('Кількість користувачів', fontsize=11)
        axes[0, 0].set_title('Розподіл користувачів по сегментах', 
                            fontsize=13, fontweight='bold')
        
        for i, v in enumerate(segment_counts.values):
            axes[0, 0].text(v, i, f' {v}', va='center', fontweight='bold')

        segment_means = rfm_segmented.groupby('segment')[
            ['recency', 'frequency', 'monetary']
        ].mean()
        
        segment_means.plot(kind='bar', ax=axes[0, 1], rot=45)
        axes[0, 1].set_title('Середні RFM метрики по сегментах', 
                            fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Сегмент', fontsize=11)
        axes[0, 1].set_ylabel('Значення', fontsize=11)
        axes[0, 1].legend(title='Метрика', loc='best')
        axes[0, 1].grid(True, alpha=0.3)

        segments_unique = rfm_segmented['segment'].unique()
        colors = plt.cm.tab10(range(len(segments_unique)))
        
        for segment, color in zip(segments_unique, colors):
            segment_data = rfm_segmented[rfm_segmented['segment'] == segment]
            axes[1, 0].scatter(
                segment_data['recency'], 
                segment_data['frequency'],
                label=segment,
                alpha=0.6,
                s=50,
                color=color
            )
        
        axes[1, 0].set_xlabel('Recency (днів)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency (подій)', fontsize=11)
        axes[1, 0].set_title('Recency vs Frequency по сегментах', 
                            fontsize=13, fontweight='bold')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        for segment, color in zip(segments_unique, colors):
            segment_data = rfm_segmented[rfm_segmented['segment'] == segment]
            axes[1, 1].scatter(
                segment_data['frequency'], 
                segment_data['monetary'],
                label=segment,
                alpha=0.6,
                s=50,
                color=color
            )
        
        axes[1, 1].set_xlabel('Frequency (подій)', fontsize=11)
        axes[1, 1].set_ylabel('Monetary (цінність)', fontsize=11)
        axes[1, 1].set_title('Frequency vs Monetary по сегментах', 
                            fontsize=13, fontweight='bold')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"The schedule is saved: {save_path}")
        plt.close()
    
    def plot_clusters_3d(self, clustered_df: pd.DataFrame,
                        save_name: str = 'clusters_3d.html',
                        max_points: int = 200000):
        logger.info("Creating 3D clusters visualization")
        
        if len(clustered_df) > max_points:
            logger.info(f"Too many points ({len(clustered_df):,}). Sampling {max_points:,} points (stratified by cluster)...")

            sampled_dfs = []
            for cluster_id in clustered_df['cluster'].unique():
                cluster_df = clustered_df[clustered_df['cluster'] == cluster_id]
                cluster_size = len(cluster_df)

                n_samples = max(1, int(max_points * cluster_size / len(clustered_df)))
                
                if cluster_size <= n_samples:
                    sampled_dfs.append(cluster_df)
                else:
                    sampled_dfs.append(cluster_df.sample(n=n_samples, random_state=42))
            
            plot_df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"Sampled {len(plot_df):,} points for visualization")
        else:
            plot_df = clustered_df
        
        fig = px.scatter_3d(
            plot_df,
            x='recency',
            y='frequency',
            z='monetary',
            color='cluster',
            title=f'3D Візуалізація кластерів користувачів (показано {len(plot_df):,} з {len(clustered_df):,})',
            labels={
                'recency': 'Recency (днів)',
                'frequency': 'Frequency (подій)',
                'monetary': 'Monetary (цінність)',
                'cluster': 'Кластер'
            },
            hover_data=['user_id'],
            color_continuous_scale='Viridis' if plot_df['cluster'].dtype != 'object' else None
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))

        if len(clustered_df) > max_points:
            fig.add_annotation(
                text=f"Репрезентативна вибірка: {len(plot_df):,} з {len(clustered_df):,} точок",
                xref="paper", yref="paper",
                x=0.5, y=1.05, 
                showarrow=False,
                font=dict(size=10, color="gray")
            )
        
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path))
        logger.info(f"The interactive schedule has been saved: {save_path}")
        
        return fig
    
    def plot_anomaly_timeline(self, anomalies_df: pd.DataFrame,
                             save_name: str = 'anomaly_timeline.png'):
        logger.info("Creating a timeline of anomalies")
        
        plt.figure(figsize=(16, 6))

        plt.plot(anomalies_df.index, anomalies_df['count'], 
                label='Фактичні значення', linewidth=2, color='blue', alpha=0.7)

        plt.plot(anomalies_df.index, anomalies_df['rolling_mean'], 
                label='Ковзне середнє', linewidth=2, color='green', linestyle='--')

        plt.fill_between(
            anomalies_df.index,
            anomalies_df['lower_bound'],
            anomalies_df['upper_bound'],
            alpha=0.2,
            color='green',
            label='Нормальний діапазон'
        )

        anomaly_points = anomalies_df[anomalies_df['is_anomaly']]
        plt.scatter(
            anomaly_points.index,
            anomaly_points['count'],
            color='red',
            s=100,
            marker='X',
            label='Аномалії',
            zorder=5
        )
        
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Кількість подій', fontsize=12)
        plt.title('Виявлення аномалій у часовому ряді', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"The schedule is saved: {save_path}")
        plt.close()
    
    def create_summary_dashboard(self, metrics: dict,
                                save_name: str = 'summary_dashboard.png'):
        logger.info("Creating summary dashboard")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        metric_color = '#3498db'

        def add_metric(ax, title, value, subtitle=''):
            ax.text(0.5, 0.6, str(value), 
                   ha='center', va='center', 
                   fontsize=36, fontweight='bold', color=metric_color)
            ax.text(0.5, 0.3, title, 
                   ha='center', va='center', 
                   fontsize=14, fontweight='bold')
            if subtitle:
                ax.text(0.5, 0.15, subtitle, 
                       ha='center', va='center', 
                       fontsize=10, color='gray')
            ax.axis('off')

        ax1 = fig.add_subplot(gs[0, 0])
        add_metric(ax1, 'Всього користувачів', 
                  metrics.get('total_users', 'N/A'))
        
        ax2 = fig.add_subplot(gs[0, 1])
        add_metric(ax2, 'Всього подій', 
                  metrics.get('total_events', 'N/A'))
        
        ax3 = fig.add_subplot(gs[0, 2])
        add_metric(ax3, 'Конверсія', 
                  f"{metrics.get('conversion_rate', 0):.1f}%",
                  'від перегляду до покупки')
        
        ax4 = fig.add_subplot(gs[1, 0])
        add_metric(ax4, 'Середня сесія', 
                  f"{metrics.get('avg_session_length', 0):.1f} хв")
        
        ax5 = fig.add_subplot(gs[1, 1])
        add_metric(ax5, 'Утримання (30 днів)', 
                  f"{metrics.get('retention_30d', 0):.1f}%")
        
        ax6 = fig.add_subplot(gs[1, 2])
        add_metric(ax6, 'Виявлено аномалій', 
                  metrics.get('anomalies_detected', 'N/A'))

        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        insights = metrics.get('key_insights', [])
        if insights:
            insights_text = "КЛЮЧОВІ ВИСНОВКИ:\n\n" + "\n".join(
                f"• {insight}" for insight in insights
            )
            ax7.text(0.05, 0.95, insights_text, 
                    ha='left', va='top', 
                    fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                    transform=ax7.transAxes)
        
        plt.suptitle('Загальний дашборд аналізу активності користувачів', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"The dashboard is saved: {save_path}")
        plt.close()