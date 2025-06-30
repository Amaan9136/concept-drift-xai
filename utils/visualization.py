"""
Visualization utilities for drift detection and cybersecurity analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any
import datetime

class DriftVisualization:
    """Create visualizations for drift detection and explanations"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff9900',
            'danger': '#d62728',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.severity_colors = {
            'LOW': self.color_palette['success'],
            'MEDIUM': self.color_palette['warning'],
            'HIGH': self.color_palette['danger']
        }
    
    def create_drift_timeline(self, drift_history: List[Dict]) -> Dict:
        """Create timeline visualization of drift events"""
        if not drift_history:
            return self._empty_chart_data("No drift history available")
        
        timestamps = []
        feature_counts = []
        max_psi_scores = []
        severities = []
        
        for record in drift_history:
            timestamps.append(record['timestamp'])
            
            # Count features with drift
            drift_results = record.get('drift_results', {})
            feature_count = len([
                f for f, r in drift_results.items()
                if r.get('drift_severity', 'LOW') != 'LOW'
            ])
            feature_counts.append(feature_count)
            
            # Get max PSI score
            psi_scores = [r.get('psi_score', 0) for r in drift_results.values()]
            max_psi_scores.append(max(psi_scores) if psi_scores else 0)
            
            # Determine overall severity
            high_count = len([f for f, r in drift_results.items() if r.get('drift_severity') == 'HIGH'])
            medium_count = len([f for f, r in drift_results.items() if r.get('drift_severity') == 'MEDIUM'])
            
            if high_count > 0:
                severities.append('HIGH')
            elif medium_count > 0:
                severities.append('MEDIUM')
            else:
                severities.append('LOW')
        
        # Create the plot data
        fig_data = {
            'data': [
                {
                    'x': timestamps,
                    'y': feature_counts,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Features Affected',
                    'line': {'color': self.color_palette['primary'], 'width': 3},
                    'marker': {'size': 8}
                },
                {
                    'x': timestamps,
                    'y': max_psi_scores,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Max PSI Score',
                    'yaxis': 'y2',
                    'line': {'color': self.color_palette['secondary'], 'width': 2},
                    'marker': {'size': 6}
                }
            ],
            'layout': {
                'title': 'Drift Detection Timeline',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Features Affected', 'side': 'left'},
                'yaxis2': {
                    'title': 'PSI Score',
                    'side': 'right',
                    'overlaying': 'y'
                },
                'hovermode': 'x unified',
                'height': 400
            }
        }
        
        return fig_data
    
    def create_feature_distribution_comparison(self, baseline_data: pd.DataFrame,
                                             current_data: pd.DataFrame, 
                                             feature: str) -> Dict:
        """Create feature distribution comparison chart"""
        if feature not in baseline_data.columns or feature not in current_data.columns:
            return self._empty_chart_data(f"Feature {feature} not found in data")
        
        fig_data = {
            'data': [
                {
                    'x': baseline_data[feature].values.tolist(),
                    'type': 'histogram',
                    'name': 'Baseline',
                    'opacity': 0.7,
                    'marker': {'color': self.color_palette['primary']},
                    'nbinsx': 30
                },
                {
                    'x': current_data[feature].values.tolist(),
                    'type': 'histogram',
                    'name': 'Current',
                    'opacity': 0.7,
                    'marker': {'color': self.color_palette['secondary']},
                    'nbinsx': 30
                }
            ],
            'layout': {
                'title': f'Distribution Comparison: {feature}',
                'xaxis': {'title': feature},
                'yaxis': {'title': 'Frequency'},
                'barmode': 'overlay',
                'height': 400
            }
        }
        
        return fig_data
    
    def create_drift_heatmap(self, drift_results: Dict[str, Dict]) -> Dict:
        """Create heatmap of drift severity across features"""
        if not drift_results:
            return self._empty_chart_data("No drift results available")
        
        features = list(drift_results.keys())
        psi_scores = [drift_results[f].get('psi_score', 0) for f in features]
        severities = [drift_results[f].get('drift_severity', 'LOW') for f in features]
        
        # Convert severities to numeric values for heatmap
        severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        severity_values = [severity_map[s] for s in severities]
        
        fig_data = {
            'data': [
                {
                    'z': [severity_values],
                    'x': features,
                    'y': ['Drift Severity'],
                    'type': 'heatmap',
                    'colorscale': [
                        [0, self.severity_colors['LOW']],
                        [0.5, self.severity_colors['MEDIUM']],
                        [1, self.severity_colors['HIGH']]
                    ],
                    'hovertemplate': 'Feature: %{x}<br>Severity: %{text}<br>PSI Score: %{customdata}<extra></extra>',
                    'text': [severities],
                    'customdata': [psi_scores]
                }
            ],
            'layout': {
                'title': 'Feature Drift Severity Heatmap',
                'xaxis': {'title': 'Features', 'tickangle': 45},
                'yaxis': {'title': ''},
                'height': 300
            }
        }
        
        return fig_data
    
    def create_feature_importance_chart(self, feature_importance: Dict) -> Dict:
        """Create feature importance visualization"""
        if not feature_importance:
            return self._empty_chart_data("No feature importance data available")
        
        features = list(feature_importance['feature_names'])
        importance_values = feature_importance['mean_abs_shap']
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance_values), key=lambda x: x[1], reverse=True)
        features_sorted, importance_sorted = zip(*sorted_data)
        
        fig_data = {
            'data': [
                {
                    'x': list(importance_sorted),
                    'y': list(features_sorted),
                    'type': 'bar',
                    'orientation': 'h',
                    'marker': {'color': self.color_palette['info']},
                    'name': 'Feature Importance'
                }
            ],
            'layout': {
                'title': 'Feature Importance (SHAP Values)',
                'xaxis': {'title': 'Mean |SHAP value|'},
                'yaxis': {'title': 'Features'},
                'height': 500
            }
        }
        
        return fig_data
    
    def create_performance_trend_chart(self, performance_history: List[Dict]) -> Dict:
        """Create model performance trend chart"""
        if not performance_history:
            return self._empty_chart_data("No performance history available")
        
        timestamps = [p['timestamp'] for p in performance_history]
        accuracies = [p['accuracy'] for p in performance_history]
        
        fig_data = {
            'data': [
                {
                    'x': timestamps,
                    'y': accuracies,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Model Accuracy',
                    'line': {'color': self.color_palette['success'], 'width': 3},
                    'marker': {'size': 8}
                }
            ],
            'layout': {
                'title': 'Model Performance Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Accuracy', 'range': [0, 1]},
                'height': 400
            }
        }
        
        return fig_data
    
    def create_alerts_summary_chart(self, alerts: List[Dict]) -> Dict:
        """Create alerts summary visualization"""
        if not alerts:
            return self._empty_chart_data("No alerts available")
        
        # Count alerts by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'LOW')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = [self.severity_colors.get(s, self.color_palette['info']) for s in severities]
        
        fig_data = {
            'data': [
                {
                    'labels': severities,
                    'values': counts,
                    'type': 'pie',
                    'marker': {'colors': colors},
                    'hovertemplate': 'Severity: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                }
            ],
            'layout': {
                'title': 'Alert Distribution by Severity',
                'height': 400
            }
        }
        
        return fig_data
    
    def create_feature_status_grid(self, feature_status: Dict) -> Dict:
        """Create grid visualization of feature status"""
        if not feature_status:
            return self._empty_chart_data("No feature status data available")
        
        features = list(feature_status.keys())
        severities = [feature_status[f]['severity'] for f in features]
        psi_scores = [feature_status[f]['psi_score'] for f in features]
        
        # Create grid layout
        grid_size = int(np.ceil(np.sqrt(len(features))))
        
        # Prepare data for grid
        grid_data = []
        for i, feature in enumerate(features):
            row = i // grid_size
            col = i % grid_size
            severity = severities[i]
            psi_score = psi_scores[i]
            
            grid_data.append({
                'x': col,
                'y': row,
                'feature': feature,
                'severity': severity,
                'psi_score': psi_score,
                'color': self.severity_colors[severity]
            })
        
        fig_data = {
            'data': [
                {
                    'x': [d['x'] for d in grid_data],
                    'y': [d['y'] for d in grid_data],
                    'mode': 'markers',
                    'marker': {
                        'size': 30,
                        'color': [d['color'] for d in grid_data],
                        'line': {'width': 2, 'color': 'white'}
                    },
                    'text': [d['feature'] for d in grid_data],
                    'hovertemplate': 'Feature: %{text}<br>Severity: %{customdata[0]}<br>PSI Score: %{customdata[1]:.3f}<extra></extra>',
                    'customdata': [[d['severity'], d['psi_score']] for d in grid_data],
                    'type': 'scatter'
                }
            ],
            'layout': {
                'title': 'Feature Status Grid',
                'xaxis': {'showticklabels': False, 'showgrid': False},
                'yaxis': {'showticklabels': False, 'showgrid': False},
                'height': 400,
                'showlegend': False
            }
        }
        
        return fig_data
    
    def get_feature_drift_data(self, baseline_data: pd.DataFrame, feature: str, 
                              drift_result: Dict) -> Dict:
        """Get comprehensive drift data for a feature"""
        if feature not in baseline_data.columns:
            return {'error': f'Feature {feature} not found'}
        
        baseline_values = baseline_data[feature].values
        
        # Statistical summary
        summary = {
            'feature_name': feature,
            'baseline_stats': {
                'mean': float(np.mean(baseline_values)),
                'std': float(np.std(baseline_values)),
                'min': float(np.min(baseline_values)),
                'max': float(np.max(baseline_values)),
                'median': float(np.median(baseline_values))
            },
            'drift_metrics': drift_result,
            'distribution_data': {
                'baseline': baseline_values.tolist()
            }
        }
        
        return summary
    
    def _empty_chart_data(self, message: str) -> Dict:
        """Return empty chart data with message"""
        return {
            'data': [],
            'layout': {
                'title': message,
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [
                    {
                        'text': message,
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'middle',
                        'showarrow': False,
                        'font': {'size': 16}
                    }
                ],
                'height': 400
            }
        }
    
    def create_security_dashboard_summary(self, dashboard_data: Dict) -> Dict:
        """Create comprehensive security dashboard summary"""
        alerts = dashboard_data.get('alerts', [])
        drift_timeline = dashboard_data.get('drift_timeline', [])
        feature_status = dashboard_data.get('feature_status', {})
        
        # Create summary metrics
        total_alerts = len(alerts)
        high_severity_alerts = len([a for a in alerts if a.get('severity') == 'HIGH'])
        features_with_drift = len([f for f, s in feature_status.items() if s.get('severity') != 'LOW'])
        
        summary = {
            'metrics': {
                'total_alerts': total_alerts,
                'high_severity_alerts': high_severity_alerts,
                'features_with_drift': features_with_drift,
                'total_features_monitored': len(feature_status)
            },
            'charts': {
                'alerts_trend': self.create_alerts_summary_chart(alerts),
                'feature_status': self.create_feature_status_grid(feature_status),
                'drift_timeline': self.create_drift_timeline(drift_timeline)
            },
            'status': {
                'overall_health': 'CRITICAL' if high_severity_alerts > 3 else 
                                'WARNING' if high_severity_alerts > 0 else 'HEALTHY',
                'last_update': datetime.datetime.now().isoformat()
            }
        }
        
        return summary