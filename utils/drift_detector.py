"""
Drift detection module for cybersecurity data
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
from dataclasses import dataclass
import datetime

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime.datetime

@dataclass
class DriftAlert:
    """Structure for drift detection alerts"""
    timestamp: datetime.datetime
    feature_name: str
    drift_type: str
    severity: str
    confidence: float
    explanation: str
    recommended_action: str

class DriftDetector:
    """Core drift detection engine using statistical tests and model performance"""
    
    def __init__(self, config):
        self.config = config
        self.baseline_stats = {}
        self.performance_history = []
        self.drift_alerts = []
    
    def set_baseline(self, baseline_data: pd.DataFrame):
        """Establish baseline statistics for drift detection"""
        self.baseline_stats = {}
        features = [col for col in baseline_data.columns if col != 'is_malicious']
        
        for feature in features:
            self.baseline_stats[feature] = {
                'mean': baseline_data[feature].mean(),
                'std': baseline_data[feature].std(),
                'quantiles': baseline_data[feature].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                'distribution': baseline_data[feature].values,
                'min': baseline_data[feature].min(),
                'max': baseline_data[feature].max()
            }
    
    def calculate_psi(self, baseline_dist: np.ndarray, current_dist: np.ndarray, bins=10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on baseline distribution
            bin_edges = np.histogram_bin_edges(baseline_dist, bins=bins)
            
            # Calculate frequencies
            baseline_freq = np.histogram(baseline_dist, bins=bin_edges)[0]
            current_freq = np.histogram(current_dist, bins=bin_edges)[0]
            
            # Convert to proportions
            baseline_prop = baseline_freq / len(baseline_dist)
            current_prop = current_freq / len(current_dist)
            
            # Avoid division by zero
            baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
            current_prop = np.where(current_prop == 0, 0.0001, current_prop)
            
            # Calculate PSI
            psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
            return abs(psi)
            
        except Exception:
            return 0.0
    
    def calculate_wasserstein_distance(self, baseline_dist: np.ndarray, current_dist: np.ndarray) -> float:
        """Calculate Wasserstein distance between distributions"""
        try:
            return stats.wasserstein_distance(baseline_dist, current_dist)
        except Exception:
            return 0.0
    
    def detect_statistical_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """Detect drift using statistical tests"""
        drift_results = {}
        features = [col for col in current_data.columns if col != 'is_malicious']
        
        for feature in features:
            if feature in self.baseline_stats:
                baseline_dist = self.baseline_stats[feature]['distribution']
                current_dist = current_data[feature].values
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p_value = stats.ks_2samp(baseline_dist, current_dist)
                
                # Population Stability Index
                psi_score = self.calculate_psi(baseline_dist, current_dist)
                
                # Wasserstein distance
                wasserstein_dist = self.calculate_wasserstein_distance(baseline_dist, current_dist)
                
                # Statistical moments comparison
                baseline_mean = self.baseline_stats[feature]['mean']
                baseline_std = self.baseline_stats[feature]['std']
                current_mean = current_data[feature].mean()
                current_std = current_data[feature].std()
                
                mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
                std_shift = abs(current_std - baseline_std) / (baseline_std + 1e-8)
                
                # Determine drift severity
                drift_severity = 'LOW'
                drift_confidence = 0.0
                
                # Multiple criteria for drift detection
                criteria_met = 0
                if ks_p_value < self.config.KS_THRESHOLD:
                    criteria_met += 1
                if psi_score > self.config.PSI_THRESHOLD:
                    criteria_met += 1
                if mean_shift > 0.5:
                    criteria_met += 1
                if wasserstein_dist > baseline_std:
                    criteria_met += 1
                
                if criteria_met >= 3:
                    drift_severity = 'HIGH'
                    drift_confidence = 0.9
                elif criteria_met >= 2:
                    drift_severity = 'MEDIUM'
                    drift_confidence = 0.7
                elif criteria_met >= 1:
                    drift_severity = 'LOW'
                    drift_confidence = 0.5
                
                drift_results[feature] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'psi_score': psi_score,
                    'wasserstein_distance': wasserstein_dist,
                    'mean_shift': current_mean - baseline_mean,
                    'std_shift': current_std - baseline_std,
                    'mean_shift_normalized': mean_shift,
                    'std_shift_normalized': std_shift,
                    'drift_severity': drift_severity,
                    'drift_confidence': drift_confidence,
                    'criteria_met': criteria_met
                }
        
        return drift_results
    
    def detect_performance_drift(self, model_performance: ModelPerformance) -> bool:
        """Detect drift based on model performance degradation"""
        if len(self.performance_history) == 0:
            self.performance_history.append(model_performance)
            return False
        
        # Calculate performance degradation
        baseline_accuracy = self.performance_history[0].accuracy
        current_accuracy = model_performance.accuracy
        
        performance_drop = baseline_accuracy - current_accuracy
        
        self.performance_history.append(model_performance)
        
        return performance_drop > self.config.PERFORMANCE_THRESHOLD
    
    def generate_drift_summary(self, drift_results: Dict[str, Dict]) -> Dict:
        """Generate summary statistics for drift detection results"""
        summary = {
            'total_features': len(drift_results),
            'high_drift_count': 0,
            'medium_drift_count': 0,
            'low_drift_count': 0,
            'avg_psi_score': 0,
            'max_psi_score': 0,
            'most_drifted_feature': '',
            'drift_severity_distribution': {},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if not drift_results:
            return summary
        
        psi_scores = []
        severities = []
        
        for feature, result in drift_results.items():
            severity = result['drift_severity']
            psi_score = result['psi_score']
            
            if severity == 'HIGH':
                summary['high_drift_count'] += 1
            elif severity == 'MEDIUM':
                summary['medium_drift_count'] += 1
            else:
                summary['low_drift_count'] += 1
            
            psi_scores.append(psi_score)
            severities.append(severity)
        
        summary['avg_psi_score'] = np.mean(psi_scores)
        summary['max_psi_score'] = np.max(psi_scores)
        
        # Find most drifted feature
        max_psi_feature = max(drift_results.keys(), key=lambda x: drift_results[x]['psi_score'])
        summary['most_drifted_feature'] = max_psi_feature
        
        # Severity distribution
        unique_severities, counts = np.unique(severities, return_counts=True)
        summary['drift_severity_distribution'] = dict(zip(unique_severities, counts.tolist()))
        
        return summary
    
    def create_drift_alert(self, feature: str, drift_result: Dict) -> DriftAlert:
        """Create a drift alert for a feature"""
        severity = drift_result['drift_severity']
        confidence = drift_result['drift_confidence']
        
        # Generate explanation based on drift characteristics
        explanation = f"Feature '{feature}' shows {severity.lower()} drift. "
        
        if drift_result['psi_score'] > 0.5:
            explanation += "Distribution has significantly changed. "
        
        if abs(drift_result['mean_shift_normalized']) > 1:
            shift_direction = "increased" if drift_result['mean_shift'] > 0 else "decreased"
            explanation += f"Mean value has {shift_direction} substantially. "
        
        # Security-specific explanations
        security_implications = self._get_security_implications(feature, drift_result)
        if security_implications:
            explanation += security_implications
        
        # Recommended actions
        if severity == 'HIGH':
            action = "Immediate investigation and model retraining required"
        elif severity == 'MEDIUM':
            action = "Monitor closely and prepare for potential retraining"
        else:
            action = "Continue monitoring with current parameters"
        
        return DriftAlert(
            timestamp=datetime.datetime.now(),
            feature_name=feature,
            drift_type='statistical',
            severity=severity,
            confidence=confidence,
            explanation=explanation,
            recommended_action=action
        )
    
    def _get_security_implications(self, feature: str, drift_result: Dict) -> str:
        """Get security-specific implications for drifted features"""
        implications = {
            'packet_size': "May indicate new attack vectors or protocol changes.",
            'payload_entropy': "Could suggest increased encryption or obfuscation techniques.",
            'suspicious_ports': "Indicates potential new attack services or port scanning.",
            'failed_connections': "May signal brute force attacks or reconnaissance.",
            'connection_frequency': "Could indicate botnet activity or automated attacks.",
            'unique_destinations': "May suggest scanning behavior or lateral movement.",
            'flow_duration': "Could indicate new attack patterns or service changes.",
            'unusual_timing_patterns': "May signal coordinated attacks or automation."
        }
        
        return implications.get(feature, "Requires security analysis to determine impact.")
    
    def get_drift_trends(self, window_size: int = 10) -> Dict:
        """Analyze drift trends over time"""
        if len(self.drift_alerts) < window_size:
            return {'insufficient_data': True}
        
        recent_alerts = self.drift_alerts[-window_size:]
        
        # Analyze trends
        feature_frequency = {}
        severity_trends = []
        
        for alert in recent_alerts:
            feature = alert.feature_name
            feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
            severity_trends.append(alert.severity)
        
        # Most frequently drifting features
        top_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Severity trend analysis
        severity_counts = {s: severity_trends.count(s) for s in set(severity_trends)}
        
        return {
            'window_size': window_size,
            'total_alerts': len(recent_alerts),
            'top_drifting_features': top_features,
            'severity_distribution': severity_counts,
            'trend_direction': self._calculate_trend_direction(severity_trends)
        }
    
    def _calculate_trend_direction(self, severity_trends: List[str]) -> str:
        """Calculate if drift severity is trending up or down"""
        severity_values = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        values = [severity_values[s] for s in severity_trends]
        
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'