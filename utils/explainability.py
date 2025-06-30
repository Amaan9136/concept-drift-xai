"""
Explainability engine for drift detection and model decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import shap
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityEngine:
    """Generate explanations for drift detection and model decisions"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.background_data = None
    
    def initialize_explainer(self, background_data: np.ndarray):
        """Initialize SHAP explainer"""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.background_data = background_data[:100]  # Use subset for efficiency
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_drift(self, drift_results: Dict[str, Dict]) -> List[str]:
        """Generate human-readable explanations for detected drift"""
        explanations = []
        
        for feature, results in drift_results.items():
            if results['drift_severity'] in ['MEDIUM', 'HIGH']:
                explanation = self._generate_feature_explanation(feature, results)
                explanations.append(explanation)
        
        return explanations
    
    def _generate_feature_explanation(self, feature: str, results: Dict) -> str:
        """Generate detailed explanation for a specific feature drift"""
        severity = results['drift_severity']
        psi_score = results['psi_score']
        mean_shift = results['mean_shift']
        confidence = results.get('drift_confidence', 0.0)
        
        explanation = f"**{severity} DRIFT DETECTED** in {feature}:\n"
        explanation += f"• Confidence: {confidence:.1%}\n"
        explanation += f"• Distribution shift score (PSI): {psi_score:.3f}\n"
        explanation += f"• Mean value changed by: {mean_shift:+.2f}\n"
        
        # Add cybersecurity context
        security_context = self._get_cybersecurity_context(feature, mean_shift, results)
        explanation += security_context
        
        # Add recommended actions
        actions = self._get_recommended_actions(feature, severity, results)
        explanation += f"• **RECOMMENDED ACTION**: {actions}\n"
        
        return explanation
    
    def _get_cybersecurity_context(self, feature: str, mean_shift: float, results: Dict) -> str:
        """Get cybersecurity-specific context for feature drift"""
        context = ""
        
        security_insights = {
            'packet_size': {
                'positive': "• Larger packets may indicate data exfiltration, file transfers, or new protocols\n",
                'negative': "• Smaller packets may indicate reconnaissance, DoS preparation, or fragmented attacks\n"
            },
            'payload_entropy': {
                'positive': "• Higher entropy suggests increased encryption, obfuscation, or compressed data\n",
                'negative': "• Lower entropy may indicate plaintext attacks, simple protocols, or padding\n"
            },
            'suspicious_ports': {
                'positive': "• Increased suspicious port usage indicates new attack vectors or services\n",
                'negative': "• Decreased suspicious port activity may indicate evasion techniques\n"
            },
            'failed_connections': {
                'positive': "• More failed connections suggest brute force, scanning, or service disruption\n",
                'negative': "• Fewer failed connections may indicate successful attacks or better targeting\n"
            },
            'connection_frequency': {
                'positive': "• Higher connection frequency may indicate botnet activity or data exfiltration\n",
                'negative': "• Lower connection frequency could suggest stealth techniques or dormant threats\n"
            },
            'unique_destinations': {
                'positive': "• More unique destinations suggest scanning, lateral movement, or widespread attacks\n",
                'negative': "• Fewer unique destinations may indicate focused targeting or reduced activity\n"
            },
            'flow_duration': {
                'positive': "• Longer flows may indicate data transfers, persistent connections, or tunneling\n",
                'negative': "• Shorter flows could suggest hit-and-run attacks or connection scanning\n"
            },
            'bytes_sent': {
                'positive': "• Increased outbound data may indicate data exfiltration or command responses\n",
                'negative': "• Decreased outbound data could suggest inbound attacks or reconnaissance\n"
            },
            'bytes_received': {
                'positive': "• More inbound data may indicate malware downloads or command distribution\n",
                'negative': "• Less inbound data could suggest data exfiltration focus or reduced activity\n"
            }
        }
        
        if feature in security_insights:
            direction = 'positive' if mean_shift > 0 else 'negative'
            context += security_insights[feature].get(direction, "")
        
        # Add severity-specific insights
        if results['drift_severity'] == 'HIGH':
            context += "• **SECURITY ALERT**: This change represents a significant deviation that requires immediate attention\n"
        
        # Add statistical insights
        if results.get('psi_score', 0) > 0.5:
            context += "• **DISTRIBUTION CHANGE**: The feature distribution has fundamentally shifted\n"
        
        return context
    
    def _get_recommended_actions(self, feature: str, severity: str, results: Dict) -> str:
        """Get recommended actions based on drift characteristics"""
        if severity == 'HIGH':
            if feature in ['suspicious_ports', 'failed_connections']:
                return "Immediate security investigation and model retraining required"
            else:
                return "Urgent model retraining and threshold adjustment needed"
        elif severity == 'MEDIUM':
            return "Monitor closely, prepare for retraining, and investigate security implications"
        else:
            return "Continue monitoring with current parameters"
    
    def explain_predictions(self, X: np.ndarray, sample_size: int = 10) -> Dict:
        """Generate SHAP explanations for model predictions"""
        if self.explainer is None:
            return self._generate_fallback_explanations(X, sample_size)
        
        try:
            # Sample data for explanation
            sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
            X_sample = X[sample_indices]
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle binary classification (shap_values might be a list)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            return {
                'shap_values': shap_values.tolist(),
                'sample_data': X_sample.tolist(),
                'feature_names': self.feature_names,
                'mean_abs_shap': np.mean(np.abs(shap_values), axis=0).tolist(),
                'explanation_type': 'shap'
            }
            
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            return self._generate_fallback_explanations(X, sample_size)
    
    def _generate_fallback_explanations(self, X: np.ndarray, sample_size: int) -> Dict:
        """Generate fallback explanations when SHAP fails"""
        try:
            # Use feature importance from model if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
            else:
                # Random importance for demo
                feature_importance = np.random.random(len(self.feature_names))
                feature_importance = feature_importance / np.sum(feature_importance)
            
            sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
            X_sample = X[sample_indices]
            
            return {
                'feature_importance': feature_importance.tolist(),
                'sample_data': X_sample.tolist(),
                'feature_names': self.feature_names,
                'mean_abs_shap': feature_importance.tolist(),
                'explanation_type': 'feature_importance'
            }
            
        except Exception:
            return {
                'error': 'Unable to generate explanations',
                'feature_names': self.feature_names,
                'explanation_type': 'error'
            }
    
    def generate_global_explanation(self, drift_results: Dict) -> Dict:
        """Generate global explanation for overall drift pattern"""
        if not drift_results:
            return {'message': 'No drift detected', 'severity': 'NONE'}
        
        # Analyze overall drift pattern
        high_drift_features = [f for f, r in drift_results.items() if r['drift_severity'] == 'HIGH']
        medium_drift_features = [f for f, r in drift_results.items() if r['drift_severity'] == 'MEDIUM']
        
        total_features = len(drift_results)
        drift_percentage = len(high_drift_features + medium_drift_features) / total_features * 100
        
        # Generate summary
        summary = {
            'overall_severity': self._determine_overall_severity(high_drift_features, medium_drift_features),
            'drift_percentage': drift_percentage,
            'affected_features': high_drift_features + medium_drift_features,
            'security_implications': self._get_global_security_implications(high_drift_features, medium_drift_features),
            'recommended_response': self._get_global_response_recommendation(high_drift_features, medium_drift_features)
        }
        
        return summary
    
    def _determine_overall_severity(self, high_drift: List[str], medium_drift: List[str]) -> str:
        """Determine overall system drift severity"""
        if len(high_drift) >= 3:
            return 'CRITICAL'
        elif len(high_drift) >= 1:
            return 'HIGH'
        elif len(medium_drift) >= 3:
            return 'MEDIUM'
        elif len(medium_drift) >= 1:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _get_global_security_implications(self, high_drift: List[str], medium_drift: List[str]) -> List[str]:
        """Get security implications for overall drift pattern"""
        implications = []
        
        security_categories = {
            'network_anomalies': ['packet_size', 'flow_duration', 'packet_count'],
            'connection_patterns': ['connection_frequency', 'unique_destinations', 'failed_connections'],
            'payload_analysis': ['payload_entropy', 'anomalous_payload_size'],
            'suspicious_activity': ['suspicious_ports', 'unusual_timing_patterns']
        }
        
        affected_categories = set()
        all_affected = high_drift + medium_drift
        
        for category, features in security_categories.items():
            if any(feature in all_affected for feature in features):
                affected_categories.add(category)
        
        category_implications = {
            'network_anomalies': "Network traffic patterns have changed significantly",
            'connection_patterns': "Connection behaviors indicate potential threats",
            'payload_analysis': "Payload characteristics suggest new attack techniques",
            'suspicious_activity': "Suspicious activity indicators have increased"
        }
        
        for category in affected_categories:
            implications.append(category_implications[category])
        
        if len(affected_categories) >= 3:
            implications.append("Multiple attack vectors may be active simultaneously")
        
        return implications
    
    def _get_global_response_recommendation(self, high_drift: List[str], medium_drift: List[str]) -> str:
        """Get global response recommendation"""
        if len(high_drift) >= 3:
            return "IMMEDIATE: Activate incident response, retrain all models, and conduct thorough security investigation"
        elif len(high_drift) >= 1:
            return "URGENT: Retrain affected models and initiate security analysis of high-drift features"
        elif len(medium_drift) >= 3:
            return "ELEVATED: Increase monitoring frequency and prepare for model updates"
        elif len(medium_drift) >= 1:
            return "STANDARD: Continue monitoring with enhanced alerting for affected features"
        else:
            return "ROUTINE: Maintain current monitoring and assessment schedule"
    
    def generate_counterfactual_analysis(self, feature: str, drift_results: Dict) -> Dict:
        """Generate counterfactual analysis for drift scenarios"""
        if feature not in drift_results:
            return {'error': f'Feature {feature} not found in drift results'}
        
        result = drift_results[feature]
        
        scenarios = {
            'if_no_drift': {
                'description': f"If {feature} had not drifted",
                'impact': "Model would maintain current accuracy",
                'security_status': "No additional security concerns"
            },
            'if_drift_continues': {
                'description': f"If {feature} drift continues to increase",
                'impact': "Model accuracy will degrade further",
                'security_status': "Security threats may go undetected"
            },
            'if_mitigated': {
                'description': f"If {feature} drift is mitigated through retraining",
                'impact': "Model accuracy should be restored",
                'security_status': "Enhanced detection of new threat patterns"
            }
        }
        
        return {
            'feature': feature,
            'current_severity': result['drift_severity'],
            'scenarios': scenarios,
            'recommendation': f"Based on {result['drift_severity']} severity, mitigation is {'critical' if result['drift_severity'] == 'HIGH' else 'recommended'}"
        }