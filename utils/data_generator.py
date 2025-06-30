"""
Cybersecurity data generation module
"""

import numpy as np
import pandas as pd
from typing import Dict
from config import FEATURE_NAMES

class CybersecurityDataGenerator:
    """Generate realistic cybersecurity dataset with concept drift"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.feature_names = FEATURE_NAMES
    
    def generate_baseline_data(self, n_samples=5000):
        """Generate baseline network traffic data"""
        data = {}
        
        # Normal traffic patterns
        data['packet_size'] = np.random.normal(500, 100, n_samples)
        data['flow_duration'] = np.random.exponential(2, n_samples)
        data['bytes_sent'] = np.random.lognormal(8, 1, n_samples)
        data['bytes_received'] = np.random.lognormal(8, 1, n_samples)
        data['packet_count'] = np.random.poisson(10, n_samples)
        data['protocol_type'] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        data['port_number'] = np.random.choice([80, 443, 22, 25, 53], n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1])
        data['tcp_flags'] = np.random.randint(0, 32, n_samples)
        data['payload_entropy'] = np.random.beta(2, 5, n_samples)
        data['connection_frequency'] = np.random.gamma(2, 2, n_samples)
        data['time_between_packets'] = np.random.exponential(0.1, n_samples)
        data['unique_destinations'] = np.random.poisson(3, n_samples)
        data['failed_connections'] = np.random.poisson(0.5, n_samples)
        data['suspicious_ports'] = np.random.binomial(1, 0.05, n_samples)
        data['anomalous_payload_size'] = np.random.binomial(1, 0.03, n_samples)
        data['unusual_timing_patterns'] = np.random.binomial(1, 0.02, n_samples)
        
        # Generate labels (0 = benign, 1 = malicious)
        threat_probability = (
            0.1 * (data['suspicious_ports'] + data['anomalous_payload_size'] + 
                   data['unusual_timing_patterns']) +
            0.05 * (data['payload_entropy'] > 0.8) +
            0.03 * (data['failed_connections'] > 2)
        )
        data['is_malicious'] = np.random.binomial(1, np.clip(threat_probability, 0, 0.3), n_samples)
        
        return pd.DataFrame(data)
    
    def generate_drifted_data(self, n_samples=1000, drift_type='gradual'):
        """Generate data with concept drift"""
        data = {}
        
        if drift_type == 'gradual':
            # Gradual shift in attack patterns
            data['packet_size'] = np.random.normal(600, 120, n_samples)  # Larger packets
            data['flow_duration'] = np.random.exponential(3, n_samples)  # Longer flows
            data['payload_entropy'] = np.random.beta(3, 4, n_samples)    # Higher entropy
            
        elif drift_type == 'sudden':
            # Sudden emergence of new attack type
            data['packet_size'] = np.random.normal(200, 50, n_samples)   # Smaller packets
            data['suspicious_ports'] = np.random.binomial(1, 0.2, n_samples)  # More suspicious ports
            data['unusual_timing_patterns'] = np.random.binomial(1, 0.15, n_samples)  # More timing anomalies
            
        elif drift_type == 'ddos':
            # DDoS attack pattern
            data['packet_count'] = np.random.poisson(50, n_samples)      # High packet count
            data['packet_size'] = np.random.normal(64, 10, n_samples)    # Small packets
            data['flow_duration'] = np.random.exponential(0.1, n_samples) # Short flows
            data['failed_connections'] = np.random.poisson(10, n_samples) # Many failures
            
        elif drift_type == 'exfiltration':
            # Data exfiltration pattern
            data['bytes_sent'] = np.random.lognormal(10, 1, n_samples)   # Large outbound data
            data['payload_entropy'] = np.random.beta(4, 2, n_samples)    # High entropy
            data['connection_frequency'] = np.random.gamma(5, 3, n_samples) # Frequent connections
        
        # Fill in other features with slight variations
        for feature in self.feature_names:
            if feature not in data:
                if feature == 'bytes_sent':
                    data[feature] = np.random.lognormal(8.2, 1.1, n_samples)
                elif feature == 'bytes_received':
                    data[feature] = np.random.lognormal(8.2, 1.1, n_samples)
                elif feature == 'packet_count':
                    data[feature] = np.random.poisson(12, n_samples)
                elif feature == 'protocol_type':
                    data[feature] = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])
                elif feature == 'port_number':
                    data[feature] = np.random.choice([80, 443, 22, 25, 53], n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1])
                elif feature == 'tcp_flags':
                    data[feature] = np.random.randint(0, 32, n_samples)
                elif feature == 'connection_frequency':
                    data[feature] = np.random.gamma(2.5, 2.2, n_samples)
                elif feature == 'time_between_packets':
                    data[feature] = np.random.exponential(0.12, n_samples)
                elif feature == 'unique_destinations':
                    data[feature] = np.random.poisson(4, n_samples)
                elif feature == 'failed_connections':
                    data[feature] = np.random.poisson(0.6, n_samples)
                else:
                    data[feature] = np.random.binomial(1, 0.06, n_samples)
        
        # Generate labels with very high threat probability for attack scenarios
        data['is_malicious'] = np.random.binomial(1, 0.9, n_samples)
        
        return pd.DataFrame(data)
    
    def generate_attack_scenario(self, attack_type: str, n_samples: int = 500) -> pd.DataFrame:
        """Generate data for specific attack scenarios"""
        scenarios = {
            'port_scan': {
                'suspicious_ports': lambda x: np.random.binomial(1, 0.8, x),
                'unique_destinations': lambda x: np.random.poisson(50, x),
                'failed_connections': lambda x: np.random.poisson(20, x),
                'packet_size': lambda x: np.random.normal(60, 10, x)
            },
            'brute_force': {
                'failed_connections': lambda x: np.random.poisson(100, x),
                'connection_frequency': lambda x: np.random.gamma(10, 1, x),
                'unusual_timing_patterns': lambda x: np.random.binomial(1, 0.9, x)
            },
            'malware_c2': {
                'payload_entropy': lambda x: np.random.beta(5, 1, x),
                'connection_frequency': lambda x: np.random.gamma(1, 10, x),
                'suspicious_ports': lambda x: np.random.binomial(1, 0.6, x)
            }
        }
        
        if attack_type not in scenarios:
            return self.generate_drifted_data(n_samples, 'sudden')
        
        # Generate base data
        data = {}
        scenario = scenarios[attack_type]
        
        # Apply scenario-specific modifications
        for feature, generator in scenario.items():
            data[feature] = generator(n_samples)
        
        # Fill remaining features
        for feature in self.feature_names:
            if feature not in data:
                # Use slightly modified baseline patterns
                if feature == 'packet_size':
                    data[feature] = np.random.normal(520, 110, n_samples)
                elif feature == 'flow_duration':
                    data[feature] = np.random.exponential(2.2, n_samples)
                elif feature == 'bytes_sent':
                    data[feature] = np.random.lognormal(8.1, 1.05, n_samples)
                elif feature == 'bytes_received':
                    data[feature] = np.random.lognormal(8.1, 1.05, n_samples)
                elif feature == 'packet_count':
                    data[feature] = np.random.poisson(11, n_samples)
                elif feature == 'protocol_type':
                    data[feature] = np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.35, 0.1])
                elif feature == 'port_number':
                    data[feature] = np.random.choice([80, 443, 22, 25, 53], n_samples, p=[0.35, 0.35, 0.15, 0.1, 0.05])
                elif feature == 'tcp_flags':
                    data[feature] = np.random.randint(0, 32, n_samples)
                elif feature == 'payload_entropy':
                    data[feature] = np.random.beta(2.2, 4.8, n_samples)
                elif feature == 'connection_frequency':
                    data[feature] = np.random.gamma(2.2, 2.1, n_samples)
                elif feature == 'time_between_packets':
                    data[feature] = np.random.exponential(0.11, n_samples)
                elif feature == 'unique_destinations':
                    data[feature] = np.random.poisson(3.2, n_samples)
                elif feature == 'failed_connections':
                    data[feature] = np.random.poisson(0.8, n_samples)
                elif feature not in data:
                    data[feature] = np.random.binomial(1, 0.08, n_samples)
        
        # Generate labels with higher threat probability
        threat_probability = (
            0.15 * (data['suspicious_ports'] + data['anomalous_payload_size'] + 
                    data['unusual_timing_patterns']) +
            0.08 * (data['payload_entropy'] > 0.8) +
            0.05 * (data['failed_connections'] > 2)
        )
        data['is_malicious'] = np.random.binomial(1, np.clip(threat_probability, 0, 0.5), n_samples)
        
        return pd.DataFrame(data)
            