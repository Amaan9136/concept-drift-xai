"""
Configuration settings for the Cybersecurity Drift Detection System
"""

import os
from dataclasses import dataclass

@dataclass
class DriftConfig:
    """Configuration parameters for drift detection"""
    # Drift detection thresholds
    KS_THRESHOLD: float = 0.05
    PSI_THRESHOLD: float = 0.2
    PERFORMANCE_THRESHOLD: float = 0.1
    
    # Model parameters
    WINDOW_SIZE: int = 1000
    RETRAIN_THRESHOLD: float = 0.15
    
    # Explainability parameters
    SHAP_SAMPLE_SIZE: int = 100
    TOP_FEATURES: int = 10

@dataclass
class FlaskConfig:
    """Flask application configuration"""
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    DEBUG: bool = os.environ.get('FLASK_ENV') == 'development'
    HOST: str = '0.0.0.0'
    PORT: int = int(os.environ.get('PORT', 5000))

@dataclass
class AppConfig:
    """Main application configuration"""
    # Data generation settings
    BASELINE_SAMPLES: int = 5000
    DRIFT_SAMPLES: int = 1000
    RANDOM_SEED: int = 42
    
    # UI settings
    MAX_CHART_POINTS: int = 1000
    REFRESH_INTERVAL: int = 30  # seconds
    
    # System settings
    LOG_LEVEL: str = 'INFO'
    DATA_DIR: str = 'data'
    MODEL_DIR: str = 'models'

# Global configuration instances
drift_config = DriftConfig()
flask_config = FlaskConfig()
app_config = AppConfig()

# Feature names for cybersecurity data
FEATURE_NAMES = [
    'packet_size', 'flow_duration', 'bytes_sent', 'bytes_received',
    'packet_count', 'protocol_type', 'port_number', 'tcp_flags',
    'payload_entropy', 'connection_frequency', 'time_between_packets',
    'unique_destinations', 'failed_connections', 'suspicious_ports',
    'anomalous_payload_size', 'unusual_timing_patterns'
]

# Cybersecurity feature descriptions
FEATURE_DESCRIPTIONS = {
    'packet_size': 'Average size of network packets',
    'flow_duration': 'Duration of network flows',
    'bytes_sent': 'Total bytes sent in connection',
    'bytes_received': 'Total bytes received in connection',
    'packet_count': 'Number of packets in flow',
    'protocol_type': 'Network protocol type (TCP/UDP/ICMP)',
    'port_number': 'Destination port number',
    'tcp_flags': 'TCP flag combinations',
    'payload_entropy': 'Entropy of packet payload',
    'connection_frequency': 'Frequency of connections',
    'time_between_packets': 'Interval between packets',
    'unique_destinations': 'Number of unique destinations',
    'failed_connections': 'Number of failed connection attempts',
    'suspicious_ports': 'Connections to suspicious ports',
    'anomalous_payload_size': 'Unusual payload size indicator',
    'unusual_timing_patterns': 'Abnormal timing patterns indicator'
}