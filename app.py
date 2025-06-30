"""
Main Flask application for Cybersecurity Drift Detection System
"""

from flask import Flask, render_template, jsonify, request
import json
import datetime
from typing import Dict, List

# Import utility modules
from utils.data_generator import CybersecurityDataGenerator
from utils.drift_detector import DriftDetector
from utils.explainability import ExplainabilityEngine
from utils.adaptive_learning import AdaptiveLearningSystem
from utils.visualization import DriftVisualization
from models.cybersecurity_model import CybersecurityModel
from config import drift_config, flask_config, app_config, FEATURE_NAMES

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = flask_config.SECRET_KEY
app.config['DEBUG'] = flask_config.DEBUG

# Global system components
system_state = {
    'initialized': False,
    'model': None,
    'drift_detector': None,
    'explainability_engine': None,
    'adaptive_system': None,
    'data_generator': None,
    'visualization': None,
    'baseline_data': None,
    'current_alerts': [],
    'performance_history': [],
    'drift_history': []
}

def initialize_system():
    """Initialize the cybersecurity drift detection system"""
    global system_state
    
    if system_state['initialized']:
        return True
    
    try:
        # Initialize components
        system_state['data_generator'] = CybersecurityDataGenerator()
        system_state['drift_detector'] = DriftDetector(drift_config)
        system_state['model'] = CybersecurityModel()
        system_state['visualization'] = DriftVisualization()
        
        # Generate baseline data
        system_state['baseline_data'] = system_state['data_generator'].generate_baseline_data(
            app_config.BASELINE_SAMPLES
        )
        
        # Train initial model
        system_state['model'].train_initial_model(system_state['baseline_data'])
        
        # Set up drift detection
        system_state['drift_detector'].set_baseline(system_state['baseline_data'])
        
        # Initialize explainability
        system_state['explainability_engine'] = ExplainabilityEngine(
            system_state['model'].model,
            FEATURE_NAMES
        )
        X_baseline = system_state['baseline_data'].drop('is_malicious', axis=1)
        system_state['explainability_engine'].initialize_explainer(
            system_state['model'].scaler.transform(X_baseline.values)
        )
        
        # Initialize adaptive learning
        system_state['adaptive_system'] = AdaptiveLearningSystem(
            system_state['model'].model,
            drift_config
        )
        
        system_state['initialized'] = True
        return True
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False

@app.route('/')
def dashboard():
    """Main dashboard page"""
    if not system_state['initialized']:
        initialize_system()
    
    return render_template('dashboard.html', 
                         feature_names=FEATURE_NAMES,
                         system_status='Initialized' if system_state['initialized'] else 'Initializing')

@app.route('/drift-analysis')
def drift_analysis():
    """Drift analysis page"""
    return render_template('drift_analysis.html')

@app.route('/api/system/status')
def system_status():
    """Get system status"""
    return jsonify({
        'initialized': system_state['initialized'],
        'timestamp': datetime.datetime.now().isoformat(),
        'alerts_count': len(system_state['current_alerts']),
        'features_monitored': len(FEATURE_NAMES)
    })

@app.route('/api/system/initialize', methods=['POST'])
def api_initialize():
    """Initialize the system via API"""
    success = initialize_system()
    return jsonify({
        'success': success,
        'message': 'System initialized successfully' if success else 'Failed to initialize system',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/drift/simulate', methods=['POST'])
def simulate_drift():
    """Simulate concept drift and analyze results"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        data = request.get_json()
        drift_type = data.get('drift_type', 'gradual')
        
        # Generate new data with drift
        new_data = system_state['data_generator'].generate_drifted_data(
            app_config.DRIFT_SAMPLES, drift_type
        )
        
        # Detect drift
        drift_results = system_state['drift_detector'].detect_statistical_drift(new_data)
        
        # Evaluate model performance
        performance = system_state['model'].evaluate_on_data(new_data)
        
        # Generate explanations
        explanations = system_state['explainability_engine'].explain_drift(drift_results)
        
        # Check if adaptation is needed
        adaptation_needed = system_state['adaptive_system'].should_retrain(
            drift_results, 
            performance['accuracy'] < 0.8
        )
        
        # Store results
        drift_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'drift_type': drift_type,
            'drift_results': drift_results,
            'performance': performance,
            'explanations': explanations,
            'adaptation_needed': adaptation_needed
        }
        
        system_state['drift_history'].append(drift_record)
        
        # Create alerts for high severity drift
        new_alerts = []
        for feature, result in drift_results.items():
            if result['drift_severity'] in ['HIGH', 'MEDIUM']:
                alert = {
                    'id': f"alert_{len(system_state['current_alerts']) + len(new_alerts)}",
                    'timestamp': datetime.datetime.now().isoformat(),
                    'feature': feature,
                    'severity': result['drift_severity'],
                    'psi_score': result['psi_score'],
                    'message': f"Drift detected in {feature} - {result['drift_severity']} severity"
                }
                new_alerts.append(alert)
        
        system_state['current_alerts'].extend(new_alerts)
        
        return jsonify({
            'success': True,
            'drift_results': drift_results,
            'performance': performance,
            'explanations': explanations,
            'adaptation_needed': adaptation_needed,
            'new_alerts': new_alerts,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations/drift/<feature>')
def get_drift_visualization(feature):
    """Get drift visualization data for a specific feature"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Get latest drift data
        if not system_state['drift_history']:
            return jsonify({'error': 'No drift data available'}), 404
        
        latest_drift = system_state['drift_history'][-1]
        
        # Generate visualization data
        viz_data = system_state['visualization'].get_feature_drift_data(
            system_state['baseline_data'],
            feature,
            latest_drift['drift_results'].get(feature, {})
        )
        
        return jsonify(viz_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations/dashboard')
def get_dashboard_data():
    """Get data for dashboard visualizations"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Prepare dashboard data
        dashboard_data = {
            'alerts': system_state['current_alerts'][-10:],  # Last 10 alerts
            'drift_timeline': [
                {
                    'timestamp': record['timestamp'],
                    'features_affected': len([
                        f for f, r in record['drift_results'].items()
                        if r['drift_severity'] != 'LOW'
                    ]),
                    'max_psi_score': max([
                        r['psi_score'] for r in record['drift_results'].values()
                    ]) if record['drift_results'] else 0
                }
                for record in system_state['drift_history'][-20:]  # Last 20 records
            ],
            'feature_status': {},
            'model_performance': system_state['performance_history'][-10:]
        }
        
        # Get current feature status
        if system_state['drift_history']:
            latest_drift = system_state['drift_history'][-1]
            for feature, result in latest_drift['drift_results'].items():
                dashboard_data['feature_status'][feature] = {
                    'severity': result['drift_severity'],
                    'psi_score': result['psi_score'],
                    'status': 'warning' if result['drift_severity'] == 'MEDIUM' else 
                             'danger' if result['drift_severity'] == 'HIGH' else 'success'
                }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get current system alerts"""
    return jsonify({
        'alerts': system_state['current_alerts'],
        'total_count': len(system_state['current_alerts']),
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    for alert in system_state['current_alerts']:
        if alert.get('id') == alert_id:
            alert['acknowledged'] = True
            alert['acknowledged_at'] = datetime.datetime.now().isoformat()
            return jsonify({'success': True, 'message': 'Alert acknowledged'})
    
    return jsonify({'error': 'Alert not found'}), 404

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Get recent drift data
        if not system_state['drift_history']:
            return jsonify({'error': 'No drift data for retraining'}), 400
        
        # Simulate retraining with latest data
        latest_record = system_state['drift_history'][-1]
        
        # In a real implementation, this would retrain the model
        # For demo purposes, we'll simulate successful retraining
        retrain_result = {
            'success': True,
            'timestamp': datetime.datetime.now().isoformat(),
            'message': 'Model retrained successfully',
            'new_accuracy': 0.95,
            'features_updated': len(FEATURE_NAMES)
        }
        
        return jsonify(retrain_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(
        host=flask_config.HOST,
        port=flask_config.PORT,
        debug=flask_config.DEBUG
    )