"""
Main Flask application for Cybersecurity Drift Detection System
"""

from flask import Flask, render_template, jsonify, request, flash
import json
import datetime
import traceback
from typing import Dict, List
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules with error handling
try:
    from utils.data_generator import CybersecurityDataGenerator
    from utils.drift_detector import DriftDetector
    from utils.explainability import ExplainabilityEngine
    from utils.adaptive_learning import AdaptiveLearningSystem
    from utils.visualization import DriftVisualization
    from models.cybersecurity_model import CybersecurityModel
    from config import drift_config, flask_config, app_config, FEATURE_NAMES
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

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
    'drift_history': [],
    'initialization_error': None
}

def initialize_system():
    """Initialize the cybersecurity drift detection system"""
    global system_state
    
    if system_state['initialized']:
        return True
    
    try:
        print("Initializing cybersecurity drift detection system...")
        
        # Initialize components
        system_state['data_generator'] = CybersecurityDataGenerator()
        system_state['drift_detector'] = DriftDetector(drift_config)
        system_state['model'] = CybersecurityModel()
        system_state['visualization'] = DriftVisualization()
        
        print("Components initialized, generating baseline data...")
        
        # Generate baseline data
        system_state['baseline_data'] = system_state['data_generator'].generate_baseline_data(
            app_config.BASELINE_SAMPLES
        )
        
        print("Training initial model...")
        
        # Train initial model
        training_result = system_state['model'].train_initial_model(system_state['baseline_data'])
        if not training_result.get('success', False):
            raise Exception(f"Model training failed: {training_result.get('error', 'Unknown error')}")
        
        print("Setting up drift detection...")
        
        # Set up drift detection
        system_state['drift_detector'].set_baseline(system_state['baseline_data'])
        
        print("Initializing explainability engine...")
        
        # Initialize explainability
        system_state['explainability_engine'] = ExplainabilityEngine(
            system_state['model'].model,
            FEATURE_NAMES
        )
        
        # Prepare data for explainability
        X_baseline = system_state['baseline_data'].drop('is_malicious', axis=1)
        X_baseline_scaled = system_state['model'].scaler.transform(X_baseline.values)
        system_state['explainability_engine'].initialize_explainer(X_baseline_scaled)
        
        print("Setting up adaptive learning...")
        
        # Initialize adaptive learning
        system_state['adaptive_system'] = AdaptiveLearningSystem(
            system_state['model'].model,
            drift_config
        )
        
        # Add initial performance record
        initial_performance = training_result.get('metrics', {})
        system_state['performance_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'accuracy': initial_performance.get('accuracy', 0.0),
            'precision': initial_performance.get('precision', 0.0),
            'recall': initial_performance.get('recall', 0.0),
            'f1_score': initial_performance.get('f1_score', 0.0)
        })
        
        system_state['initialized'] = True
        system_state['initialization_error'] = None
        print("System initialization completed successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Error initializing system: {str(e)}"
        print(error_msg)
        print("Traceback:", traceback.format_exc())
        system_state['initialization_error'] = error_msg
        system_state['initialized'] = False
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
    try:
        return jsonify({
            'initialized': system_state['initialized'],
            'timestamp': datetime.datetime.now().isoformat(),
            'alerts_count': len(system_state['current_alerts']),
            'features_monitored': len(FEATURE_NAMES),
            'error': system_state.get('initialization_error'),
            'drift_history_count': len(system_state['drift_history']),
            'performance_history_count': len(system_state['performance_history'])
        })
    except Exception as e:
        return jsonify({
            'initialized': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/system/initialize', methods=['POST'])
def api_initialize():
    """Initialize the system via API"""
    try:
        success = initialize_system()
        return jsonify({
            'success': success,
            'message': 'System initialized successfully' if success else f'Failed to initialize system: {system_state.get("initialization_error", "Unknown error")}',
            'timestamp': datetime.datetime.now().isoformat(),
            'error': system_state.get('initialization_error') if not success else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

@app.route('/api/drift/simulate', methods=['POST'])
def simulate_drift():
    """Simulate concept drift and analyze results"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized', 'details': system_state.get('initialization_error')}), 400
    
    try:
        data = request.get_json() or {}
        drift_type = data.get('drift_type', 'gradual')
        
        print(f"Simulating {drift_type} drift...")
        
        # Generate new data with drift
        new_data = system_state['data_generator'].generate_drifted_data(
            app_config.DRIFT_SAMPLES, drift_type
        )
        
        print("Detecting drift...")
        
        # Detect drift
        drift_results = system_state['drift_detector'].detect_statistical_drift(new_data)
        
        print("Evaluating model performance...")
        
        # Evaluate model performance
        performance_result = system_state['model'].evaluate_on_data(new_data)
        performance = performance_result.get('metrics', {})
        
        print("Generating explanations...")
        
        # Generate explanations
        explanations = system_state['explainability_engine'].explain_drift(drift_results)
        
        print("Checking adaptation needs...")
        
        # Check if adaptation is needed
        adaptation_needed = system_state['adaptive_system'].should_retrain(
            drift_results, 
            performance.get('accuracy', 0) < 0.8
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
        
        # Add performance to history
        system_state['performance_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            **performance
        })
        
        # Create alerts for high severity drift
        new_alerts = []
        for feature, result in drift_results.items():
            if result.get('drift_severity') in ['HIGH', 'MEDIUM']:
                alert = {
                    'id': f"alert_{len(system_state['current_alerts']) + len(new_alerts) + 1}",
                    'timestamp': datetime.datetime.now().isoformat(),
                    'feature': feature,
                    'severity': result.get('drift_severity', 'LOW'),
                    'psi_score': result.get('psi_score', 0),
                    'message': f"Drift detected in {feature} - {result.get('drift_severity', 'LOW')} severity"
                }
                new_alerts.append(alert)
        
        system_state['current_alerts'].extend(new_alerts)
        
        # Keep only last 50 alerts
        if len(system_state['current_alerts']) > 50:
            system_state['current_alerts'] = system_state['current_alerts'][-50:]
        
        print(f"Drift simulation completed. Found {len(new_alerts)} new alerts.")
        
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
        error_msg = f"Error simulating drift: {str(e)}"
        print(error_msg)
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': error_msg}), 500

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
        
        # Check if feature exists in drift results
        if feature not in latest_drift['drift_results']:
            return jsonify({'error': f'Feature {feature} not found in drift results'}), 404
        
        # Generate visualization data
        viz_data = system_state['visualization'].get_feature_drift_data(
            system_state['baseline_data'],
            feature,
            latest_drift['drift_results'][feature]
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
            'drift_timeline': [],
            'feature_status': {},
            'model_performance': system_state['performance_history'][-10:]
        }
        
        # Build drift timeline
        for record in system_state['drift_history'][-20:]:  # Last 20 records
            timeline_entry = {
                'timestamp': record['timestamp'],
                'features_affected': len([
                    f for f, r in record['drift_results'].items()
                    if r.get('drift_severity', 'LOW') != 'LOW'
                ])
            }
            
            # Calculate max PSI score
            psi_scores = [r.get('psi_score', 0) for r in record['drift_results'].values()]
            timeline_entry['max_psi_score'] = max(psi_scores) if psi_scores else 0
            
            dashboard_data['drift_timeline'].append(timeline_entry)
        
        # Get current feature status
        if system_state['drift_history']:
            latest_drift = system_state['drift_history'][-1]
            for feature, result in latest_drift['drift_results'].items():
                severity = result.get('drift_severity', 'LOW')
                dashboard_data['feature_status'][feature] = {
                    'severity': severity,
                    'psi_score': result.get('psi_score', 0),
                    'status': 'danger' if severity == 'HIGH' else 
                             'warning' if severity == 'MEDIUM' else 'success'
                }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        error_msg = f"Error getting dashboard data: {str(e)}"
        print(error_msg)
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get current system alerts"""
    try:
        return jsonify({
            'alerts': system_state['current_alerts'],
            'total_count': len(system_state['current_alerts']),
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        for alert in system_state['current_alerts']:
            if alert.get('id') == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.datetime.now().isoformat()
                return jsonify({'success': True, 'message': 'Alert acknowledged'})
        
        return jsonify({'error': 'Alert not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Get recent drift data
        if not system_state['drift_history']:
            return jsonify({'error': 'No drift data for retraining'}), 400
        
        # Generate new training data
        new_training_data = system_state['data_generator'].generate_baseline_data(1000)
        
        # Retrain model
        retrain_result = system_state['model'].retrain_model(
            new_training_data,
            system_state['baseline_data']
        )
        
        if retrain_result.get('success'):
            # Update performance history
            new_performance = retrain_result.get('metrics', {})
            system_state['performance_history'].append({
                'timestamp': datetime.datetime.now().isoformat(),
                **new_performance
            })
            
            return jsonify({
                'success': True,
                'timestamp': datetime.datetime.now().isoformat(),
                'message': 'Model retrained successfully',
                'new_accuracy': new_performance.get('accuracy', 0),
                'features_updated': len(FEATURE_NAMES),
                'training_samples': retrain_result.get('total_samples', 0)
            })
        else:
            return jsonify({
                'success': False,
                'error': retrain_result.get('error', 'Unknown error'),
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
        
    except Exception as e:
        error_msg = f"Error retraining model: {str(e)}"
        print(error_msg)
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/visualizations/feature-importance')
def get_feature_importance():
    """Get feature importance data"""
    if not system_state['initialized']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        importance_data = system_state['model'].get_feature_importance()
        
        if 'error' in importance_data:
            return jsonify(importance_data), 400
        
        # Format for visualization
        viz_data = {
            'feature_names': FEATURE_NAMES,
            'importance_values': importance_data.get('importance_values', []),
            'model_type': importance_data.get('model_type', 'unknown')
        }
        
        return jsonify(viz_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/reset', methods=['POST'])
def reset_system():
    """Reset the system state"""
    try:
        global system_state
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
            'drift_history': [],
            'initialization_error': None
        }
        
        return jsonify({
            'success': True,
            'message': 'System reset successfully',
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    return render_template('dashboard.html', 
                         feature_names=FEATURE_NAMES,
                         system_status='Error'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"Internal server error: {error}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('dashboard.html', 
                         feature_names=FEATURE_NAMES,
                         system_status='Error'), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize system before starting the app
    print("Starting cybersecurity drift detection system...")
    print("Initializing system components...")
    initialize_system()
    
    print("Starting Flask application...")
    print(f"Dashboard will be available at: http://localhost:{flask_config.PORT}")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        app.run(
            host=flask_config.HOST,
            port=flask_config.PORT,
            debug=flask_config.DEBUG
        )
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nError starting application: {e}")
        sys.exit(1)