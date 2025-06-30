"""
Cybersecurity machine learning model management
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import joblib
import os

class CybersecurityModel:
    """Manage cybersecurity machine learning models"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_initial_model(self, data: pd.DataFrame) -> Dict:
        """Train the initial cybersecurity model"""
        try:
            # Prepare features and target
            X = data.drop('is_malicious', axis=1)
            y = data['is_malicious']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Store training history
            training_record = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': metrics,
                'feature_importance': self.model.feature_importances_.tolist()
            }
            
            self.training_history.append(training_record)
            self.is_trained = True
            
            return {
                'success': True,
                'metrics': metrics,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': list(X.columns)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_on_data(self, data: pd.DataFrame) -> Dict:
        """Evaluate model performance on new data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            X = data.drop('is_malicious', axis=1)
            y = data['is_malicious']
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred)
            
            return {
                'success': True,
                'metrics': metrics,
                'samples_evaluated': len(data),
                'predictions': y_pred.tolist(),
                'prediction_probabilities': y_pred_proba.tolist()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        if hasattr(self.model, 'feature_importances_'):
            return {
                'importance_values': self.model.feature_importances_.tolist(),
                'model_type': self.model_type
            }
        else:
            return {'error': 'Model does not support feature importance'}
    
    def retrain_model(self, new_data: pd.DataFrame, 
                     historical_data: pd.DataFrame = None) -> Dict:
        """Retrain the model with new data"""
        try:
            # Combine data if historical data is provided
            if historical_data is not None:
                combined_data = pd.concat([historical_data, new_data], ignore_index=True)
            else:
                combined_data = new_data
            
            # Train model
            result = self.train_initial_model(combined_data)
            
            if result['success']:
                result['retrain'] = True
                result['total_samples'] = len(combined_data)
                result['new_samples'] = len(new_data)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to disk"""
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from disk"""
        try:
            if not os.path.exists(filepath):
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.training_history = model_data.get('training_history', [])
            self.is_trained = model_data.get('is_trained', False)
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model performance metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
        except Exception as e:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        summary = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_sessions': len(self.training_history)
        }
        
        if self.is_trained and self.training_history:
            latest_training = self.training_history[-1]
            summary.update({
                'latest_training': latest_training['timestamp'],
                'latest_metrics': latest_training['metrics'],
                'total_training_samples': latest_training['training_samples'],
                'feature_count': len(latest_training.get('feature_importance', []))
            })
        
        return summary
    
    def detect_model_drift(self, current_performance: Dict, 
                          threshold: float = 0.05) -> Dict:
        """Detect if model performance has drifted"""
        if not self.training_history:
            return {'drift_detected': False, 'reason': 'No baseline performance'}
        
        baseline_metrics = self.training_history[0]['metrics']
        current_metrics = current_performance.get('metrics', {})
        
        drift_detected = False
        drift_details = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_value = baseline_metrics.get(metric, 0)
            current_value = current_metrics.get(metric, 0)
            
            performance_drop = baseline_value - current_value
            
            if performance_drop > threshold:
                drift_detected = True
                drift_details[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'drop': performance_drop
                }
        
        return {
            'drift_detected': drift_detected,
            'threshold': threshold,
            'drift_details': drift_details,
            'recommendation': 'Model retraining recommended' if drift_detected else 'Model performance stable'
        }
    
    def get_training_history(self) -> List[Dict]:
        """Get model training history"""
        return self.training_history.copy()
    
    def update_model_incrementally(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict:
        """Perform incremental model update (simplified implementation)"""
        if not self.is_trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # For RandomForest, we can't do true incremental learning
            # This is a simplified approach that would need improvement for production
            
            # Scale new data
            X_new_scaled = self.scaler.transform(X_new)
            
            # Get current performance before update
            current_score = self.model.score(X_new_scaled, y_new)
            
            # In a real implementation, you would use online learning algorithms
            # For now, we simulate incremental learning
            
            return {
                'success': True,
                'method': 'simulated_incremental',
                'samples_processed': len(X_new),
                'performance_score': current_score,
                'note': 'Incremental learning simulated - use online learning algorithms for production'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }