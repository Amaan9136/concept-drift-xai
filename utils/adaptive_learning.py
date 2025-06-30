"""
Adaptive learning system for model retraining and adaptation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import datetime
from dataclasses import dataclass

@dataclass
class AdaptationRecord:
    """Record of model adaptation activities"""
    timestamp: datetime.datetime
    adaptation_type: str
    affected_features: List[str]
    data_size: int
    performance_before: float
    performance_after: float
    success: bool
    notes: str

class AdaptiveLearningSystem:
    """Handle model adaptation and retraining"""
    
    def __init__(self, base_model, config):
        self.base_model = base_model
        self.config = config
        self.model_history = []
        self.adaptation_log = []
        self.adaptation_strategies = {
            'incremental': self._incremental_update,
            'full_retrain': self._full_retrain,
            'ensemble': self._ensemble_update,
            'transfer': self._transfer_learning
        }
    
    def should_retrain(self, drift_results: Dict[str, Dict], performance_drift: bool) -> bool:
        """Determine if model retraining is necessary"""
        high_drift_features = [
            feature for feature, results in drift_results.items()
            if results['drift_severity'] == 'HIGH'
        ]
        
        medium_drift_features = [
            feature for feature, results in drift_results.items()
            if results['drift_severity'] == 'MEDIUM'
        ]
        
        # Decision criteria
        if performance_drift:
            return True
        if len(high_drift_features) >= 2:
            return True
        if len(medium_drift_features) >= 4:
            return True
        
        return False
    
    def determine_adaptation_strategy(self, drift_results: Dict[str, Dict], 
                                    performance_drift: bool) -> str:
        """Determine the best adaptation strategy"""
        high_drift_count = sum(1 for r in drift_results.values() if r['drift_severity'] == 'HIGH')
        medium_drift_count = sum(1 for r in drift_results.values() if r['drift_severity'] == 'MEDIUM')
        
        if performance_drift or high_drift_count >= 3:
            return 'full_retrain'
        elif high_drift_count >= 1 or medium_drift_count >= 3:
            return 'incremental'
        elif medium_drift_count >= 1:
            return 'ensemble'
        else:
            return 'transfer'
    
    def adapt_model(self, drift_results: Dict[str, Dict], new_data: pd.DataFrame,
                   historical_data: pd.DataFrame = None, strategy: str = None) -> Dict:
        """Execute model adaptation based on drift analysis"""
        
        if strategy is None:
            strategy = self.determine_adaptation_strategy(drift_results, False)
        
        if strategy not in self.adaptation_strategies:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")
        
        # Get performance before adaptation
        performance_before = self._evaluate_current_performance(new_data)
        
        # Execute adaptation strategy
        adaptation_result = self.adaptation_strategies[strategy](
            drift_results, new_data, historical_data
        )
        
        # Get performance after adaptation
        performance_after = self._evaluate_current_performance(new_data)
        
        # Log adaptation
        affected_features = [f for f, r in drift_results.items() if r['drift_severity'] != 'LOW']
        
        record = AdaptationRecord(
            timestamp=datetime.datetime.now(),
            adaptation_type=strategy,
            affected_features=affected_features,
            data_size=len(new_data),
            performance_before=performance_before,
            performance_after=performance_after,
            success=adaptation_result['success'],
            notes=adaptation_result.get('notes', '')
        )
        
        self.adaptation_log.append(record)
        
        return {
            'strategy': strategy,
            'success': adaptation_result['success'],
            'performance_improvement': performance_after - performance_before,
            'affected_features': affected_features,
            'timestamp': record.timestamp.isoformat(),
            'details': adaptation_result
        }
    
    def _incremental_update(self, drift_results: Dict, new_data: pd.DataFrame, 
                           historical_data: pd.DataFrame = None) -> Dict:
        """Perform incremental model update"""
        try:
            affected_features = [f for f, r in drift_results.items() if r['drift_severity'] in ['HIGH', 'MEDIUM']]
            
            # In a real implementation, this would use online learning algorithms
            # For demo purposes, we simulate incremental learning
            
            return {
                'success': True,
                'method': 'incremental_learning',
                'features_updated': affected_features,
                'notes': f'Incrementally updated {len(affected_features)} features'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'notes': 'Incremental update failed'
            }
    
    def _full_retrain(self, drift_results: Dict, new_data: pd.DataFrame,
                     historical_data: pd.DataFrame = None) -> Dict:
        """Perform full model retraining"""
        try:
            # Combine historical and new data if available
            if historical_data is not None:
                combined_data = pd.concat([historical_data, new_data], ignore_index=True)
            else:
                combined_data = new_data
            
            # Simulate retraining process
            training_size = len(combined_data)
            
            return {
                'success': True,
                'method': 'full_retrain',
                'training_samples': training_size,
                'notes': f'Full retraining completed with {training_size} samples'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'notes': 'Full retraining failed'
            }
    
    def _ensemble_update(self, drift_results: Dict, new_data: pd.DataFrame,
                        historical_data: pd.DataFrame = None) -> Dict:
        """Update ensemble model weights"""
        try:
            affected_features = [f for f, r in drift_results.items() if r['drift_severity'] != 'LOW']
            
            return {
                'success': True,
                'method': 'ensemble_update',
                'features_reweighted': affected_features,
                'notes': f'Updated ensemble weights for {len(affected_features)} features'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'notes': 'Ensemble update failed'
            }
    
    def _transfer_learning(self, drift_results: Dict, new_data: pd.DataFrame,
                          historical_data: pd.DataFrame = None) -> Dict:
        """Apply transfer learning techniques"""
        try:
            return {
                'success': True,
                'method': 'transfer_learning',
                'notes': 'Applied transfer learning from related cybersecurity domains'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'notes': 'Transfer learning failed'
            }
    
    def _evaluate_current_performance(self, test_data: pd.DataFrame) -> float:
        """Evaluate current model performance"""
        try:
            # Simulate performance evaluation
            # In real implementation, this would evaluate the actual model
            return np.random.uniform(0.85, 0.95)
        except Exception:
            return 0.0
    
    def get_adaptation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent adaptation history"""
        recent_adaptations = self.adaptation_log[-limit:] if self.adaptation_log else []
        
        return [
            {
                'timestamp': record.timestamp.isoformat(),
                'type': record.adaptation_type,
                'affected_features': record.affected_features,
                'performance_improvement': record.performance_after - record.performance_before,
                'success': record.success,
                'notes': record.notes
            }
            for record in recent_adaptations
        ]
    
    def get_adaptation_statistics(self) -> Dict:
        """Get adaptation statistics"""
        if not self.adaptation_log:
            return {'total_adaptations': 0}
        
        successful_adaptations = [r for r in self.adaptation_log if r.success]
        
        avg_improvement = np.mean([
            r.performance_after - r.performance_before 
            for r in successful_adaptations
        ]) if successful_adaptations else 0
        
        strategy_counts = {}
        for record in self.adaptation_log:
            strategy = record.adaptation_type
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_adaptations': len(self.adaptation_log),
            'successful_adaptations': len(successful_adaptations),
            'success_rate': len(successful_adaptations) / len(self.adaptation_log),
            'average_improvement': avg_improvement,
            'strategy_distribution': strategy_counts,
            'last_adaptation': self.adaptation_log[-1].timestamp.isoformat() if self.adaptation_log else None
        }