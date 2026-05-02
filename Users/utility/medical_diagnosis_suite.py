"""
MEDICAL-GRADE PILL IDENTIFICATION - COMPREHENSIVE ANALYSIS & TESTING
======================================================================

This module provides comprehensive testing and analysis tools to identify
and fix model prediction errors, ensuring medical-grade safety.

Key Problems Identified & Solutions:
1. OVERCONFIDENCE: Model predicts high-confidence wrong pills
   - Solution: Add confidence threshold (70%), reject below threshold
   
2. CLASS IMBALANCE: Some pills under-represented
   - Solution: Compute class weights, data augmentation
   
3. VISUAL SIMILARITY: Visually similar pills confuse model
   - Solution: Contrastive learning, hard negative mining
   
4. MISSING IMPRINT DETECTION: Model overfits to shape/color
   - Solution: Add attention to imprint region, data focus
   
5. OVERFITTING ON BACKGROUND: Model depends on background features
   - Solution: Background removal preprocessing, augmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import json

logger = logging.getLogger(__name__)


class PredictionErrorAnalyzer:
    """
    Analyze model prediction errors to identify issues.
    """
    
    def __init__(self, y_true, y_pred, y_pred_probs, reverse_label_map):
        """
        Initialize analyzer.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_probs: Prediction probabilities
            reverse_label_map: Label mapping
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_probs = y_pred_probs
        self.reverse_label_map = reverse_label_map
        self.confidences = np.max(y_pred_probs, axis=1)
        
        # Identify errors
        self.correct_predictions = (y_true == y_pred)
        self.incorrect_predictions = ~self.correct_predictions
        self.num_errors = np.sum(self.incorrect_predictions)
        self.error_rate = self.num_errors / len(y_true)
        
        logger.info(f"Found {self.num_errors} errors ({self.error_rate*100:.2f}%)")
    
    def find_overconfident_errors(self, conf_threshold=0.70):
        """
        Find WRONG predictions that are HIGHLY CONFIDENT.
        
        Medical Risk: HIGH - System confident about wrong diagnosis
        
        Returns:
            pd.DataFrame: Overconfident errors
        """
        overconf_mask = (
            self.incorrect_predictions & 
            (self.confidences >= conf_threshold)
        )
        
        overconf_errors = []
        for idx in np.where(overconf_mask)[0]:
            true_label = self.reverse_label_map[self.y_true[idx]]
            pred_label = self.reverse_label_map[self.y_pred[idx]]
            conf = float(self.confidences[idx])
            
            overconf_errors.append({
                'index': idx,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': conf,
                'error_severity': 'CRITICAL' if conf > 0.90 else 'HIGH'
            })
        
        df = pd.DataFrame(overconf_errors)
        logger.warning(f"Found {len(df)} OVERCONFIDENT ERRORS (High Risk)")
        return df.sort_values('confidence', ascending=False)
    
    def find_confused_classes(self):
        """
        Identify which pill pairs are most commonly confused.
        
        Returns:
            pd.DataFrame: Class pairs and confusion frequency
        """
        cm = confusion_matrix(self.y_true, self.y_pred, 
                             labels=range(len(self.reverse_label_map)))
        
        confusions = []
        for i in range(len(self.reverse_label_map)):
            for j in range(len(self.reverse_label_map)):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        'true_class': self.reverse_label_map[i],
                        'predicted_class': self.reverse_label_map[j],
                        'frequency': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100)
                    })
        
        df = pd.DataFrame(confusions)
        df = df.sort_values('frequency', ascending=False)
        logger.info(f"Found {len(df)} class confusion pairs")
        return df.head(20)  # Top 20 confusions
    
    def analyze_confidence_distribution(self):
        """
        Analyze confidence distribution for correct vs incorrect predictions.
        
        Medical Insight: If wrong predictions are high-confidence,
        the model is poorly calibrated and unreliable.
        """
        correct_confs = self.confidences[self.correct_predictions]
        incorrect_confs = self.confidences[self.incorrect_predictions]
        
        stats = {
            'correct_mean': float(np.mean(correct_confs)),
            'correct_std': float(np.std(correct_confs)),
            'correct_min': float(np.min(correct_confs)),
            'correct_max': float(np.max(correct_confs)),
            'incorrect_mean': float(np.mean(incorrect_confs)) if len(incorrect_confs) > 0 else 0,
            'incorrect_std': float(np.std(incorrect_confs)) if len(incorrect_confs) > 0 else 0,
            'incorrect_min': float(np.min(incorrect_confs)) if len(incorrect_confs) > 0 else 1,
            'incorrect_max': float(np.max(incorrect_confs)) if len(incorrect_confs) > 0 else 1,
        }
        
        # Check calibration problem
        overlap = (
            (stats['incorrect_max'] > stats['correct_mean']) and 
            len(incorrect_confs) > 0
        )
        if overlap:
            logger.warning("CALIBRATION ISSUE: Wrong predictions have high confidence")
        
        return stats
    
    def plot_error_analysis(self, save_path='error_analysis.png'):
        """Plot comprehensive error analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confidence distribution
        correct_confs = self.confidences[self.correct_predictions]
        incorrect_confs = self.confidences[self.incorrect_predictions]
        
        axes[0, 0].hist(correct_confs, bins=30, alpha=0.7, label='Correct', color='green')
        axes[0, 0].hist(incorrect_confs, bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error rate by confidence bin
        bins = np.linspace(0, 1, 11)
        error_rates = []
        bin_centers = []
        for i in range(len(bins)-1):
            mask = (self.confidences >= bins[i]) & (self.confidences < bins[i+1])
            if mask.sum() > 0:
                error_rate = (1 - accuracy_score(
                    self.y_true[mask], self.y_pred[mask]
                ))
                error_rates.append(error_rate)
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        axes[0, 1].plot(bin_centers, error_rates, 'o-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Confidence Bin')
        axes[0, 1].set_ylabel('Error Rate')
        axes[0, 1].set_title('Error Rate by Confidence (Calibration Analysis)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top confused classes
        confused = self.find_confused_classes().head(10)
        y_pos = np.arange(len(confused))
        axes[1, 0].barh(y_pos, confused['frequency'].values)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([
            f"{row['true_class']} → {row['predicted_class']}" 
            for _, row in confused.iterrows()
        ], fontsize=9)
        axes[1, 0].set_xlabel('Confusion Frequency')
        axes[1, 0].set_title('Top 10 Class Confusions')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Summary statistics
        stats = self.analyze_confidence_distribution()
        summary_text = f"""
        PREDICTION ERROR ANALYSIS SUMMARY
        ═══════════════════════════════════
        
        OVERALL:
        • Total Errors: {self.num_errors} ({self.error_rate*100:.2f}%)
        
        CORRECT PREDICTIONS:
        • Mean Confidence: {stats['correct_mean']:.3f}
        • Std Dev: {stats['correct_std']:.3f}
        • Range: [{stats['correct_min']:.3f}, {stats['correct_max']:.3f}]
        
        INCORRECT PREDICTIONS:
        • Mean Confidence: {stats['incorrect_mean']:.3f}
        • Std Dev: {stats['incorrect_std']:.3f}
        • Range: [{stats['incorrect_min']:.3f}, {stats['incorrect_max']:.3f}]
        
        CALIBRATION:
        • Overlap: {stats['incorrect_max'] > stats['correct_mean']}
        • Issue: Overconfident on wrong predictions
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error analysis plot saved to {save_path}")
        plt.show()


class ClassImbalanceReport:
    """
    Analyze class imbalance and recommend fixes.
    """
    
    @staticmethod
    def analyze_dataset(train_df):
        """
        Generate comprehensive class imbalance report.
        
        Args:
            train_df: Training dataframe with 'label' column
            
        Returns:
            dict: Detailed imbalance analysis
        """
        class_counts = train_df['label'].value_counts().sort_index()
        total = len(train_df)
        
        report = {
            'total_samples': total,
            'num_classes': len(class_counts),
            'imbalance_ratio': float(class_counts.max() / class_counts.min()),
            'class_distribution': class_counts.to_dict(),
            'class_percentages': {
                idx: float(count/total*100) 
                for idx, count in class_counts.items()
            },
            'minority_classes': class_counts[class_counts < class_counts.quantile(0.25)].index.tolist(),
            'majority_classes': class_counts[class_counts > class_counts.quantile(0.75)].index.tolist()
        }
        
        logger.info(f"Class Imbalance Ratio: {report['imbalance_ratio']:.2f}x")
        return report


class RecoveryRecommendations:
    """
    Generate specific recommendations to fix model issues.
    """
    
    @staticmethod
    def recommend_fixes(error_analysis, class_report):
        """
        Generate actionable recommendations based on analysis.
        
        Returns:
            dict: Prioritized list of fixes
        """
        recommendations = {
            'critical_fixes': [],
            'important_fixes': [],
            'optimizations': []
        }
        
        # Check for overconfident errors
        # (This would be called with actual error data)
        
        # Check for class imbalance
        if class_report['imbalance_ratio'] > 5:
            recommendations['critical_fixes'].append({
                'issue': 'SEVERE CLASS IMBALANCE',
                'impact': 'Minority classes have poor accuracy',
                'fixes': [
                    '1. Apply class weight balancing in loss function',
                    '2. Perform oversampling on minority classes',
                    '3. Use focal loss instead of cross-entropy',
                    '4. Collect more data for minority classes'
                ]
            })
        
        recommendations['important_fixes'].append({
            'issue': 'CALIBRATION ERROR',
            'impact': 'Model overconfident on wrong predictions',
            'fixes': [
                '1. Apply confidence threshold (70%) before prediction',
                '2. Use temperature scaling to recalibrate confidence',
                '3. Train with contrastive loss to improve embeddings',
                '4. Return UNKNOWN for low-confidence predictions'
            ]
        })
        
        recommendations['important_fixes'].append({
            'issue': 'VISUAL SIMILARITY CONFUSION',
            'impact': 'Similar pills frequently confused',
            'fixes': [
                '1. Mine hard negatives for focused training',
                '2. Use contrastive learning (triplet loss)',
                '3. Add data augmentation for confusing pairs',
                '4. Focus on imprint/texture features'
            ]
        })
        
        recommendations['optimizations'].append({
            'issue': 'BACKGROUND DEPENDENCY',
            'impact': 'Model may overfit to background patterns',
            'fixes': [
                '1. Apply background removal preprocessing',
                '2. Add aggressive augmentation (rotation, shift)',
                '3. Use U-Net for background masking',
                '4. Test on different backgrounds'
            ]
        })
        
        return recommendations


# Diagnostic function
def diagnose_model_issues(model, test_images, test_labels, reverse_label_map, train_df):
    """
    Run comprehensive diagnostic on model.
    
    Args:
        model: Trained Keras model
        test_images: Test image array
        test_labels: Test labels
        reverse_label_map: Label mapping
        train_df: Training dataframe
        
    Returns:
        dict: Complete diagnosis report
    """
    logger.info("="*70)
    logger.info("STARTING COMPREHENSIVE MODEL DIAGNOSIS")
    logger.info("="*70)
    
    # Get predictions
    y_pred_probs = model.predict(test_images, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 1. Error analysis
    logger.info("\n[1/3] Analyzing prediction errors...")
    error_analyzer = PredictionErrorAnalyzer(
        test_labels, y_pred, y_pred_probs, reverse_label_map
    )
    error_analyzer.plot_error_analysis('diagnosis_error_analysis.png')
    
    overconf_errors = error_analyzer.find_overconfident_errors()
    logger.info(f"Found {len(overconf_errors)} overconfident errors (HIGH RISK)")
    
    confused_classes = error_analyzer.find_confused_classes()
    logger.info(f"Top confused classes:\n{confused_classes.head()}")
    
    # 2. Class imbalance analysis
    logger.info("\n[2/3] Analyzing class imbalance...")
    class_report = ClassImbalanceReport.analyze_dataset(train_df)
    logger.info(f"Imbalance Ratio: {class_report['imbalance_ratio']:.2f}x")
    logger.info(f"Minority classes: {len(class_report['minority_classes'])}")
    
    # 3. Recommendations
    logger.info("\n[3/3] Generating recommendations...")
    recommendations = RecoveryRecommendations.recommend_fixes(error_analyzer, class_report)
    
    # Compile full diagnosis
    diagnosis = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'overall_accuracy': float(accuracy_score(test_labels, y_pred)),
        'num_errors': error_analyzer.num_errors,
        'error_rate': float(error_analyzer.error_rate),
        'overconfident_errors': len(overconf_errors),
        'top_confused_pairs': confused_classes.head(5).to_dict('records'),
        'class_imbalance_ratio': class_report['imbalance_ratio'],
        'minority_classes_count': len(class_report['minority_classes']),
        'recommendations': recommendations
    }
    
    # Save diagnosis
    with open('model_diagnosis.json', 'w') as f:
        json.dump(diagnosis, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS COMPLETE - See model_diagnosis.json")
    logger.info("="*70 + "\n")
    
    return diagnosis
