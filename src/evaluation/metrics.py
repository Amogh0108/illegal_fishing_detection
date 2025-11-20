"""Evaluation metrics and comparison"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/evaluation.log")

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def compare_models(self, y_true, ml_pred, rule_pred, ml_proba=None):
        """Compare ML model vs rule-based baseline"""
        logger.info("=" * 50)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 50)
        
        # ML model metrics
        ml_metrics = self.calculate_metrics(y_true, ml_pred, ml_proba)
        logger.info("\nML Model Performance:")
        for metric, value in ml_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Rule-based metrics
        rule_metrics = self.calculate_metrics(y_true, rule_pred)
        logger.info("\nRule-Based Performance:")
        for metric, value in rule_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Improvement
        logger.info("\nImprovement over Rule-Based:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            improvement = ((ml_metrics[metric] - rule_metrics[metric]) / rule_metrics[metric] * 100)
            logger.info(f"  {metric}: {improvement:+.2f}%")
        
        return ml_metrics, rule_metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, title, output_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_roc_curve(self, y_true, y_proba, output_path):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved ROC curve to {output_path}")
    
    def generate_report(self, y_true, ml_pred, rule_pred, ml_proba=None):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Calculate metrics
        ml_metrics, rule_metrics = self.compare_models(y_true, ml_pred, rule_pred, ml_proba)
        
        # Create output directory
        output_dir = Path("outputs/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(y_true, ml_pred, "ML Model Confusion Matrix", 
                                   output_dir / "ml_confusion_matrix.png")
        self.plot_confusion_matrix(y_true, rule_pred, "Rule-Based Confusion Matrix",
                                   output_dir / "rule_confusion_matrix.png")
        
        # Plot ROC curve
        if ml_proba is not None:
            self.plot_roc_curve(y_true, ml_proba, output_dir / "roc_curve.png")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Model': ['ML Model', 'Rule-Based'],
            'Accuracy': [ml_metrics['accuracy'], rule_metrics['accuracy']],
            'Precision': [ml_metrics['precision'], rule_metrics['precision']],
            'Recall': [ml_metrics['recall'], rule_metrics['recall']],
            'F1-Score': [ml_metrics['f1_score'], rule_metrics['f1_score']]
        })
        
        if ml_proba is not None:
            metrics_df.loc[metrics_df['Model'] == 'ML Model', 'ROC-AUC'] = ml_metrics['roc_auc']
        
        metrics_df.to_csv(output_dir / "metrics_comparison.csv", index=False)
        logger.info(f"Saved metrics comparison to {output_dir / 'metrics_comparison.csv'}")
        
        return ml_metrics, rule_metrics

def main():
    """Run evaluation"""
    config = load_config()
    evaluator = ModelEvaluator(config)
    
    # Load predictions
    ml_pred_path = Path("outputs/anomaly_predictions.csv")
    rule_pred_path = Path("outputs/rule_based_predictions.csv")
    
    ml_df = pd.read_csv(ml_pred_path)
    rule_df = pd.read_csv(rule_pred_path)
    
    # Load ground truth (using synthetic labels from training)
    features_path = Path(config.get('data', 'output_dir')) / "ais_all_features.csv"
    df = pd.read_csv(features_path)
    
    # Align data
    y_true = df['anomaly'].values[:len(ml_df)]
    ml_pred = ml_df['anomaly'].values
    ml_proba = ml_df['ensemble_score'].values
    rule_pred = rule_df['rule_anomaly'].values[:len(ml_df)]
    
    # Generate report
    evaluator.generate_report(y_true, ml_pred, rule_pred, ml_proba)

if __name__ == "__main__":
    main()
