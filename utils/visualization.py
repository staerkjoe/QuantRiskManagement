import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
from typing import Tuple, Optional
import pandas as pd

class ModelVisualizer:
    """Class for generating model evaluation visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            labels: Optional[list] = None,
                            normalize: bool = False,
                            title: str = "Confusion Matrix") -> plt.Figure:
        """
        Generate a confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(cm, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=labels or ['Bad Credit', 'Good Credit'],
                   yticklabels=labels or ['Bad Credit', 'Good Credit'],
                   ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_threshold(self,
                                    y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    pos_label: int = 1,
                                    title: str = "Precision-Recall Curve with Best Threshold") -> plt.Figure:
        """
        Generate a precision-recall curve with optimal threshold marked.
        
        Args:
            y_true: True binary labels (0=Good Credit, 1=Bad Credit after preprocessing)
            y_prob: Predicted probabilities for the positive class (class 1)
            pos_label: Label of the positive class (default: 1 for bad credit)
            title: Plot title
        """
        from sklearn.metrics import average_precision_score
        
        # Debug information
        print(f"\nDEBUG: Unique y_true values: {np.unique(y_true)}")
        print(f"DEBUG: y_prob shape: {y_prob.shape}")
        print(f"DEBUG: y_prob range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
        print(f"DEBUG: pos_label: {pos_label}")
        
        # If y_prob is 2D (probabilities for both classes), extract probability for positive class
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # For pos_label=0, we want the probability of class 0 (first column)
            y_prob_positive = y_prob[:, pos_label]
            print(f"DEBUG: Extracted probabilities for class {pos_label}")
        else:
            y_prob_positive = y_prob
        
        print(f"DEBUG: Final y_prob_positive range: [{y_prob_positive.min():.3f}, {y_prob_positive.max():.3f}]")
        
        # Precision-Recall Curve and Best Threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob_positive, pos_label=pos_label)
        ap_score = average_precision_score(y_true, y_prob_positive, pos_label=pos_label)
        
        # Find F1-optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        print(f"\nBest threshold = {best_threshold:.3f}")
        print(f"Precision @ best threshold = {precision[best_idx]:.3f}")
        print(f"Recall @ best threshold = {recall[best_idx]:.3f}")
        print(f"F1 @ best threshold = {f1_scores[best_idx]:.3f}")
        print(f"Average Precision Score = {ap_score:.3f}")
        
        # Create plot with better styling
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(recall, precision, linewidth=2, label=f"AP={ap_score:.3f}")
        ax.scatter(recall[best_idx], precision[best_idx], marker="o", color="red", s=100, zorder=5,
                  label=f"Best F1={f1_scores[best_idx]:.3f} @ thr={best_threshold:.3f}")
        
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self,
                               model,
                               feature_names: list,
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               show_direction: bool = False) -> plt.Figure:
        """
        Plot feature importance for both logistic regression and tree-based models.
        
        Args:
            model: Trained model (LogisticRegression, XGBoost, or Pipeline)
            feature_names: List of feature names
            top_n: Number of top features to display
            title: Plot title
            show_direction: If True, show coefficient direction (only for LogReg)
            
        Returns:
            matplotlib Figure object
        """
        # Extract the actual model if it's a pipeline
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps.get('classifier') or model.named_steps.get('model') or model[-1]
        else:
            actual_model = model
        
        # Extract feature importances based on model type
        if hasattr(actual_model, 'feature_importances_'):
            # Tree-based models (XGBoost, RandomForest, etc.)
            importances = actual_model.feature_importances_
            coefficients = None
            model_type = "Tree-based"
        elif hasattr(actual_model, 'coef_'):
            # Linear models (LogisticRegression, etc.)
            if len(actual_model.coef_.shape) > 1:
                coefficients = actual_model.coef_[0]
            else:
                coefficients = actual_model.coef_
            importances = np.abs(coefficients)
            model_type = "Logistic Regression"
        else:
            raise ValueError("Model must have either 'feature_importances_' or 'coef_' attribute")
        
        # Debug: Check length mismatch
        print(f"\nDEBUG - Feature names length: {len(feature_names)}")
        print(f"DEBUG - Importances length: {len(importances)}")
        
        # Handle length mismatch
        if len(feature_names) != len(importances):
            print(f"WARNING: Feature names ({len(feature_names)}) don't match importances ({len(importances)})")
            print("Using generic feature names instead.")
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'coefficient': coefficients if coefficients is not None else importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        if show_direction and coefficients is not None:
            # Plot with direction (for logistic regression)
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in importance_df['coefficient']]
            bars = ax.barh(range(len(importance_df)), importance_df['coefficient'], color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Coefficient Value', fontsize=11)
            plot_title = f'{title}\n(Green=Increases Risk, Red=Decreases Risk)'
        else:
            # Plot absolute importance
            bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                          color='steelblue', alpha=0.7)
            ax.set_xlabel('Importance Score', fontsize=11)
            plot_title = f'{title} ({model_type})'
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=9)
        ax.set_ylabel('Features', fontsize=11)
        ax.set_title(plot_title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig