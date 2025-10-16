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