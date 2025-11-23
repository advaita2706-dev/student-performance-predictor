"""Visualization Module for Student Performance Predictor

This module provides comprehensive data visualization capabilities including
charts, graphs, and statistical plots for data analysis and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

# Configure visualization style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
    """Handles all visualization tasks for the student performance predictor."""
    
    def __init__(self, style: str = 'darkgrid'):
        """Initialize visualizer with specified style."""
        sns.set_style(style)
        self.figures = []
        logger.info("Visualizer initialized")
    
    def plot_distribution(self, data: pd.Series, title: str, xlabel: str, 
                         save_path: str = None) -> plt.Figure:
        """Plot distribution of a variable with histogram and KDE."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data, kde=True, ax=ax, color='skyblue')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, title: str = 'Correlation Matrix',
                               save_path: str = None) -> plt.Figure:
        """Create correlation heatmap for numeric features."""
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray,
                               title: str = 'Feature Importance', save_path: str = None) -> plt.Figure:
        """Plot feature importance bar chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        indices = np.argsort(importances)[::-1]
        
        colors = sns.color_palette('viridis', len(feature_names))
        ax.barh(range(len(importances)), importances[indices], color=colors)
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def plot_model_comparison(self, model_names: List[str], metrics: Dict[str, List[float]],
                             save_path: str = None) -> plt.Figure:
        """Create bar chart comparing different models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metric_list = ['rmse', 'mae', 'r2_score', 'mse']
        titles = ['Root Mean Squared Error (Lower is Better)', 
                 'Mean Absolute Error (Lower is Better)',
                 'R² Score (Higher is Better)',
                 'Mean Squared Error (Lower is Better)']
        
        for idx, (metric, title) in enumerate(zip(metric_list, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [metrics[model][metric] for model in model_names]
            colors = sns.color_palette('Set2', len(model_names))
            ax.bar(range(len(model_names)), values, color=colors)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray,
                                title: str = 'Actual vs Predicted', save_path: str = None) -> plt.Figure:
        """Scatter plot of actual vs predicted values."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: str = None) -> plt.Figure:
        """Plot residuals distribution and residual plot."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Residual distribution
        sns.histplot(residuals, kde=True, ax=axes[1], color='coral')
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def plot_learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                          val_scores: np.ndarray, save_path: str = None) -> plt.Figure:
        """Plot learning curve for model training."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, label='Training Score', marker='o', linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        ax.plot(train_sizes, val_mean, label='Validation Score', marker='s', linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def create_dashboard(self, df: pd.DataFrame, predictions: Dict,
                        save_path: str = None) -> plt.Figure:
        """Create comprehensive dashboard with multiple plots."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Distribution plots
        ax1 = fig.add_subplot(gs[0, :])
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        df[numeric_cols].plot(kind='box', ax=ax1)
        ax1.set_title('Feature Distributions', fontsize=14, fontweight='bold')
        
        # More dashboard components can be added here
        
        plt.suptitle('Student Performance Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig)
        return fig
    
    def save_all_figures(self, directory: str):
        """Save all generated figures to directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        for idx, fig in enumerate(self.figures):
            fig.savefig(f"{directory}/figure_{idx+1}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved {len(self.figures)} figures to {directory}")
    
    def close_all(self):
        """Close all figure windows."""
        plt.close('all')
        self.figures = []
        logger.info("All figures closed")
