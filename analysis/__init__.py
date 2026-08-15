"""Analysis and visualization for StudentRisk."""
from .compare import build_comparison, build_comparison_markdown
from .visualize import (
    plot_confusion_matrices, plot_roc_curves, plot_pr_curves,
    plot_metric_comparison, plot_per_fold_stability,
    generate_all_visualizations,
)

__all__ = [
    'build_comparison', 'build_comparison_markdown',
    'plot_confusion_matrices', 'plot_roc_curves', 'plot_pr_curves',
    'plot_metric_comparison', 'plot_per_fold_stability',
    'generate_all_visualizations',
]