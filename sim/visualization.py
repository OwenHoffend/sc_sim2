import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional

def plot_scc_histogram(
    scc_values: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    bins: int = 20,
    title: str = "SCC Frequency Distribution",
    xlabel: str = "SCC Value",
    ylabel: str = "Frequency",
    figsize: tuple = (10, 6),
    alpha: float = 0.7,
    density: bool = False,
    show_grid: bool = True,
    color_palette: Optional[List[str]] = None
) -> None:
    """
    Generate a frequency histogram for SCC (Stochastic Correlation Coefficient) values.
    
    Args:
        scc_values: Single array or list of arrays containing SCC values (range: [-1, 1])
        labels: List of labels for each series (optional)
        bins: Number of histogram bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        alpha: Transparency of histogram bars
        density: If True, normalize the histogram to form a probability density
        show_grid: Whether to show grid lines
        color_palette: List of colors for different series (optional)
    
    Returns:
        None (displays the plot)
    """
    # Convert single array to list for consistent processing
    if isinstance(scc_values, np.ndarray):
        scc_values = [scc_values]
    
    # Set default labels if not provided
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(scc_values))]
    
    # Set default color palette if not provided
    if color_palette is None:
        color_palette = plt.cm.tab10.colors[:len(scc_values)]
    
    # Create figure and axis
    plt.figure(figsize=figsize)
    
    # Plot histogram for each series
    for values, label, color in zip(scc_values, labels, color_palette):
        plt.hist(
            values,
            bins=bins,
            range=(-1, 1),
            alpha=alpha,
            label=label,
            color=color,
            density=density
        )
    
    # Customize plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(show_grid, alpha=0.3)
    plt.legend()
    
    # Set x-axis limits to ensure full range is visible
    plt.xlim(-1.1, 1.1)
    
    # Add vertical line at x=0 for reference
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    scc1 = np.random.normal(0.3, 0.2, 1000)  # Centered around 0.3
    scc2 = np.random.normal(-0.2, 0.2, 1000)  # Centered around -0.2
    
    # Clip values to [-1, 1] range
    scc1 = np.clip(scc1, -1, 1)
    scc2 = np.clip(scc2, -1, 1)
    
    # Plot histogram
    plot_scc_histogram(
        [scc1, scc2],
        labels=['Positive Correlation', 'Negative Correlation'],
        title='Example SCC Distributions',
        bins=30
    ) 