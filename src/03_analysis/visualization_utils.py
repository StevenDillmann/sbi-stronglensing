import numpy as np
import matplotlib.pyplot as plt
import corner


# === CORNER PLOT FUNCTIONS ===

def update_figure_legend(fig, color, legend_label, lw = 2, fontsize = 20):

    # Retrieve old handles & labels if they exist
    old_handles = getattr(fig, '_my_legend_handles', [])
    old_labels = getattr(fig, '_my_legend_labels', [])

    # Remove any existing figure-level legends (if present)
    if hasattr(fig, 'legends') and fig.legends:
        for legend in fig.legends:
            legend.remove()

    # Create a new handle
    new_handle = plt.Line2D([0], [0], color=color, lw=lw)

    # Append new handle & label
    old_handles.append(new_handle)
    old_labels.append(legend_label)

    # Save them back onto the figure object
    fig._my_legend_handles = old_handles
    fig._my_legend_labels = old_labels

    # Draw a fresh single legend containing all handles/labels so far
    fig.legend(
        handles=fig._my_legend_handles,
        labels=fig._my_legend_labels,
        loc='upper right',
        fontsize=fontsize,
        frameon=False
    )

def plot_corner(samples, truths, labels, color, legend_label=None):

    if labels is None:
        labels = [
            r'$\theta_\mathrm{E}$', r'$\gamma_1$', r'$\gamma_2$', 
            r'$\gamma_\mathrm{lens}$', r'$e_1$', r'$e_2$', 
            r'$x_\mathrm{lens}$', r'$y_\mathrm{lens}$', 
            r'$x_\mathrm{src}$', r'$y_\mathrm{src}$'
        ]

    fig = corner.corner(
        np.array(samples),
        labels=labels,
        truths=truths,
        show_titles=True,
        title_fmt='.2f',
        title_kwargs=dict(fontsize=15),
        label_kwargs=dict(fontsize=20),
        truth_color='k',
        levels=[0.68, 0.95],
        bins=20,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        smooth=1.0,
        hist_kwargs=dict(density=True, color=color, linewidth=2, histtype='step'),
        color=color,
        fig=None
    )
    
    # Delegate legend creation to our helper
    update_figure_legend(fig, color, legend_label=legend_label)

    return fig
    
def plot_corner_overlay(samples, fig, color, legend_label=None):

    fig = corner.corner(
        np.array(samples),
        levels=[0.68, 0.95],
        bins=20,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        smooth=1.0,
        hist_kwargs=dict(density=True, color=color, linewidth=2, histtype='step'),
        color=color,
        fig=fig
    )
    
    # Delegate legend update to our helper
    update_figure_legend(fig, color, legend_label=legend_label)

    return fig