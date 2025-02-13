import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from IPython.display import HTML

def plot_misfit(iter_loss, save_path="", show=False):
    """
    Plot and save the L2-norm misfits over iterations.

    Args:
        iter_loss (list or np.array): The L2-norm misfits for each iteration.
        save_path (str): The path where the plot image will be saved.
        plot (bool): Flag to control whether to display the plot. Default is True.

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=(8, 6))
    
    # Plot the misfit
    plt.plot(iter_loss, c='k')
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Misfits", fontsize=12)
    plt.tick_params(labelsize=12)
    
    # Save the figure
    if not save_path == "":
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    
    # If plot is True, display the plot
    if show:
        plt.show()
    else:
        # Close the plot
        plt.close()

def plot_initial_and_inverted(vp_init, iter_vp, save_path="", show=False,cmap='jet_r'):
    """
    Plot and save the initial model and inverted results.

    Args:
        vp_init (np.array): The initial velocity model.
        iter_vp (list or np.array): List of inverted results, with the last element being the final model.
        save_path (str): The path where the plot image will be saved.
        plot (bool): Flag to control whether to display the plot. Default is False.

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=(12, 8))
    
    # Plot initial model
    plt.subplot(121)
    plt.imshow(vp_init, cmap=cmap)
    plt.title("Initial Model")
    
    # Plot inverted results
    plt.subplot(122)
    plt.imshow(iter_vp[-1], cmap=cmap)
    plt.title("Inverted Model")

    # Save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    
    # If plot is True, display the plot
    if show:
        plt.show()
    else:
        # Close the plot
        plt.close()

def animate_inversion_process(iter_vp, vmin=None,vmax=None, save_path="", fps=10, interval=150):
    """
    Create an animation of the inversion process.

    Args:
        iter_vp (list or np.array): List of inverted results over iterations.
        vp_true (np.array): The true velocity model for color scaling.
        save_path (str): The path where the animation will be saved.
        fps (int): Frames per second for the animation. Default is 10.

    Returns:
        None
    """
    # Set up the figure for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin = vmin if vmin is not None else iter_vp.min()
    vmax = vmax if vmax is not None else iter_vp.max()
    cax = ax.imshow(iter_vp[0], aspect='equal', cmap='jet_r', vmin=vmin, vmax=vmax)
    ax.set_title('Inversion Process Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')

    # Create a horizontal colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', fraction=0.046, pad=0.2)
    cbar.set_label('Velocity (m/s)')

    # Adjust the layout to minimize white space
    plt.subplots_adjust(top=0.85, bottom=0.2, left=0.1, right=0.9)

    # Initialization function
    def init():
        cax.set_array(iter_vp[0])  # Use the 2D array directly
        return cax,

    # Animation function
    def animate(i):
        cax.set_array(iter_vp[i])  # Update with the i-th iteration directly
        return cax,

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(iter_vp), interval=interval, blit=True)

    # Save the animation as a video file (e.g., GIF format)
    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)

    plt.close(fig)  # Prevents static display of the last frame
