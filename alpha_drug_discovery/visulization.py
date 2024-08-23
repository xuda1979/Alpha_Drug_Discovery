import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, title="Data Visualization", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plot data using matplotlib.

    Parameters:
    data (list or np.ndarray): Data to plot.
    title (str): Title of the plot.
    xlabel (str): Label for the X-axis.
    ylabel (str): Label for the Y-axis.

    Returns:
    None
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("Data should be a list or numpy array.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
