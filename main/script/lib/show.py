# display number given a single flatten matrix

def show(matrix: torch.Tensor):
    """Display a number given a single flatten matrix

    Args:
        matrix (torch.Tensor): 1-D array of 784 elements representing a 28x28 image

    Returns:
        plt.imshow object: Display the image
    """
    return plt.imshow(matrix.reshape(28, 28), cmap="gray")