import numpy as np
from scipy import ndimage


def largest_connected_component(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labels, count = ndimage.label(mask, structure)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def keep_largest_components(prediction: np.ndarray, foreground_labels=(1, 2)) -> np.ndarray:
    output = prediction.copy()
    for label in foreground_labels:
        component = largest_connected_component(output == label)
        output[output == label] = 0
        output[component] = label
    return output
