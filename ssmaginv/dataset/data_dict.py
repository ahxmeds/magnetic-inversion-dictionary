import numpy as np

def save_npz_true_data(filepath, x_true, data_true, data_true_noise):
    """
    Save magnetic inversion data to a standardized .npz format.

    Parameters:
        filepath (str): Path to save the .npz file.
        x_true (ndarray): Ground truth model.
        data_true (ndarray): Forward model output without noise.
        data_true_noise (ndarray): Forward model output with added noise.
    """
    save_dict = {
        "x_true": x_true,
        "data_true": data_true,
        "data_true_noise": data_true_noise,
    }
    np.savez(filepath, **save_dict)


def load_npz_true_data(filepath):
    """
    Load magnetic inversion data from a standardized .npz file.

    Parameters:
        filepath (str): Path to the .npz file.

    Returns:
        tuple: (x_true, data_true, data_true_noise)
    """
    d = np.load(filepath)
    return d["x_true"], d["data_true"], d["data_true_noise"]

def load_npz_pred_data(filepath):
    """
    Load predicted data from a standardized .npz file.

    Parameters:
        filepath (str): Path to the .npz file.

    Returns:
        tuple: (x_pred, data_pred)
    """
    d = np.load(filepath)
    pred_dict = {key: d[key] for key in d.files}
    x_true, x = pred_dict['x_true'], pred_dict['x']
    data_true, data = pred_dict['data_true'], pred_dict['data'] 
    return x_true, x, data_true, data