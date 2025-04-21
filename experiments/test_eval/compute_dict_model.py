import torch
from collections import OrderedDict
import os

import ssmaginv.config as config
import ssmaginv.solvers.variational as l2
from ssmaginv.magnetics import Magnetics
from ssmaginv.solvers.dictionary import SingleDictRecovery, ScaleSpaceRecovery
    
def get_inference_model_fn(model_type=SingleDictRecovery, saved_weights_path=None, device="cpu"):
    """
    Get the inference model function for a given dictionary model type.

    Parameters:
        model_type (callable): The dictionary model type to use (e.g., SingleDictRecovery).
        saved_weights (str): Path to the saved model weights.

    Returns:
        callable: The inference model function.
    """
    model_cfg = config.get_default_magnetics_config(device=device)
    model_cfg["device"] = device
    forMod = Magnetics(**model_cfg)  # Forward model operator    
    model = model_type(forMod, device=device) 

    
    results_dir = config.RESULTS_DIR
    weight_path = os.path.join(results_dir, "WEIGHTS", saved_weights_path)
    
    # load model weights
    state_dict = torch.load(weight_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Strip off 'module.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    # Turn off gradient tracking
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    def inference(x_true, data_true):
        """
        Perform inference using the loaded model.

        Parameters:
            x_true (torch.Tensor): The true mag sus model tensor.
            data_true (torch.Tensor): The true mag forward data tensor.

        Returns:
            tuple: The predicted model and forward data.
        """
        z_latent, x_pred, data_pred = model(data_true)
        return x_pred, data_pred
    
    return inference

if __name__ == "__main__":
    # Example usage
    model_type = SingleDictRecovery
    saved_weights_path = "path/to/saved/weights.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inference_model = get_inference_model_fn(model_type, saved_weights_path, device)
    
    # Assuming x_true and data_true are defined
    x_true = torch.randn(1, 1, 64, 64, 64).to(device)  # Example tensor
    data_true = torch.randn(1, 1, 64, 64).to(device)  # Example tensor
    
    x_pred, data_pred = inference_model(x_true, data_true)