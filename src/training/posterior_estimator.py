import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# write script to define the posterior estimator

def get_posterior_estimator(density_estimator="maf", z_score_theta=None, z_score_x=None, hidden_features=128, num_transforms=5, num_components=1, num_bins=10, embedding_net=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Creates and returns a Neural Posterior Estimation (NPE) model.
    
    Args:
        density_estimator (str): Type of density estimator to use ('maf', 'nsf', or 'mdn')
        hidden_features (int): Number of hidden features in each layer
        num_transforms (int): Number of transforms in the flow-based model
        embedding_net: Neural network for embedding the images (e.g. ResNet feature extractor)
        
    Returns:
        model: The posterior estimator model
    """
    model = posterior_nn(
            model=density_estimator,
            z_score_theta=z_score_theta,
            z_score_x=z_score_x,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_components=num_components,
            num_bins=num_bins,
            embedding_net=embedding_net,
            device = device
        )
    
    return model