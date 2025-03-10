import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
from sbi.inference import NPE
from src.training.feature_extractor import get_feature_extractor
from src.training.posterior_estimator import get_posterior_estimator


def get_npe_model(feature_dim=64,
              pretrained=False,
              density_estimator="maf",
              z_score_theta=None,
              z_score_x=None,
              hidden_features=50,
              num_transforms=5,
              num_components=1,
              num_bins=10,
              device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Trains a Neural Posterior Estimation model using a feature extractor and density estimator.
    
    Args:
        theta_train: Training parameters (N x param_dim tensor)
        x_train: Training data (N x C x H x W tensor)
        feature_dim: Dimension of extracted features
        batch_size: Batch size for training
        density_estimator: Type of density estimator ('maf', 'nsf', or 'mdn')
        hidden_features: Number of hidden features in density estimator
        num_transforms: Number of transforms in flow-based models
        num_components: Number of components for MDN
        num_bins: Number of bins for NSF
        device: Device to train on
    """
    
    # Get feature extractor
    embedding_net = get_feature_extractor(output_dim=feature_dim, pretrained=pretrained)
    
    # Get density estimator with feature extractor as embedding network
    density_net = get_posterior_estimator(
        density_estimator=density_estimator,
        z_score_theta=None,
        z_score_x=None,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_components=num_components,
        num_bins=num_bins,
        embedding_net=embedding_net,
    )

    # Build inference object
    inference = NPE(density_estimator=density_net)
   

    return inference



def train_npe_model(inference, theta, x, 
              batch_size = 200,
              learning_rate = 5e-4,
              validation_fraction = 0.1,
              stop_after_epochs = 20,
              max_num_epochs = 2**31 - 1,
              resume_training = False,
              retrain_from_scratch = False,
              show_train_summary = False
              ):
    
    # Append simulations to inference object
    inference.append_simulations(theta, x)
    
    # Train
    estimator = inference.train(training_batch_size=batch_size,
                    learning_rate=learning_rate,
                    validation_fraction=validation_fraction,
                    stop_after_epochs=stop_after_epochs,
                    max_num_epochs=max_num_epochs,
                    resume_training=resume_training,
                    retrain_from_scratch=retrain_from_scratch,
                    show_train_summary=show_train_summary)
    
    # Build posterior
    posterior = inference.build_posterior(
        density_estimator=estimator
    )

    
    return inference, estimator, posterior
