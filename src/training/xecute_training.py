import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.data_loader import load_data
from src.training.feature_extractor import get_feature_extractor
from src.training.posterior_estimator import get_posterior_estimator
from src.training.sbi_trainer import get_npe_model, train_npe_model

import yaml
from datetime import datetime
import os

# full script to execute training

def execute_training(config_path: str):

    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # load data
    theta, x = load_data(**config['data_loader'])

    # get NPE model 
    npe_model = get_npe_model(**config['feature_extractor'], **config['posterior_estimator'])

    # train NPE model
    inference, estimator, posterior = train_npe_model(npe_model, theta, x, **config['training'])

    # save NPE model
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #create directory for saving models
    os.makedirs(config['model_directory'], exist_ok=True)
    #save model
    torch.save(npe_model, os.path.join(config['model_save_path'], f"npe_model_{date_time}.pth"))

    


