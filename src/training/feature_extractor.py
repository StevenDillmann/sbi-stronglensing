import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet34


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64, pretrained=False):
        super(ResNetFeatureExtractor, self).__init__()
        
        # Load ResNet34 with configurable pretrained weights
        self.resnet = resnet34(pretrained=pretrained)
        
        # Modify first conv layer to accept single channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Use the stored number of features for the FC layer
        self.fc = nn.Linear(512, output_dim)

        
    def forward(self, x):
        # Ensure input is float and normalized
        x = x.float()
        if x.max() > 1:
            x = x / 255.0
            
        # Extract features through ResNet
        x = self.features(x)
        
        # Flatten 
        x = x.view(x.size(0), -1)
        
        # Final fully connected layer
        x = self.fc(x)
        
        return x

def get_feature_extractor(output_dim=64, pretrained=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Creates and returns a ResNet34-based feature extractor for 180x180 images.
    
    Args:
        output_dim (int): Dimension of the output feature vector
        device (str): Device to put the model on ('cuda' or 'cpu')
    
    Returns:
        model: The feature extractor model
    """
    model = ResNetFeatureExtractor(output_dim=output_dim, pretrained=pretrained)
    # model = model.to(device)
    return model
