import torch
import torch.nn as nn
import torchvision.models as models

def YourModelClass(num_classes=10):
    """
    Returns the EfficientNet B5 model with modifying the final 
    classification layer to match the number of classes.
    """
    model = models.efficientnet_b5(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
