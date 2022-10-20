import torch
from torch import optim , nn
from torchvision import models , transforms


class FeatureExtractor(nn.Module) :
    def __init__(self , model) :
        super(FeatureExtractor , self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self , x) :
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        print(out.shape)
        return out

    # Initialize the model


#model = models.vgg16(pretrained=True)
#new_model = FeatureExtractor(model)
