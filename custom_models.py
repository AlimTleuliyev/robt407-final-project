import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MLP, self).__init__()

        layers = [nn.Linear(input_size, hidden_layers[0]), nn.BatchNorm1d(hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class CustomCNN(nn.Module):
    def __init__(self, input_size, conv_layers, fc_layers, num_classes):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.convs = nn.Sequential()
        in_channels = 3 # RGB
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.convs.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channels))
            self.convs.add_module(f'relu{i}', nn.ReLU(inplace=True))
            self.convs.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=3, stride=2))
            in_channels = out_channels

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fcs = nn.Sequential()
        fc_input_features = self._get_conv_output_size(conv_layers, input_size=input_size)
        for i, fc_output_features in enumerate(fc_layers):
            self.fcs.add_module(f'fc{i}', nn.Linear(fc_input_features, fc_output_features))
            self.fcs.add_module(f'relu_fc{i}', nn.ReLU(inplace=True))
            fc_input_features = fc_output_features

        # Final layer
        self.final_fc = nn.Linear(fc_input_features, num_classes)

    def _get_conv_output_size(self, conv_layers, input_size=32, pool_kernel_size=3, pool_stride=2):
        output_size = input_size
        for _, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            output_size = (output_size - kernel_size + 2 * padding) // stride + 1

            # Apply pooling reduction
            output_size = (output_size - pool_kernel_size) // pool_stride + 1

        return output_size * output_size * out_channels

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = self.fcs(x)
        x = self.final_fc(x)
        return x
    

def create_mlp_model(config, image_size, num_classes=12):
    hidden_layers, _ = config
    return MLP(image_size*image_size*3, hidden_layers, num_classes)

def create_cnn_model(config, image_size, num_classes=12):
    conv_layers, fc_layers, _ = config
    return CustomCNN(image_size, conv_layers, fc_layers, num_classes)

def create_efficientnet_model(model_name, num_classes=12):
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def create_resnet_model(model_name, num_classes=12):
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_mobile_net_model(model_name, num_classes=12):
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model

def create_googlenet_model(model_name, num_classes=12):
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def get_model(model_name, image_size, num_classes=12, config = None):
    if 'mlp' in model_name:
        return create_mlp_model(config, image_size, num_classes)
    elif 'cnn' in model_name:
        return create_cnn_model(config, image_size, num_classes)
    elif 'efficientnet' in model_name:
        return create_efficientnet_model(model_name, num_classes)
    elif 'resnet' in model_name:
        return create_resnet_model(model_name, num_classes)
    elif 'mobilenet' in model_name:
        return create_mobile_net_model(model_name, num_classes)
    elif 'googlenet' in model_name:
        return create_googlenet_model(model_name, num_classes)
    else:
        raise Exception('Model not found')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)