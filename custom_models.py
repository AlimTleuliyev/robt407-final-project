import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A Multilayer Perceptron (MLP) class implemented using PyTorch.

    This class creates a fully connected neural network with a specified number of hidden layers and neurons in each layer. 
    Batch normalization and ReLU activation are applied after each hidden layer.

    Parameters:
    - input_size (int): The size of the input features.
    - hidden_layers (list of int): A list where each element represents the number of neurons in the corresponding hidden layer.
    - num_classes (int): The number of output neurons/classes for the MLP.

    The network architecture is constructed dynamically based on the `hidden_layers` list. Each hidden layer is followed by batch normalization and ReLU activation. The final layer is a linear layer that outputs logits for classification tasks.

    Example usage:
        # Create an MLP with an input size of 100, two hidden layers with 50 and 30 neurons respectively, and 10 output classes
        mlp = MLP(input_size=100, hidden_layers=[50, 30], num_classes=10)

    Methods:
    - forward(x): Defines the forward pass of the MLP. Takes an input tensor `x` and returns the MLP's output.
    """
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
    """
    A custom Convolutional Neural Network (CNN) class implemented using PyTorch.

    This class constructs a CNN based on specified convolutional and fully connected (FC) layer configurations. It supports the addition of multiple convolutional layers with batch normalization and ReLU activation, followed by a series of fully connected layers ending in an output layer for classification.

    Parameters:
    - input_size (int): The size (width/height) of the square input image. Assumes input images are square and have 3 channels (RGB).
    - conv_layers (list of tuples): Each tuple in the list should specify the parameters for a convolutional layer in the format (out_channels, kernel_size, stride, padding).
    - fc_layers (list of int): A list where each element represents the number of neurons in the corresponding fully connected layer.
    - num_classes (int): The number of output neurons/classes for the CNN.

    The network initializes with a series of convolutional layers as defined in `conv_layers`, each followed by batch normalization, ReLU activation, and max pooling. After the convolutional layers, the network flattens the output and passes it through the fully connected layers as defined in `fc_layers`, ending with a linear layer for classification.

    Example usage:
        # Define a CNN with input size 32x32, two convolutional layers, and two fully connected layers
        conv_config = [(16, 3, 1, 1), (32, 3, 1, 1)] # (out_channels, kernel_size, stride, padding)
        fc_config = [120, 60]
        custom_cnn = CustomCNN(input_size=32, conv_layers=conv_config, fc_layers=fc_config, num_classes=10)

    Methods:
    - forward(x): Defines the forward pass of the CNN. Takes an input tensor `x` and returns the CNN's output.
    - _get_conv_output_size(...): Helper method to calculate the size of the flattened output after the convolutional layers.
    """
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
    """
    Create a multi-layer perceptron (MLP) model.

    Args:
        config (tuple): A tuple containing the hidden layer sizes of the MLP.
        image_size (int): The size of the input image (assumed to be square).
        num_classes (int, optional): The number of output classes. Defaults to 12.

    Returns:
        MLP: The created MLP model.
    """
    hidden_layers, _ = config
    return MLP(image_size*image_size*3, hidden_layers, num_classes)

def create_cnn_model(config, image_size, num_classes=12):
    """
    Create a CNN model based on the given configuration.

    Args:
        config (tuple): A tuple containing the configuration of the model.
        image_size (int): The size of the input images.
        num_classes (int, optional): The number of output classes. Defaults to 12.

    Returns:
        CustomCNN: The created CNN model.
    """
    conv_layers, fc_layers, _ = config
    return CustomCNN(image_size, conv_layers, fc_layers, num_classes)

def create_efficientnet_model(model_name, num_classes=12):
    """
    Create an EfficientNet model with a custom classifier.

    Args:
        model_name (str): Name of the EfficientNet model to load.
        num_classes (int, optional): Number of output classes. Defaults to 12.

    Returns:
        torch.nn.Module: The created EfficientNet model with the custom classifier.
    """
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def create_resnet_model(model_name, num_classes=12):
    """
    Create a ResNet model with a specified number of classes.

    Args:
        model_name (str): The name of the ResNet model to load.
        num_classes (int, optional): The number of output classes. Defaults to 12.

    Returns:
        torch.nn.Module: The created ResNet model.
    """
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_mobile_net_model(model_name, num_classes=12):
    """
    Create a MobileNet model with a custom classifier.

    Args:
        model_name (str): Name of the MobileNet model architecture.
        num_classes (int, optional): Number of output classes. Defaults to 12.

    Returns:
        torch.nn.Module: MobileNet model with a custom classifier.
    """
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model

def create_googlenet_model(model_name, num_classes=12):
    """
    Create a GoogleNet model with a modified fully connected layer.

    Args:
        model_name (str): The name of the GoogleNet model to load.
        num_classes (int, optional): The number of output classes. Defaults to 12.

    Returns:
        torch.nn.Module: The modified GoogleNet model.
    """
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def get_model(model_name, image_size, num_classes=12, config=None):
    """
    Returns a model based on the given model_name.

    Args:
        model_name (str): The name of the model to be created.
        image_size (int): The size of the input images.
        num_classes (int, optional): The number of output classes. Defaults to 12.
        config (dict, optional): Configuration parameters for the model. Defaults to None.

    Returns:
        model: The created model.

    Raises:
        Exception: If the model_name is not found.
    """
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
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)