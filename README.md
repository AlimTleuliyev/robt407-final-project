# README for Plant Seedlings Classification Project

## Project Overview and Context
Welcome to the repository for the Plant Seedlings Classification project, our final project for the ROBT407 course. This project is centered around the Kaggle Plant Seedlings Classification competition, where the primary goal is to effectively distinguish between different species of plant seedlings. Achieving this can lead to better crop yields and enhance environmental stewardship.

For more details about the Plant Seedlings Classification competition, you can visit the [Kaggle Competition Page](https://www.kaggle.com/competitions/plant-seedlings-classification).

## Repository Structure

### 1. `custom_datasets.py`
This file includes two key classes:
- `TestDataset`: Facilitates handling test image datasets without labels. It is designed for loading images, applying optional transformations, and returning both the image and its filename. This is particularly useful for inference tasks where identification is key, but labels are not needed.
- `WrapperDataset`: Serves as a wrapper to apply varied transformations to different subsets of a dataset, such as training and validation splits. This flexibility is essential for cases requiring distinct data processing strategies.

### 2. `custom_models.py`
This file encompasses multiple classes and functions for model creation:
- `MLP Class`: A versatile Multilayer Perceptron with configurable layers, batch normalization, and ReLU activation. Suitable for various classification tasks.
- `CustomCNN Class`: A custom Convolutional Neural Network adaptable for different image classification needs.
- Model Creation Functions: These include `create_mlp_model`, `create_cnn_model`, and functions for creating modified versions of popular models like EfficientNet, ResNet, MobileNet, and GoogleNet.
- `get_model Function`: A utility function for selecting and instantiating models based on specified configurations.
- `count_parameters Function`: Aids in determining the number of trainable parameters in a model.

### 3. `utils.py`
This file is crucial for various operational aspects:
- Data Preparation: Setting up data loaders with appropriate image transformations.
- Model Evaluation: Computing key metrics to assess model performance.
- Training Loop: Managing the training process, including optimization and model saving.
- Model Testing and Submission File Creation: For generating predictions and preparing submission files.
- Ensemble Prediction: Enhances prediction accuracy by combining multiple model outputs.
- Performance Tracking and Visualization: Includes functionality for saving and visualizing training metrics.

### 4. `train.ipynb`
The main Jupyter notebook that utilizes the defined functions and classes for training models on the dataset.

### 5. `kaggle-run-train.ipynb`
A variation of `train.ipynb`, this notebook contains logs from the model training executed on Kaggle.

### 6. `experiments` Folder
Contains subfolders for each model, with each subfolder housing training graphs, training metrics CSVs, and submission files.

### 7. `ensemble` Folder
Holds CSV files from various voting classifiers, each generated using a different combination of models.

### 8. `results` Folder
Includes CSV files detailing the evaluated metrics of top models, their parameters, test results from submissions, and ensemble model test results.

Continuing from the overview and context, the next section of your README can detail the specific methodologies and model configurations you employed in your project. Here's how you can structure and present this information:

---

## Methodology and Model Configurations

### Dataset Split
The training dataset, consisting of 4750 images, was split into a training set and a validation set. The validation set comprised 15% of the total data, ensuring a robust evaluation of the models trained.

### Model Architectures
In this project, a variety of models were trained, including custom MLPs (Multilayer Perceptrons), custom CNNs (Convolutional Neural Networks), and several popular pre-trained models like EfficientNet, ResNet, GoogleNet, and MobileNet. The custom model configurations were as follows:

#### Custom MLP Configurations
The MLP models varied in complexity and size, defined by the number of neurons in each layer:
- `mlp_n`: [512, 256]
- `mlp_s`: [1024, 512, 256]
- `mlp_m`: [2048, 1024, 512, 256]
- `mlp_l`: [4096, 2048, 1024, 512, 256]
- `mlp_xl`: [8192, 4096, 2048, 1024, 512, 256]

Each configuration represents a different MLP model, with the number and size of fully connected layers varying.

#### Custom CNN Configurations
The CNN models were defined by their convolutional and fully connected layers:
- `cnn_small`: Conv Layers - [(16, 3, 1, 1), (32, 3, 1, 1)], FC Layers - [64]
- `cnn_medium`: Conv Layers - [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)], FC Layers - [128, 64]
- `cnn_large`: Conv Layers - [(64, 6, 1, 1), (128, 3, 1, 1), (256, 3, 1, 1), (512, 3, 1, 1)], FC Layers - [256, 128, 64]
- `cnn_extra_large`: Conv Layers - [(128, 6, 1, 1), (256, 6, 1, 1), (512, 3, 1, 1), (1024, 3, 1, 1), (2048, 3, 1, 1)], FC Layers - [512, 256, 128, 64]

The CNN configurations varied from small to extra-large, with an increasing number of convolutional and fully connected layers.

## Training Process

The training of the models was carried out through a rigorous and methodical approach, with the primary objective of optimizing the F1 score on the validation set. Below are the key components of the training process:

### Training Setup
- **Model Initialization**: Each model, whether it's a custom MLP, CNN, or a pre-trained model, is initialized with a specific configuration.
- **Data Loaders**: Data loaders for both training and validation datasets are prepared, ensuring the efficient feeding of data to the models during training.
- **Loss Function**: Cross-Entropy Loss is used as the criterion for training, catering to the multi-class classification nature of the problem.
- **Optimizer**: Adam optimizer is employed for updating model parameters, with an initial learning rate `lr0` and weight decay.
- **Learning Rate Scheduler**: A Cosine Annealing Warm Restarts scheduler is used to adjust the learning rate in a cosine manner from `lr0` to `lrf` over `num_epochs` epochs.

### Training Loop (`train_loop`)
- **Device Compatibility**: The training is conducted on the specified device (CPU or GPU).
- **Autocast & GradScaler**: For mixed precision training, autocast and GradScaler are utilized to enhance training speed and reduce memory usage.
- **Early Stopping**: Patience-based early stopping is implemented to halt training if the validation F1 score does not improve for a specified number of epochs (default: 20).
- **Metrics Tracking**: Key metrics like loss, accuracy, precision, recall, and F1 score are tracked for both training and validation sets.
- **Model Saving**: The best model based on validation F1 score is saved during training. The last model state is also saved at the end of training.
- **Visualization**: Training results are visualized in plots showing Loss, Accuracy, Precision, Recall, and F1 Score against epochs.

### Training Execution (`train`)
- **Model and Parameter Configuration**: Each model is configured with its specific parameters, including learning rates, weight decay, batch size, image size, and validation set size.
- **Parameter Counting**: The total number of parameters in each model is displayed, providing insight into the model's complexity.
- **Training Invocation**: The `train_loop` function is called with all necessary parameters to commence the training process.

### Key Highlights
- **Mixed Precision Training**: By using autocast and GradScaler, the training process is optimized for performance efficiency.
- **Adaptive Learning Rate**: The Cosine Annealing Warm Restarts scheduler ensures a dynamic adjustment of the learning rate, potentially leading to better convergence.
- **Comprehensive Metrics Tracking**: The tracking of various performance metrics helps in thoroughly understanding the model's strengths and weaknesses.


### Results
The performance of each model was rigorously evaluated based on F1 micro scores on the training, validation, and test sets. The test set lacked labels, and the performance metrics were derived from the submission results on Kaggle. Below is a table summarizing the performance of each model:

| Model             | Params       | Image Size | Train F1 | Val F1  | Test F1 |
|-------------------|--------------|------------|----------|---------|---------|
| efficientnet_b2   | 7,717,902    | 256x256    | 0.99331  | 0.97335 | 0.97732 |
| efficientnet_b0   | 4,022,920    | 256x256    | 0.99207  | 0.98457 | 0.97229 |
| efficientnet_b1   | 6,528,556    | 256x256    | 0.99579  | 0.98177 | 0.97103 |
| efficientnet_b3   | 10,714,676   | 256x256    | 0.99827  | 0.98177 | 0.96977 |
| efficientnet_b4   | 17,570,132   | 256x256    | 0.99331  | 0.96774 | 0.96851 |
| resnet50          | 23,532,620   | 256x256    | 0.99802  | 0.98597 | 0.96851 |
| googlenet         | 5,612,204    | 256x256    | 0.99876  | 0.97475 | 0.96599 |
| resnet18          | 11,182,668   | 256x256    | 0.99876  | 0.97616 | 0.96599 |
| resnet101         | 42,524,748   | 256x256    | 0.99851  | 0.96774 | 0.96347 |
| resnet34          | 21,290,828   | 256x256    | 0.99331  | 0.97896 | 0.95969 |
| mobilenet_v2      | 2,239,244    | 256x256    | 0.99777  | 0.97195 | 0.95843 |
| cnn_large         | 6,318,860    | 128x128    | 0.94823  | 0.92286 | 0.92695 |
| cnn_extra_large   | 30,346,060   | 128x128    | 0.89200  | 0.91445 | 0.91561 |
| cnn_medium        | 3,789,260    | 128x128    | 0.92990  | 0.89902 | 0.90176 |
| cnn_small         | 1,974,156    | 128x128    | 0.92395  | 0.82609 | 0.85894 |
| mlp_l             | 212,494,604  | 128x128    | 0.95516  | 0.72791 | 0.73047 |
| mlp_s             | 50,995,468   | 128x128    | 0.87590  | 0.70126 | 0.7034  |
| mlp_m             | 103,430,412  | 128x128    | 0.76170  | 0.68022 | 0.69395 |
| mlp_xl            | 447,400,204  | 128x128    | 0.88382  | 0.71809 | 0.69269 |
| mlp_n             | 25,302,284   | 128x128    | 0.81719  | 0.68443 | 0.67506 |

## Voting Classifier Implementation

The project employed a sophisticated voting classifier to enhance prediction accuracy on the test dataset. This method involved aggregating predictions from multiple models and selecting the most commonly predicted class for each instance. Here's a brief overview of the process:

1. **Model Selection**: A set of pre-trained and custom models were chosen based on their individual performance. These models include variations of MLPs, CNNs, and well-known architectures like EfficientNet, ResNet, GoogleNet, and MobileNet.

2. **Ensemble Approach**: Each selected model contributes to the final decision. The ensemble method ensures that the biases or weaknesses of individual models are mitigated, leading to a more reliable and robust prediction.

3. **Prediction Aggregation**: For each test image, predictions from all the models are collected. The final class prediction for each image is determined by a majority vote among the models.

4. **Result Compilation**: The aggregated predictions are compiled into a structured format, typically a CSV file, mapping each test image to its predicted class.

5. **Performance Evaluation**: The effectiveness of the voting classifier is assessed by comparing its predictions against the actual labels (where available) or through the performance metrics obtained from competition submissions.

This voting classifier is a testament to the project's innovative approach, leveraging the strengths of multiple models to achieve high accuracy in plant seedlings classification.

### Results
The voting classifier was implemented using different combinations of models, with each combination producing a different set of predictions. The table below summarizes the performance of each voting classifier:

| Models                                      | Image Size | Test F1 |
|---------------------------------------------|------------|---------|
| efficientnet_b0+b1+b2                       | 256x256    | 0.98236 |
| efficientnet_b0+b1+b2+b3                    | 256x257    | 0.97984 |
| efficientnet_b0+b1+b2 + resnet18+34+50+101  | 256x258    | 0.97732 |
| resnet18+34+50+101                          | 256x259    | 0.97481 |

Voting classifiers often outperform single models, especially in complex tasks like plant seedlings classification. By combining the strengths of multiple models, they reduce the impact of individual biases and errors, leading to more accurate and robust predictions. This ensemble approach harnesses the collective insights of various models, making it a powerful strategy in advanced machine learning applications.


## Conclusion

In conclusion, this project represents a comprehensive effort in the realm of machine learning for plant seedlings classification. By employing a mix of custom and pre-trained models, and further enhancing prediction accuracy with a voting classifier, the project successfully navigates the complexities of distinguishing between various plant species. This work not only demonstrates the practical application of advanced machine learning techniques but also contributes to the broader field of agricultural technology, potentially aiding in better crop management and environmental conservation.

## Team

This project was brought to life through the collaborative efforts of:

- Maiya Goloburda - [maiya.goloburda@nu.edu.kz](mailto:maiya.goloburda@nu.edu.kz)
- Batyr Bodaubay - [batyr.bodaubay@nu.edu.kz](mailto:batyr.bodaubay@nu.edu.kz)
- Alim Tleuliyev - [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)

Their individual contributions have been integral to the success of this project.