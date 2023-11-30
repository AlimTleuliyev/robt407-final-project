import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import gc
from custom_models import *
from custom_datasets import TestDataset, WrapperDataset
    
def prepare_data_loaders(data_path, image_size=128, batch_size=32, val_size=0.15):
    """
    Prepare data loaders for training and validation.

    Args:
        data_path (str): Path to the dataset.
        image_size (int, optional): Size of the input images. Defaults to 128.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        val_size (float, optional): Proportion of the dataset to use for validation. Defaults to 0.15.

    Returns:
        train_loader (torch.utils.data.DataLoader): Data loader for training set.
        val_loader (torch.utils.data.DataLoader): Data loader for validation set.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for validation set, we don't need data augmentation
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_path)

    # split the dataset into train and validation sets
    train_size = int((1 - val_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # we pass full dataset, created subsets, and transforms to the created by us WrapperDataset
    train_dataset = WrapperDataset(full_dataset, train_subset.indices, train_transforms)
    val_dataset = WrapperDataset(full_dataset, val_subset.indices, val_transforms)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def get_metrics(y_true, y_pred):
    """
    Calculate various metrics for evaluating the performance of a classification model.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - accuracy (float): Accuracy score.
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1 (float): F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    return accuracy, precision, recall, f1

def evaluate_model(model, model_name, criterion, val_loader, device):
    """
    Evaluate the performance of a model on a validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        model_name (str): The name of the model.
        criterion (torch.nn.Module): The loss function used for evaluation.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to perform evaluation on.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    val_bar = tqdm(val_loader, desc=f'Evaluation [VALID|{model_name}]', total=len(val_loader))

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            
            # flatten images if using MLP
            if 'mlp' in model_name:
                images = images.view(images.size(0), -1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = val_loss / len(val_loader)
    accuracy, precision, recall, f1 = get_metrics(all_labels, all_preds)
    return avg_loss, accuracy, precision, recall, f1

def train_loop(model, model_name, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs, patience=20):
    """
    Trains the model for a specified number of epochs using the given data loaders and optimization parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        model_name (str): The name of the model.
        criterion: The loss function used for training.
        optimizer: The optimizer used for updating the model's parameters.
        scheduler: The learning rate scheduler.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        device: The device on which the model and data will be loaded.
        num_epochs (int): The number of epochs to train the model.
        patience (int, optional): The number of epochs to wait for improvement in validation F1 score before early stopping. Defaults to 20.
    
    Returns:
        None
    """
    os.makedirs(model_name, exist_ok=True)
    scaler = GradScaler()   # create a GradScaler for autocast
                            # autocast is used for mixed precision training 
                            # to speed up training and reduce memory usage
    best_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_labels = []
        all_preds = []
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN|{model_name}]', total=len(train_loader))

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            # flatten images if using MLP
            if 'mlp' in model_name:
                images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()

            # Use autocast for the forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Scale the loss and call backward
            scaler.step(optimizer)  # Optimizer step with scaler
            scaler.update()  # Update the scaler
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            train_accuracy = accuracy_score(all_labels, all_preds) * 100
            train_bar.set_postfix(loss=loss.item(), accuracy=f'{train_accuracy:.2f}%') # update progress bar

        # save training metrics
        train_avg_loss = train_loss / len(train_loader)
        train_accuracy, train_precision, train_recall, train_f1 = get_metrics(all_labels, all_preds)
        train_losses.append(train_avg_loss)
        train_accs.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}')

        # if using cuda clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Evaluate after each epoch and save
        val_avg_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, model_name, criterion, val_loader, device)
        val_losses.append(val_avg_loss)
        val_accs.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}')
        
        # check for best model and patience
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            print(f'Saving best model at epoch {epoch+1}...')
            torch.save(model.state_dict(), f'{model_name}/{model_name}_best_model.pt')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print('Early stopping...')
                break
        
        scheduler.step()   # step the scheduler
        
    # save last model
    print('Saving Last Model')
    torch.save(model.state_dict(), f'{model_name}/{model_name}_last_model.pt')

    # After the training loop, create a DataFrame from the collected metrics
    epochs = range(1, len(train_losses) + 1)

    metrics = {
        'Epoch': list(epochs),
        'Train Loss': train_losses,
        'Train Accuracy': train_accs,
        'Train Precision': train_precisions,
        'Train Recall': train_recalls,
        'Train F1': train_f1s,
        'Val Loss': val_losses,
        'Val Accuracy': val_accs,
        'Val Precision': val_precisions,
        'Val Recall': val_recalls,
        'Val F1': val_f1s
    }

    df = pd.DataFrame(metrics)

    # Save the DataFrame to a CSV file
    csv_file = f'{model_name}/{model_name}_training_metrics.csv'
    df.to_csv(csv_file, index=False)

    print(f'Metrics saved to {csv_file}')
    
    # Plot the training results
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}/{model_name}_training_results.png')
    plt.show()

def train(
        train_path,
        model,
        model_name,
        lr0,
        lrf,
        weight_decay,
        num_epochs,
        patience,
        batch_size,
        image_size,
        val_size,
        device
):  
    """
    Trains the model using the provided parameters.

    Args:
        train_path (str): The path to the training data.
        model (nn.Module): The model to be trained.
        model_name (str): The name of the model.
        lr0 (float): The initial learning rate.
        lrf (float): The final learning rate.
        weight_decay (float): The weight decay for the optimizer.
        num_epochs (int): The number of training epochs.
        patience (int): The number of epochs to wait for improvement in validation loss before early stopping.
        batch_size (int): The batch size for training.
        image_size (int): The size of the input images.
        val_size (float): The proportion of data to be used for validation.
        device (str): The device to be used for training (e.g., 'cpu', 'cuda').

    Returns:
        None
    """

    print('\n'*3)
    print(f'Starting training for {model_name}')
    print(f'Number of parameters: {count_parameters(model):,}')

    train_loader, val_loader = prepare_data_loaders(train_path, image_size, batch_size, val_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)

    # create a learning rate scheduler
    # in this case we set T_0 to the number of epochs
    # which means that learning rate will go from lr0 to lrf in num_epochs epochs
    # in a cosine manner

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs, T_mult=1, eta_min=lrf) 

    train_loop(model, model_name, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs, patience)

    print(f'Training complete for {model_name}')

def test_submit(model, model_name, image_size, test_path, class2id, device):
    """
    Test the model on the test dataset and create a submission file.

    Args:
        model (torch.nn.Module): The trained model.
        model_name (str): The name of the model.
        image_size (int): The size of the input images.
        test_path (str): The path to the test dataset.
        class2id (dict): A dictionary mapping class names to class IDs.
        device (torch.device): The device to run the model on.

    Returns:
        None
    """
    print('Creating submission file for', model_name)
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = TestDataset(test_path, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model.eval()
    all_preds = []
    all_image_names = []
    test_bar = tqdm(test_loader, desc=f'Test [TEST|{model_name}]', total=len(test_loader))

    with torch.no_grad():
        for images, image_names in test_bar:
            images = images.to(device)

            # flatten images if using MLP
            if 'mlp' in model_name:
                images = images.view(images.size(0), -1)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_image_names.extend(image_names)

    df = pd.DataFrame({'file': all_image_names, 'species': all_preds})
    
    # convert class IDs to class names
    id2class = {v: k for k, v in class2id.items()}
    df['species'] = df['species'].apply(lambda x: id2class[x])
    
    # save submission file
    df.to_csv(f'{model_name}/{model_name}_test_results.csv', index=False)
    print(f'Test results saved to {model_name}/{model_name}_test_results.csv')

def predict_voting(model_names, image_size, test_path, class2id, device):
    """
    Predicts the species of images in the test dataset using a voting ensemble of multiple models.

    Args:
        model_names (list): List of model names to use for voting.
        image_size (int): Size of the input images.
        test_path (str): Path to the test dataset.
        class2id (dict): Dictionary mapping class names to class IDs.
        device (torch.device): Device to use for prediction.

    Returns:
        None
    """
    print('Creating submission file for', model_names)
    print('Votes from:')
    for model_name in model_names:
        print('\t' + model_name)

    # load models and put them on the device
    models = [get_model(model_name, image_size, len(class2id)) for model_name in model_names]
    models = [model.to(device) for model in models]
    
    # load best models and put them in evaluation mode
    for model_name, model in zip(model_names, models):
        model.load_state_dict(torch.load(f'{model_name}/{model_name}_best_model.pt'))
        model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # use of custom dataset called TestDataset from custom_datasets.py
    test_dataset = TestDataset(test_path, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    all_preds = []
    all_image_names = []
    test_bar = tqdm(test_loader, desc=f'Test [TEST|VOTING]', total=len(test_loader))

    with torch.no_grad():
        for images, image_names in test_bar:
            images = images.to(device)
            logits = torch.zeros(images.size(0), len(class2id)).to(device)
            
            # sum logits from all models
            for model in models:
                outputs = model(images)
                logits += outputs
            
            # get the class with the highest logit
            _, predicted = torch.max(logits.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_image_names.extend(image_names)
    
    df = pd.DataFrame({'file': all_image_names, 'species': all_preds})
    
    # convert class IDs to class names
    id2class = {v: k for k, v in class2id.items()}
    df['species'] = df['species'].apply(lambda x: id2class[x])
    
    # save submission file
    model_names_str = '_'.join(model_names)
    os.makedirs(model_names_str, exist_ok=True)
    df.to_csv(f'{model_names_str}/{model_names_str}_test_results.csv', index=False)