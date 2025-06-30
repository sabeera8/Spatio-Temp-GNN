import numpy as np
import torch
import optuna
from model import PrebuiltGraphNN
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from preprocessing.helper_funcs import save_metrics_to_json, plot_training_curves, plot_confusion_matrix
from preprocessing.helper_funcs import plot_roc_curve, plot_precision_recall_curve, feature_importance_analysis, plot_feature_importance



train_loader = torch.load('train_loader.pt')
test_loader = torch.load('test_loader.pt')

data = np.load('patient_data.npy', allow_pickle=True)
mni_coordinates = np.load('mni_coordinates.npy', allow_pickle=True)
eeg_channels = np.load('eeg_channels.npy', allow_pickle=True)
region_names = np.load('region_names.npy', allow_pickle=True)


def evaluate_model(train_loader, test_loader, best_params, epochs, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Best parameters: {best_params}")
    
    # Extract parameters
    learning_rate = best_params['learning_rate']
    hidden_dim = best_params['hidden_dim']
    model_type = best_params['model_type']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    
    # Get input dimensions from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]
    num_classes = 2  # As specified in your code
    
    # Initialize model with best parameters
    model = PrebuiltGraphNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        model_type=model_type,
        num_layers=num_layers
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training metrics tracking
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            
            epoch_loss += loss.item()
            batch = batch.cpu()
            
        train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        val_accuracy = validate(model, test_loader, device)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.4f}')
    
    # Final evaluation
    test_metrics = final_evaluation(model, test_loader, device)
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, val_accuracies)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'])
    
    # Plot ROC curve
    plot_roc_curve(test_metrics['y_true'], test_metrics['y_scores'])
    
    # Plot precision-recall curve
    plot_precision_recall_curve(test_metrics['y_true'], test_metrics['y_scores'])
    
    return model, test_metrics

def validate(model, data_loader, device):
    """Validate the model on the given data loader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            batch = batch.cpu()
            
    return correct / total

def final_evaluation(model, test_loader, device):
    """Perform comprehensive evaluation on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_scores.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
            
            batch = batch.cpu()
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': report['accuracy'],
        'f1_score': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'y_true': all_labels,
        'y_pred': all_preds,
        'y_scores': all_scores
    }

def perform_kfold_cross_validation(dataset, model_class, best_params, epochs, 
                                    output_dir='results/seeg-mri', seed=42):
    """
    Perform 10-fold cross-validation on the dataset
    
    Parameters:
    -----------
    dataset : torch_geometric.Dataset
        The input dataset
    model_class : torch.nn.Module
        The graph neural network model class
    best_params : dict
        Best hyperparameters for the model
    epochs : int
        Number of training epochs
    output_dir : str, optional
        Directory to save results
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Averaged results across all folds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare labels for stratification
    labels = [data.y.item() for data in dataset]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    # Tracking metrics for each fold
    fold_metrics = {
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    # Tracking detailed results
    detailed_results = {}
    
    # Perform 10-fold cross-validation
    for fold, (train_indices, test_indices) in enumerate(skf.split(dataset, labels), 1):
        print(f"\n{'='*20} Fold {fold} {'='*20}")
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Split dataset
        train_dataset = [dataset[i] for i in train_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Train and evaluate model
        model, test_metrics = evaluate_model(
            train_loader, 
            test_loader, 
            best_params, 
            epochs, 
            device=device
        )
        
        # Store fold-specific metrics
        detailed_results[f'fold_{fold}'] = test_metrics
        
        # Store performance metrics
        fold_metrics['accuracy'].append(test_metrics['accuracy'])
        fold_metrics['f1_score'].append(test_metrics['f1_score'])
        fold_metrics['precision'].append(test_metrics['precision'])
        fold_metrics['recall'].append(test_metrics['recall'])
        
        # Calculate ROC and PR AUC if not available
        if 'roc_auc' not in test_metrics:
            fpr, tpr, _ = roc_curve(test_metrics['y_true'], test_metrics['y_scores'])
            fold_metrics['roc_auc'].append(auc(fpr, tpr))
        else:
            fold_metrics['roc_auc'].append(test_metrics['roc_auc'])
        
        if 'pr_auc' not in test_metrics:
            precision, recall, _ = precision_recall_curve(test_metrics['y_true'], test_metrics['y_scores'])
            fold_metrics['pr_auc'].append(auc(recall, precision))
        else:
            fold_metrics['pr_auc'].append(test_metrics['pr_auc'])
        
        # Save fold-specific results
        fold_results_path = os.path.join(fold_output_dir, 'fold_results.json')
        save_metrics_to_json(test_metrics, fold_results_path)
        
        # Save training curves, confusion matrix, ROC, and PR curves
        plt.close('all')  # Close any existing plots
    
    # Calculate averaged metrics
    averaged_metrics = {
        metric: np.mean(values) for metric, values in fold_metrics.items()
    }
    
    # Add standard deviation
    for metric, values in fold_metrics.items():
        averaged_metrics[f'{metric}_std'] = np.std(values)
    
    # Save averaged metrics
    averaged_results_path = os.path.join(output_dir, 'averaged_results.json')
    save_metrics_to_json(averaged_metrics, averaged_results_path)
    
    # Visualize cross-validation results
    plot_cross_validation_results(fold_metrics, output_dir)
    
    return {
        'averaged_metrics': averaged_metrics,
        'detailed_results': detailed_results
    }

def plot_cross_validation_results(fold_metrics, output_dir):
    """
    Visualize cross-validation results
    
    Parameters:
    -----------
    fold_metrics : dict
        Metrics for each fold
    output_dir : str
        Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Metrics to plot
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'pr_auc']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = fold_metrics[metric]
        plt.bar(range(1, 11), values)
        plt.title(f'{metric.replace("_", " ").title()} per Fold')
        plt.xlabel('Fold')
        plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_metrics.png'), dpi=300)
    plt.close()



if __name__ == "__main__":
    best_params = {} # Replace with actual best parameters from hyperparameter tuning
    epochs = 100
    model, metrics = evaluate_model(train_loader, test_loader, best_params, epochs)

    # Optional: Feature importance analysis (if using GCN)
    if best_params['model_type'] == 'gcn':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        importance_scores = feature_importance_analysis(model, test_loader, device)
        plot_feature_importance(importance_scores)

    # Print summary metrics
    print("\nSummary Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")


    # Uncomment the following lines to perform k-fold cross-validation
    # results = perform_kfold_cross_validation(
    # dataset=graph_data_list, 
    # model_class=PrebuiltGraphNN, 
    # best_params=best_params, 
    # epochs=100, 
    # output_dir='results/seeg2', 
    # seed=42
    # )

    # # Access averaged metrics
    # print(results['averaged_metrics'])