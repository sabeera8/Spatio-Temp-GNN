import torch
import torch.optim as optim
import optuna
from model import PrebuiltGraphNN
from torch_geometric.loader import DataLoader


train_loader = torch.load('train_loader.pt')
test_loader = torch.load('test_loader.pt')
train_data_list = next(iter(train_loader))  # Get a single batch to determine input dimensions



def objective(trial):
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1)
    hidden_dim =  trial.suggest_categorical('hidden_dim', [32, 64, 128])
    model_type = trial.suggest_categorical('model_type', ['gcn', 'gat', 'graphconv'])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    
    # Determine input dimension and number of classes
    input_dim = train_data_list[0].x.shape[1]
    #input_dim = 11
    #num_classes = len(torch.unique(torch.tensor([data.y for data in train_data_list])))
    num_classes = 2
    
    # Initialize model
    
    model = PrebuiltGraphNN(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes,
        model_type=model_type,
        num_layers=num_layers
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Clear unnecessary CUDA cache
    torch.cuda.empty_cache()

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    try:
        # Training loop
        model.train()
        for epoch in range(50):
            total_loss = 0
            
            for batch in train_loader:
                # Move batch to device
                batch = batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                loss.backward()
                optimizer.step()
                
                # Move batch back to CPU and clear cache
                batch = batch.cpu()
                torch.cuda.empty_cache()
                
                total_loss += loss.item()
                
        # Validation
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch.y).sum().item()
                total_samples += batch.y.size(0)
                batch = batch.cpu()
                
        accuracy = total_correct / total_samples
        return accuracy
        
    except Exception as e:
        # Clean up on error
        torch.cuda.empty_cache()
        raise e
    

if __name__ == "__main__":
    print("Checking data dimensions:")
    sample_batch = next(iter(train_loader))
    print(f"Sample batch x shape: {sample_batch.x.shape}")
    print(f"Sample batch edge_index shape: {sample_batch.edge_index.shape}")
    print(f"Sample batch y shape: {sample_batch.y.shape}")
    print(f"Memory used by sample batch: {sample_batch.x.element_size() * sample_batch.x.nelement() / 1024**2:.2f} MB")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('\nStudy statistics: ')
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    best_trial = study.best_trial

    print('  Value: ', best_trial.value)
    print('  Params: ')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')