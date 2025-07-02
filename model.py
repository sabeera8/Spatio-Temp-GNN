import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GAT, GraphConv, global_mean_pool
from torch_geometric.data import Batch



class PrebuiltGraphNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, model_type='gcn', num_layers=2):
        super(PrebuiltGraphNN, self).__init__()
        
        self.model_type = model_type
        
        # Input projection layer to standardize input dimensions
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        # Choose model type
        if model_type == 'gcn':
            self.convs = torch.nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) 
                for _ in range(num_layers)
            ])
        elif model_type == 'gat':
            self.convs = torch.nn.ModuleList([
                GAT(hidden_dim, hidden_dim, heads=4, num_layers=1) 
                for _ in range(num_layers)
            ])
        elif model_type == 'graphconv':
            self.convs = torch.nn.ModuleList([
                GraphConv(hidden_dim, hidden_dim) 
                for _ in range(num_layers)
            ])
        
        # Final classification layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, data):
        # Ensure data is a Batch object
        if not isinstance(data, Batch):
            data = Batch.from_data_list([data])
        
        # Project input to consistent dimension
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        batch = data.batch
        
        # Apply graph convolution layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

