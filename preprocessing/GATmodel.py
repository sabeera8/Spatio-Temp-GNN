import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(torch.nn.Module):
    """
    Graph Attention Network with attention weight extraction capability.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, heads=4, dropout=0.6):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        
        # Last layer (uses single head for classification)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        
        # For storing attention weights
        self.attention_weights = None
        self._attention_hook_handles = []
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Store attention weights for each layer
        self.attention_weights = []
        
        # Process through GAT layers
        for i, conv in enumerate(self.convs):
            # Forward pass with attention weights
            x_with_attention = self._forward_with_attention(conv, x, edge_index, i)
            if x_with_attention is not None:
                x = x_with_attention
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final classification layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def _forward_with_attention(self, conv_layer, x, edge_index, layer_idx):
        """
        Custom forward pass that extracts attention weights.
        Returns transformed node features.
        """
        # Try to access the attention mechanism based on PyTorch Geometric version
        try:
            # For newer PyG versions
            out, attention_weights = conv_layer(x, edge_index, return_attention_weights=True)
            edge_index_with_weights, attention_values = attention_weights
            
            # Store weights with their corresponding edges
            self.attention_weights.append((edge_index_with_weights, attention_values, layer_idx))
            
            return out
        except:
            # If return_attention_weights is not supported, return None
            # and let the normal forward pass handle it
            return None
    
    def get_attention_weights(self, data=None):
        """
        Return the stored attention weights or compute them if not available.
        
        Args:
            data: Optional data to compute attention weights if not already stored
        
        Returns:
            List of tuples (edge_index, attention_weights, layer_index)
        """
        if self.attention_weights is None and data is not None:
            # Forward pass to compute attention weights
            _ = self(data)
        
        return self.attention_weights
    
    def get_node_attention_scores(self, attention_weights=None):
        """
        Convert edge attention weights to node importance scores by aggregating
        incoming attention for each node.
        
        Args:
            attention_weights: Optional attention weights, uses stored weights if None
            
        Returns:
            Dictionary mapping node indices to attention-based importance scores
        """
        if attention_weights is None:
            attention_weights = self.attention_weights
        
        if not attention_weights:
            return {}
        
        # Use the last layer's attention weights (typically most informative)
        edge_index, attention_values, _ = attention_weights[-1]
        
        # Get target nodes (incoming edges)
        target_nodes = edge_index[1]
        
        # Initialize importance scores
        node_importance = {}
        
        # Sum incoming attention for each node
        for i, node_idx in enumerate(target_nodes):
            node_idx = node_idx.item()
            if node_idx not in node_importance:
                node_importance[node_idx] = 0
                
            # For multi-head attention, sum across all heads
            if attention_values.dim() > 1:
                attention_sum = attention_values[i].sum().item()
            else:
                attention_sum = attention_values[i].item()
                
            node_importance[node_idx] += attention_sum
        
        return node_importance

# Example usage in evaluate_model function:
def evaluate_model_with_attention(train_loader, test_loader, best_params, epochs, region_names, device=None):
    """Modified evaluate_model function that uses GAT model with attention weights"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Best parameters: {best_params}")
    
    # Extract parameters
    learning_rate = best_params['learning_rate']
    hidden_dim = best_params['hidden_dim']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    
    # Get input dimensions from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]
    num_classes = 2  # As specified in your code
    
    # Initialize GAT model instead of PrebuiltGraphNN
    model = GATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout_rate
    )
    model = model.to(device)
    
    # Rest of the function remains the same as evaluate_model
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
                
    
    # Add node importance analysis at the end
    print("\nAnalyzing node importance...")
    sample_batch = next(iter(test_loader)).to(device)
    
    # Get region names from the batch if available
    #region_names = getattr(sample_batch, 'region_names', None)
    #region_names = sample_batch.region_names
    
    # Run node importance analysis
    importance_results = node_importance_analysis(model, test_loader, device, region_names)
    
    region_names = list(region_names.keys())
    # Visualization with attention weights
    attention_scores = model.get_node_attention_scores()
    if attention_scores:
        print("\nTop nodes by attention score:")
        sorted_nodes = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)
        print(sorted_nodes)
        for node_idx, score in sorted_nodes:
            node_name = region_names[node_idx] if region_names else f"Node {node_idx}"
            print(f"{node_name}: {score:.4f}")
        
        # Visualize on graph
        visualize_important_nodes_on_graph(attention_scores, sample_batch.edge_index, region_names)
    
    return model, importance_results