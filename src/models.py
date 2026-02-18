import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, GCNConv, GATv2Conv, TransformerConv, LayerNorm
from torch_geometric.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__name__ = 'RecurrentGCN'
        self.predictor_window = out_channels
        self.in_channels = in_channels
        self.recurrent1 = GConvGRU(in_channels, hidden_channels, K=2)
        self.recurrent2 = GConvGRU(hidden_channels, hidden_channels//2, K=1)
        self.linear = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight, h=None):
        # Pass the hidden state `h` to the recurrent layer
        h_next = self.recurrent1(x, edge_index, edge_weight, h)
        h_relu = F.relu(h_next)
        # The final prediction for the current time step
        out = self.linear(h_relu)
        # Return both the prediction and the new hidden state
        return out, h_next

class GNNGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout_rate=0.1):
        super(GNNGAT, self).__init__()
        self.__name__ = 'GNNGAT'
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1)
        self.lin = Linear(hidden_channels, out_channels)
        # self.dropout = Dropout(p=dropout_rate)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        return self.lin(x)

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super(SimpleGCN, self).__init__()
        self.__name__ = 'SimpleGCN'
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.lin = Linear(hidden_channels//2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        assert edge_weight is not None
        h = self.conv1(x, edge_index, edge_weight)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_weight)
        h = h.relu()
        return self.lin(h)

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GraphTransformer, self).__init__()
        self.__name__ = 'GraphTransformer'
        
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, edge_dim=1, dropout=0.6)
        self.norm1 = LayerNorm(hidden_channels * heads)
        
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, dropout=0.6)
        self.norm2 = LayerNorm(hidden_channels * heads)
        
        self.out_lin = Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_attr): # Renamed from edge_weight
        # Layer 1
        # Pass the argument as a keyword `edge_attr`
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        
        # Output
        return self.out_lin(x)
