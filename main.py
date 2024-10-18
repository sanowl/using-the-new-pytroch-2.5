import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        attn_output, _ = self.attention(x, x, x)  # Use scaled dot-product attention
        return self.linear2(attn_output)

model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters())

input_data = torch.randn(5, 10, 20).to(device)  # Adjust shape
