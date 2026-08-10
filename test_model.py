import torch
from model.EvoBrain import EvoBrain_classification

class DummyArgs:
    num_nodes = 19
    num_rnn_layers = 2
    rnn_units = 64
    input_dim = 100
    max_diffusion_step = 2
    dropout = 0.0
    dcgru_activation = 'tanh'
    filter_type = 'dual_random_walk'
    agg = 'max'

args = DummyArgs()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Building EvoBrain model ..")
model = EvoBrain_classification(args=args, num_classes=1, device=device)
model = model.to(device)

    # dummy data
batch_size = 4
seq_len = 12
print(f"Generating fake EEG data on {device} ..")
x = torch.randn(batch_size, seq_len, args.num_nodes, args.input_dim).to(device)
seq_lengths = torch.tensor([seq_len] * batch_size).to(device)
adj = torch.rand(batch_size, seq_len, args.num_nodes, args.num_nodes).to(device)


print("Running forward pass through the network ..")
logits, hidden = model(x, seq_lengths, adj)
print(logits)
