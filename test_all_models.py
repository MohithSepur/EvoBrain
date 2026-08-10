import torch
import traceback

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
    dataset = 'TUSZ'
    cl_decay_steps = 3000
    use_curriculum_learning = False
    max_seq_len = 12

args = DummyArgs()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 1
batch_size = 4
seq_len = args.max_seq_len

print(f"Testing on device: {device}\n")

# Dummy Inputs
x = torch.randn(batch_size, seq_len, args.num_nodes, args.input_dim).to(device)
seq_lengths = torch.tensor([seq_len] * batch_size).to(device)
adj = torch.rand(batch_size, seq_len, args.num_nodes, args.num_nodes).to(device)
supports = adj

models_to_test = [
    "evobrain", "lstm", "cnnlstm", "dcrnn", 
    "evolvegcn", "BIOT", "gru_gcn", "graphs4mer"
]

for model_name in models_to_test:
    print(f"--- Testing {model_name} ---")
    try:
        if model_name == "dcrnn":
            from model.DCRNN import DCRNNModel_classification
            model = DCRNNModel_classification(args=args, num_classes=num_classes, device=device)
        elif model_name == "evolvegcn":
            from model.EGCN import EvolveGCN_Model_classification
            model = EvolveGCN_Model_classification(args=args, num_classes=num_classes, device=device)
        elif model_name == "evobrain":
            from model.EvoBrain import EvoBrain_classification
            model = EvoBrain_classification(args=args, num_classes=num_classes, device=device)
        elif model_name == "graphs4mer":
            from model.graphs4mer import GraphS4mer
            model = GraphS4mer(num_classes=num_classes, max_seq_len=args.max_seq_len, num_nodes=args.num_nodes)
        elif model_name == "gru_gcn":
            from model.gru_gcn import GRU_GCN_classification
            model = GRU_GCN_classification(args=args, num_classes=num_classes, device=device)
        elif model_name == "BIOT":
            from model.BIOT import BIOTClassifier
            model = BIOTClassifier(n_classes=num_classes, n_channels=args.num_nodes, n_fft=args.input_dim, hop_length=int(args.input_dim / 2))
        elif model_name == "lstm":
            from model.lstm import LSTMModel
            model = LSTMModel(args, num_classes, device)
        elif model_name == "cnnlstm":
            from model.cnnlstm import CNN_LSTM
            model = CNN_LSTM(num_classes, args.dataset)
        else:
            continue
            
        model = model.to(device)
        
        # Forward pass
        if model_name in ["evobrain", "evolvegcn", "gru_gcn"]:
            logits, _ = model(x, seq_lengths, adj)
        elif model_name == "dcrnn":
            dcrnn_supports = [torch.rand(args.num_nodes, args.num_nodes).to(device)]
            logits, _ = model(x, seq_lengths, dcrnn_supports)
        elif model_name == "BIOT":
            logits, _ = model(x)
        elif model_name in ["lstm", "cnnlstm", "graphs4mer"]:
            logits, _ = model(x, seq_lengths)
            
        print(logits)

    except ImportError as e:
        print(f"Skipped {model_name}: Module not available ({e})\n")
    except Exception as e:
        print(f" Failed {model_name}: {e}")
        # traceback.print_exc()
        print("\n")
