import torch
import torch.nn as nn
from torch.nn import Linear
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch.nn.functional as F
from data_utils import load_data
import argparse

# ── GNN Layer ─────────────────────────────────────────────────────────────────
# Implements: x_v^k = g_θ( x_v^(k-1) + Σ_{j∈N(v)} x_j^(k-1) )
 
class GNNLayer(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.linear = Linear(dim, dim)
        self.dropout = dropout
 
    def forward(self, x, edge_index):
        src, dst = edge_index
 
        # Sum neighbour features into each destination node
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
 
        # Self + neighbour sum, then apply g_θ, then dropout
        return F.dropout(F.relu(self.linear(x + agg)), p=self.dropout, training=self.training)

# ── Full GNN Model ────────────────────────────────────────────────────────────
 
class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, k, dropout=0.5):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GNNLayer(hidden_dim, dropout) for _ in range(k)])
        self.classifier = Linear(hidden_dim, num_classes)  # w in the loss formula
        self.dropout = dropout
 
    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.input_proj(x)), p=self.dropout, training=self.training)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.classifier(h)  # raw logits

# ── Helpers ───────────────────────────────────────────────────────────────────
 
def macro_f1(logits, y, mask):
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    truth = y[mask].cpu().numpy()
    return f1_score(truth, preds, average='macro', zero_division=0)
 
@torch.no_grad()
def evaluate(model, data, criterion):
    model.eval()
    logits = model(data.x, data.edge_index)
    train_loss = criterion(logits[data.train_mask], data.y[data.train_mask]).item()
    val_loss   = criterion(logits[data.val_mask],   data.y[data.val_mask]).item()
    train_f1   = macro_f1(logits, data.y, data.train_mask)
    val_f1     = macro_f1(logits, data.y, data.val_mask)
    test_f1    = macro_f1(logits, data.y, data.test_mask)
    return train_loss, val_loss, train_f1, val_f1, test_f1

 # ── Training ──────────────────────────────────────────────────────────────────
 
def train(data, k, epochs=3000, hidden_dim=128, lr=0.01, weight_decay=5e-3, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data   = data.to(device)
 
    num_classes = int(data.y.max().item()) + 1
    model       = GNN(data.num_features, hidden_dim, num_classes, k).to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion   = nn.CrossEntropyLoss()  # expects logits, applies softmax internally
 
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_loss, best_state, best_metrics = float('inf'), None, None
 
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
 
        if epoch % 20 == 0:
            tr_loss, vl_loss, tr_f1, vl_f1, te_f1 = evaluate(model, data, criterion)
            history['epoch'].append(epoch)
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_f1'].append(tr_f1)
            history['val_f1'].append(vl_f1)
 
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                best_state    = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
                best_metrics  = {'epoch': epoch, 'train_f1': tr_f1, 'val_f1': vl_f1, 'test_f1': te_f1}
 
    # Reload best checkpoint and re-evaluate
    model.load_state_dict(best_state)
    _, _, tr_f1, vl_f1, te_f1 = evaluate(model.to(device), data, criterion)
    best_metrics = {'epoch': best_metrics['epoch'], 'train_f1': tr_f1, 'val_f1': vl_f1, 'test_f1': te_f1}
 
    return history, best_metrics   

# ── Plots ─────────────────────────────────────────────────────────────────────
 
def plot_f1(history, tag):
    plt.figure()
    plt.plot(history['epoch'], history['train_f1'], label='Train F1')
    plt.plot(history['epoch'], history['val_f1'],   label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('Macro F1'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f'F1 vs Epoch [{tag}]')
    plt.savefig(f'f1_{tag}.png', dpi=150); plt.close()
 
def plot_loss(history, tag):
    plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f'Loss vs Epoch [{tag}]')
    plt.savefig(f'loss_{tag}.png', dpi=150); plt.close()
 
def plot_topology(k_values, test_f1s):
    plt.figure()
    plt.plot(k_values, test_f1s, marker='o')
    plt.xlabel('k (GNN layers)'); plt.ylabel('Test Macro F1'); plt.grid(alpha=0.3)
    plt.title('Impact of Topology — Cora')
    plt.savefig('topology_experiment.png', dpi=150); plt.close()
 

# ── Main ──────────────────────────────────────────────────────────────────────
 
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--k",       type=int)
args = parser.parse_args()
 
dataset = load_data(args.dataset)
print(f'\nDataset: {dataset}:')
print(f' Number of GNN layers {args.k}')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
 
data = dataset[0]
print(f'\n{data}')
print('=' * 107)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
 
# Main training run
tag = f"{args.dataset}_k{args.k}"
print(f"\nTraining GNN ({tag}) ...")
history, best = train(data, k=args.k)
 
print(f"\n=== Results ===")
print(f"Best epoch : {best['epoch']}")
print(f"Train F1   : {best['train_f1']:.4f}")
print(f"Val F1     : {best['val_f1']:.4f}")
print(f"Test F1    : {best['test_f1']:.4f}")
 
plot_f1(history, tag)
plot_loss(history, tag)
 
# Topology experiment — only runs on Cora
if args.dataset == 'Cora':
    print("\n=== Topology Experiment ===")
    cora_data = load_data('Cora')[0]
    k_values, test_f1s = [0, 1, 2, 3, 4, 5], []
 
    for k in k_values:
        _, bm = train(cora_data, k=k)
        print(f"k={k}  Test F1={bm['test_f1']:.4f}  (best epoch {bm['epoch']})")
        test_f1s.append(bm['test_f1'])
 
    plot_topology(k_values, test_f1s)
