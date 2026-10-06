import random
from dataclasses import dataclass

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class GraphSample:
    x: torch.Tensor          # [n, in_dim]
    a_hat: torch.Tensor      # [n, n]
    y: torch.Tensor          # [n]


def normalized_adjacency(g: nx.Graph) -> torch.Tensor:
    n = g.number_of_nodes()
    node_list = list(g.nodes())
    index = {node: i for i, node in enumerate(node_list)}

    a = torch.zeros((n, n), dtype=torch.float32)
    for u, v in g.edges():
        i, j = index[u], index[v]
        a[i, j] = 1.0
        a[j, i] = 1.0

    # Add self loops for message passing stability.
    a = a + torch.eye(n, dtype=torch.float32)
    deg = a.sum(dim=1)
    d_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return d_inv_sqrt @ a @ d_inv_sqrt


def best_response_labels(g: nx.Graph, initial_a: set[int], q: float) -> torch.Tensor:
    labels = []
    for v in g.nodes():
        neighbors = list(g.neighbors(v))
        if not neighbors:
            labels.append(0.0)
            continue
        frac_a = sum(1 for w in neighbors if w in initial_a) / len(neighbors)
        labels.append(1.0 if frac_a >= q else 0.0)
    return torch.tensor(labels, dtype=torch.float32)


def build_sample(min_n: int = 12, max_n: int = 25) -> GraphSample:
    n = random.randint(min_n, max_n)
    p = random.uniform(0.15, 0.35)
    g = nx.erdos_renyi_graph(n=n, p=p)

    # Ensure graph has at least one edge so neighborhood fractions are meaningful.
    if g.number_of_edges() == 0:
        u, v = random.sample(list(g.nodes()), 2)
        g.add_edge(u, v)

    q = random.uniform(0.35, 0.65)
    p_init = random.uniform(0.2, 0.5)
    initial_a = {v for v in g.nodes() if random.random() < p_init}

    deg = dict(g.degree())
    max_deg = max(1, max(deg.values()))

    # Node features: current action, normalized degree, global threshold q.
    features = []
    for v in g.nodes():
        features.append([
            1.0 if v in initial_a else 0.0,
            deg[v] / max_deg,
            q,
        ])

    x = torch.tensor(features, dtype=torch.float32)
    y = best_response_labels(g, initial_a, q)
    a_hat = normalized_adjacency(g)
    return GraphSample(x=x, a_hat=a_hat, y=y)


def make_dataset(size: int) -> list[GraphSample]:
    return [build_sample() for _ in range(size)]


class TinyGCN(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = a_hat @ x
        h = self.w1(h)
        h = F.relu(h)
        h = a_hat @ h
        logits = self.w2(h).squeeze(-1)
        return logits


class NodeMLP(nn.Module):
    """Baseline that ignores graph edges (not a GNN)."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, _a_hat: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def evaluate(model: nn.Module, dataset: list[GraphSample]) -> tuple[float, float]:
    model.eval()
    total_nodes = 0
    correct_nodes = 0
    exact_graphs = 0

    with torch.no_grad():
        for sample in dataset:
            logits = model(sample.x, sample.a_hat)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct_nodes += int((pred == sample.y).sum().item())
            total_nodes += sample.y.numel()
            if bool(torch.all(pred == sample.y).item()):
                exact_graphs += 1

    node_acc = correct_nodes / max(1, total_nodes)
    graph_exact = exact_graphs / max(1, len(dataset))
    return node_acc, graph_exact


def train_model(model: nn.Module, train_data: list[GraphSample], val_data: list[GraphSample], epochs: int = 20) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for sample in train_data:
            opt.zero_grad()
            logits = model(sample.x, sample.a_hat)
            loss = F.binary_cross_entropy_with_logits(logits, sample.y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            val_acc, val_exact = evaluate(model, val_data)
            avg_loss = total_loss / len(train_data)
            print(
                f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
                f"val_node_acc={val_acc:.4f} | val_graph_exact={val_exact:.4f}"
            )


def run_experiment() -> None:
    set_seed(42)

    train_data = make_dataset(160)
    val_data = make_dataset(60)
    test_data = make_dataset(80)

    print("Training TinyGCN (uses graph structure)...")
    gnn = TinyGCN()
    train_model(gnn, train_data, val_data)
    gnn_node_acc, gnn_graph_exact = evaluate(gnn, test_data)

    print("\nTraining NodeMLP baseline (no edge information)...")
    mlp = NodeMLP()
    train_model(mlp, train_data, val_data)
    mlp_node_acc, mlp_graph_exact = evaluate(mlp, test_data)

    print("\n=== Test Results ===")
    print(f"TinyGCN node accuracy:      {gnn_node_acc:.4f}")
    print(f"TinyGCN exact-graph score:  {gnn_graph_exact:.4f}")
    print(f"NodeMLP node accuracy:      {mlp_node_acc:.4f}")
    print(f"NodeMLP exact-graph score:  {mlp_graph_exact:.4f}")

    print("\nConclusion:")
    if gnn_node_acc > mlp_node_acc + 0.05:
        print("The GNN clearly learns the game best-response rule from data better than a non-graph baseline.")
    else:
        print("The GNN does not clearly outperform the baseline under this setup; further tuning or data is needed.")


if __name__ == "__main__":
    run_experiment()
