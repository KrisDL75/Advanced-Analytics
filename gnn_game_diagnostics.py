import statistics
from dataclasses import dataclass

import torch

import gnn_game_experiment as exp


@dataclass
class DatasetStats:
    samples: int
    avg_nodes: float
    min_nodes: int
    max_nodes: int
    positive_label_rate: float
    init_a_rate: float
    symmetric_adj_rate: float
    finite_adj_rate: float


def analyze_dataset(dataset):
    node_counts = []
    pos_labels = 0
    total_labels = 0
    init_a_sum = 0.0
    init_a_total = 0
    symmetric_count = 0
    finite_count = 0

    for sample in dataset:
        n = sample.x.shape[0]
        node_counts.append(n)

        # y in {0,1}
        pos_labels += int(sample.y.sum().item())
        total_labels += int(sample.y.numel())

        # first feature is initial A state
        init_a_sum += float(sample.x[:, 0].sum().item())
        init_a_total += int(sample.x[:, 0].numel())

        if torch.allclose(sample.a_hat, sample.a_hat.T, atol=1e-6):
            symmetric_count += 1
        if torch.isfinite(sample.a_hat).all().item():
            finite_count += 1

    return DatasetStats(
        samples=len(dataset),
        avg_nodes=statistics.mean(node_counts),
        min_nodes=min(node_counts),
        max_nodes=max(node_counts),
        positive_label_rate=pos_labels / total_labels,
        init_a_rate=init_a_sum / init_a_total,
        symmetric_adj_rate=symmetric_count / len(dataset),
        finite_adj_rate=finite_count / len(dataset),
    )


def run_single(seed: int, epochs: int = 15):
    exp.set_seed(seed)
    train_data = exp.make_dataset(160)
    val_data = exp.make_dataset(60)
    test_data = exp.make_dataset(80)

    gnn = exp.TinyGCN()
    exp.train_model(gnn, train_data, val_data, epochs=epochs)
    gnn_node_acc, gnn_graph_exact = exp.evaluate(gnn, test_data)

    mlp = exp.NodeMLP()
    exp.train_model(mlp, train_data, val_data, epochs=epochs)
    mlp_node_acc, mlp_graph_exact = exp.evaluate(mlp, test_data)

    return {
        "seed": seed,
        "gnn_node_acc": gnn_node_acc,
        "gnn_graph_exact": gnn_graph_exact,
        "mlp_node_acc": mlp_node_acc,
        "mlp_graph_exact": mlp_graph_exact,
        "node_acc_gain": gnn_node_acc - mlp_node_acc,
        "graph_exact_gain": gnn_graph_exact - mlp_graph_exact,
        "train_stats": analyze_dataset(train_data),
        "val_stats": analyze_dataset(val_data),
        "test_stats": analyze_dataset(test_data),
    }


def main():
    seeds = [1, 7, 42]
    results = []

    for seed in seeds:
        print("=" * 72)
        print(f"Running diagnostics for seed={seed}")
        results.append(run_single(seed=seed, epochs=15))

    print("\n" + "=" * 72)
    print("DATA PROCESSING CHECK (from first run)")
    print("=" * 72)
    first = results[0]
    for split_name in ["train_stats", "val_stats", "test_stats"]:
        st = first[split_name]
        print(f"{split_name}: samples={st.samples}, avg_nodes={st.avg_nodes:.2f}, nodes_range=[{st.min_nodes},{st.max_nodes}]")
        print(f"  positive_label_rate={st.positive_label_rate:.3f}, init_a_rate={st.init_a_rate:.3f}")
        print(f"  symmetric_adj_rate={st.symmetric_adj_rate:.3f}, finite_adj_rate={st.finite_adj_rate:.3f}")

    print("\n" + "=" * 72)
    print("MODEL PERFORMANCE ACROSS SEEDS")
    print("=" * 72)
    for r in results:
        print(
            f"seed={r['seed']:>2} | "
            f"GNN node={r['gnn_node_acc']:.4f}, graph_exact={r['gnn_graph_exact']:.4f} | "
            f"MLP node={r['mlp_node_acc']:.4f}, graph_exact={r['mlp_graph_exact']:.4f} | "
            f"gain_node={r['node_acc_gain']:.4f}, gain_graph={r['graph_exact_gain']:.4f}"
        )

    avg_gain_node = statistics.mean(r["node_acc_gain"] for r in results)
    avg_gain_graph = statistics.mean(r["graph_exact_gain"] for r in results)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Average node-accuracy gain (GNN - MLP): {avg_gain_node:.4f}")
    print(f"Average exact-graph gain (GNN - MLP):   {avg_gain_graph:.4f}")

    if avg_gain_node > 0.05:
        print("Conclusion: YES, in this setup a GNN learns the game rule better than a non-graph baseline.")
    elif avg_gain_node > 0.0:
        print("Conclusion: PARTIAL YES, GNN is better but the margin is modest.")
    else:
        print("Conclusion: NO clear evidence in this setup; needs redesign/tuning.")


if __name__ == "__main__":
    main()
