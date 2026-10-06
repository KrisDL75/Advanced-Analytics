# Network Coordination Game and GNN

Exam project on coordination cascades, the majority game, and learning a best-response rule with a small graph neural network (GNN).

## Files

- `Exam_NetworkCoordination.ipynb`: Main notebook with the experiments and their outputs.
- `gnn_game_experiment.py`: Graph and label generation, GNN model, and baseline model.
- `gnn_game_diagnostics.py`: Dataset diagnostics and model comparison across multiple random seeds.

The saved outputs in the main notebook are kept so the results can be presented without rerunning every cell.

## Requirements

Python 3.10 or newer, with a version of PyTorch compatible with Python and your operating system. The project was prepared with PyTorch 2.14.0 and Python 3.14.4.

## Installation on Windows

From this directory, run the following commands in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name network-coordination --display-name "Python (Network Coordination)"
```

In VS Code, open the notebook and select the `Python (Network Coordination)` kernel. To start Jupyter from the terminal:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

## Running the Scripts

```powershell
.\.venv\Scripts\python.exe gnn_game_experiment.py
.\.venv\Scripts\python.exe gnn_game_diagnostics.py
```

The main notebook also contains the cascade and majority-game experiments, then imports `run_single` from `gnn_game_diagnostics.py` for the GNN section.

## Google Colab

Open the notebook in Colab, then upload `gnn_game_diagnostics.py` and `gnn_game_experiment.py` to the session. They must be in the notebook's current directory (usually `/content`) before running the GNN cell. PyTorch is usually already installed in Colab.