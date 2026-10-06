# Network Coordination Game and GNN

Projet d'examen sur les cascades de coordination, le jeu de majorite et l'apprentissage d'une regle de meilleure reponse avec un petit reseau de neurones sur graphes (GNN).

## Fichiers

- `Exam_NetworkCoordination.ipynb` : notebook principal avec les experiences et leurs sorties.
- `Exam_NetworkCoordination_Presentation.ipynb` : support de presentation.
- `gnn_game_experiment.py` : generation des graphes, labels, modele GNN et modele de reference.
- `gnn_game_diagnostics.py` : diagnostics des donnees et comparaison des modeles sur plusieurs graines.

Les sorties enregistrees dans les notebooks sont conservees pour presenter les resultats sans relancer toutes les cellules.

## Prerequis

Python 3.10 ou plus recent, avec une version de PyTorch compatible avec Python et le systeme. Le projet a ete prepare avec PyTorch 2.14.0 et Python 3.14.4.

## Installation sous Windows

Depuis ce dossier, dans PowerShell :

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name network-coordination --display-name "Python (Network Coordination)"
```

Dans VS Code, ouvrir le notebook et selectionner le kernel `Python (Network Coordination)`. Pour lancer Jupyter depuis le terminal :

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

## Execution des scripts

```powershell
.\.venv\Scripts\python.exe gnn_game_experiment.py
.\.venv\Scripts\python.exe gnn_game_diagnostics.py
```

Le notebook principal contient egalement les experiences de cascades et de jeu de majorite, puis importe `run_single` depuis `gnn_game_diagnostics.py` pour la partie GNN.

## Google Colab

Ouvrir le notebook dans Colab, puis televerser `gnn_game_diagnostics.py` et `gnn_game_experiment.py` dans la session. Ils doivent etre dans le repertoire courant du notebook (generalement `/content`) avant d'executer la cellule GNN. PyTorch est normalement deja installe dans l'environnement Colab.

