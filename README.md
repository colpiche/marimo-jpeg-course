# Compression JPEG - Cours interactif

<!-- BADGES -->
[![Marimo](https://img.shields.io/badge/marimo-notebook-blue)](https://marimo.io)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_PcrRS5A7d99UDxXtPmvJtg)

Carnet [Marimo](https://marimo.io) illustrant pas à pas les étapes de la compression JPEG.

## Utilisation en ligne

> **Exécuter ce carnet en ligne** : [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_PcrRS5A7d99UDxXtPmvJtg/app)

## Installation locale

```bash
uv sync
```

### Lancer le carnet

```bash
# Mode lecture (navigateur)
uv run marimo run main.py

# Mode édition
uv run marimo edit main.py
```

## Maintenance

### Dépendances

Les dépendances sont gérées par [uv](https://github.com/astral-sh/uv) via `pyproject.toml`. Pour ajouter un package :

```bash
uv add <package>
```

Le bloc PEP 723 en tête de `main.py` doit rester synchronisé avec `pyproject.toml` pour que le carnet soit auto-suffisant sur les plateformes cloud (molab, etc.).
