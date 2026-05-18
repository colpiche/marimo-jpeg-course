# Compression JPEG — Cours interactif

<!-- BADGES -->
[![Marimo](https://img.shields.io/badge/marimo-notebook-blue)](https://marimo.io)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_PcrRS5A7d99UDxXtPmvJtg)

Carnet marimo interactif illustrant pas à pas les six étapes de la compression JPEG. Chaque paramètre (facteur qualité, mode de sous-échantillonnage, bloc sélectionné) est modifiable en temps réel.

## Étapes couvertes

1. **Codage de la couleur** — conversion RGB → YCbCr (ITU-R BT.601)
2. **Sous-échantillonnage** — réduction des chrominances Cb/Cr (4:4:4 / 4:2:2 / 4:2:0)
3. **Découpage en blocs** — partition de l'image en blocs 8×8 pixels
4. **DCT** — transformée en cosinus discrète par bloc
5. **Quantification** — suppression des hautes fréquences via table Q
6. **Codage entropique** — parcours zigzag, RLE et codes de Huffman

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

molab ne semble pas prendre en compte le bloc PEP 723 pour l'installation automatique des dépendances. Sur molab, il faut installer les packages manuellement depuis l'éditeur (panneau latéral → *Manage packages*).
