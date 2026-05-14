# CLAUDE.md — marimo-jpeg-course

## Stack
Marimo interactive notebook (`main.py`), Pylance strict mode, Python 3.x.

## Marimo × Pylance strict — règles de cohabitation

- Toujours mettre `from __future__ import annotations` en tête de fichier pour rendre toutes les annotations paresseuses (chaînes).
- Placer les imports de types purs (`ndarray`, `ModuleType`, `LinearSegmentedColormap`…) sous un guard `if TYPE_CHECKING:` pour qu'ils soient visibles par Pylance mais absents à l'exécution.
- Les cellules marimo s'exécutent via `exec()` dans un scope isolé : les imports de niveau module ne sont pas accessibles à l'intérieur d'une cellule.
- Tout symbole utilisé dans une cellule doit provenir de ses paramètres déclarés ou d'un import local dans la cellule.
- Annoter les variables locales dans le corps d'une cellule avec des chaînes (`_x: "ndarray"`) car marimo évalue les annotations de variables à l'exécution.
- Annoter aussi les paramètres des fonctions définies dans une cellule avec des chaînes (`def f(x: "ndarray") -> "ndarray"`).
- Pour typer un tuple-unpacking, déclarer chaque variable sur sa propre ligne avant l'assignation (`_a: "ndarray"\n_b: "ndarray"\n_a, _b = f()`).
- Si un type est nécessaire à la fois comme annotation ET comme valeur à l'exécution (ex. `LinearSegmentedColormap.from_list`), l'exporter depuis une cellule normale — pas depuis `TYPE_CHECKING`.
- Ajouter `# type: ignore[import-untyped]` sur les imports de packages sans stubs (ex. `scipy`).

## Typage fort

- Tous les paramètres et valeurs de retour de fonctions doivent être annotés ; ne pas laisser de type `Unknown` ou implicite.
- Les types génériques doivent être pleinement paramétrés : `list[str]`, `dict[str, list[str | LinearSegmentedColormap]]`, `Callable[[ndarray, str], tuple[...]]` — jamais `list` ou `dict` nus.
- Les types union s'écrivent `A | B` (PEP 604) ; éviter `Optional[X]`, préférer `X | None`.
- Les cellules marimo doivent déclarer leur type de retour explicitement (`-> tuple[...]` ou `-> None`).

## Commentaires de fonctions

- Toute fonction exportée depuis une cellule doit avoir une docstring décrivant ce qu'elle fait, ses plages d'entrée/sortie et ses invariants importants.
- La docstring décrit l'état actuel du code ; elle ne fait pas référence à des modifications passées ni à l'historique de développement.
- Une ligne courte suffit pour les fonctions simples ; utiliser un bloc multi-lignes uniquement si les paramètres ou les contraintes le justifient.

## Commentaires génériques

- N'ajouter un commentaire que si le « pourquoi » est non évident : contrainte cachée, invariant subtil, contournement d'un bug connu.
- Les commentaires sont factuels et décrivent l'état courant du code — jamais une action passée ("ajouté pour…", "corrigé le…").
- Ne pas commenter ce que le nom de la variable ou de la fonction exprime déjà.

## Style général

- Pas de gestion d'erreur pour des cas impossibles ; valider uniquement aux frontières système.
- Ne pas introduire d'abstractions au-delà du besoin immédiat.
