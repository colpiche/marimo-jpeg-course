import marimo
from collections.abc import Callable
from numpy import ndarray
from types import ModuleType

app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np: ModuleType) -> tuple[Callable[[ndarray, str], tuple[list[ndarray], list[str], list[str], list[tuple[int, int]]]]]:
    def rgb_to_ycbcr(img: "ndarray") -> "ndarray":
        """Conversion RGB → YCbCr selon la norme ITU-R BT.601."""
        r: ndarray = img[..., 0].astype(float)
        g: ndarray = img[..., 1].astype(float)
        b: ndarray = img[..., 2].astype(float)
        Y: ndarray  =  16 + ( 65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        Cb: ndarray = 128 + (-37.797 * r -  74.203 * g + 112.0  * b) / 255.0
        Cr: ndarray = 128 + (112.0   * r -  93.786 * g - 18.214 * b) / 255.0
        return np.stack([Y, Cb, Cr], axis=-1)

    def get_channels(
        img: "ndarray",
        space: str,
    ) -> "tuple[list[ndarray], list[str], list[str], list[tuple[int, int]]]":
        """Retourne (canaux, noms, colormaps, plages) pour le modèle colorimétrique donné."""
        channels: list[ndarray]
        names: list[str]
        cmaps: list[str]
        ranges: list[tuple[int, int]]
        if space == "RGB":
            channels = [img[..., 0].astype(float),
                        img[..., 1].astype(float),
                        img[..., 2].astype(float)]
            names  = ["R - Rouge", "G - Vert", "B - Bleu"]
            cmaps  = ["Reds", "Greens", "Blues"]
            ranges = [(0, 255), (0, 255), (0, 255)]
        else:  # YCbCr
            ycbcr: ndarray = rgb_to_ycbcr(img)
            channels = [ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]]
            names    = ["Y - Luminance", "Cb - Chroma bleue", "Cr - Chroma rouge"]
            cmaps    = ["gray", "RdBu", "RdYlBu_r"]
            ranges   = [(16, 235), (16, 240), (16, 240)]
        return channels, names, cmaps, ranges

    return (get_channels,)


@app.cell
def _(np: ModuleType) -> tuple[ndarray]:
    from scipy import datasets as _datasets
    _img_full: "ndarray" = _datasets.face()   # 768x1024 RGB uint8
    image: "ndarray" = _img_full[::2, ::2]    # -> 384x512, plus fluide en UI
    return (image,)


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    # Compression JPEG - Cours interactif

    Ce carnet illustre pas a pas les étapes de la compression JPEG.
    Modifiez les paramètres pour observer les effets en temps réel.

    | # | Etape | Description | Statut |
    |---|-------|-------------|--------|
    | 1 | Codage de la couleur | Conversion RGB vers YCbCr | OK |
    | 2 | Sous-échantillonnage | Reduction des chrominances Cb/Cr | à venir |
    | 3 | Découpage en blocs | Partition en blocs 8x8 pixels | à venir |
    | 4 | DCT | Transformée en cosinus discrète | à venir |
    | 5 | Quantification | Suppression des hautes fréquences | à venir |
    | 6 | Codage entropique | Huffman + RLE | à venir |
    """)
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Etape 1 : Codage de la couleur
    """)
    return


@app.cell
def _(mo: ModuleType) -> tuple[marimo.ui.radio, marimo.ui.checkbox]:
    color_space = mo.ui.radio(
        options=["RGB", "YCbCr"],
        value="RGB",
        label="Modèle colorimétrique",
    )
    show_hist = mo.ui.checkbox(
        label="Afficher les histogrammes de canaux",
        value=True,
    )
    mo.hstack([color_space, show_hist], gap="3rem", justify="start")
    return color_space, show_hist


@app.cell
def _(
    color_space: marimo.ui.radio,
    get_channels: Callable[[ndarray, str], tuple[list[ndarray], list[str], list[str], list[tuple[int, int]]]],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    show_hist: marimo.ui.checkbox,
) -> None:
    _channels: "list[ndarray]"
    _names: "list[str]"
    _cmaps: "list[str]"
    _ranges: "list[tuple[int, int]]"
    _channels, _names, _cmaps, _ranges = get_channels(image, color_space.value)
    _hist_colors: dict[str, list[str]] = {
        "RGB":   ["#cc3333", "#33aa33", "#3333cc"],
        "YCbCr": ["#555555", "#4169e1", "#dc143c"],
    }
    _hc: list[str] = _hist_colors[color_space.value]

    _n_rows: int = 2 if show_hist.value else 1
    _fig = plt.figure(figsize=(16, 4.5 * _n_rows + 0.8), layout="constrained")
    _gs = _fig.add_gridspec(
        _n_rows, 4,
        width_ratios=[1.6, 1, 1, 1],
    )

    # Image originale
    _ax_orig = _fig.add_subplot(_gs[:, 0])
    _ax_orig.imshow(image)
    _ax_orig.set_title(
        f"Image originale (RGB)\n{image.shape[1]}x{image.shape[0]} px",
        fontsize=11, fontweight="bold",
    )
    _ax_orig.axis("off")

    # Canaux separés (ligne 0) et histogrammes optionnels (ligne 1)
    for _i, (_ch, _name, _cmap, (_vmin, _vmax)) in enumerate(
        zip(_channels, _names, _cmaps, _ranges)
    ):
        _ax_ch = _fig.add_subplot(_gs[0, _i + 1])
        _im = _ax_ch.imshow(_ch, cmap=_cmap, vmin=_vmin, vmax=_vmax)
        _ax_ch.set_title(_name, fontsize=10)
        _ax_ch.axis("off")
        _fig.colorbar(_im, ax=_ax_ch, fraction=0.046, pad=0.04)
        _ax_ch.text(
            0.5, -0.03,
            f"min={float(np.min(_ch)):.1f}  max={float(np.max(_ch)):.1f}",
            ha="center", va="top", transform=_ax_ch.transAxes,
            fontsize=8, color="#555555", clip_on=False,
        )

        if show_hist.value:
            _ax_h = _fig.add_subplot(_gs[1, _i + 1])
            _ax_h.hist(
                _ch.ravel(), bins=64, range=(_vmin, _vmax),
                color=_hc[_i], edgecolor="none", alpha=0.85,
            )
            _ax_h.set_xlim(_vmin, _vmax)
            _ax_h.set_xlabel("Valeur du canal", fontsize=8)
            _ax_h.set_ylabel("Nb pixels", fontsize=8)
            _ax_h.tick_params(labelsize=7)
            _ax_h.spines[["top", "right"]].set_visible(False)

    _fig.suptitle(
        f"Décomposition en canaux - espace {color_space.value}",
        fontsize=13, fontweight="bold",
    )
    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)
    return


@app.cell
def _(color_space: marimo.ui.radio, mo: ModuleType) -> None:
    _explanations = {
        "RGB": mo.md("""
    **Espace RGB** - représentation native des capteurs et des écrans.

    Chaque pixel est défini par trois composantes *Rouge (R)*, *Vert (G)*, *Bleu (B)* codées sur 8 bits, soit dans l'intervalle [0, 255].
    Cet espace est **peu adapté à la compression** : les trois canaux sont fortement corrélés entre eux
    et l'oeil humain n'est pas sensible de la même manière aux trois couleurs primaires.

    Basculez sur **YCbCr** pour voir comment JPEG sépare luminance et chrominance.
    """),
        "YCbCr": mo.md("""
    **Espace YCbCr** - représentation utilisée par le standard JPEG (norme ITU-R BT.601).

    | Canal | Plage | Rôle |
    |-------|-------|------|
    | **Y** - Luminance | [16, 235] | Information de luminosité - perçue avec la plus haute acuité visuelle |
    | **Cb** - Chroma bleue | [16, 240] | Différence de couleur vers le bleu |
    | **Cr** - Chroma rouge | [16, 240] | Différence de couleur vers le rouge |

    L'oeil humain est **beaucoup plus sensible** aux variations de luminance (Y) qu'aux variations de
    chrominance (Cb, Cr). JPEG exploite cette propriété : à l'étape suivante, Cb et Cr seront
    **sous-échantillonnés** (réduits en résolution spatiale) sans perte perceptible notable.
    """),
    }
    mo.callout(_explanations[color_space.value], kind="info")
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ---
    ## Etape 2 : Sous-échantillonnage de la chrominance - *à venir*

    Réduction spatiale des canaux Cb et Cr (modes 4:4:4 / 4:2:2 / 4:2:0).
    L'oeil étant peu sensible à la chrominance, on peut réduire sa résolution sans artefact visible.
    """)
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Etape 3 : Découpage en blocs 8x8 - *à venir*

    Partition de chaque canal en blocs de 8x8 pixels.
    C'est l'unite de traitement atomique de JPEG.
    """)
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Etape 4 : Transformée en cosinus discrète (DCT) - *à venir*

    Projection de chaque bloc 8x8 sur une base de fréquences spatiales.
    Les coefficients basse frequence concentrent l'essentiel de l'énergie visible.
    """)
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Etape 5 : Quantification - *à venir*

    Division des coefficients DCT par une table de quantification.
    Les hautes fréquences (peu perceptibles) sont fortement réduites ou annulées.
    """)
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Etape 6 : Codage entropique - *à venir*

    Codage de Huffman + RLE (Run-Length Encoding) pour compresser les coefficients quantifiés
    sans perte supplémentaire.
    """)
    return


if __name__ == "__main__":
    app.run()
