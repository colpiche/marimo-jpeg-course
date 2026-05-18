from __future__ import annotations

import marimo
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import LinearSegmentedColormap
    from numpy import ndarray
    from types import ModuleType

__generated_with = "0.23.6"

app = marimo.App(width="wide")


@app.cell
def _() -> tuple[ModuleType, ModuleType, ModuleType, type[LinearSegmentedColormap]]:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    return mo, np, plt, LinearSegmentedColormap


@app.cell
def _(
    np: ModuleType,
    LinearSegmentedColormap: type[LinearSegmentedColormap],
) -> tuple[
    Callable[[ndarray, str], tuple[list[ndarray], list[str], list[str | LinearSegmentedColormap], list[tuple[int, int]]]],
    Callable[[ndarray, ndarray, ndarray], ndarray],
    Callable[[ndarray], ndarray],
    LinearSegmentedColormap,
    LinearSegmentedColormap,
]:

    # Matrice ITU-R BT.601 : lignes = (Y, Cb, Cr), colonnes = (R, G, B).
    # Divisée par 255 pour opérer directement sur des uint8 convertis en float.
    _M = np.array([
        [ 65.481, 128.553,  24.966],
        [-37.797, -74.203, 112.000],
        [112.000, -93.786, -18.214],
    ]) / 255.0

    # Offsets CCIR : Y démarre à 16 (headroom bas réservé), Cb/Cr centrés à 128 (valeur neutre).
    _OFF = np.array([16.0, 128.0, 128.0])

    def rgb_to_ycbcr(img: "ndarray") -> "ndarray":
        """Convertit une image RGB uint8 (HxWx3) en YCbCr float selon ITU-R BT.601.

        Plages de sortie : Y ∈ [16, 235], Cb ∈ [16, 240], Cr ∈ [16, 240].
        """

        # @ _M.T applique la combinaison linéaire sur le dernier axe (HxWx3).
        return img.astype(float) @ _M.T + _OFF

    def ycbcr_to_rgb(Y: "ndarray", Cb: "ndarray", Cr: "ndarray") -> "ndarray":
        """Convertit trois canaux YCbCr float en une image RGB uint8 (HxWx3).

        Inverse de rgb_to_ycbcr. Accepte des tableaux 2D de même forme pour Y, Cb, Cr.
        """

        r = 1.164*(Y - 16) + 1.596*(Cr - 128)
        g = 1.164*(Y - 16) - 0.813*(Cr - 128) - 0.392*(Cb - 128)
        b = 1.164*(Y - 16) + 2.017*(Cb - 128)

        # Le clip est nécessaire : les couleurs en bord de gamme YCbCr peuvent
        # produire des valeurs RGB hors de [0, 255].
        return np.clip(np.round(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)

    # RGB   : dégradé noir → couleur primaire (0 = canal absent, 255 = saturation pleine).
    # YCbCr : 3 ancres par canal (min=16, neutre=128, max=240) calculées avec
    # ycbcr_to_rgb(Y=128, autre_chroma=128) pour refléter la teinte réellement encodée.
    cb_cmap: "LinearSegmentedColormap" = LinearSegmentedColormap.from_list("Cb_BT601", [
        (130/255, 174/255,   0/255),  # Cb=16  : jaune-vert
        (130/255, 130/255, 130/255),  # Cb=128 : gris neutre
        (130/255,  86/255, 255/255),  # Cb=240 : bleu vif
    ])
    cr_cmap: "LinearSegmentedColormap" = LinearSegmentedColormap.from_list("Cr_BT601", [
        (  0/255, 221/255, 130/255),  # Cr=16  : cyan-vert
        (130/255, 130/255, 130/255),  # Cr=128 : gris neutre
        (255/255,  39/255, 130/255),  # Cr=240 : rouge vif
    ])
    _CMAPS: dict[str, list[str | LinearSegmentedColormap]] = {
        "RGB": [
            LinearSegmentedColormap.from_list("R", [(0, 0, 0), (1, 0, 0)]),
            LinearSegmentedColormap.from_list("G", [(0, 0, 0), (0, 1, 0)]),
            LinearSegmentedColormap.from_list("B", [(0, 0, 0), (0, 0, 1)]),
        ],
        "YCbCr": ["gray", cb_cmap, cr_cmap],
    }

    def get_channels(img: "ndarray", space: str) -> "tuple[list[ndarray], list[str], list[str | LinearSegmentedColormap], list[tuple[int, int]]]":
        """Décompose une image RGB uint8 en ses trois canaux scalaires 2D.

        Retourne (channels, names, cmaps, ranges) où :
        - channels : liste de 3 tableaux float 2D (HxW), un par canal
        - names    : étiquettes des canaux
        - cmaps    : colormaps matplotlib associées (depuis _CMAPS)
        - ranges   : plages (vmin, vmax) pour la normalisation de l'affichage
        """

        if space == "RGB":
            channels = [img[..., i].astype(float) for i in range(3)]
            names    = ["R - Rouge", "G - Vert", "B - Bleu"]
            ranges   = [(0, 255)] * 3
        else:
            ycbcr = rgb_to_ycbcr(img)
            channels = [ycbcr[..., i] for i in range(3)]
            names    = ["Y - Luminance", "Cb - Chroma bleue", "Cr - Chroma rouge"]
            ranges   = [(16, 235), (16, 240), (16, 240)]

        return channels, names, _CMAPS[space], ranges

    return (get_channels, ycbcr_to_rgb, rgb_to_ycbcr, cb_cmap, cr_cmap)


@app.cell
def _(np: ModuleType) -> tuple[ndarray]:
    from scipy import datasets as _datasets   # type: ignore[import-untyped]
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
    |:--|:------|:------------|:-------|
    | 1 | Codage de la couleur | Conversion RGB vers YCbCr | OK |
    | 2 | Sous-échantillonnage | Réduction des chrominances Cb/Cr | OK |
    | 3 | Découpage en blocs | Partition en blocs 8x8 pixels | OK |
    | 4 | DCT | Transformée en cosinus discrète | OK |
    | 5 | Quantification | Suppression des hautes fréquences | OK |
    | 6 | Codage entropique | RLE + Huffman | OK |
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
        value=False,
    )
    mo.hstack([color_space, show_hist], gap="3rem", justify="start")
    return color_space, show_hist


@app.cell
def _(
    color_space: marimo.ui.radio,
    get_channels: Callable[[ndarray, str], tuple[list[ndarray], list[str], list[str | LinearSegmentedColormap], list[tuple[int, int]]]],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    show_hist: marimo.ui.checkbox,
) -> None:
    _channels, _names, _cmaps, _ranges = get_channels(image, color_space.value)
    _hist_colors: dict[str, list[str]] = {
        "RGB":   ["#cc3333", "#33aa33", "#3333cc"],
        "YCbCr": ["#555555", "#4169e1", "#dc143c"],
    }
    _hc: list[str] = _hist_colors[color_space.value]

    _row_h: float = 4.0
    _hist_h: float = 2.0

    # 2 sous-figures (originale + canaux) ou 3 si les histogrammes sont activés.
    _n_sfs: int = 3 if show_hist.value else 2
    _h_ratios: list[float] = [_row_h, _row_h, _hist_h] if show_hist.value else [_row_h, _row_h]
    _fig_h: float = 2 * _row_h + (_hist_h if show_hist.value else 0.0) + 0.6
    _fig = plt.figure(figsize=(12, _fig_h), layout="constrained")
    _sfs = _fig.subfigures(_n_sfs, 1, height_ratios=_h_ratios)

    # L'originale occupe la colonne centrale (pos. 2/3) pour s'aligner visuellement
    # avec les trois canaux affichés dans la sous-figure du dessous.
    _ax_orig = _sfs[0].add_subplot(1, 3, 2)
    _ax_orig.imshow(image)
    _ax_orig.set_title(
        f"Image originale (RGB)\n{image.shape[1]}x{image.shape[0]} px",
        fontsize=11, fontweight="bold",
    )
    _ax_orig.axis("off")

    # Ligne 1 : canaux séparés
    for _i, (_ch, _name, _cmap, (_vmin, _vmax)) in enumerate(
        zip(_channels, _names, _cmaps, _ranges)
    ):
        _ax_ch = _sfs[1].add_subplot(1, 3, _i + 1)
        _im = _ax_ch.imshow(_ch, cmap=_cmap, vmin=_vmin, vmax=_vmax)
        # Colorbar à gauche pour ne pas empiéter sur le titre du subplot voisin.
        _sfs[1].colorbar(_im, ax=_ax_ch, fraction=0.046, pad=0.04, location="left")
        _ax_ch.set_title(_name, fontsize=10)
        _ax_ch.axis("off")
        _ax_ch.text(
            0.5, -0.03,
            f"min={float(np.min(_ch)):.1f}  max={float(np.max(_ch)):.1f}",
            ha="center", va="top", transform=_ax_ch.transAxes,
            fontsize=8, color="#555555", clip_on=False,
        )

    # Ligne 2 : histogrammes optionnels
    if show_hist.value:
        for _i, (_ch, _, (_vmin, _vmax)) in enumerate(
            zip(_channels, _names, _ranges)
        ):
            _ax_h = _sfs[2].add_subplot(1, 3, _i + 1)
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
        f"Décomposition en canaux - espace {color_space.value}\n\n",
        fontsize=13, fontweight="bold",
    )
    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)  # replace() évite l'accumulation de figures à chaque ré-exécution réactive
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
    |:------|:------|:-----|
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
    ## Etape 2 : Sous-échantillonnage de la chrominance
    """)
    return


@app.cell
def _(np: ModuleType) -> tuple[
    Callable[[ndarray, str], tuple[ndarray, ndarray, ndarray]],
    Callable[[ndarray, ndarray, ndarray, str], ndarray],
]:
    def chroma_subsample(
        ycbcr: "ndarray", mode: str
    ) -> "tuple[ndarray, ndarray, ndarray]":
        """Sous-échantillonne Cb et Cr d'une image YCbCr selon le mode donné.

        Retourne (Y, Cb_sub, Cr_sub) :
        - 4:4:4 : pass-through, aucune réduction
        - 4:2:2 : résolution horizontale de Cb/Cr divisée par 2 → H x W/2
        - 4:2:0 : résolution divisée par 2 dans les deux axes → H/2 x W/2
        """
        Y  = ycbcr[..., 0]
        Cb = ycbcr[..., 1]
        Cr = ycbcr[..., 2]
        if mode == "4:4:4":
            return Y, Cb, Cr
        elif mode == "4:2:2":
            return Y, Cb[:, ::2], Cr[:, ::2]
        else:  # 4:2:0
            return Y, Cb[::2, ::2], Cr[::2, ::2]

    def chroma_upsample(
        Y: "ndarray", Cb_sub: "ndarray", Cr_sub: "ndarray", mode: str
    ) -> "ndarray":
        """Reconstruit un tableau YCbCr (HxWx3) par upsampling nearest-neighbor.

        Inverse de chroma_subsample. Chaque valeur sous-échantillonnée est dupliquée
        sur ses voisins. Le crop final gère les dimensions impaires de l'image.
        """
        H, W = Y.shape
        if mode == "4:4:4":
            Cb_up, Cr_up = Cb_sub, Cr_sub
        elif mode == "4:2:2":
            # Duplication horizontale uniquement
            Cb_up = np.repeat(Cb_sub, 2, axis=1)[:, :W]
            Cr_up = np.repeat(Cr_sub, 2, axis=1)[:, :W]
        else:  # 4:2:0
            # Duplication horizontale et verticale
            Cb_up = np.repeat(np.repeat(Cb_sub, 2, axis=0), 2, axis=1)[:H, :W]
            Cr_up = np.repeat(np.repeat(Cr_sub, 2, axis=0), 2, axis=1)[:H, :W]
        return np.stack([Y, Cb_up, Cr_up], axis=-1)

    return (chroma_subsample, chroma_upsample)


@app.cell
def _(mo: ModuleType) -> tuple[marimo.ui.radio]:
    sampling_mode = mo.ui.radio(
        options=["4:4:4", "4:2:2", "4:2:0"],
        value="4:2:0",
        label="Mode de sous-échantillonnage",
    )
    sampling_mode
    return (sampling_mode,)


@app.cell
def _(
    cb_cmap: LinearSegmentedColormap,
    cr_cmap: LinearSegmentedColormap,
    chroma_subsample: Callable[[ndarray, str], tuple[ndarray, ndarray, ndarray]],
    chroma_upsample: Callable[[ndarray, ndarray, ndarray, str], ndarray],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
    sampling_mode: marimo.ui.radio,
    ycbcr_to_rgb: Callable[[ndarray, ndarray, ndarray], ndarray],
) -> None:
    # Bruit 8x8 : Y uniforme (128), Cb/Cr aléatoires sur toute la plage [16, 240].
    # Couleurs maximalement saturées → l'effet du sous-échantillonnage est immédiatement visible.
    _rng = np.random.default_rng(42)
    _Y_p: "ndarray"  = _rng.uniform(16.0, 235.0, (8, 8))
    _Cb_p: "ndarray" = _rng.uniform(16.0, 240.0, (8, 8))
    _Cr_p: "ndarray" = _rng.uniform(16.0, 240.0, (8, 8))
    _patch: "ndarray" = ycbcr_to_rgb(_Y_p, _Cb_p, _Cr_p)

    # Crop 8x8 centré : pixel central ± 4 dans chaque axe.
    _H, _W = image.shape[:2]
    _cy, _cx = _H // 2, _W // 2
    _crop: "ndarray" = image[_cy - 4:_cy + 4, _cx - 4:_cx + 4]

    _block_label: "str"
    _gs_w: "int"
    _gs_h: "int"
    _block_label, _gs_w, _gs_h = {
        "4:4:4": ("1x1", 1, 1),
        "4:2:2": ("2x1", 2, 1),
        "4:2:0": ("2x2", 2, 2),
    }[sampling_mode.value]
    _PH, _PW = 8, 8

    _fig = plt.figure(figsize=(12, 24), layout="constrained", dpi=200)
    # height_ratios sections = somme des ratios internes de chaque section (4+4=8, 4+4+5=13).
    # Garantit que les lignes de ratio 4 ont la même hauteur absolue dans les deux sections.
    _sections = _fig.subfigures(2, 1, hspace=0.15, height_ratios=[8, 13])

    for _row_idx, (_src, _section_title) in enumerate([
        (_patch, "Exemple synthétique — bruit aléatoire en YCbCr"),
        (_crop,  "Image originale — crop 8x8 px central"),
    ]):
        _section = _sections[_row_idx]
        _section.suptitle(_section_title, fontsize=11, fontweight="bold")
        # Section 2 : 3 sous-figures (8x8 avant/après, canaux, image complète avant/après).
        _sfs = _section.subfigures(
            3 if _row_idx == 1 else 2, 1,
            height_ratios=[4.0, 4.0, 5.0] if _row_idx == 1 else [4.0, 4.0],
        )

        _ycbcr_s: "ndarray" = rgb_to_ycbcr(_src)
        _Ys: "ndarray"
        _Cb_s: "ndarray"
        _Cr_s: "ndarray"
        _Ys, _Cb_s, _Cr_s = chroma_subsample(_ycbcr_s, sampling_mode.value)
        _ycbcr_up: "ndarray" = chroma_upsample(_Ys, _Cb_s, _Cr_s, sampling_mode.value)
        _rec: "ndarray" = ycbcr_to_rgb(
            _ycbcr_up[..., 0], _ycbcr_up[..., 1], _ycbcr_up[..., 2]
        )

        for _i, (_img, _title) in enumerate([
            (_src, "Image de départ"),
            (_rec, f"Après sous-échantillonnage ({sampling_mode.value})"),
        ]):
            _ax = _sfs[0].add_subplot(1, 2, _i + 1)
            _ax.imshow(_img, interpolation="nearest")
            _ax.set_title(_title, fontsize=10)
            _ax.axis("off")

        for _i, (_ch, _cmap, _name, _vmin, _vmax, _gsw, _gsh) in enumerate([
            (_ycbcr_up[..., 0], "gray",   "Y — Luminance",              16, 235, 1,     1    ),
            (_ycbcr_up[..., 1], cb_cmap, "Cb — blocs " + _block_label, 16, 240, _gs_w, _gs_h),
            (_ycbcr_up[..., 2], cr_cmap, "Cr — blocs " + _block_label, 16, 240, _gs_w, _gs_h),
        ]):
            _ax = _sfs[1].add_subplot(1, 3, _i + 1)
            _im = _ax.imshow(_ch, cmap=_cmap, vmin=_vmin, vmax=_vmax, interpolation="nearest")
            _sfs[1].colorbar(_im, ax=_ax, fraction=0.046, pad=0.04, location="left")
            _ax.set_title(_name, fontsize=10)
            _ax.axis("off")
            for _x in range(_gsw, _PW, _gsw):
                _ax.axvline(_x - 0.5, color="white", linewidth=0.8, alpha=0.8)
            for _y in range(_gsh, _PH, _gsh):
                _ax.axhline(_y - 0.5, color="white", linewidth=0.8, alpha=0.8)

        if _row_idx == 1:
            _ycbcr_full: "ndarray" = rgb_to_ycbcr(image)
            _Yf: "ndarray"
            _Cb_sf: "ndarray"
            _Cr_sf: "ndarray"
            _Yf, _Cb_sf, _Cr_sf = chroma_subsample(_ycbcr_full, sampling_mode.value)
            _ycbcr_up_full: "ndarray" = chroma_upsample(_Yf, _Cb_sf, _Cr_sf, sampling_mode.value)
            _rec_full: "ndarray" = ycbcr_to_rgb(
                _ycbcr_up_full[..., 0], _ycbcr_up_full[..., 1], _ycbcr_up_full[..., 2]
            )
            for _i, (_img, _title) in enumerate([
                (image,     "Image originale complète"),
                (_rec_full, f"Après sous-échantillonnage ({sampling_mode.value})"),
            ]):
                _ax = _sfs[2].add_subplot(1, 2, _i + 1)
                _ax.imshow(_img, interpolation="nearest")
                _ax.set_title(_title, fontsize=10)
                _ax.axis("off")

    _fig.suptitle(
        f"Sous-échantillonnage de la chrominance — mode {sampling_mode.value}\n\n",
        fontsize=13, fontweight="bold",
    )
    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)  # replace() évite l'accumulation de figures à chaque ré-exécution réactive
    return


@app.cell
def _(mo: ModuleType, sampling_mode: marimo.ui.radio) -> None:
    _explanations = {
        "4:4:4": mo.md("""
    **Mode 4:4:4** — aucun sous-échantillonnage.

    Chaque pixel conserve ses trois composantes Y, Cb, Cr à pleine résolution.
    Cb et Cr sont stockés à résolution pleine **(W x H)** — gain en chrominance : **0 %**.
    Utilisé en photographie professionnelle et en vidéo haut de gamme.
    """),
        "4:2:2": mo.md("""
    **Mode 4:2:2** — sous-échantillonnage horizontal uniquement.

    Pour chaque ligne, une valeur Cb et Cr est retenue tous les 2 pixels horizontaux.
    Cb et Cr sont stockés à **(W/2 x H)** — gain en données chrominance : **~33 %**.
    Utilisé en production vidéo broadcast. La perte est peu perceptible sur les images naturelles.
    """),
        "4:2:0": mo.md("""
    **Mode 4:2:0** — sous-échantillonnage horizontal et vertical.

    Une valeur Cb et Cr est retenue pour chaque bloc 2x2 pixels.
    Cb et Cr sont stockés à **(W/2 x H/2)** — gain en données chrominance : **~50 %**.
    C'est le mode utilisé par **JPEG et la majorité des codecs vidéo** (H.264, H.265, VP9).
    Il représente le meilleur compromis qualité/compression pour les contenus grand public.
    """),
    }
    mo.callout(_explanations[sampling_mode.value], kind="info")
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Étape 3 : Découpage en blocs 8x8
    """)
    return


@app.function
def split_into_blocks(channel: "ndarray", block_size: int = 8) -> "ndarray":
    """Découpe un canal 2D en blocs block_size x block_size.

    Retourne un tableau (n_h, n_w, block_size, block_size).
    Les pixels en bordure hors multiple de block_size sont ignorés.
    """
    H, W = channel.shape
    n_h, n_w = H // block_size, W // block_size
    return (
        channel[:n_h * block_size, :n_w * block_size]
        .reshape(n_h, block_size, n_w, block_size)
        .transpose(0, 2, 1, 3)
    )


@app.function
def extract_y_block(
    image: "ndarray",
    ycbcr_fn: "Callable[[ndarray], ndarray]",
    block_idx: int,
) -> "tuple[ndarray, int, int]":
    """Extrait un bloc 8x8 du canal Y centré (level-shift -128).

    Retourne (block_8x8, n_blocs_h, n_blocs_w).
    """
    _ycbcr: "ndarray" = ycbcr_fn(image)
    _Y: "ndarray" = _ycbcr[..., 0]
    _h, _w = _Y.shape
    _n_h, _n_w = _h // 8, _w // 8
    _by = (block_idx // _n_w) * 8
    _bx = (block_idx % _n_w) * 8
    _block: "ndarray" = _Y[_by:_by + 8, _bx:_bx + 8].astype(float) - 128.0
    return _block, _n_h, _n_w


@app.function
def annotate_spatial_block(
    ax: "Axes",
    block: "ndarray",
    threshold: int = 128,
    fontsize: int = 6,
) -> None:
    """Annote chaque cellule d'un heatmap 8x8 spatial avec sa valeur entière.

    Texte blanc si la valeur est inférieure au seuil, noir sinon.
    """
    for r in range(8):
        for c in range(8):
            v: int = int(round(float(block[r, c])))
            ax.text(c, r, str(v), ha="center", va="center", fontsize=fontsize,
                    fontweight="bold", color="white" if v < threshold else "black")


@app.function
def annotate_dct_block(
    ax: "Axes",
    C: "ndarray",
    fontsize: int = 6,
    zero_color: str = "",
    zorder: int = 1,
) -> None:
    """Annote chaque cellule d'un heatmap 8x8 DCT avec la valeur du coefficient.

    Couleur du texte dérivée de la luminance plasma sous-jacente.
    Si zero_color est fourni, les zéros sont affichés dans cette couleur.
    """
    import matplotlib.pyplot as _plt
    _plasma = _plt.cm.plasma  # type: ignore[attr-defined]
    for r in range(8):
        for c in range(8):
            cv: int = int(round(float(C[r, c])))
            if cv == 0 and zero_color:
                ax.text(c, r, "0", ha="center", va="center", fontsize=fontsize,
                        fontweight="bold", color=zero_color, zorder=zorder)
            else:
                amp: float = abs(float(C[r, c]))
                rgba: "tuple[float, float, float, float]" = _plasma(min(amp / 200.0, 1.0))
                lum: float = (0.299 * float(rgba[0]) + 0.587 * float(rgba[1])
                              + 0.114 * float(rgba[2]))
                ax.text(c, r, str(cv), ha="center", va="center", fontsize=fontsize,
                        fontweight="bold", color="black" if lum > 0.45 else "white",
                        zorder=zorder)


@app.cell
def _(mo: ModuleType) -> tuple[Callable[[], int], Callable[[int], None]]:
    get_block_idx: "Callable[[], int]"
    set_block_idx: "Callable[[int], None]"
    get_block_idx, set_block_idx = mo.state(0)
    return get_block_idx, set_block_idx


@app.cell
def _(
    get_block_idx: Callable[[], int],
    image: ndarray,
    mo: ModuleType,
    set_block_idx: Callable[[int], None],
) -> None:
    _h: "int"
    _w: "int"
    _h, _w = image.shape[:2]
    _n_blocks: "int" = (_h // 8) * (_w // 8)
    _current: "int" = get_block_idx()

    _slider = mo.ui.slider(
        0, _n_blocks - 1,
        value=_current,
        label="Bloc sélectionné",
        on_change=set_block_idx,
    )
    def _on_prev(_val: int) -> None:
        set_block_idx(max(0, get_block_idx() - 1))

    def _on_next(_val: int) -> None:
        set_block_idx(min(_n_blocks - 1, get_block_idx() + 1))

    _btn_prev = mo.ui.button(label="◀", on_click=_on_prev)
    _btn_next = mo.ui.button(label="▶", on_click=_on_next)
    mo.hstack([_btn_prev, _slider, _btn_next], justify="start")
    return


@app.cell
def _(
    get_block_idx: Callable[[], int],
    image: ndarray,
    mo: ModuleType,
    plt: ModuleType,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
) -> None:
    import matplotlib.patches as _patches

    _ycbcr: "ndarray" = rgb_to_ycbcr(image)
    _Y: "ndarray" = _ycbcr[..., 0]
    _h: "int"
    _w: "int"
    _h, _w = _Y.shape
    _n_w: "int" = _w // 8
    _n_h: "int" = _h // 8

    _idx: "int" = get_block_idx()
    _block_row: "int" = _idx // _n_w
    _block_col: "int" = _idx % _n_w
    _by: "int" = _block_row * 8
    _bx: "int" = _block_col * 8
    _block: "ndarray" = _Y[_by:_by + 8, _bx:_bx + 8]

    _fig = plt.figure(figsize=(12, 9), layout="constrained", dpi=150)
    _sfs = _fig.subfigures(2, 1, height_ratios=[5, 4])

    # Canal Y complet avec grille de blocs 8x8 et rectangle sur le bloc sélectionné
    _ax0 = _sfs[0].add_subplot(1, 1, 1)
    _ax0.imshow(_Y, cmap="gray", vmin=16, vmax=235, interpolation="nearest")
    for _x in range(0, _w, 8):
        _ax0.axvline(_x - 0.5, color="white", linewidth=0.4, alpha=0.5)
    for _y in range(0, _h, 8):
        _ax0.axhline(_y - 0.5, color="white", linewidth=0.4, alpha=0.5)
    _ax0.add_patch(_patches.Rectangle(
        (_bx - 0.5, _by - 0.5), 8, 8,
        linewidth=2, edgecolor="#ff6600", facecolor="none",
    ))
    _ax0.set_title(
        f"Canal Y — grille {_n_h}x{_n_w} blocs — bloc {_idx} "
        f"[ligne {_block_row}, col {_block_col}]",
        fontsize=10,
    )
    _ax0.axis("off")

    # Zoom sur le bloc | heatmap avec valeurs numériques
    for _i, (_title, _show_vals) in enumerate([
        ("Bloc sélectionné", False),
        ("Valeurs Y", True),
    ]):
        _ax = _sfs[1].add_subplot(1, 2, _i + 1)
        _ax.imshow(_block, cmap="gray", vmin=16, vmax=235, interpolation="nearest")
        if _show_vals:
            annotate_spatial_block(_ax, _block)
        _ax.set_title(_title, fontsize=10)
        _ax.axis("off")

    _fig.suptitle(
        f"Découpage en blocs 8x8 — bloc {_idx} / {_n_h * _n_w - 1}",
        fontsize=13, fontweight="bold",
    )
    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)
    return


@app.cell
def _(image: ndarray, mo: ModuleType) -> None:
    _h: "int"
    _w: "int"
    _h, _w = image.shape[:2]
    _n_h: "int" = _h // 8
    _n_w: "int" = _w // 8
    mo.callout(mo.md(f"""
    **Pourquoi des blocs 8x8 ?**

    La DCT (étape suivante) travaille sur des blocs de taille fixe. La taille 8x8 est un compromis :
    des blocs plus petits perdraient la cohérence spatiale des fréquences ; des blocs plus grands
    augmenteraient la complexité sans gain perceptible — l'œil humain discrimine mal les variations
    spatiales au-delà d'environ 8 cycles par degré visuel.

    Sur cette image ({_w}x{_h} px), le canal Y produit **{_n_h}x{_n_w} = {_n_h * _n_w} blocs**.
    Chaque canal Y, Cb, Cr est traité **indépendamment**, bloc par bloc.
    """), kind="info")
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Étape 4 : Transformée en cosinus discrète (DCT)
    """)
    return


@app.cell
def _(np: ModuleType) -> tuple[
    Callable[[ndarray], ndarray],
    Callable[[ndarray], ndarray],
    ndarray,
]:
    from scipy.fft import dctn as _dctn, idctn as _idctn  # type: ignore[import-untyped]

    def dct2(block: "ndarray") -> "ndarray":
        """DCT-II 2D orthonormale sur un bloc 8x8 centré en 0."""
        return _dctn(block, norm="ortho")

    def idct2(coeffs: "ndarray") -> "ndarray":
        """DCT-III 2D orthonormale — inverse de dct2."""
        return _idctn(coeffs, norm="ortho")

    # Indices zigzag standard JPEG : chaque cellule contient l'ordre de passage (0 = DC).
    zigzag_order: "ndarray" = np.array([
        [ 0,  1,  5,  6, 14, 15, 27, 28],
        [ 2,  4,  7, 13, 16, 26, 29, 42],
        [ 3,  8, 12, 17, 25, 30, 41, 43],
        [ 9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ])
    return dct2, idct2, zigzag_order


@app.cell
def _(
    idct2: Callable[[ndarray], ndarray],
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
) -> None:
    _fig = plt.figure(figsize=(11, 5.5), dpi=120)
    _sfs = _fig.subfigures(1, 2, width_ratios=[3, 2])

    # Catalogue 8x8 des bases DCT
    _axes = _sfs[0].subplots(8, 8, gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    for _i in range(8):
        for _j in range(8):
            _delta: "ndarray" = np.zeros((8, 8))
            _delta[_i, _j] = 1.0
            _basis: "ndarray" = idct2(_delta)
            _axes[_i, _j].imshow(_basis, cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
            _axes[_i, _j].axis("off")
    _sfs[0].suptitle("Catalogue des 64 motifs DCT 8x8", fontsize=10, fontweight="bold")

    # Annotations aux 4 coins via les axes de coin, clip_on=False pour déborder hors du subplot
    _axes[0, 0].text(-0.1, 1.25, "DC\n(fréquence nulle)",
        transform=_axes[0, 0].transAxes, fontsize=9,
        ha="left", va="bottom", clip_on=False, color="#333333")
    _axes[0, 7].text(1.1, 1.25, "Détail fin horizontal →",
        transform=_axes[0, 7].transAxes, fontsize=9,
        ha="right", va="bottom", clip_on=False, color="#333333")
    _axes[7, 0].text(-0.1, -0.25, "Détail fin vertical ↓",
        transform=_axes[7, 0].transAxes, fontsize=9,
        ha="left", va="top", clip_on=False, color="#333333")
    _axes[7, 7].text(1.1, -0.25, "Détail fin diagonal ↘",
        transform=_axes[7, 7].transAxes, fontsize=9,
        ha="right", va="top", clip_on=False, color="#333333")

    # Courbe CSF schématique — axe x sans valeurs, juste la tendance décroissante
    _ax_csf = _sfs[1].add_subplot(1, 1, 1)
    _freq: "ndarray" = np.linspace(0, 7, 100)
    _csf: "ndarray" = np.exp(-0.5 * _freq)
    _ax_csf.plot(_freq, _csf, color="#e07020", linewidth=2)
    _ax_csf.fill_between(_freq, _csf, alpha=0.12, color="#e07020")
    _ax_csf.annotate(
        "Peu sensible\naux détails fins",
        xy=(4.0, float(np.exp(-2.0))),
        xytext=(5.0, 0.55),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "color": "#888888"},
        ha="center",
    )
    _ax_csf.set_xlabel("Fréquence spatiale", fontsize=8)
    _ax_csf.set_ylabel("Sensibilité de l'œil", fontsize=7)
    _ax_csf.set_xticks([])
    _ax_csf.set_yticks([])
    _ax_csf.spines[["top", "right"]].set_visible(False)
    _sfs[1].suptitle("Sensibilité\nvisuelle", fontsize=9, fontweight="bold")

    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)
    return


@app.cell
def _(
    dct2: Callable[[ndarray], ndarray],
    get_block_idx: Callable[[], int],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
) -> None:
    _idx: "int" = get_block_idx()
    _block: "ndarray"
    _n_h: "int"
    _n_w: "int"
    _block, _n_h, _n_w = extract_y_block(image, rgb_to_ycbcr, _idx)
    _C: "ndarray" = dct2(_block)

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", dpi=150)

    # Panneau 1 — Bloc Y original avec valeurs numériques
    _axs[0].imshow(_block + 128.0, cmap="gray", vmin=16, vmax=235, interpolation="nearest")
    annotate_spatial_block(_axs[0], _block + 128.0)
    _axs[0].set_title(f"Valeurs Y", fontsize=9)
    _axs[0].axis("off")

    # Panneau 2 — Amplitude |C| en couleur (plasma), valeur entière signée en texte.
    _axs[1].imshow(np.abs(_C), cmap="plasma", vmin=0, vmax=200, interpolation="nearest")
    annotate_dct_block(_axs[1], _C)
    _fig.colorbar(
        plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 200)),
        ax=_axs[1], fraction=0.046, pad=0.04, label="Amplitude |C|",
    )
    _axs[1].set_title("Coefficients DCT", fontsize=9)
    _axs[1].axis("off")

    _fig.suptitle(
        f"DCT 8x8 — bloc {_idx} / {_n_h * _n_w - 1}",
        fontsize=12, fontweight="bold",
    )
    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.callout(mo.md("""
    **Principe : décomposer un bloc en motifs élémentaires**

    La DCT transforme un bloc 8x8 de pixels en 64 coefficients, chacun associé à l'un des 64
    motifs du catalogue. Chaque motif est une onde cosinus 2D de fréquence différente — du motif
    uniforme (DC, coin haut-gauche) aux motifs à rayures très fines (coins bas-droite).
    Un coefficient fort signifie que le motif correspondant est très présent dans le bloc ;
    un coefficient proche de zéro signifie qu'il est quasi-absent. La DCT est une opération
    réversible : les 64 coefficients suffisent à reconstruire exactement le bloc d'origine.

    **Le coefficient DC : la moyenne du bloc**

    Le coefficient (0, 0), appelé DC (*Direct Current*), est proportionnel à la moyenne des
    pixels du bloc. C'est presque toujours le plus grand en amplitude. Les autres coefficients
    (AC, *Alternating Current*) représentent les variations autour de cette moyenne.

    **Amplitude = énergie du motif**

    La couleur dans la heatmap encode l'amplitude de chaque coefficient — c'est-à-dire la
    quantité d'énergie que ce motif apporte à l'image. Plus la couleur est chaude, plus le motif
    est présent. Les coefficients de faible amplitude, souvent nombreux dans le coin bas-droite,
    peuvent être supprimés sans impact perceptible : l'œil ne détecte pas les très hautes
    fréquences spatiales, comme le montre la courbe de sensibilité visuelle.

    **Compaction d'énergie : la propriété clé**

    Pour une image naturelle, presque toute l'énergie du bloc est concentrée dans les
    coefficients de basse fréquence (coin haut-gauche). Les coefficients de haute fréquence
    (coin bas-droite) sont souvent très faibles, voire nuls — comme on peut le constater en
    faisant varier le bloc sélectionné. C'est cette concentration qui rend la DCT efficace
    pour la compression.

    **Ce qui arrive ensuite : quantification (étape 5)**

    JPEG exploite la compaction d'énergie en divisant chaque coefficient par une valeur issue
    d'une *table de quantification*. Les diviseurs sont grands pour les hautes fréquences
    (souvent > 10) et petits pour les basses fréquences. Après arrondi à l'entier, les
    coefficients de haute fréquence tombent naturellement à zéro — c'est l'étape suivante
    qui détaillera ce mécanisme et montrera son impact sur la qualité d'image.
    """), kind="info")
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Étape 5 : Quantification
    """)
    return


@app.cell
def _(mo: ModuleType) -> tuple[marimo.ui.slider]:
    quality_factor = mo.ui.slider(
        1, 100,
        value=50,
        label="Facteur de qualité JPEG",
        show_value=True,
    )
    quality_factor
    return (quality_factor,)


@app.cell
def _(np: ModuleType) -> tuple[
    ndarray,
    ndarray,
    Callable[[ndarray, int], ndarray],
]:
    _Q_LUMA_BASE: "ndarray" = np.array([
        [16, 11, 10, 16,  24,  40,  51,  61],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99],
    ], dtype=float)

    _Q_CHROMA_BASE: "ndarray" = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ], dtype=float)

    def scale_q(Q_base: "ndarray", quality: int) -> "ndarray":
        """Redimensionne une table JPEG selon la formule Annex K (quality ∈ [1, 95]).

        Valeurs de sortie clampées dans [1, 255].
        """
        _s: float = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        return np.clip(np.floor((Q_base * _s + 50.0) / 100.0), 1.0, 255.0)

    Q_luma:   "ndarray" = _Q_LUMA_BASE
    Q_chroma: "ndarray" = _Q_CHROMA_BASE

    return Q_luma, Q_chroma, scale_q


@app.function
def reconstruct_channel(
    channel: "ndarray",
    Q_scaled: "ndarray",
) -> "ndarray":
    """Quantifie et reconstruit un canal 2D complet par blocs 8x8.

    Applique un level-shift de -128 avant la DCT et +128 après l'iDCT,
    conformément au standard JPEG. Retourne le canal clipé à [0, 255].
    """
    import numpy as _np
    from scipy.fft import dctn as _dctn, idctn as _idctn  # type: ignore[import-untyped]
    H, W = channel.shape
    n_h, n_w = H // 8, W // 8
    _flat: "ndarray" = (
        (channel[:n_h * 8, :n_w * 8] - 128.0)
        .reshape(n_h, 8, n_w, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )
    _coeffs: "ndarray" = _dctn(_flat, axes=(1, 2), norm="ortho")
    _quant: "ndarray" = _np.round(_coeffs / Q_scaled[_np.newaxis, :, :])
    _dequant: "ndarray" = _quant * Q_scaled[_np.newaxis, :, :]
    _recon: "ndarray" = _idctn(_dequant, axes=(1, 2), norm="ortho") + 128.0
    _recon_2d: "ndarray" = (
        _recon
        .reshape(n_h, n_w, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(n_h * 8, n_w * 8)
    )
    _out: "ndarray" = channel.copy().astype(float)
    _out[:n_h * 8, :n_w * 8] = _recon_2d
    return _np.clip(_out, 0.0, 255.0)


@app.cell
def _(
    dct2: Callable[[ndarray], ndarray],
    get_block_idx: Callable[[], int],
    idct2: Callable[[ndarray], ndarray],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    quality_factor: marimo.ui.slider,
    Q_luma: ndarray,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
    scale_q: Callable[[ndarray, int], ndarray],
) -> None:
    _idx: "int" = get_block_idx()
    _q: "int" = int(quality_factor.value)
    _block: "ndarray"
    _n_h: "int"
    _n_w: "int"
    _block, _n_h, _n_w = extract_y_block(image, rgb_to_ycbcr, _idx)
    _C: "ndarray" = dct2(_block)
    _Ql: "ndarray" = scale_q(Q_luma, _q)
    _Cq: "ndarray" = np.round(_C / _Ql)
    _Cdq: "ndarray" = _Cq * _Ql
    _recon_block: "ndarray" = idct2(_Cdq) + 128.0
    _err: "ndarray" = (_block + 128.0) - _recon_block
    _n_nz: "int" = int(np.count_nonzero(_Cq))
    _err_max: "float" = float(np.max(np.abs(_err)))
    _err_mean: "float" = float(np.mean(np.abs(_err)))

    # Exemples concrets du seuil Q/2 : un coefficient supprimé, un gardé
    _example_text: "str" = ""
    _has_zero: "bool" = bool(((np.abs(_Cq) == 0) & (np.abs(_C) > 0.5)).any())
    _has_kept: "bool" = bool((np.abs(_Cq) > 0).any())
    if _has_zero:
        _iz_arr: "ndarray" = np.argwhere((np.abs(_Cq) == 0) & (np.abs(_C) > 0.5))[0]
        _iz_r: "int" = int(_iz_arr[0])
        _iz_c: "int" = int(_iz_arr[1])
        _cz: "float" = float(np.round(_C[_iz_r, _iz_c]))
        _qz: "int" = int(_Ql[_iz_r, _iz_c])
        _example_text += f"Supprimé : {_cz:+.0f} ÷ {_qz} → 0  (|{abs(_cz):.0f}| < seuil {_qz // 2})"
    if _has_kept:
        if _has_zero:
            _example_text += "\n"
        _ik_arr: "ndarray" = np.argwhere(np.abs(_Cq) > 0)[0]
        _ik_r: "int" = int(_ik_arr[0])
        _ik_c: "int" = int(_ik_arr[1])
        _ck: "float" = float(np.round(_C[_ik_r, _ik_c]))
        _qk: "int" = int(_Ql[_ik_r, _ik_c])
        _cqk: "int" = int(_Cq[_ik_r, _ik_c])
        _example_text += f"Gardé : {_ck:+.0f} ÷ {_qk} → {_cqk:+d}  (|{abs(_ck):.0f}| ≥ seuil {_qk // 2})"

    _fig, _axes = plt.subplots(2, 4, figsize=(13, 9), layout="constrained", dpi=150,
                               gridspec_kw={"width_ratios": [1, 0.12, 1, 1]})
    _axes[0, 1].axis("off")
    _axes[1, 1].axis("off")

    # Colonne gauche : table Q (opérateur) et erreur (résultat de la perte)
    _q00: "int" = int(_Ql[0, 0])
    _q77: "int" = int(_Ql[7, 7])
    _vmax_q: "float" = float(max(_Ql.max(), 1.0))
    _axes[0, 0].imshow(_Ql, cmap="Reds", vmin=1, vmax=_vmax_q, interpolation="nearest")
    for _r in range(8):
        for _c in range(8):
            _qtv: "int" = int(_Ql[_r, _c])
            _rel: "float" = (_qtv - 1.0) / max(_vmax_q - 1.0, 1.0)
            _axes[0, 0].text(_c, _r, str(_qtv), ha="center", va="center", fontsize=6,
                             fontweight="bold", color="white" if _rel > 0.55 else "black")
    _axes[0, 0].set_title(
        f"Table Q (diviseurs pour qualité {_q})\nSeuil DC ±{_q00 // 2} · Seuil HF ±{_q77 // 2}",
        fontsize=10,
    )
    _axes[0, 0].axis("off")

    _axes[1, 0].imshow(_err, cmap="RdBu_r", vmin=-30, vmax=30, interpolation="nearest")
    for _r in range(8):
        for _c in range(8):
            _ev: "int" = int(round(float(_err[_r, _c])))
            _axes[1, 0].text(_c, _r, str(_ev), ha="center", va="center", fontsize=6,
                             fontweight="bold", color="white" if abs(_ev) > 15 else "black")
    _axes[1, 0].set_title(f"Erreur pixel (orig - recon)\nmax {_err_max:.1f} · moy {_err_mean:.1f}", fontsize=10)
    _axes[1, 0].axis("off")

    # Colonne centrale : domaine spatial et fréquentiel AVANT quantification
    _axes[0, 2].imshow(_block + 128.0, cmap="gray", vmin=16, vmax=235, interpolation="nearest")
    annotate_spatial_block(_axes[0, 2], _block + 128.0)
    _axes[0, 2].set_title("Valeurs Y — avant", fontsize=10)
    _axes[0, 2].axis("off")

    _axes[1, 2].imshow(np.abs(_C), cmap="plasma", vmin=0, vmax=200, interpolation="nearest")
    annotate_dct_block(_axes[1, 2], _C)
    _axes[1, 2].set_title("Coefficients DCT — avant", fontsize=10)
    _axes[1, 2].axis("off")

    # Colonne droite : domaine spatial et fréquentiel APRÈS quantification
    _axes[0, 3].imshow(_recon_block, cmap="gray", vmin=16, vmax=235, interpolation="nearest")
    annotate_spatial_block(_axes[0, 3], _recon_block)
    _axes[0, 3].set_title("Valeurs Y — après", fontsize=10)
    _axes[0, 3].axis("off")

    _axes[1, 3].imshow(np.abs(_Cq), cmap="plasma", vmin=0, vmax=200, interpolation="nearest")
    annotate_dct_block(_axes[1, 3], _Cq, zero_color="#aaaaaa")
    _pct_nuls: "int" = 100 * (64 - _n_nz) // 64
    _axes[1, 3].set_title(f"Coefficients DCT — après ({_pct_nuls} % nuls)", fontsize=10)
    _axes[1, 3].axis("off")
    if _example_text:
        _axes[1, 3].text(0.5, -0.04, _example_text, ha="center", va="top", fontsize=7.5,
                         transform=_axes[1, 3].transAxes, fontfamily="monospace")

    _fig.suptitle(
        f"Bloc {_idx} / {_n_h * _n_w - 1} — qualité {_q} — {_pct_nuls} % de coefficients nuls",
        fontsize=12, fontweight="bold",
    )
    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)
    return


@app.cell
def _(
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    quality_factor: marimo.ui.slider,
    Q_chroma: ndarray,
    Q_luma: ndarray,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
    scale_q: Callable[[ndarray, int], ndarray],
    ycbcr_to_rgb: Callable[[ndarray, ndarray, ndarray], ndarray],
) -> None:
    _q_img: "int" = int(quality_factor.value)
    _ycbcr_f: "ndarray" = rgb_to_ycbcr(image).astype(float)
    _Ql_img: "ndarray" = scale_q(Q_luma,   _q_img)
    _Qc_img: "ndarray" = scale_q(Q_chroma, _q_img)

    _Y_rec:  "ndarray" = reconstruct_channel(_ycbcr_f[..., 0], _Ql_img)
    _Cb_rec: "ndarray" = reconstruct_channel(_ycbcr_f[..., 1], _Qc_img)
    _Cr_rec: "ndarray" = reconstruct_channel(_ycbcr_f[..., 2], _Qc_img)
    _recon_rgb: "ndarray" = ycbcr_to_rgb(_Y_rec, _Cb_rec, _Cr_rec)

    _diff: "ndarray" = np.clip(
        np.abs(image.astype(float) - _recon_rgb.astype(float)) * 5, 0, 255
    ).astype(np.uint8)

    _H_img: "int"
    _W_img: "int"
    _H_img, _W_img = image.shape[:2]
    _n_h_img: "int" = _H_img // 8
    _n_w_img: "int" = _W_img // 8
    from scipy.fft import dctn as _dctn  # type: ignore[import-untyped]
    _flat_Y: "ndarray" = (
        (_ycbcr_f[..., 0][:_n_h_img * 8, :_n_w_img * 8] - 128.0)
        .reshape(_n_h_img, 8, _n_w_img, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )
    _Cq_Y: "ndarray" = np.round(_dctn(_flat_Y, axes=(1, 2), norm="ortho") / _Ql_img)
    _pct_zero: "float" = float(100.0 * (_Cq_Y == 0).sum() / _Cq_Y.size)

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained", dpi=150)
    for _i, (_img_data, _title) in enumerate([
        (image,      "Image originale"),
        (_recon_rgb, "Image reconstruite"),
        (_diff,      "Écarts pixel |original - reconstruit| x 5"),
    ]):
        _axes[_i].imshow(_img_data, interpolation="nearest")
        _axes[_i].set_title(_title, fontsize=10)
        _axes[_i].axis("off")

    _info = mo.md(f"**Qualité : {_q_img}** — {_pct_zero:.0f} % de coefficients nuls dans le canal Y")
    _out_fig = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(mo.vstack([_info, _out_fig]))
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.callout(mo.md("""
    **La quantification : un filtre par seuillage**

    Chaque coefficient DCT est divisé par le diviseur Q de sa case, puis arrondi à
    l'entier le plus proche : `Cq = arrondi(C / Q)`. Ce simple arrondi agit comme un filtre :
    tout coefficient dont la valeur absolue est inférieure au seuil **Q / 2** tombe à zéro
    et est définitivement éliminé. Ceux qui dépassent ce seuil survivent, arrondis au multiple
    de Q le plus proche — avec une imprécision proportionnelle à Q.

    Dans une image naturelle, les coefficients hautes fréquences ont généralement de faibles
    amplitudes. Un grand diviseur Q place un seuil haut, et la plupart de ces coefficients
    tombent à zéro. Les zéros résultants sont encodés de façon extrêmement compacte à l'étape
    suivante : c'est ce mécanisme qui produit la compression.

    **Pourquoi les hautes fréquences ont-elles de grands diviseurs ?**

    Les grandes valeurs dans le coin bas-droit de la table Q correspondent aux variations
    rapides d'un pixel à l'autre (textures fines, contours nets). L'œil humain y est peu
    sensible — les effacer ne se remarque quasiment pas. Les petites valeurs en haut à gauche
    protègent le contraste global et les tons moyens, que l'œil discrimine beaucoup mieux.

    **Un niveau de qualité = une table de diviseurs**

    À chaque niveau de qualité correspond une table Q unique, calculée à partir d'une table
    de référence. Cette table de référence fixe les *proportions*
    entre les diviseurs : les hautes fréquences ont toujours des diviseurs plus grands que les
    basses fréquences. Le niveau de qualité en détermine l'*amplitude globale* : un niveau bas de qualité
    gonfle tous les diviseurs (seuils élevés, peu de coefficients survivent), un niveau élevé de qualité
    les réduit (seuils bas, presque tout est conservé).
    À qualité 100, chaque diviseur vaut 1 — quasiment aucun coefficient n'est éliminé.
    Les valeurs DCT ne sont toutefois pas des entiers, et l'arrondi au multiple de 1
    le plus proche introduit de légères imprécisions (inférieures à 1 niveau de gris,
    imperceptibles à l'œil).

    **Artéfacts de bloc (*blocking*)**

    En dessous de qualité 20, chaque bloc 8x8 est reconstruit avec très peu de coefficients,
    parfois uniquement le terme DC (la luminosité moyenne du bloc). Les discontinuités aux
    frontières entre blocs deviennent visibles — c'est le défaut caractéristique de la
    compression JPEG agressive.
    """), kind="info")
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.md("""
    ## Étape 6 : Codage entropique
    """)
    return


@app.cell
def _(np: ModuleType) -> tuple[
    ndarray,
    Callable[[ndarray], ndarray],
    Callable[[ndarray], list[tuple[int, int]]],
    Callable[[int], int],
    Callable[[int, int], int],
]:
    # Ordre de parcours zigzag JPEG (position 0 = DC, 63 = coin bas-droit)
    _ZZ_SCAN: "list[tuple[int, int]]" = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
    ]

    # ZIGZAG_IDX[r, c] = numéro de position dans la séquence zigzag (0-63)
    ZIGZAG_IDX: "ndarray" = np.zeros((8, 8), dtype=int)
    for _pos, _rc in enumerate(_ZZ_SCAN):
        _zr: "int" = _rc[0]
        _zc: "int" = _rc[1]
        ZIGZAG_IDX[_zr, _zc] = _pos

    def zigzag_scan(block: "ndarray") -> "ndarray":
        """Retourne les 64 coefficients d'un bloc 8x8 dans l'ordre zigzag JPEG (DC en tête)."""
        import numpy as _np
        return _np.array([block[r, c] for r, c in _ZZ_SCAN])

    def rle_encode_ac(seq: "ndarray") -> "list[tuple[int, int]]":
        """Encode les 63 coefficients AC (positions 1-63) en paires RLE JPEG.

        Chaque paire (run, valeur) : run = zéros précédents (0-15).
        ZRL = (15, 0) si 16 zéros consécutifs précèdent une valeur non nulle.
        EOB = (0, 0) en terminaison ; les zéros de queue ne génèrent pas de ZRL.
        """
        import numpy as _np
        _ac: "ndarray" = seq[1:]
        _nz: "ndarray" = _np.where(_ac != 0)[0]
        _last_nz: "int" = int(_nz[-1]) if len(_nz) > 0 else -1
        _pairs: "list[tuple[int, int]]" = []
        _run: "int" = 0
        for _i, _v in enumerate(_ac):
            if _i > _last_nz:
                break
            _iv: "int" = int(round(float(_v)))
            if _iv == 0:
                _run += 1
                if _run == 16:
                    _pairs.append((15, 0))
                    _run = 0
            else:
                _pairs.append((_run, _iv))
                _run = 0
        _pairs.append((0, 0))
        return _pairs

    # Tables de Huffman luminance JPEG Annexe K — longueurs de code en bits (hors bits de catégorie)
    _DC_LENS: "list[int]" = [2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9]

    _AC_LENS: "dict[tuple[int, int], int]" = {
        (0, 0): 4,   (15, 0): 11,
        (0,1):2,  (0,2):2,  (0,3):3,  (0,4):4,  (0,5):5,  (0,6):7,  (0,7):8,  (0,8):10,
        (1,1):4,  (1,2):5,  (1,3):7,  (1,4):9,  (1,5):11,
        (2,1):5,  (2,2):8,  (2,3):10, (2,4):12,
        (3,1):6,  (3,2):9,  (3,3):12,
        (4,1):6,  (4,2):10,
        (5,1):7,  (5,2):11,
        (6,1):7,  (6,2):12,
        (7,1):8,  (7,2):12,
        (8,1):9,  (8,2):15,
        (9,1):9,  (10,1):9, (11,1):10, (12,1):10,
        (13,1):11, (14,1):11, (15,1):12,
    }

    def huffman_dc_bits(diff: int) -> int:
        """Bits totaux pour encoder un différentiel DC (table Annexe K luminance).

        Total = longueur_code(catégorie) + catégorie.
        Catégorie = nombre de bits pour représenter abs(diff).
        """
        _cat: "int" = int(abs(diff)).bit_length()
        return _DC_LENS[min(_cat, 11)] + _cat

    def huffman_ac_bits(run: int, value: int) -> int:
        """Bits totaux pour encoder une paire RLE AC (table Annexe K luminance).

        EOB (0,0) retourne la longueur EOB seule. Autres : longueur_code + catégorie.
        Renvoie 16 pour les paires absentes de la table standard.
        """
        if run == 0 and value == 0:
            return _AC_LENS.get((0, 0), 4)
        _cat: "int" = int(abs(value)).bit_length()
        return _AC_LENS.get((run, _cat), 16) + _cat

    return ZIGZAG_IDX, zigzag_scan, rle_encode_ac, huffman_dc_bits, huffman_ac_bits


@app.cell
def _(
    ZIGZAG_IDX: ndarray,
    dct2: Callable[[ndarray], ndarray],
    get_block_idx: Callable[[], int],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    plt: ModuleType,
    quality_factor: marimo.ui.slider,
    Q_luma: ndarray,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
    scale_q: Callable[[ndarray, int], ndarray],
    zigzag_scan: Callable[[ndarray], ndarray],
) -> None:
    from matplotlib.collections import LineCollection as _LineCollection

    _idx: "int" = get_block_idx()
    _q: "int" = int(quality_factor.value)
    _block: "ndarray"
    _n_h: "int"
    _n_w: "int"
    _block, _n_h, _n_w = extract_y_block(image, rgb_to_ycbcr, _idx)
    _Ql: "ndarray" = scale_q(Q_luma, _q)
    _Cq: "ndarray" = np.round(dct2(_block) / _Ql)
    _seq: "ndarray" = zigzag_scan(_Cq)
    _n_nz: "int" = int(np.count_nonzero(_Cq))

    # Ordre zigzag reconstruit depuis ZIGZAG_IDX pour les segments du chemin
    _flat_order: "ndarray" = np.argsort(ZIGZAG_IDX.ravel())
    _ZZ_RC: "list[tuple[int, int]]" = [(int(k // 8), int(k % 8)) for k in _flat_order]

    _fig, _axd = plt.subplot_mosaic(
        [[".", "block", "."], ["seq", "seq", "seq"]],
        figsize=(12, 9), layout="constrained", dpi=150,
        width_ratios=[1, 1, 1], height_ratios=[1.2, 1],
    )
    _ax_b: "Axes" = _axd["block"]
    _ax_s: "Axes" = _axd["seq"]

    # Panneau gauche : coefficients + chemin zigzag
    _ax_b.imshow(np.abs(_Cq), cmap="plasma", vmin=0, vmax=200, interpolation="nearest")
    annotate_dct_block(_ax_b, _Cq, fontsize=10, zero_color="#aaaaaa", zorder=4)

    _segs: "list[list[tuple[float, float]]]" = [
        [(_ZZ_RC[_i][1], _ZZ_RC[_i][0]), (_ZZ_RC[_i + 1][1], _ZZ_RC[_i + 1][0])]
        for _i in range(63)
    ]
    _lc = _LineCollection(_segs, cmap="cool", linewidths=1.5, alpha=0.8, zorder=2)
    _lc.set_array(np.linspace(0.0, 1.0, 63))
    _ax_b.add_collection(_lc)
    _ax_b.scatter([_ZZ_RC[0][1]], [_ZZ_RC[0][0]], s=35, color="white", zorder=3)
    _ax_b.set_title(
        f"Bloc {_idx} / {_n_h * _n_w - 1} — qualité {_q}\n"
        "Chemin zigzag",
        fontsize=10,
    )
    _ax_b.axis("off")

    # Panneau bas : séquence complète (position 0 = DC, 1-63 = AC)
    _bar_colors: "list[str]" = [
        "#cccccc" if v == 0 else ("#d04020" if v > 0 else "#2050c0")
        for v in _seq
    ]
    _ax_s.bar(np.arange(64), _seq, color=_bar_colors, width=0.85)
    _ax_s.axhline(0, color="black", linewidth=0.5)
    _nz_idx: "ndarray" = np.where(_seq != 0)[0]
    _last_nz: "int" = int(_nz_idx[-1]) if len(_nz_idx) > 0 else 0
    _ax_s.axvline(_last_nz + 0.5, color="#cc3333", linewidth=1.2, linestyle=":")
    _yrange: "float" = float(max(abs(float(_seq.max())), abs(float(_seq.min())), 1.0))
    _ax_s.text(_last_nz + 1.2, _yrange * 0.92, "EOB", fontsize=7, color="#cc3333", ha="left")
    for _i in range(64):
        _sv: "int" = int(_seq[_i])
        if _sv > 0:
            _ax_s.text(_i, float(_sv) + _yrange * 0.03, str(_sv),
                       ha="center", va="bottom", fontsize=10, color="#d04020")
        elif _sv < 0:
            _ax_s.text(_i, float(_sv) - _yrange * 0.03, str(_sv),
                       ha="center", va="top", fontsize=10, color="#2050c0")
        else:
            _ax_s.text(_i, _yrange * 0.03, "0",
                       ha="center", va="bottom", fontsize=10, color="#aaaaaa")
    _ax_s.set_ylim(-_yrange * 1.35, _yrange * 1.35)
    _ax_s.set_xlim(-0.5, 63.5)
    _ax_s.set_xlabel("Position zigzag", fontsize=9)
    _ax_s.set_ylabel("Valeur", fontsize=9)
    _ax_s.set_title(
        f"Séquence 1D",
        fontsize=10,
    )
    _ax_s.spines[["top", "right"]].set_visible(False)
    _ax_s.tick_params(labelsize=7)

    _out = mo.as_html(_fig)
    plt.close(_fig)
    mo.output.replace(_out)
    return


@app.cell
def _(
    dct2: Callable[[ndarray], ndarray],
    get_block_idx: Callable[[], int],
    huffman_ac_bits: Callable[[int, int], int],
    huffman_dc_bits: Callable[[int], int],
    image: ndarray,
    mo: ModuleType,
    np: ModuleType,
    quality_factor: marimo.ui.slider,
    Q_luma: ndarray,
    rgb_to_ycbcr: Callable[[ndarray], ndarray],
    rle_encode_ac: Callable[[ndarray], list[tuple[int, int]]],
    scale_q: Callable[[ndarray, int], ndarray],
    zigzag_scan: Callable[[ndarray], ndarray],
) -> None:
    _idx: "int" = get_block_idx()
    _q: "int" = int(quality_factor.value)
    _block: "ndarray"
    _n_h: "int"
    _n_w: "int"
    _block, _n_h, _n_w = extract_y_block(image, rgb_to_ycbcr, _idx)
    _Ql: "ndarray" = scale_q(Q_luma, _q)
    _Cq: "ndarray" = np.round(dct2(_block) / _Ql)
    _seq: "ndarray" = zigzag_scan(_Cq)
    _pairs: "list[tuple[int, int]]" = rle_encode_ac(_seq)

    _dc_bits: "int" = huffman_dc_bits(int(round(float(_Cq[0, 0]))))
    _ac_bits: "int" = sum(huffman_ac_bits(r, v) for r, v in _pairs)
    _total_bits: "int" = _dc_bits + _ac_bits
    _n_pairs: "int" = len(_pairs)

    # Paires RLE formatées
    _chips: "list[str]" = []
    for _rr, _vv in _pairs:
        if _rr == 0 and _vv == 0:
            _chips.append("**`EOB`**")
        elif _rr == 15 and _vv == 0:
            _chips.append("`ZRL`")
        else:
            _chips.append(f"`({_rr}, {_vv:+d})`")

    _rle_md = mo.md(
        f"**RLE — {_n_pairs} paires** (dont EOB) : "
        + " · ".join(_chips)
    )

    _huffman_table_md = mo.md("""
Chaque coefficient non nul occupe **deux zones** dans le fichier :

1. **① Identifier le type** : un code Huffman qui dit *quel événement* — combien de zéros précèdent, et dans quelle plage de magnitude se trouve la valeur. Les événements fréquents reçoivent les codes les plus courts (norme JPEG Annexe K).
2. **② Préciser la valeur** : quelques bits supplémentaires pour donner la valeur exacte à l'intérieur de la plage. `(0, ±1)` a deux possibilités : `(0, -1)` ou `(0, +1)` → 1 bit suffit. `(0, ±2/±3)` a quatre possibilités → 2 bits.

*Extrait — 8 événements parmi les plus courants (table complète : 162 entrées).*

| Paire RLE<br>(type d'évènement) | ①<br>Code Huffman | ①<br>Longueur<br>(bits) | ②<br>Valeurs possibles | ②<br>Bits pour la valeur | Total<br>① + ②<br>(bits) |
|-----------|:--------------:|:-----------------:|:-----------------:|:---------------------:|:----------------------------:|
| `EOB` | `1010` | 4 | 1 | 0 | **4** |
| `(0, ±1)` | `00` | 2 | 2 | 1 | **3** |
| `(0, ±2)` ou `(0, ±3)` | `01` | 2 | 4 | 2 | **4** |
| `(0, ±4)` à `(0, ±7)` | `100` | 3 | 8 | 3 | **6** |
| `(1, ±1)` | `1100` | 4 | 2 | 1 | **5** |
| `(1, ±2)` ou `(1, ±3)` | `11011` | 5 | 4 | 2 | **7** |
| `(2, ±1)` | `11100` | 5 | 2 | 1 | **6** |
| `(3, ±1)` | `111010` | 6 | 2 | 1 | **7** |
""")

    _ratio: "float" = 512.0 / _total_bits if _total_bits > 0 else float("inf")
    _pct_final: "float" = 100.0 * _total_bits / 512.0
    _compression_md = mo.md(
        f"**Volume du bloc — avant / après**\n\n"
        f"| | Bits |\n"
        f"|---|---:|\n"
        f"| Brut (8 bits x 64 coefficients) | **512** |\n"
        f"| Encodé (DC {_dc_bits} bits + AC {_ac_bits} bits) | **{_total_bits}** |\n"
        f"| Poids final / poids original | **{_pct_final:.1f} %** |\n"
        f"| Ratio de compression | **{_ratio:.1f}x** |"
    )

    mo.output.replace(mo.vstack([_rle_md, mo.md("<br><br>"), _huffman_table_md, mo.md("<br><br>"), _compression_md]))
    return


@app.cell
def _(mo: ModuleType) -> None:
    mo.callout(mo.md("""
    **Étape 1 — le parcours zigzag**

    Les 64 coefficients d'un bloc 8x8 sont lus dans un ordre en zigzag : on part du coin
    haut-gauche (DC — luminosité moyenne) et on parcourt les diagonales successives jusqu'au
    coin bas-droit (hautes fréquences). Cet ordre place en tête les coefficients les plus
    importants et regroupe en queue les coefficients hautes fréquences — souvent nuls après
    quantification — créant de longues plages de zéros consécutifs.

    **Étape 2 — le codage RLE** *(Run-Length Encoding — codage par plages)*

    La séquence AC (63 valeurs après le DC) est encodée en paires `(run, valeur)` :
    `run` = nombre de zéros qui précèdent la valeur non nulle. Une seule paire remplace
    souvent 5, 10 ou 20 cases brutes.

    Deux symboles spéciaux complètent l'encodage :
    - `EOB` (*End Of Block*) : tous les coefficients restants sont nuls. 4 bits suffisent
      pour clore le bloc, quelle que soit la longueur de la plage de zéros finale.
    - `ZRL` (*Zero Run Length*) : exactement 16 zéros consécutifs sans valeur non nulle
      entre eux. Nécessaire car `run` est limité à 15 — au-delà, on empile des `ZRL`.

    **Étape 3 — le codage Huffman**

    Principe : les événements fréquents reçoivent des codes courts, les rares des codes longs.
    Comme le code Morse où le « E » (lettre la plus commune) est encodé sur un seul point,
    tandis que le « Q » en demande quatre.

    En JPEG, après quantification, la grande majorité des coefficients AC est nulle.
    Les valeurs survivantes sont presque toujours petites (±1, ±2). Un coefficient ±1
    sans zéro devant ne coûte que **3 bits** au total (deuxième ligne du tableau).
    Un coefficient ±1 précédé de trois zéros en coûte **7** (dernière ligne du tableau).
    Les événements non listés — valeurs plus grandes ou davantage de zéros — coûtent encore plus.

    Le code Huffman n'encode pas la valeur exacte : il encode *le type d'événement* —
    combien de zéros précèdent, et dans quelle plage de magnitude se trouve la valeur.
    Les bits suivant servent à préciser la valeur exacte parmi les différentes possibilités de l'évènement.
    Plus la quantification est forte, plus les coefficients non nuls sont rares
    et petits — plus les codes sont courts et le fichier léger.
    """), kind="info")
    return


if __name__ == "__main__":
    app.run()
