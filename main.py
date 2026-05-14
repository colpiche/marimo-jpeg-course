from __future__ import annotations

import marimo
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.colors import LinearSegmentedColormap
    from numpy import ndarray
    from types import ModuleType

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
    _CMAPS: dict[str, list[str | LinearSegmentedColormap]] = {
        "RGB": [
            LinearSegmentedColormap.from_list("R", [(0, 0, 0), (1, 0, 0)]),
            LinearSegmentedColormap.from_list("G", [(0, 0, 0), (0, 1, 0)]),
            LinearSegmentedColormap.from_list("B", [(0, 0, 0), (0, 0, 1)]),
        ],
        "YCbCr": [
            "gray",
            LinearSegmentedColormap.from_list("Cb_BT601", [
                (130/255, 174/255,   0/255),  # Cb=16  : jaune-vert
                (130/255, 130/255, 130/255),  # Cb=128 : gris neutre
                (130/255,  86/255, 255/255),  # Cb=240 : bleu vif
            ]),
            LinearSegmentedColormap.from_list("Cr_BT601", [
                (  0/255, 221/255, 130/255),  # Cr=16  : cyan-vert
                (130/255, 130/255, 130/255),  # Cr=128 : gris neutre
                (255/255,  39/255, 130/255),  # Cr=240 : rouge vif
            ]),
        ],
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

    return (get_channels, ycbcr_to_rgb, rgb_to_ycbcr)


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
    mo.hstack([sampling_mode], justify="start")
    return (sampling_mode,)


@app.cell
def _(
    LinearSegmentedColormap: type[LinearSegmentedColormap],
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

    _block_label: "str" = {"4:4:4": "1x1", "4:2:2": "2x1", "4:2:0": "2x2"}[sampling_mode.value]
    _gs_w: "int" = {"4:4:4": 1, "4:2:2": 2, "4:2:0": 2}[sampling_mode.value]
    _gs_h: "int" = {"4:4:4": 1, "4:2:2": 1, "4:2:0": 2}[sampling_mode.value]
    _PH, _PW = 8, 8

    # Colormaps BT.601 pour Cb et Cr (mêmes ancres que l'étape 1)
    _cb_cmap = LinearSegmentedColormap.from_list("Cb_BT601", [
        (130/255, 174/255,   0/255),
        (130/255, 130/255, 130/255),
        (130/255,  86/255, 255/255),
    ])
    _cr_cmap = LinearSegmentedColormap.from_list("Cr_BT601", [
        (  0/255, 221/255, 130/255),
        (130/255, 130/255, 130/255),
        (255/255,  39/255, 130/255),
    ])

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
            (_ycbcr_up[..., 1], _cb_cmap, "Cb — blocs " + _block_label, 16, 240, _gs_w, _gs_h),
            (_ycbcr_up[..., 2], _cr_cmap, "Cr — blocs " + _block_label, 16, 240, _gs_w, _gs_h),
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
            for _r in range(8):
                for _c in range(8):
                    _val = int(round(float(_block[_r, _c])))
                    _ax.text(
                        _c, _r, str(_val), ha="center", va="center",
                        fontsize=6, fontweight="bold",
                        # blanc sur foncé (< 128), noir sur clair
                        color="white" if _val < 128 else "black",
                    )
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
    mo.md("## Étape 4 : Transformée en cosinus discrète (DCT)")
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
    _ycbcr: "ndarray" = rgb_to_ycbcr(image)
    _Y: "ndarray" = _ycbcr[..., 0]
    _h: "int"
    _w: "int"
    _h, _w = _Y.shape
    _n_w: "int" = _w // 8
    _n_h: "int" = _h // 8
    _idx: "int" = get_block_idx()
    _by: "int" = (_idx // _n_w) * 8
    _bx: "int" = (_idx % _n_w) * 8

    _block: "ndarray" = _Y[_by:_by + 8, _bx:_bx + 8].astype(float) - 128.0
    _C: "ndarray" = dct2(_block)

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", dpi=150)

    # Panneau 1 — Bloc Y original avec valeurs numériques
    _axs[0].imshow(_block + 128.0, cmap="gray", vmin=16, vmax=235, interpolation="nearest")
    for _r in range(8):
        for _c in range(8):
            _v: "int" = int(round(float(_block[_r, _c] + 128.0)))
            _axs[0].text(_c, _r, str(_v), ha="center", va="center", fontsize=6,
                         fontweight="bold", color="white" if _v < 128 else "black")
    _axs[0].set_title(f"Valeurs Y", fontsize=9)
    _axs[0].axis("off")

    # Panneau 2 — Amplitude |C| en couleur (plasma), valeur entière signée en texte.
    # Luminance perceptuelle de la couleur plasma pour garantir la lisibilité du texte.
    _axs[1].imshow(np.abs(_C), cmap="plasma", vmin=0, vmax=200, interpolation="nearest")
    for _r in range(8):
        for _c in range(8):
            _cv: "int" = int(round(float(_C[_r, _c])))
            _amp: "float" = abs(float(_C[_r, _c]))
            _rgba_bg: "tuple[float, float, float, float]" = plt.cm.plasma(min(_amp / 200.0, 1.0))
            _lum: "float" = 0.299 * float(_rgba_bg[0]) + 0.587 * float(_rgba_bg[1]) + 0.114 * float(_rgba_bg[2])
            _axs[1].text(_c, _r, str(_cv), ha="center", va="center", fontsize=6,
                         fontweight="bold", color="black" if _lum > 0.45 else "white")
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
