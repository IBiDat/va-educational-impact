#########################################################################################################################################################

import math
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from scipy import stats

sns.set_style('whitegrid')

#########################################################################################################################################################
# ---------------------------------------------------------------------------
# Palette & theme constants
# ---------------------------------------------------------------------------
_DARK_BG     = "#0e1117"
_CARD_BG     = "#1a1f2e"
_RICH_COLORS = [
    "#5B8CFF", "#FF6B6B", "#43E97B", "#FFD93D",
    "#C77DFF", "#FF8C42", "#00D4FF", "#FF4D9E",
]

# ---------------------------------------------------------------------------
# Fixed colours for control / experimental groups.
# These take precedence over any auto-generated palette everywhere in the file.
# ---------------------------------------------------------------------------
GROUP_PALETTE = {
    "control":      "#66C2A5",   # Set2-1 — teal green
    "experimental": "#FC8D62",   # Set2-2 — warm orange

    "ExpA":  "#8DA0CB",          # Set2-3 — muted blue
    "ExpAA": "#8DA0CB",

    "ExpAB": "#FFD92FFF",          # Set2-4 — soft pink
    "ExpBA": "#A6D854",          # Set2-5 — lime green

    "ExpB":  "#E78AC3",          # Set2-6 — yellow
    "ExpBB": "#E78AC3",

    "ExpNotUsed": "#B3B3B3",     # Set2-7 — tan/beige
}

def resolve_palette(keys, base_palette="Set2", override=None):
    """
    Build a colour dict for *keys*, guaranteeing that any key present in
    GROUP_PALETTE (case-insensitive match) receives its fixed colour.
    All other keys are assigned colours from *base_palette*.

    Parameters
    ----------
    keys : list[str]
        The category / group values that need a colour.
    base_palette : str
        Seaborn palette name used for keys NOT in GROUP_PALETTE.
    override : dict | None
        Caller-supplied palette that takes absolute precedence over everything,
        including GROUP_PALETTE (pass-through for explicit ``cat_palette``).

    Returns
    -------
    dict[str, colour]
    """
    if override:
        return override

    result = {}
    # Assign base colours first so every key has *something*
    base_colors = sns.color_palette(base_palette, len(keys))
    for k, c in zip(keys, base_colors):
        result[k] = c

    #Lower case every element in GROUP_PALETTE
    LOWERCASE_PALETE = {
        key.lower(): value 
        if isinstance(value, str) else value 
        for key, value in GROUP_PALETTE.items()
    }

    # Overwrite with GROUP_PALETTE where applicable (case-insensitive)
    for k in keys:
        match = LOWERCASE_PALETE.get(str(k).lower())
        
        if match:
            result[k] = match

    return result

#########################################################################################################################################################
def rename_df(
    df: pl.DataFrame
) -> pl.DataFrame:
    
    #Replace traditional scale
    mapping = {
        "sobresaliente": "Excellent",
        "notable": "Very Good",
        "bien": "Good",
        "suficiente": "Pass",
        "suspenso": "Fail"
    }
    
    df = df.with_columns(
        pl.col("puntuacion_tc_cat_trad_scale_pre").replace(mapping),
        pl.col("puntuacion_tc_cat_trad_scale_post").replace(mapping)
    )
    
    #Replace mejora_hake_gain
    mapping_mejora = {
        "Mejora": "Improve",
        "No Mejora": "Not Improve",
        "Empeora": "Worsen"
    }
    
    df = df.with_columns(
        pl.col("mejora_hake_gain_v2").replace(mapping_mejora)
    )
    
    #Replace WSDI_cat
    mapping_wdsi = {
        "No Usado": "Not Used",
        "Out_of_context": "Out of context",
        "Profunda": "Deep"
    }
    
    df = df.with_columns(
        pl.col("WSDI_cat_v2").replace(mapping_wdsi)
    )
    
    #Replace Alta/Media/Baja
    mapping_levels = {
        "No Usado": "Not Used",
        "Baja": "Low",
        "Media": "Medium",
        "Alta": "High"
    }
    
    df = df.with_columns(
        pl.col("chat_freq_use").replace(mapping_levels),
        pl.col("puntuacion_tc_cat_pre").replace(mapping_levels),
        pl.col("puntuacion_tc_retencion_cat_pre").replace(mapping_levels),
        pl.col("puntuacion_tc_transferencia_cat_pre").replace(mapping_levels),
        pl.col("puntuacion_tc_cat_post").replace(mapping_levels),
        pl.col("puntuacion_tc_retencion_cat_post").replace(mapping_levels),
        pl.col("puntuacion_tc_transferencia_cat_post").replace(mapping_levels),
        pl.col("puntuacion_tcc_rel_cat_post").replace(mapping_levels),
        pl.col("puntuacion_tcc_ext_cat_post").replace(mapping_levels),
        pl.col("puntuacion_tcc_int_cat_post").replace(mapping_levels),
    )
    
    
    #Transform puntuation into score in every column
    df = df.rename({
        col: col.replace('puntuacion', 'score') for col in df.columns
    })
    
    df = df.rename({
        col: col.replace('retencion', 'retention') for col in df.columns
    })
    
    df = df.rename({
        col: col.replace('transferencia', 'transfer') for col in df.columns
    })
    
    return df
    


#########################################################################################################################################################

def group_stats(df, cols, group_by):
    stats = [
        expr
        for col in cols
        for expr in [
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).quantile(0.25).alias(f"{col}_q25"),
            pl.col(col).quantile(0.50).alias(f"{col}_median"),
            pl.col(col).quantile(0.75).alias(f"{col}_q75"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
        ]
    ]
    return df.group_by(group_by).agg(stats)

#########################################################################################################################################################

def plot_cat_distribution(
    df, 
    cat_cols, order=None, max_cols=3, 
    palette="Set2", sharey=False, x_rotation=30, 
    title=True, show_counts=False, save_path=None):
    """
    Crea un multiplot adaptativo mostrando la proporción de las categorías 
    para una lista de columnas de un DataFrame de Polars.
    
    Parámetros:
    - df: DataFrame de Polars.
    - cat_cols: Lista de strings con los nombres de las columnas a graficar.
    - max_cols: Número máximo de gráficos por fila (por defecto 3).
    - palette: Paleta de colores de Seaborn a utilizar (por defecto "Set2").
    """
    
    n_vars = len(cat_cols)
    
    if n_vars == 0:
        print("Aviso: La lista de columnas está vacía.")
        return

    # --- 1. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_vars, max_cols)
    n_rows_fig = math.ceil(n_vars / n_cols_fig) 

    fig, axes = plt.subplots(nrows=n_rows_fig, ncols=n_cols_fig, figsize=(6 * n_cols_fig, 5 * n_rows_fig), sharey=sharey)

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes[None, :]
    elif n_cols_fig == 1:
        axes = axes[:, None]

    # --- 2. DIBUJAR LOS GRÁFICOS ---
    for i, col in enumerate(cat_cols):
        
        r = i // n_cols_fig
        c = i % n_cols_fig
        ax = axes[r, c]
        
        serie_str = df[col].fill_null("Nulo").cast(pl.String).to_pandas()
        total = len(serie_str)
        
        sns.countplot(
            x=serie_str, 
            ax=ax, 
            palette=palette, 
            hue=serie_str, 
            legend=False, 
            stat='proportion',
            order=order
        )
        
        if show_counts:
            # Superponer el count en cada barra
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:
                    count = round(height * total)
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,  # centro horizontal
                        height,                          # justo encima de la barra
                        f"n={count}",
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold',
                        color='#333333'
                    )
        
        if title:
            ax.set_title(col.upper(), fontsize=12, fontweight='bold')
            
        ax.set_xlabel('')
        ax.set_ylabel('Proporción')
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=12)

    # --- 3. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_fig * n_cols_fig
    for i in range(n_vars, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    # --- 4. RENDERIZADO ---
    plt.tight_layout()
    plt.show()
    
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

#########################################################################################################################################################

def plot_quant_distribution(
    df, quant_cols, max_cols=3, box_color="skyblue", 
    hist_color="salmon", save_path=None):
    """
    Crea un multiplot adaptativo mostrando un Boxplot y un Histograma (con proporciones)
    para una lista de columnas cuantitativas de un DataFrame de Polars.
    
    Parámetros:
    - df: DataFrame de Polars.
    - quant_cols: Lista de strings con los nombres de las columnas a graficar.
    - max_cols: Número máximo de columnas de variables por fila (por defecto 3).
    - box_color: Color para los boxplots (por defecto "skyblue").
    - hist_color: Color para los histogramas (por defecto "salmon").
    """
    n_vars = len(quant_cols)
    
    # Validación por si la lista viene vacía
    if n_vars == 0:
        print("Aviso: La lista de columnas cuantitativas está vacía.")
        return

    # --- 1. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_vars, max_cols)
    
    # Calculamos cuántas "filas de variables" necesitamos
    n_rows_vars = math.ceil(n_vars / n_cols_fig) 
    
    # Como cada variable usa 2 filas reales (boxplot + hist), multiplicamos por 2
    n_rows_fig = n_rows_vars * 2

    # Ajustamos el tamaño de la figura dinámicamente
    fig, axes = plt.subplots(nrows=n_rows_fig, ncols=n_cols_fig, figsize=(6 * n_cols_fig, 3 * n_rows_fig))

    # Forzamos a que axes sea siempre un array 2D
    if n_rows_fig == 2 and n_cols_fig == 1:
        axes = axes.reshape(2, 1)

    # --- 2. DIBUJAR LOS GRÁFICOS ---
    for i, col in enumerate(quant_cols):
        
        # Mágia matemática para encontrar la coordenada exacta
        r = (i // n_cols_fig) * 2  
        c = i % n_cols_fig         
        
        ax_box = axes[r, c]
        ax_hist = axes[r + 1, c] 
        
        # Compatibilidad Polars -> Pandas (limpiamos nulos para que el KDE no falle)
        serie_num = df[col].drop_nulls().to_pandas()
        
        # --- BOXPLOT ---
        sns.boxplot(x=serie_num, ax=ax_box, color=box_color)
        ax_box.set_title(col.upper(), fontsize=12, fontweight='bold')
        ax_box.set_xlabel('')
        ax_box.tick_params(axis='x', labelsize=12)
        
        # --- HISTOGRAMA ---
        sns.histplot(x=serie_num, kde=True, ax=ax_hist, color=hist_color, edgecolor="black", stat='proportion')
        ax_hist.set_title('', fontsize=12, fontweight='bold')
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel('Proporción')
        ax_hist.tick_params(axis='x', labelsize=12)

    # --- 3. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_vars * n_cols_fig
    for i in range(n_vars, total_blocks):
        r = (i // n_cols_fig) * 2
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])
        fig.delaxes(axes[r + 1, c])

    # --- 4. RENDERIZADO ---
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

#########################################################################################################################################################

def plot_quant_comparison(
    df, comparisons, group_by=None, figsize=None, showfliers=True, 
    order=None, labelbottom=True, xlabel_rotation=30, max_cols=3, 
    title=None, palette="Set2", bbox_to_anchor=(0.5, -0.03), 
    save_path=None):
    
    n_blocks = len(comparisons)

    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_blocks = math.ceil(n_blocks / n_cols_fig)
    n_rows_fig = n_rows_blocks

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(6 * n_cols_fig, 4 * n_rows_fig) if not figsize else figsize
    )

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    if group_by:
        group_vals = df[group_by].drop_nulls().unique().sort().to_list()
        hue_order = group_vals
        x_order_grouped = order if order else group_vals
        group_color_map = resolve_palette(x_order_grouped, base_palette=palette)

    # --- 2. DIBUJAR LOS BLOQUES ---
    for i, col_group in enumerate(comparisons):

        r = i // n_cols_fig
        c = i % n_cols_fig
        ax_box = axes[r, c]

        # --- CONSTRUCCIÓN DEL TIDY ---
        if group_by:
            pdf = df.select(col_group + [group_by]).to_pandas()
            tidy = pdf.melt(
                id_vars=group_by,
                value_vars=col_group,
                var_name="variable",
                value_name="valor"
            ).dropna(subset=["valor", group_by])
        else:
            pdf = df.select(col_group).to_pandas()
            tidy = pdf.melt(
                var_name="variable",
                value_name="valor"
            ).dropna()

        # --- TÍTULO ---
        if title is True:
            block_title = " vs ".join(col.upper() for col in col_group)
            if group_by:
                block_title += f"\n(por {group_by})"
            if not showfliers:
                block_title += "\n(Outliers Hidden)"
            ax_box.set_title(block_title, fontsize=11, fontweight="bold", y=1.05)
        else:
            if title is not None:
                ax_box.set_title(title, fontsize=11, fontweight="bold", y=1.05)

        # --- BOXPLOT ---
        if group_by and len(col_group) == 1:
            x_order_filtered = [g for g in x_order_grouped if g in tidy[group_by].values]

            for j, group_val in enumerate(x_order_filtered):
                mask = tidy[group_by] == group_val
                subset = tidy.loc[mask, "valor"].dropna()
                if len(subset) == 0:
                    continue

                ax_box.boxplot(
                    subset,
                    positions=[j],
                    widths=0.5,
                    showfliers=showfliers,
                    patch_artist=True,
                    boxprops=dict(facecolor=group_color_map[group_val], alpha=0.7),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker='o', markerfacecolor=group_color_map[group_val], markersize=4, alpha=0.5)
                )

                mean_val = subset.mean()
                ax_box.plot(
                    j, mean_val, marker="D",
                    color=group_color_map[group_val],
                    markersize=6,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=5
                )

            ax_box.set_xticks(range(len(x_order_filtered)))
            ax_box.set_xticklabels(x_order_filtered)

        elif group_by and len(col_group) > 1:
            sns.boxplot(
                data=tidy, x="variable", y="valor",
                hue=group_by, showfliers=showfliers,
                palette=group_color_map, ax=ax_box,
                width=0.5, linewidth=1.5,
                legend=False, hue_order=hue_order,
                order=col_group
            )
            for j, col in enumerate(col_group):
                for g, group_val in enumerate(hue_order):
                    mask = (tidy["variable"] == col) & (tidy[group_by] == group_val)
                    if mask.sum() == 0:
                        continue
                    mean_val = tidy.loc[mask, "valor"].mean()
                    n_groups = len(hue_order)
                    offset = (g - (n_groups - 1) / 2) * (0.5 / n_groups)
                    ax_box.plot(
                        j + offset, mean_val, marker="D",
                        color=group_color_map[group_val],
                        markersize=6,
                        markeredgecolor="black", markeredgewidth=0.8,
                        zorder=5
                    )

        else:
            x_order = order if order else col_group
            var_color_map = resolve_palette(col_group, base_palette=palette)
            sns.boxplot(
                data=tidy, x="variable", y="valor",
                hue="variable", showfliers=showfliers,
                palette=var_color_map, ax=ax_box,
                width=0.5, linewidth=1.5,
                legend=False,
                order=x_order
            )
            for j, col in enumerate(x_order):
                mean_val = tidy[tidy["variable"] == col]["valor"].mean()
                ax_box.plot(
                    j, mean_val, marker="D",
                    color=var_color_map[col],
                    markersize=7,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=5
                )

        ax_box.set_xlabel("")
        ax_box.set_ylabel("")
        ax_box.tick_params(axis="x", labelbottom=labelbottom, rotation=xlabel_rotation)

    # --- 3. LEYENDA GLOBAL MANUAL ---
    if group_by:
        legend_keys = x_order_grouped if len(comparisons[0]) == 1 else hue_order
        handles = [
            mpatches.Patch(color=group_color_map[g], alpha=0.7, label=str(g))
            for g in legend_keys if g in group_color_map
        ]
        legend_ncol = len(handles)
    else:
        all_vars = [col for col_group in comparisons for col in col_group]
        unique_vars = list(dict.fromkeys(all_vars))
        legend_color_map = resolve_palette(unique_vars, base_palette=palette)
        handles = [
            mpatches.Patch(color=legend_color_map[var], alpha=0.7, label=var)
            for var in unique_vars
        ]
        legend_ncol = len(unique_vars)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor
    )

    # --- 4. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_blocks * n_cols_fig
    for i in range(n_blocks, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

#########################################################################################################################################################


def plot_cat_comparison(
    df, comparisons, group_by=None, max_cols=3, 
    title=None, subplots_title=True, order=None,
    hue_order=None, palette="Set2", cat_palette=None, 
    bbox_to_anchor=(0.5, -0.03), sharey=False, x_rotation=30, 
    show_counts=False, save_path=None):

    n_blocks = len(comparisons)

    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. PALETA FIJA POR CATEGORÍA ---
    if group_by:
        group_vals = df[group_by].drop_nulls().cast(pl.String).unique().sort().to_list()
        fixed_palette = resolve_palette(group_vals, base_palette=palette, override=cat_palette)
        legend_keys = group_vals
    else:
        all_cats = (
            df.select([pl.col(col).cast(pl.String) for col_group in comparisons for col in col_group])
            .to_pandas().stack().unique()
        )
        fixed_palette = resolve_palette(sorted(all_cats), base_palette=palette, override=cat_palette)
        legend_keys = list(fixed_palette.keys())

    # --- 2. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_blocks = math.ceil(n_blocks / n_cols_fig)
    n_rows_fig = n_rows_blocks

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(6 * n_cols_fig, 5 * n_rows_fig),
        sharey=sharey
    )

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    # --- 3. DIBUJAR LOS BLOQUES ---
    for i, col_group in enumerate(comparisons):

        r = i // n_cols_fig
        c = i % n_cols_fig
        ax = axes[r, c]

        if group_by:
            for col in col_group:
                pdf = df.select([col, group_by]).to_pandas()
                pdf[col] = pdf[col].fillna("Nulo").astype(str)

                # Proporción condicional por grupo: n(cat, grupo) / n(grupo)
                prop = (
                    pdf.groupby([group_by, col])
                    .size()
                    .reset_index(name="n")
                )
                prop["proporcion"] = prop.groupby(group_by)["n"].transform(lambda x: x / x.sum())

                # order → eje X (valores de col)
                col_order = order if order else (
                    pdf[col].value_counts().sort_values(ascending=False).index.tolist()
                )

                # hue_order → orden de los grupos (valores de group_by)
                resolved_hue_order = hue_order if hue_order else group_vals
                local_palette = {k: fixed_palette[k] for k in group_vals if k in fixed_palette}

                sns.barplot(
                    data=prop, x=col, y="proporcion", hue=group_by,
                    palette=local_palette,
                    order=col_order,
                    hue_order=resolved_hue_order,
                    ax=ax,
                    legend=False
                )

                # --- ANOTACIÓN DE COUNT (GROUP_BY) ---
                if show_counts:
                    num_x = len(col_order)
                    for idx, patch in enumerate(ax.patches):
                        height = patch.get_height()
                        if pd.notna(height) and height > 0:
                            hue_idx = idx // num_x
                            x_idx = idx % num_x
                            
                            if hue_idx < len(resolved_hue_order) and x_idx < num_x:
                                hue_val = resolved_hue_order[hue_idx]
                                x_val = col_order[x_idx]
                                
                                # Buscar el count 'n' exacto en el dataframe prop
                                row = prop[(prop[group_by] == hue_val) & (prop[col] == x_val)]
                                if not row.empty:
                                    count = row["n"].values[0]
                                    ax.text(
                                        patch.get_x() + patch.get_width() / 2, 
                                        height, 
                                        f"n={count}",
                                        ha='center', va='bottom',
                                        fontsize=10, fontweight='bold', color='#333333'
                                    )
        else:
            frames = []
            for col in col_group:
                serie = df[col].fill_null("Nulo").cast(pl.String).to_pandas()
                frames.append(pd.DataFrame({"valor": serie, "variable": col}))
            tidy = pd.concat(frames, ignore_index=True)

            # Proporción condicional por variable: n(cat, variable) / n(variable)
            prop = (
                tidy.groupby(["variable", "valor"])
                .size()
                .reset_index(name="n")
            )
            prop["proporcion"] = prop.groupby("variable")["n"].transform(lambda x: x / x.sum())

            col_order = order if order else (
                tidy["valor"].value_counts().sort_values(ascending=False).index.tolist()
            )

            sns.barplot(
                data=prop, x="valor", y="proporcion", hue="variable",
                palette=palette,
                order=col_order,
                hue_order=col_group, # <-- Forzamos el hue_order para garantizar alineación en los patches
                ax=ax,
                legend=False
            )

            # --- ANOTACIÓN DE COUNT (ELSE) ---
            if show_counts:
                num_x = len(col_order)
                resolved_vars = col_group  # Las variables actúan como hue
                for idx, patch in enumerate(ax.patches):
                    height = patch.get_height()
                    if pd.notna(height) and height > 0:
                        hue_idx = idx // num_x
                        x_idx = idx % num_x
                        
                        if hue_idx < len(resolved_vars) and x_idx < num_x:
                            hue_val = resolved_vars[hue_idx]
                            x_val = col_order[x_idx]
                            
                            # Buscar el count 'n' exacto en el dataframe prop
                            row = prop[(prop["variable"] == hue_val) & (prop["valor"] == x_val)]
                            if not row.empty:
                                count = row["n"].values[0]
                                ax.text(
                                    patch.get_x() + patch.get_width() / 2, 
                                    height, 
                                    f"n={count}",
                                    ha='center', va='bottom',
                                    fontsize=10, fontweight='bold', color='#333333'
                                )

        if subplots_title:
            block_title = " vs ".join(col.upper() for col in col_group)
            if group_by:
                block_title += f"\n(por {group_by})"
            ax.set_title(block_title, fontsize=11, fontweight="bold")

        ax.set_xlabel("")
        ax.set_ylabel("Proporción condicional")
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=10)

    # --- 4. LEYENDA GLOBAL MANUAL ---
    if group_by:
        legend_order = hue_order if hue_order else legend_keys
        handles = [
            mpatches.Patch(color=fixed_palette[g], alpha=0.8, label=str(g))
            for g in legend_order if g in fixed_palette
        ]
    else:
        all_vars = [col for col_group in comparisons for col in col_group]
        unique_vars = list(dict.fromkeys(all_vars))
        var_colors = sns.color_palette(palette, len(unique_vars))
        handles = [
            mpatches.Patch(color=var_colors[j], alpha=0.8, label=var)
            for j, var in enumerate(unique_vars)
        ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor
    )

    # --- 5. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_blocks * n_cols_fig
    for i in range(n_blocks, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    if title:
        fig.suptitle(title , fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
#########################################################################################################################################################

def plot_quant_scatter(
    df, comparisons, group_by=None, figsize=None,
    order=None, max_cols=3, title=None, subplots_title=True,
    palette="Set2", alpha=0.6, show_regression=True, corr_annotation=True,
    bbox_to_anchor=(0.5, -0.03), save_path=None
):

    n_blocks = len(comparisons)
    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. LAYOUT ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_fig = math.ceil(n_blocks / n_cols_fig)

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(5 * n_cols_fig, 4.5 * n_rows_fig) if not figsize else figsize
    )

    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    # --- 2. GRUPOS Y COLORES ---
    if group_by:
        group_vals = df[group_by].drop_nulls().unique().sort().to_list()
        hue_order  = order if order else group_vals
        group_color_map = resolve_palette(hue_order, base_palette=palette)
    else:
        pair_colors = sns.color_palette(palette, n_blocks)

    # --- 3. DIBUJAR BLOQUES ---
    for i, (x_col, y_col) in enumerate(comparisons):
        r = i // n_cols_fig
        c = i % n_cols_fig
        ax = axes[r, c]

        # Construir pandas subset
        cols = [x_col, y_col] + ([group_by] if group_by else [])
        pdf = df.select(cols).to_pandas().dropna(subset=[x_col, y_col])

        # --- TÍTULO ---
        if subplots_title:
            block_title = f"{x_col.upper()} vs {y_col.upper()}"
            if group_by:
                block_title += f"\n(por {group_by})"
            ax.set_title(block_title, fontsize=11, fontweight="bold", y=1.02)

        # --- SCATTER + REGRESIÓN ---
        if group_by:
            for g in hue_order:
                mask = pdf[group_by] == g
                sub  = pdf[mask]
                if sub.empty:
                    continue
                color = group_color_map[g]
                ax.scatter(sub[x_col], sub[y_col],
                           color=color, alpha=alpha, s=25,
                           linewidths=0, label=str(g), zorder=3)
                if show_regression and len(sub) > 2:
                    m, b, r_val, *_ = stats.linregress(sub[x_col], sub[y_col])
                    x_line = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                    ax.plot(x_line, m * x_line + b,
                            color=color, linewidth=1.6, zorder=4)
            # r global (todos los grupos)
            r_val, p_val = stats.pearsonr(pdf[x_col], pdf[y_col])
        else:
            color = pair_colors[i]
            ax.scatter(pdf[x_col], pdf[y_col],
                       color=color, alpha=alpha, s=25,
                       linewidths=0, zorder=3)
            if show_regression and len(pdf) > 2:
                m, b, r_val, p_val, _ = stats.linregress(pdf[x_col], pdf[y_col])
                x_line = np.linspace(pdf[x_col].min(), pdf[x_col].max(), 100)
                ax.plot(x_line, m * x_line + b,
                        color=color, linewidth=1.8, zorder=4)
            r_val, p_val = stats.pearsonr(pdf[x_col], pdf[y_col])

        # --- LÍNEAS DE MEDIA ---
        ax.axvline(pdf[x_col].mean(), color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=2)
        ax.axhline(pdf[y_col].mean(), color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=2)

        # --- ANOTACIÓN r ---
        if corr_annotation:
            p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            ax.annotate(f"r = {r_val:.2f}  ({p_str})",
                        xy=(0.04, 0.96), xycoords="axes fraction",
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                fc="white", ec="lightgray", alpha=0.8))

        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.tick_params(labelsize=8)

    # --- 4. LEYENDA GLOBAL ---
    if group_by:
        handles = [
            mpatches.Patch(color=group_color_map[g], alpha=0.8, label=str(g))
            for g in hue_order
        ]
    else:
        handles = [
            mpatches.Patch(color=pair_colors[j], alpha=0.8,
                           label=f"{x} vs {y}")
            for j, (x, y) in enumerate(comparisons)
        ]

    fig.legend(handles=handles,
               loc="lower center",
               ncol=len(handles),
               fontsize=9,
               frameon=True,
               bbox_to_anchor=bbox_to_anchor)

    # --- 5. LIMPIAR SUBPLOTS VACÍOS ---
    total = n_rows_fig * n_cols_fig
    for i in range(n_blocks, total):
        fig.delaxes(axes[i // n_cols_fig, i % n_cols_fig])

    if title:
        fig.suptitle(title , fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

#########################################################################################################################################################

def plot_quant_comparison_faceted(
    df,
    comparisons,
    outer_group_by,
    group_by=None,
    outer_order=None,
    order=None,
    figsize=None,
    showfliers=True,
    labelbottom=True,
    xlabel_rotation=30,
    sharey="row",        # "row" | "all" | "none"
    title=True,
    palette="Set2",
    bbox_to_anchor=(0.5, -0.03),
    save_path=None
):
    """
    sharey : {"row", "all", "none"}, default "row"
        "row"  → misma escala Y por fila (mismo bloque de variables,
                 distintos outer groups). Comparación directa entre columnas.
        "all"  → un único eje Y global para toda la figura.
        "none" → cada subplot tiene su propio eje Y independiente.
    """

    # --- 0. OUTER VALS ---
    outer_vals = (
        outer_order
        if outer_order is not None
        else df[outer_group_by].drop_nulls().unique().sort().to_list()
    )

    n_cols = len(outer_vals)
    n_rows = len(comparisons)

    if n_cols == 0 or n_rows == 0:
        print("Aviso: comparisons u outer_group_by están vacíos.")
        return

    # --- 1. COLORES ---
    if group_by:
        group_vals  = df[group_by].drop_nulls().unique().sort().to_list()
        hue_order   = order if order else group_vals
        group_color_map = resolve_palette(hue_order, base_palette=palette)
    else:
        all_vars = list(dict.fromkeys(v for cg in comparisons for v in cg))
        var_color_map = resolve_palette(all_vars, base_palette=palette)

    # --- 2. FIGURA ---
    # "row" y "all" se delegan a matplotlib directamente.
    # "none" también, pero añadimos un paso posterior para romper
    # cualquier linkeo residual.
    sharey_flag = {"row": "row", "all": True, "none": False}[sharey]

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows + 0.5) if not figsize else figsize,
        sharey=sharey_flag,
        squeeze=False,
    )

    # --- 3. TÍTULOS DE COLUMNA ---
    for c, val in enumerate(outer_vals):
        axes[0, c].set_title(
            f"{outer_group_by} = {val}",
            fontsize=11, fontweight="bold", pad=8,
        )

    # --- 4. DIBUJAR ---
    for r, col_group in enumerate(comparisons):
        for c, val in enumerate(outer_vals):
            ax  = axes[r, c]
            df_sub = df.filter(pl.col(outer_group_by) == val)

            cols = col_group + ([group_by] if group_by else [])
            pdf  = df_sub.select(cols).to_pandas()

            if group_by:
                tidy = pdf.melt(
                    id_vars=group_by,
                    value_vars=col_group,
                    var_name="variable",
                    value_name="valor",
                ).dropna(subset=["valor", group_by])
            else:
                tidy = pdf.melt(
                    var_name="variable",
                    value_name="valor",
                ).dropna()

            if group_by:
                sns.boxplot(
                    data=tidy, x="variable", y="valor",
                    hue=group_by, showfliers=showfliers,
                    palette=palette, ax=ax,
                    width=0.5, linewidth=1.2,
                    legend=False, hue_order=hue_order,
                )
                for j, col in enumerate(col_group):
                    for g, gval in enumerate(hue_order):
                        mask = (tidy["variable"] == col) & (tidy[group_by] == gval)
                        if mask.sum() == 0:
                            continue
                        mean_val = tidy.loc[mask, "valor"].mean()
                        n_groups = len(hue_order)
                        offset   = (g - (n_groups - 1) / 2) * (0.5 / n_groups)
                        ax.plot(
                            j + offset, mean_val,
                            marker="D", color=group_color_map[gval],
                            markersize=5,
                            markeredgecolor="black", markeredgewidth=0.7,
                            zorder=5,
                        )
            else:
                block_palette = [var_color_map[v] for v in col_group]
                sns.boxplot(
                    data=tidy, x="variable", y="valor",
                    hue="variable", showfliers=showfliers,
                    palette=block_palette, ax=ax,
                    width=0.5, linewidth=1.2,
                    legend=False,
                )
                for j, col in enumerate(col_group):
                    mean_val = tidy[tidy["variable"] == col]["valor"].mean()
                    ax.plot(
                        j, mean_val,
                        marker="D", color=var_color_map[col],
                        markersize=6,
                        markeredgecolor="black", markeredgewidth=0.7,
                        zorder=5,
                    )

            if title and c == 0:
                ax.set_ylabel(" vs ".join(col_group), fontsize=9, labelpad=6)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=labelbottom, rotation=xlabel_rotation)

    # Reemplaza el paso 5 completo por esto:

    # --- 5. DESACOPLAR EJES SI sharey="none" ---
    # En matplotlib >= 3.7 get_shared_y_axes() devuelve un GrouperView
    # inmutable. La solución es no linkear desde el principio (sharey=False)
    # y forzar autoscale por subplot tras dibujar.
    if sharey == "none":
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                # Recalcular límites Y solo con los datos de este subplot
                ax.relim()
                ax.autoscale_view(scaley=True)
                # Restaurar etiquetas Y que sharey=False puede haber ocultado
                ax.tick_params(axis="y", labelleft=True)

    # --- 6. LEYENDA GLOBAL ---
    if group_by:
        handles = [
            mpatches.Patch(color=group_color_map[g], alpha=0.75, label=str(g))
            for g in hue_order
        ]
    else:
        handles = [
            mpatches.Patch(color=var_color_map[v], alpha=0.75, label=v)
            for v in all_vars
        ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

######################################################################################################################################################### 
 
def plot_quant_pie(df, columns, figsize=None, max_cols=3, title=None,
                   subplots_title=True, palette=None, min_pct_label=3.0,
                   donut=True, style="light", bbox_to_anchor=(0.5, -0.22),
                   save_path=None):
    """
    Plot one donut / pie chart per column in `columns`.
 
    Parameters
    ----------
    df : polars.DataFrame
        Source dataframe.
    columns : list[str]
        Categorical columns to plot; one chart each.
    figsize : tuple[float, float] | None
        Overall figure size. Defaults to (5.2 * n_cols, 5.0 * n_rows).
    max_cols : int
        Maximum number of charts per row.
    title : str | None
        Overall figure super-title.
    subplots_title : bool
        Whether to show a title above each individual chart.
    palette : list[str] | None
        List of hex colours. Defaults to a built-in vivid palette.
    min_pct_label : float
        Slices smaller than this percentage are not labelled.
    donut : bool
        If True (default) renders as a donut with a centre annotation.
    style : {"dark", "light"}
        Overall colour theme.
    """
 
    n_blocks = len(columns)
    if n_blocks == 0:
        print("Aviso: La lista de columnas está vacía.")
        return
 
    # --- 1. THEME ---
    IS_DARK  = (style == "dark")
    bg       = _DARK_BG  if IS_DARK else "#F8F7F4"
    card_bg  = _CARD_BG  if IS_DARK else "#FFFFFF"
    fg       = "#E8ECF4" if IS_DARK else "#1a1a2e"
    sub_fg   = "#8892AA" if IS_DARK else "#666680"
    edge_col = _DARK_BG  if IS_DARK else "#F0EEE8"
    leg_bg   = _DARK_BG  if IS_DARK else "#F0EEE8"
 
    colors = palette if palette else _RICH_COLORS
 
    # --- 2. LAYOUT ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_fig = math.ceil(n_blocks / n_cols_fig)
 
    fw = 5.2 * n_cols_fig
    fh = 5.0 * n_rows_fig + (1.0 if title else 0)
    fig = plt.figure(figsize=figsize or (fw, fh), facecolor=bg)
 
    axes = [fig.add_subplot(n_rows_fig, n_cols_fig, i + 1)
            for i in range(n_blocks)]
 
    # --- 3. DRAW EACH CHART ---
    for i, col in enumerate(columns):
        ax = axes[i]
        ax.set_facecolor(card_bg)
        for sp in ax.spines.values():
            sp.set_visible(False)
 
        counts = (
            df.select(col)
            .to_pandas()[col]
            .dropna()
            .value_counts()
            .sort_values(ascending=False)
        )
        labels       = [str(lbl) for lbl in counts.index.tolist()]
        values       = counts.values.tolist()
        total        = sum(values)
        n_cat        = len(labels)
        slice_colors = [colors[j % len(colors)] for j in range(n_cat)]
 
        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=slice_colors,
            autopct=lambda p: f"{p:.1f}%" if p >= min_pct_label else "",
            pctdistance=0.78,
            startangle=90,
            explode=[0.03] * n_cat,
            wedgeprops=dict(
                width=0.52 if donut else 1.0,
                linewidth=2.5,
                edgecolor=edge_col,
            ),
        )
 
        # --- pct label styling ---
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color(fg)
            at.set_fontweight("bold")
            at.set_path_effects([pe.withStroke(linewidth=2, foreground=card_bg)])
 
        # --- donut centre annotation ---
        if donut:
            ax.text(0,  0.06, f"{total:,}",
                    ha="center", va="center", fontsize=15,
                    fontweight="bold", color=fg, transform=ax.transData)
            ax.text(0, -0.14, "total",
                    ha="center", va="center", fontsize=8,
                    color=sub_fg, transform=ax.transData)
 
        # --- subplot title ---
        if subplots_title:
            ax.set_title(col.upper(), fontsize=12, fontweight="bold",
                         color=fg, pad=14, loc="center",
                         fontfamily="monospace")
 
        # --- per-chart legend ---
        legend_handles = [
            mpatches.Patch(
                facecolor=c, edgecolor="none",
                label=f"{lbl}   {val:,}  ({val/total*100:.1f}%)"
            )
            for lbl, val, c in zip(labels, values, slice_colors)
        ]
        leg = ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            fontsize=8,
            frameon=True,
            ncol=min(n_cat, 3),
            facecolor=leg_bg,
            edgecolor="none",
            labelcolor=fg,
            handlelength=1.0,
            handleheight=0.9,
            borderpad=0.6,
            columnspacing=1.0,
        )
        leg.get_frame().set_alpha(0.85)
 
    # --- 4. HIDE EMPTY AXES ---
    for i in range(n_blocks, n_rows_fig * n_cols_fig):
        fig.add_subplot(n_rows_fig, n_cols_fig, i + 1).set_visible(False)
 
    # --- 5. GLOBAL TITLE ---
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold",
                     color=fg, y=1.01, fontfamily="monospace")
 
    plt.tight_layout(pad=2.2)
    plt.subplots_adjust(bottom=0.15, hspace=0.45)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

#########################################################################################################################################################
    
def plot_cat_comparison_faceted(
    df,
    comparisons,
    outer_group_by,
    group_by=None,
    outer_order=None,
    order=None,
    hue_order=None,
    palette="Set2",
    cat_palette=None,
    figsize=None,
    bbox_to_anchor=(0.5, -0.03),
    sharey=False,
    x_rotation=30,
    subplots_title=True,
    title=None,
    show_counts=False, 
    save_path=None
):
    # --- 0. OUTER VALS ---
    outer_vals = (
        outer_order
        if outer_order is not None
        else df[outer_group_by].drop_nulls().cast(pl.String).unique().sort().to_list()
    )

    n_cols = len(outer_vals)
    n_rows = len(comparisons)

    if n_cols == 0 or n_rows == 0:
        print("Aviso: comparisons u outer_group_by están vacíos.")
        return

    # --- 1. PALETA FIJA ---
    if group_by:
        group_vals = df[group_by].drop_nulls().cast(pl.String).unique().sort().to_list()
        resolved_hue_order = hue_order if hue_order else group_vals
        fixed_palette = resolve_palette(group_vals, base_palette=palette, override=cat_palette)
    else:
        all_cats = (
            df.select([pl.col(col).cast(pl.String) for col_group in comparisons for col in col_group])
            .to_pandas().stack().unique()
        )
        fixed_palette = resolve_palette(sorted(all_cats), base_palette=palette, override=cat_palette)

    # --- 1b. PRE-COMPUTE GLOBAL X ORDER FROM FULL DF ---
    # Keyed by (col_group tuple, col) for group_by branch, or (col_group tuple,) otherwise.
    global_col_orders = {}
    for col_group in comparisons:
        key = tuple(col_group)
        if order:
            # User-supplied order wins for everything
            if group_by:
                for col in col_group:
                    global_col_orders[(key, col)] = order
            else:
                global_col_orders[key] = order
        elif group_by:
            for col in col_group:
                serie = df[col].fill_null("Nulo").cast(pl.String).to_pandas()
                global_col_orders[(key, col)] = (
                    serie.value_counts().sort_values(ascending=False).index.tolist()
                )
        else:
            tidy = pd.concat(
                [df[col].fill_null("Nulo").cast(pl.String).to_pandas().rename("valor")
                 for col in col_group],
                ignore_index=True,
            )
            global_col_orders[key] = (
                tidy.value_counts().sort_values(ascending=False).index.tolist()
            )

    # --- 2. FIGURA ---
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.5 * n_cols, 5 * n_rows) if figsize is None else figsize,
        sharey=sharey,
        squeeze=False,
    )

    # --- 3. TÍTULOS DE COLUMNA ---
    for c, val in enumerate(outer_vals):
        axes[0, c].set_title(
            f"{outer_group_by} = {val}",
            fontsize=11, fontweight="bold", pad=8,
        )

    # --- 4. DIBUJAR ---
    for r, col_group in enumerate(comparisons):
        key = tuple(col_group)
        for c, val in enumerate(outer_vals):
            ax = axes[r, c]
            df_sub = df.filter(pl.col(outer_group_by).cast(pl.String) == str(val))

            if group_by:
                for col in col_group:
                    col_order = global_col_orders[(key, col)]   # ← global order

                    pdf = df_sub.select([col, group_by]).to_pandas()
                    pdf[col] = pdf[col].fillna("Nulo").astype(str)

                    prop = (
                        pdf.groupby([group_by, col])
                        .size()
                        .reset_index(name="n")
                    )
                    prop["proporcion"] = prop.groupby(group_by)["n"].transform(
                        lambda x: x / x.sum()
                    )

                    local_palette = {k: fixed_palette[k] for k in group_vals if k in fixed_palette}

                    sns.barplot(
                        data=prop, x=col, y="proporcion", hue=group_by,
                        palette=local_palette,
                        order=col_order,
                        hue_order=resolved_hue_order,
                        ax=ax,
                        legend=False,
                    )

                    if show_counts:
                        num_x = len(col_order)
                        for idx, patch in enumerate(ax.patches):
                            height = patch.get_height()
                            if pd.notna(height) and height > 0:
                                hue_idx = idx // num_x
                                x_idx   = idx % num_x
                                if hue_idx < len(resolved_hue_order) and x_idx < num_x:
                                    hue_val = resolved_hue_order[hue_idx]
                                    x_val   = col_order[x_idx]
                                    row = prop[(prop[group_by] == hue_val) & (prop[col] == x_val)]
                                    if not row.empty:
                                        ax.text(
                                            patch.get_x() + patch.get_width() / 2,
                                            height,
                                            f"n={row['n'].values[0]}",
                                            ha="center", va="bottom",
                                            fontsize=9, fontweight="bold", color="#333333",
                                        )

            else:
                col_order = global_col_orders[key]              # ← global order

                frames = []
                for col in col_group:
                    serie = df_sub[col].fill_null("Nulo").cast(pl.String).to_pandas()
                    frames.append(pd.DataFrame({"valor": serie, "variable": col}))
                tidy = pd.concat(frames, ignore_index=True)

                prop = (
                    tidy.groupby(["variable", "valor"])
                    .size()
                    .reset_index(name="n")
                )
                prop["proporcion"] = prop.groupby("variable")["n"].transform(
                    lambda x: x / x.sum()
                )

                sns.barplot(
                    data=prop, x="valor", y="proporcion", hue="variable",
                    palette=palette,
                    order=col_order,
                    hue_order=col_group,
                    ax=ax,
                    legend=False,
                )

                
                if show_counts:
                    num_x = len(col_order)
                    for idx, patch in enumerate(ax.patches):
                        height = patch.get_height()
                        if pd.notna(height) and height > 0:
                            hue_idx = idx // num_x
                            x_idx   = idx % num_x
                            if hue_idx < len(col_group) and x_idx < num_x:
                                hue_val = col_group[hue_idx]
                                x_val   = col_order[x_idx]
                                row = prop[(prop["variable"] == hue_val) & (prop["valor"] == x_val)]
                                if not row.empty:
                                    ax.text(
                                        patch.get_x() + patch.get_width() / 2,
                                        height,
                                        f"n={row['n'].values[0]}",
                                        ha="center", va="bottom",
                                        fontsize=9, fontweight="bold", color="#333333",
                                    )

            if subplots_title and c == 0:
                ax.set_ylabel(
                    " vs ".join(col.upper() for col in col_group),
                    fontsize=9, labelpad=6,
                )
            else:
                ax.set_ylabel("Proporción condicional" if c == 0 else "")

            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=x_rotation, labelsize=10)

    # --- 5. LEYENDA GLOBAL ---
    if group_by:
        handles = [
            mpatches.Patch(color=fixed_palette[g], alpha=0.8, label=str(g))
            for g in resolved_hue_order if g in fixed_palette
        ]
    else:
        all_vars   = list(dict.fromkeys(col for col_group in comparisons for col in col_group))
        var_colors = sns.color_palette(palette, len(all_vars))
        handles = [
            mpatches.Patch(color=var_colors[j], alpha=0.8, label=var)
            for j, var in enumerate(all_vars)
        ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=bbox_to_anchor,
    )

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()