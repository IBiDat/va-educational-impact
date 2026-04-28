#########################################################################################################################################################

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sns.set_style('whitegrid')

#########################################################################################################################################################

def plot_cat_distribution(df, cat_cols, order=None, max_cols=3, palette="Set2"):
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
    
    # Validación de seguridad
    if n_vars == 0:
        print("Aviso: La lista de columnas está vacía.")
        return

    # --- 1. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_vars, max_cols)
    n_rows_fig = math.ceil(n_vars / n_cols_fig) 

    # Ajustamos el tamaño de la figura dinámicamente
    fig, axes = plt.subplots(nrows=n_rows_fig, ncols=n_cols_fig, figsize=(6 * n_cols_fig, 5 * n_rows_fig))

    # Forzamos a que axes sea siempre un array 2D
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
        
        # Procesamiento con Polars y conversión a Pandas para Seaborn
        serie_str = df[col].fill_null("Nulo").cast(pl.String).to_pandas()
        
        # Gráfico de proporciones
        sns.countplot(
            x=serie_str, 
            ax=ax, 
            palette=palette, 
            hue=serie_str, 
            legend=False, 
            stat='proportion',
            order=order
        )
        
        # Configuración de títulos y etiquetas
        ax.set_title(col.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Proporción')
        ax.tick_params(axis='x', rotation=30, labelsize=12)

    # --- 3. LIMPIEZA DE ESPACIOS VACÍOS ---
    total_blocks = n_rows_fig * n_cols_fig
    for i in range(n_vars, total_blocks):
        r = i // n_cols_fig
        c = i % n_cols_fig
        fig.delaxes(axes[r, c])

    # --- 4. RENDERIZADO ---
    plt.tight_layout()
    plt.show()

#########################################################################################################################################################

def plot_quant_distribution(df, quant_cols, max_cols=3, box_color="skyblue", hist_color="salmon"):
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
    plt.show()

#########################################################################################################################################################

def plot_quant_comparison(df, comparisons, group_by=None, showfliers=True, labelbottom=True, xlabel_rotation=30, max_cols=3, title=False, palette="Set2", bbox_to_anchor=(0.5, -0.03)):
    
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
        figsize=(6 * n_cols_fig, 4 * n_rows_fig)
    )

    # Forzamos array 2D
    if n_rows_fig == 1 and n_cols_fig == 1:
        axes = np.array([[axes]])
    elif n_rows_fig == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_fig == 1:
        axes = axes.reshape(-1, 1)

    # Grupos únicos para leyenda (solo si hay group_by)
    if group_by:
        group_vals = df[group_by].drop_nulls().unique().sort().to_list()
        group_colors = sns.color_palette(palette, len(group_vals))
        group_color_map = dict(zip(group_vals, group_colors))

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
            ).dropna()
        else:
            pdf = df.select(col_group).to_pandas()
            tidy = pdf.melt(
                var_name="variable",
                value_name="valor"
            ).dropna()

        # --- TÍTULO ---
        if title:
            block_title = " vs ".join(col.upper() for col in col_group)
            if group_by:
                block_title += f"\n(por {group_by})"
            if not showfliers:
                block_title += "\n(Outliers Hidden)"
            ax_box.set_title(block_title, fontsize=11, fontweight="bold", y=1.05)

        # --- BOXPLOT ---
        if group_by:
            sns.boxplot(
                data=tidy, x="variable", y="valor",
                hue=group_by, showfliers=showfliers,
                palette=palette, ax=ax_box,
                width=0.5, linewidth=1.5,
                legend=False
            )
            for j, col in enumerate(col_group):
                for g, group_val in enumerate(group_vals):
                    mask = (tidy["variable"] == col) & (tidy[group_by] == group_val)
                    mean_val = tidy.loc[mask, "valor"].mean()
                    n_groups = len(group_vals)
                    offset = (g - (n_groups - 1) / 2) * (0.5 / n_groups)
                    ax_box.plot(
                        j + offset, mean_val, marker="D",
                        color=group_colors[g], markersize=6,
                        markeredgecolor="black", markeredgewidth=0.8,
                        zorder=5
                    )
        else:
            var_colors = sns.color_palette(palette, len(col_group))
            sns.boxplot(
                data=tidy, x="variable", y="valor",
                hue="variable", showfliers=showfliers,
                palette=palette, ax=ax_box,
                width=0.5, linewidth=1.5,
                legend=False
            )
            for j, col in enumerate(col_group):
                mean_val = tidy[tidy["variable"] == col]["valor"].mean()
                ax_box.plot(
                    j, mean_val, marker="D",
                    color=var_colors[j], markersize=7,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=5
                )

        ax_box.set_xlabel("")
        ax_box.set_ylabel("")
        ax_box.tick_params(axis="x", labelbottom=labelbottom, rotation=xlabel_rotation)

    # --- 3. LEYENDA GLOBAL MANUAL ---
    if group_by:
        handles = [
            mpatches.Patch(color=group_color_map[g], alpha=0.7, label=str(g))
            for g in group_vals
        ]
        legend_ncol = len(group_vals)
    else:
        all_vars = [col for col_group in comparisons for col in col_group]
        unique_vars = list(dict.fromkeys(all_vars))
        legend_colors = sns.color_palette(palette, len(unique_vars))
        handles = [
            mpatches.Patch(color=legend_colors[j], alpha=0.7, label=var)
            for j, var in enumerate(unique_vars)
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
    plt.show()

#########################################################################################################################################################

def plot_cat_comparison(df, comparisons, group_by=None, max_cols=3, title=False, order=None, palette="Set2", cat_palette=None, bbox_to_anchor=(0.5, -0.03)):

    n_blocks = len(comparisons)

    if n_blocks == 0:
        print("Aviso: La lista de comparaciones está vacía.")
        return

    # --- 1. PALETA FIJA POR CATEGORÍA ---
    if group_by:
        group_vals = df[group_by].drop_nulls().cast(pl.String).unique().sort().to_list()
        if cat_palette:
            fixed_palette = cat_palette
        else:
            fixed_palette = dict(zip(group_vals, sns.color_palette(palette, len(group_vals))))
        legend_keys = group_vals
    else:
        all_cats = (
            df.select([pl.col(col).cast(pl.String) for col_group in comparisons for col in col_group])
            .to_pandas().stack().unique()
        )
        if cat_palette:
            fixed_palette = cat_palette
        else:
            fixed_palette = dict(zip(all_cats, sns.color_palette(palette, len(all_cats))))
        legend_keys = list(fixed_palette.keys())

    # --- 2. CONFIGURACIÓN ADAPTATIVA ---
    n_cols_fig = min(n_blocks, max_cols)
    n_rows_blocks = math.ceil(n_blocks / n_cols_fig)
    n_rows_fig = n_rows_blocks

    fig, axes = plt.subplots(
        nrows=n_rows_fig, ncols=n_cols_fig,
        figsize=(6 * n_cols_fig, 5 * n_rows_fig)
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

                col_order = order if order else (
                    pdf[col].value_counts().sort_values(ascending=False).index.tolist()
                )
                local_palette = {k: fixed_palette[k] for k in group_vals if k in fixed_palette}

                sns.barplot(
                    data=prop, x=col, y="proporcion", hue=group_by,
                    palette=local_palette,
                    order=col_order,
                    ax=ax,
                    legend=False
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
                ax=ax,
                legend=False
            )

        if title:
            block_title = " vs ".join(col.upper() for col in col_group)
            if group_by:
                block_title += f"\n(por {group_by})"
            ax.set_title(block_title, fontsize=11, fontweight="bold")

        ax.set_xlabel("")
        ax.set_ylabel("Proporción condicional")
        ax.tick_params(axis='x', rotation=30, labelsize=10)

    # --- 4. LEYENDA GLOBAL MANUAL ---
    if group_by:
        handles = [
            mpatches.Patch(color=fixed_palette[g], alpha=0.8, label=str(g))
            for g in legend_keys if g in fixed_palette
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

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.show()

#########################################################################################################################################################



#########################################################################################################################################################