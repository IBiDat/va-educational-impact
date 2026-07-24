#########################################################################################################################################################

import os
import polars as pl
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
import numpy as np

#########################################################################################################################################################

def analyze_experiment_results(processed_data_dict, hashes_dir):
    """
    Cruza los datos procesados con la tabla de grupos, extrae métricas
    y calcula la variación porcentual entre el pre-test y el post-test.
    """
    # 1. Unir todos los DataFrames del diccionario en uno solo
    df_combined = pl.concat(list(processed_data_dict.values()), how="diagonal")
    
    # 2. Cargar la tabla relacional de grupos
    df_grupos = pl.read_csv(os.path.join(hashes_dir, "hashes_groups.csv"), separator=";")    
    
    # 3. Unir (Join) los resultados con el grupo correspondiente de cada estudiante
    df_completo = df_combined.join(df_grupos, on="id", how="inner")
    
    # 4. Extraer métricas (Media y Desviación Estándar) agrupadas por periodo y grupo
    cols_to_analyze = [
        "score_tc", 
        "score_ta", 
        "indice_desempeño_global"
    ]
    
    agregaciones = []
    for col in cols_to_analyze:
        agregaciones.extend([
            pl.col(col).mean().round(2).alias(f"{col}_media"),
            pl.col(col).std().round(2).alias(f"{col}_std")
        ])
        
    metricas = (
        df_completo
        .group_by(["periodo", "grupo"])
        .agg(agregaciones)
        .sort(["periodo", "grupo"]) 
    )
    
    # --- CÁLCULO DE LA VARIACIÓN PORCENTUAL ---
    
    # Separamos pre y post
    df_pre = metricas.filter(pl.col("periodo") == "pre")
    df_post = metricas.filter(pl.col("periodo") == "post")
    
    # Renombramos las columnas del 'pre' para diferenciarlas al cruzar
    renames = {f"{col}_media": f"{col}_media_pre" for col in cols_to_analyze}
    df_pre = df_pre.rename(renames)
    
    # Unimos pre y post para tener todo en la misma fila y poder operar
    df_variacion = df_post.join(
        df_pre.select(["grupo"] + list(renames.values())), 
        on="grupo", 
        how="inner"
    )
    
    # Calculamos la fórmula: ((Post - Pre) / Pre) * 100
    exprs_var = []
    for col in cols_to_analyze:
        col_post = f"{col}_media"
        col_pre = f"{col}_media_pre"
        
        exprs_var.append(
            ((pl.col(col_post) - pl.col(col_pre)) / pl.col(col_pre) * 100)
            .round(2)
            .alias(f"{col}_var_%")
        )
        
    df_variacion = df_variacion.select(["grupo"] + exprs_var)
    
    # Añadimos estas nuevas columnas a nuestra tabla original
    metricas = metricas.join(df_variacion, on="grupo", how="left")
    
    # Limpiamos las filas con nulos (que corresponden al periodo 'pre' que no tiene variación porcentual)
    limpieza_exprs = [
        pl.when(pl.col("periodo") == "pre")
        .then(pl.lit(None))
        .otherwise(pl.col(f"{col}_var_%"))
        .alias(f"{col}_var_%")
        for col in cols_to_analyze
    ]
    
    metricas = metricas.with_columns(limpieza_exprs)
    
    # -------------------------------------------------
    
    return df_completo, metricas

#########################################################################################################################################################

def test_significacion_estadistica(df_completo, metrica="score_tc"):
    """
    Realiza un T-test independiente en el post-test para la métrica elegida.
    """
    # Filtramos solo los datos del post-test
    df_post = df_completo.filter(pl.col("periodo") == "post")
    
    # Extraemos las puntuaciones en formato lista/array ignorando valores nulos
    grupo_control = df_post.filter(pl.col("grupo") == "control").select(pl.col(metrica).drop_nulls()).to_series().to_list()
    grupo_experimental = df_post.filter(pl.col("grupo") == "experimental").select(pl.col(metrica).drop_nulls()).to_series().to_list()
    
    # Realizamos el T-test
    t_stat, p_value = stats.ttest_ind(grupo_experimental, grupo_control, equal_var=False)
    
    print(f"--- Análisis Estadístico para: {metrica} (Post-test) ---")
    print(f"Media Experimental: {sum(grupo_experimental)/len(grupo_experimental):.2f}")
    print(f"Media Control: {sum(grupo_control)/len(grupo_control):.2f}")
    print(f"P-valor: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ CONCLUSIÓN: La diferencia ES estadísticamente significativa (p < 0.05). La IA tuvo un impacto real.")
    else:
        print("❌ CONCLUSIÓN: La diferencia NO ES estadísticamente significativa (p >= 0.05). Podría deberse al azar.")

#########################################################################################################################################################

def calcular_medias_puntuacion(df_resultados, df_grupos):
    """
    Cruza los resultados con la tabla de grupos y calcula la media 
    de las variables de puntuación de conocimientos.
    """
    # 1. Unimos los resultados con los grupos (usando inner join por el 'id')
    df_cruzado = df_resultados.join(df_grupos, on="id", how="inner")
    
    # 2. Agrupamos por grupo y calculamos las medias
    medias = (
        df_cruzado
        # Si también quieres separar por "pre" y "post", añade "periodo" a esta lista:
        .group_by(["periodo", "grupo"]) 
        .agg([
            pl.col("score_tc").mean().round(2).alias("media_tc_general"),
            pl.col("score_tc_retention").mean().round(2).alias("media_tc_retencion"),
            pl.col("score_tc_transfer").mean().round(2).alias("media_tc_transferencia"),
            pl.col("score_ta").mean().round(2).alias("media_ta"),
            pl.col("indice_desempeño_global").mean().round(2).alias("media_indice_desempeño_global")
        ])
        .sort(["periodo", "grupo"])
    )
    
    return medias

#########################################################################################################################################################

def realizar_analisis_hake_normalizado(
    df: pl.DataFrame,
    constructo_base: str,
    col_grupo: str = "grupo"
):
    """
    Realiza el análisis de Hake y ANCOVA para puntuaciones ya normalizadas (0,1).

    Args:
        df: DataFrame de Polars con columnas *_pre y *_post
        constructo_base: prefijo de la variable (ej. 'score_tc')
        col_grupo: columna con 'Control' / 'Experimental'
    """

    col_pre = f"{constructo_base}_pre"
    col_post = f"{constructo_base}_post"
    x_max = 1.0

    print("=" * 60)
    print(f" ANÁLISIS DE GANANCIA DE HAKE: {constructo_base.upper()}")
    print("=" * 60)

    # --------------------------------------------------
    # 1. CÁLCULO DE LA GANANCIA DE HAKE
    # --------------------------------------------------
    df_valid = (
        df
        .filter(pl.col(col_pre) < x_max)
        .with_columns(
            g=(pl.col(col_post) - pl.col(col_pre)) / (x_max - pl.col(col_pre))
        )
        .drop_nulls(subset=["g", col_grupo])
    )

    excluidos = df.height - df_valid.height
    print(f"Sujetos excluidos por efecto techo (Pre = 1.0): {excluidos}")

    # Separación por grupos
    df_ctrl = df_valid.filter(pl.col(col_grupo).str.to_lowercase() == "control")
    df_exp = df_valid.filter(pl.col(col_grupo).str.to_lowercase() == "experimental")

    grupo_ctrl = df_ctrl["g"].to_numpy()
    grupo_exp = df_exp["g"].to_numpy()

    # --------------------------------------------------
    # 2. HOMOGENEIDAD INICIAL (PRE-TEST)
    # --------------------------------------------------
    print("\n--- 1. Homogeneidad Inicial (Pre-test) ---")

    pre_ctrl = df_ctrl[col_pre].to_numpy()
    pre_exp = df_exp[col_pre].to_numpy()

    t_stat_pre, p_val_pre = stats.ttest_ind(pre_ctrl, pre_exp, nan_policy="omit")

    print(f"T-test Pre-test: t = {t_stat_pre:.3f}, p-valor = {p_val_pre:.3f}")
    if p_val_pre > 0.05:
        print("-> Grupos HOMOGÉNEOS en el Pre-test.")
    else:
        print("-> Grupos NO homogéneos. (El ANCOVA corregirá este sesgo).")

    # --------------------------------------------------
    # 3. CONTRASTE PRINCIPAL (GANANCIA DE HAKE)
    # --------------------------------------------------
    print("\n--- 2. Contraste Principal (Variable g) ---")

    _, p_sw_c = stats.shapiro(grupo_ctrl)
    _, p_sw_e = stats.shapiro(grupo_exp)
    _, p_lev = stats.levene(grupo_ctrl, grupo_exp)

    normalidad = (p_sw_c > 0.05) and (p_sw_e > 0.05)
    homocedasticidad = p_lev > 0.05

    print(
        f"Normalidad (Shapiro): Control p={p_sw_c:.3f}, "
        f"Exp p={p_sw_e:.3f} -> {'Cumple' if normalidad else 'Falla'}"
    )
    print(
        f"Homocedasticidad (Levene): p={p_lev:.3f} "
        f"-> {'Cumple' if homocedasticidad else 'Falla'}"
    )

    if normalidad and homocedasticidad:
        print("=> Aplicando T-test paramétrico (H1: Exp > Control)")
        stat_main, p_main = stats.ttest_ind(
            grupo_ctrl, grupo_exp, alternative="less"
        )
    else:
        print("=> Aplicando U de Mann-Whitney no paramétrico (H1: Exp > Control)")
        stat_main, p_main = stats.mannwhitneyu(
            grupo_ctrl, grupo_exp, alternative="less"
        )

    print(
        f"Resultado Contraste: estadístico = {stat_main:.3f}, "
        f"p-valor (unilateral) = {p_main:.3f}"
    )

    # Tamaño del efecto: d de Cohen
    n1, n2 = len(grupo_ctrl), len(grupo_exp)
    var1, var2 = np.var(grupo_ctrl, ddof=1), np.var(grupo_exp, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d_cohen = (grupo_exp.mean() - grupo_ctrl.mean()) / s_pooled

    print(f"Tamaño del Efecto (d de Cohen): {d_cohen:.3f}")

    # --------------------------------------------------
    # 4. ANCOVA
    # --------------------------------------------------
    print("\n--- 3. Contraste Complementario (ANCOVA) ---")

    df_valid = df_valid.with_columns(
        G_dummy=pl.when(pl.col(col_grupo).str.to_lowercase() == "experimental")
        .then(1)
        .otherwise(0)
    )

    # statsmodels requiere pandas → conversión puntual
    df_sm = df_valid.select([col_post, col_pre, "G_dummy"]).to_pandas()

    modelo = ols(f"{col_post} ~ G_dummy + {col_pre}", data=df_sm).fit()
    print(modelo.summary().tables[1])

    p_beta1 = modelo.pvalues["G_dummy"]
    print(f"\nEfecto del Tratamiento (beta_1): p-valor = {p_beta1/2:.4f} (unilateral)")

    anova_table = sm.stats.anova_lm(modelo, typ=2)
    ss_effect = anova_table.loc["G_dummy", "sum_sq"]
    ss_error = anova_table.loc["Residual", "sum_sq"]
    eta_sq_partial = ss_effect / (ss_effect + ss_error)

    print(
        f"Tamaño del Efecto ANCOVA (Eta Cuadrado Parcial): {eta_sq_partial:.3f}"
    )
    print("\n")

    return df_valid, modelo

#########################################################################################################################################################

def analyze_experiment_results_centros(processed_data_dict, hashes_dir):
    """
    Cruza los datos procesados con la tabla de grupos, extrae métricas
    y calcula la variación porcentual entre el pre-test y el post-test
    por centro y grupo.
    """

    # 1. Unir todos los DataFrames
    df_combined = pl.concat(list(processed_data_dict.values()), how="diagonal")

    # 2. Cargar tabla de grupos
    df_grupos = pl.read_csv(
        os.path.join(hashes_dir, "hashes_groups.csv"),
        separator=";"
    )

    # 3. Join con grupos
    df_completo = df_combined.join(df_grupos, on="id", how="inner")

    # 4. Métricas
    cols_to_analyze = [
        "score_tc",
        "score_ta",
        "indice_desempeño_global"
    ]

    agregaciones = []
    for col in cols_to_analyze:
        agregaciones.extend([
            pl.col(col).mean().round(2).alias(f"{col}_media"),
            pl.col(col).std().round(2).alias(f"{col}_std")
        ])

    metricas = (
        df_completo
        .group_by(["centro", "periodo", "grupo"])
        .agg(agregaciones)
        .sort(["centro", "periodo", "grupo"])
    )

    # --- VARIACIÓN PORCENTUAL ---

    df_pre = metricas.filter(pl.col("periodo") == "pre")
    df_post = metricas.filter(pl.col("periodo") == "post")

    renames = {f"{col}_media": f"{col}_media_pre" for col in cols_to_analyze}
    df_pre = df_pre.rename(renames)

    df_variacion = df_post.join(
        df_pre.select(["centro", "grupo"] + list(renames.values())),
        on=["centro", "grupo"],
        how="inner"
    )

    exprs_var = []
    for col in cols_to_analyze:
        col_post = f"{col}_media"
        col_pre = f"{col}_media_pre"

        exprs_var.append(
            ((pl.col(col_post) - pl.col(col_pre)) / pl.col(col_pre) * 100)
            .round(2)
            .alias(f"{col}_var_%")
        )

    df_variacion = df_variacion.select(
        ["centro", "grupo"] + exprs_var
    )

    metricas = metricas.join(
        df_variacion,
        on=["centro", "grupo"],
        how="left"
    )

    # Limpiar variación en periodo pre
    limpieza_exprs = [
        pl.when(pl.col("periodo") == "pre")
        .then(pl.lit(None))
        .otherwise(pl.col(f"{col}_var_%"))
        .alias(f"{col}_var_%")
        for col in cols_to_analyze
    ]

    metricas = metricas.with_columns(limpieza_exprs)

    return df_completo, metricas

#########################################################################################################################################################

def test_significacion_estadistica_centros(df_completo, metrica="score_tc"):
    """
    Realiza un T-test independiente en el post-test
    para cada centro y métrica elegida.
    """

    df_post = df_completo.filter(pl.col("periodo") == "post")

    centros = df_post.select("centro").unique().to_series().to_list()

    for centro in centros:

        df_centro = df_post.filter(pl.col("centro") == centro)

        grupo_control = (
            df_centro
            .filter(pl.col("grupo") == "control")
            .select(pl.col(metrica).drop_nulls())
            .to_series()
            .to_list()
        )

        grupo_experimental = (
            df_centro
            .filter(pl.col("grupo") == "experimental")
            .select(pl.col(metrica).drop_nulls())
            .to_series()
            .to_list()
        )

        if len(grupo_control) < 2 or len(grupo_experimental) < 2:
            print(f"\n⚠ Centro: {centro} → Muestras insuficientes.")
            continue

        t_stat, p_value = stats.ttest_ind(
            grupo_experimental,
            grupo_control,
            equal_var=False
        )

        print(f"\n--- Centro: {centro} | Métrica: {metrica} (Post-test) ---")
        print(f"Media Experimental: {sum(grupo_experimental)/len(grupo_experimental):.2f}")
        print(f"Media Control: {sum(grupo_control)/len(grupo_control):.2f}")
        print(f"P-valor: {p_value:.4f}")

        if p_value < 0.05:
            print("✅ Diferencia estadísticamente significativa (p < 0.05)")
        else:
            print("❌ No es estadísticamente significativa (p >= 0.05)")

#########################################################################################################################################################

def realizar_analisis_hake_por_centro(
    df: pl.DataFrame,
    constructo_base: str,
    col_grupo: str = "grupo",
    col_centro: str = "centro",
    min_n: int = 5
):
    """
    Análisis de ganancia de Hake estratificado por centro.

    Args:
        df: DataFrame Polars con *_pre y *_post
        constructo_base: prefijo del constructo
        col_grupo: 'Control' / 'Experimental'
        col_centro: columna identificadora del centro
        min_n: tamaño mínimo por grupo para contrastar
    """

    col_pre = f"{constructo_base}_pre"
    col_post = f"{constructo_base}_post"
    x_max = 1.0

    print("=" * 70)
    print(f" ANÁLISIS DE HAKE POR CENTRO: {constructo_base.upper()}")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Calcular ganancia de Hake
    # --------------------------------------------------
    df_valid = (
        df
        .filter(pl.col(col_pre) < x_max)
        .with_columns(
            g=(pl.col(col_post) - pl.col(col_pre)) / (x_max - pl.col(col_pre))
        )
        .drop_nulls(subset=["g", col_grupo, col_centro])
    )

    # --------------------------------------------------
    # 2. Análisis por centro
    # --------------------------------------------------
    resultados = []

    for centro in df_valid[col_centro].unique().sort().to_list():

        df_c = df_valid.filter(pl.col(col_centro) == centro)

        df_ctrl = df_c.filter(pl.col(col_grupo).str.to_lowercase() == "control")
        df_exp = df_c.filter(pl.col(col_grupo).str.to_lowercase() == "experimental")

        g_ctrl = df_ctrl["g"].to_numpy()
        g_exp = df_exp["g"].to_numpy()

        n_ctrl, n_exp = len(g_ctrl), len(g_exp)

        if n_ctrl < min_n or n_exp < min_n:
            print(f"Centro {centro}: tamaño insuficiente (Control={n_ctrl}, Exp={n_exp})")
            continue

        # Contraste principal
        _, p_sw_c = stats.shapiro(g_ctrl)
        _, p_sw_e = stats.shapiro(g_exp)
        _, p_lev = stats.levene(g_ctrl, g_exp)

        normalidad = (p_sw_c > 0.05) and (p_sw_e > 0.05)
        homocedasticidad = p_lev > 0.05

        if normalidad and homocedasticidad:
            stat, p_val = stats.ttest_ind(
                g_ctrl, g_exp, alternative="less"
            )
            test = "t-test"
        else:
            stat, p_val = stats.mannwhitneyu(
                g_ctrl, g_exp, alternative="less"
            )
            test = "mann-whitney"

        # Tamaño del efecto (Cohen d)
        s_pooled = np.sqrt(
            ((n_ctrl - 1) * np.var(g_ctrl, ddof=1)
             + (n_exp - 1) * np.var(g_exp, ddof=1))
            / (n_ctrl + n_exp - 2)
        )
        d = (g_exp.mean() - g_ctrl.mean()) / s_pooled

        resultados.append({
            "centro": centro,
            "n_control": n_ctrl,
            "n_experimental": n_exp,
            "media_g_control": g_ctrl.mean(),
            "media_g_experimental": g_exp.mean(),
            "test": test,
            "estadistico": stat,
            "p_valor": p_val,
            "d_cohen": d
        })

    resumen = pl.DataFrame(resultados)

    return df_valid, resumen

#########################################################################################################################################################