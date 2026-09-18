#########################################################################################################################################################

import logging
import os, sys
import polars as pl

#########################################################################################################################################################

script_path = os.path.dirname(os.path.abspath(__file__)) 
project_path = os.path.abspath(os.path.join(script_path, '..', '..', '..')) 
sys.path.insert(0, project_path)

#########################################################################################################################################################

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#########################################################################################################################################################

from src.config.config_forms import (
    TC_TIPOS,
    PERIODOS,
    MAX_PUNTUACION_TC,
    MAX_PUNTUACION_TA,
    MAX_PUNTUACION_TCC,
    TC_RESPUESTAS_CORRECTAS,
    MAX_PUNTUACION_TC_TIPO
)

#########################################################################################################################################################

data_dir = os.path.join(project_path, 'data', 'forms')
raw_data_dir = os.path.join(data_dir, 'raw_data')
processed_data_dir = os.path.join(data_dir, 'processed_data')
hashes_dir = os.path.join(project_path, 'data', 'hashes', 'processed_data')
raw_data_filenames = [f for f in os.listdir(raw_data_dir) if 'forms' in f]

#########################################################################################################################################################

def get_cols_tc(df):

    return {
        tc_tipo: [
            c for c in df.columns if 'conocimientos' in c and tc_tipo in c
        ] 
        for tc_tipo in TC_TIPOS
    }

def get_exprs_tc_tipos(cols_tc):

    return {
        periodo: {
            tc_tipo: [
                pl.col(c) == TC_RESPUESTAS_CORRECTAS[periodo][tc_tipo][i] 
                for i, c in enumerate(cols_tc[tc_tipo])
            ]
            for tc_tipo in TC_TIPOS
        } 
        for periodo in PERIODOS
    }

def get_cols_ta(df):

    return [
        c for c in df.columns if 'autoeficacia' in c
    ] 

def get_exprs_ta(cols_ta):

    return [
        pl.col(c) / MAX_PUNTUACION_TA
        for c in cols_ta
    ]

def get_cols_tcc_rel(df):

    return [
        c for c in df.columns if 'carga' in c and 'relevante' in c
    ] 

def get_exprs_tcc_rel(cols_tcc):

    return [
        pl.col(c) / MAX_PUNTUACION_TCC 
        for c in cols_tcc
    ]

def get_cols_tcc_int(df):

    return [
        c for c in df.columns if 'carga' in c and 'intrinseca' in c
    ] 

def get_exprs_tcc_int(cols_tcc):

    return [
        pl.col(c) / MAX_PUNTUACION_TCC 
        for c in cols_tcc
    ]


def get_cols_tcc_ext(df):

    return [
        c for c in df.columns if 'carga' in c and 'extrinseca' in c
    ] 

def get_exprs_tcc_ext(cols_tcc):

    return [
        pl.col(c) / MAX_PUNTUACION_TCC 
        for c in cols_tcc
    ]

#########################################################################################################################################################

def add_categorization(df):
    # Definimos las columnas que queremos categorizar
    cols_to_categorize = [
        'score_tc', 
        'score_ta', 
        'score_tc_retention', 
        'score_tc_transfer'
    ]
    
    # Añadimos las de carga cognitiva (estas existen en el pre-test como Null, y en el post-test con datos)
    if 'score_tcc_rel' in df.columns:
        cols_to_categorize.extend(['score_tcc_rel', 'score_tcc_int', 'score_tcc_ext'])

    for col in cols_to_categorize:
        # 1. VERIFICACIÓN: Si la columna no existe, es de tipo Null o está completamente vacía, la saltamos.
        if col not in df.columns or df[col].dtype == pl.Null or df[col].null_count() == df.height:
            # Creamos la columna de categoría vacía para mantener la consistencia
            df = df.with_columns(pl.lit(None).alias(f"{col}_cat"))
            continue

        # 2. Calculamos los límites de los cuantiles ignorando los posibles valores nulos
        q33 = df.select(pl.col(col).drop_nulls().quantile(0.33)).to_series()[0]
        q67 = df.select(pl.col(col).drop_nulls().quantile(0.67)).to_series()[0]
        
        # 3. Si por alguna razón los cuantiles siguen siendo nulos, también saltamos
        if q33 is None or q67 is None:
            df = df.with_columns(pl.lit(None).alias(f"{col}_cat"))
            continue

        # 4. Aplicamos la categorización
        df = df.with_columns(
            pl.when(pl.col(col) <= q33).then(pl.lit("Low"))
            .when(pl.col(col) <= q67).then(pl.lit("Medium"))
            .otherwise(pl.lit("High"))
            .alias(f"{col}_cat")
        )

    df = df.with_columns(
        pl.when(pl.col('score_tc') < 0.5).then(pl.lit('Fail'))
        .when((pl.col('score_tc') >= 0.5) & (pl.col('score_tc') < 0.6)).then(pl.lit('Pass'))
        .when((pl.col('score_tc') >= 0.6) & (pl.col('score_tc') < 0.7)).then(pl.lit('Good'))
        .when((pl.col('score_tc') >= 0.7) & (pl.col('score_tc') < 0.9)).then(pl.lit('Very Good'))
        .otherwise(pl.lit('Excellent'))
        .alias('score_tc_cat_trad_scale')
    ).with_columns(
        pl.when(pl.col('score_tc') <= 0.7).then(pl.lit('Fail-Pass-Good'))
        .otherwise(pl.lit('Very Good-Excellent'))
        .alias('score_tc_cat_trad_scale_v2')
    ).with_columns(
        pl.when(pl.col('score_tc_retention') <= 0.7).then(pl.lit('Fail-Pass-Good'))
        .otherwise(pl.lit('Very Good-Excellent'))
        .alias('score_tc_retention_cat_trad_scale_v2')
    ).with_columns(
        pl.when(pl.col('score_tc_transfer') <= 0.7).then(pl.lit('Fail-Pass-Good'))
        .otherwise(pl.lit('Very Good-Excellent'))
        .alias('score_tc_transfer_cat_trad_scale_v2')
    )

    # Generación de la Puntuación Final Sintética
    cat_mapping = {"Low": 1, "Medium": 2, "High": 3}
    
    df = df.with_columns(
        ((pl.col("score_tc_cat").replace(cat_mapping).cast(pl.Float32) + 
          pl.col("score_ta_cat").replace(cat_mapping).cast(pl.Float32)) / 2
        ).round(1).alias("indice_desempeño_global")
    )

    return df

#########################################################################################################################################################

def process_forms_data(raw_data_dir):

    raw_data_filenames = [f for f in os.listdir(raw_data_dir) if 'forms' in f]

    raw_data = {f: pl.read_csv(os.path.join(raw_data_dir, f)) for f in raw_data_filenames}

    processed_data = {}

    for raw_filename, df_raw in raw_data.items():

        print(raw_filename)

        df = df_raw.clone()
        
        df = df.rename(
            {c: c.lower().replace('retención', 'retention').replace('extrínseca', 'extrinseca').replace('intrínseca', 'intrinseca') for c in df.columns}
        ).rename({'identificador': 'id'})

        cols_tc = get_cols_tc(df)
        cols_ta = get_cols_ta(df)
        cols_tcc_rel = get_cols_tcc_rel(df)
        cols_tcc_int = get_cols_tcc_int(df)
        cols_tcc_ext = get_cols_tcc_ext(df)
        exprs_tc_tipos = get_exprs_tc_tipos(cols_tc)
        exprs_ta = get_exprs_ta(cols_ta)
        exprs_tcc_rel = get_exprs_tcc_rel(cols_tcc_rel)
        exprs_tcc_int = get_exprs_tcc_int(cols_tcc_int)
        exprs_tcc_ext = get_exprs_tcc_ext(cols_tcc_ext)
        exprs_tc_tipos = exprs_tc_tipos['pre'] if 'pre' in raw_filename else exprs_tc_tipos['post']

        df = (
            df
            .with_columns(
                [
                    pl.lit('pre').alias('periodo') if 'pre' in raw_filename else pl.lit('post').alias('periodo'),

                    pl.col("id").str.split("-").list.get(0).alias("school"),

                    (pl.col('puntuación').str.splitn(" / ", 2).struct.field("field_0").cast(pl.Int64) / MAX_PUNTUACION_TC).alias('score_tc'),

                    (pl.sum_horizontal(exprs_ta) / len(cols_ta)).round(2).alias('score_ta'),

                    (pl.sum_horizontal(exprs_tcc_rel) / len(cols_tcc_rel)).round(2).alias('score_tcc_rel') if 'post' in raw_filename else pl.lit(None).alias('score_tcc_rel'),

                    (pl.sum_horizontal(exprs_tcc_int) / len(cols_tcc_int)).round(2).alias('score_tcc_int') if 'post' in raw_filename else pl.lit(None).alias('score_tcc_int'),

                    (pl.sum_horizontal(exprs_tcc_ext) / len(cols_tcc_ext)).round(2).alias('score_tcc_ext') if 'post' in raw_filename else pl.lit(None).alias('score_tcc_ext')
                ] + 
                [
                    (pl.sum_horizontal(exprs_tc_tipos[tc_tipo]) / MAX_PUNTUACION_TC_TIPO).alias(f'score_tc_{tc_tipo}') for tc_tipo in TC_TIPOS
                ]
            )
                )
                
        df = add_categorization(df)

        processed_data[f'processed_{raw_filename}'.replace('.csv', '')] = df

    return processed_data

#########################################################################################################################################################

def add_hake_gains(df, metrics, max_score=1.0):
    """
    Calcula la Ganancia Normalizada de Hake para una lista de métricas cruzadas.
    Espera que existan las columnas con sufijos '_pre' y '_post'.
    """
    hake_exprs = []
    
    for metric in metrics:
        col_pre = f"{metric}_pre"
        col_post = f"{metric}_post"
        
        # Expresión para cada métrica
        expr_1 = (
            pl.when(pl.col(col_pre) == max_score)
            .then(pl.lit(0.0))
            .otherwise(
                ((pl.col(col_post) - pl.col(col_pre)) / (max_score - pl.col(col_pre))).round(3)
            )
            .alias(f"{metric}_hake_gain") # ej: score_tc_hake_gain
        )

        expr_2 = (
            pl.when(pl.col(col_pre) == max_score)
            .then(pl.lit(0.0))
            .otherwise(
                (pl.col(col_post) - pl.col(col_pre)).round(3)
            )
            .alias(f"{metric}_units_hake_gain") # ej: score_tc_hake_gain
        )

        hake_exprs.append(expr_1)
        hake_exprs.append(expr_2)
        
    return df.with_columns(hake_exprs)

#########################################################################################################################################################

def add_hake_gain_categorization(df):
    """
    Categoriza las columnas de Ganancia de Hake en 'Low', 'Medium' y 'High'
    usando los percentiles 33 y 67, siguiendo el mismo criterio que add_categorization.
    """
    # Detectamos automáticamente las columnas de Hake presentes en el DataFrame
    cols_to_categorize = [c for c in df.columns if c.endswith('_hake_gain')]

    for col in cols_to_categorize:
        # 1. VERIFICACIÓN: columna inexistente, tipo Null o completamente vacía
        if col not in df.columns or df[col].dtype == pl.Null or df[col].null_count() == df.height:
            df = df.with_columns(pl.lit(None).alias(f"{col}_cat"))
            continue

        # 2. Calculamos los límites de los cuantiles ignorando nulos
        q33 = df.select(pl.col(col).drop_nulls().quantile(0.33)).to_series()[0]
        q67 = df.select(pl.col(col).drop_nulls().quantile(0.67)).to_series()[0]

        # 3. Si los cuantiles son nulos, saltamos
        if q33 is None or q67 is None:
            df = df.with_columns(pl.lit(None).alias(f"{col}_cat"))
            continue

        # 4. Aplicamos la categorización
        df = df.with_columns(
            pl.when(pl.col(col) <= q33).then(pl.lit("Low"))
            .when(pl.col(col) <= q67).then(pl.lit("Medium"))
            .otherwise(pl.lit("High"))
            .alias(f"{col}_cat")
        )

    df = df.with_columns(
        pl.when(pl.col('score_tc_hake_gain') > 0)
        .then(True)
        .otherwise(False)
        .alias('improvement_hake_gain_v2')
    ).with_columns(
        pl.when(pl.col('score_tc_units_hake_gain') > 0)
        .then(True)
        .otherwise(False)
        .alias('improvement_units_hake_gain')
    ).with_columns(
            pl.when(pl.col('score_tc_hake_gain') > 0).then(pl.lit("Improve"))
            .when(pl.col('score_tc_hake_gain') == 0).then(pl.lit("Not Improve"))
            .otherwise(pl.lit("Worsen"))
            .alias('improvement_hake_gain')
    )
    
    #Improves en retention y transfer
    df = df.with_columns(
            pl.when(pl.col('score_tc_retention_hake_gain') > 0).then(pl.lit("Improve"))
            .when(pl.col('score_tc_retention_hake_gain') == 0).then(pl.lit("Not Improve"))
            .otherwise(pl.lit("Worsen"))
            .alias('improvement_retention_hake_gain')
    ).with_columns(
            pl.when(pl.col('score_tc_transfer_hake_gain') > 0).then(pl.lit("Improve"))
            .when(pl.col('score_tc_transfer_hake_gain') == 0).then(pl.lit("Not Improve"))
            .otherwise(pl.lit("Worsen"))
            .alias('improvement_transfer_hake_gain')
    )
    
    #Improves en autoconfianza
    df = df.with_columns(
            pl.when(pl.col('score_ta_hake_gain') > 0).then(pl.lit("Improve"))
            .when(pl.col('score_ta_hake_gain') == 0).then(pl.lit("Not Improve"))
            .otherwise(pl.lit("Worsen"))
            .alias('improvement_ta_hake_gain')
    )

    mejoras = df.filter(pl.col('score_tc_hake_gain') > 0)['score_tc_hake_gain']
    empeoramientos = df.filter(pl.col('score_tc_hake_gain') < 0)['score_tc_hake_gain']
    q33_mejora = mejoras.quantile(0.33)
    q50_mejora = mejoras.quantile(0.50)
    q66_mejora = mejoras.quantile(0.66)
    q33_empeoramiento = empeoramientos.quantile(0.33)
    q50_empeoramiento = empeoramientos.quantile(0.50)
    q66_empeoramiento = empeoramientos.quantile(0.66)

    df = df.with_columns(
        pl.when(pl.col('improvement_hake_gain') == 'Improve')
        .then(
            pl.when(pl.col('score_tc_hake_gain') <= q33_mejora)
            .then(pl.lit('Improve-Low'))
            .when(pl.col('score_tc_hake_gain') <= q66_mejora)
            .then(pl.lit('Improve-Medium'))
            .otherwise(pl.lit('Improve-High'))
        )
        .when(pl.col('improvement_hake_gain') == 'Worsen')
        .then(
            pl.when(pl.col('score_tc_hake_gain') >= q66_empeoramiento)
            .then(pl.lit('Worsenmiento-Bajo'))
            .when(pl.col('score_tc_hake_gain') >= q33_empeoramiento)
            .then(pl.lit('Worsenmiento-Medio'))
            .otherwise(pl.lit('Worsenmiento-Alto'))
        )
        .otherwise(pl.lit('Not Improve'))
        .alias('improvement_levels_hake_gain')
    )

    df = df.with_columns(
        pl.when(pl.col('improvement_hake_gain') == 'Improve')
        .then(
            pl.when(pl.col('score_tc_hake_gain') <= q50_mejora)
            .then(pl.lit('Improve-Low'))
            .otherwise(pl.lit('Improve-High'))
        )
        .when(pl.col('improvement_hake_gain') == 'Worsen')
        .then(
            pl.when(pl.col('score_tc_hake_gain') >= q50_empeoramiento)
            .then(pl.lit('Worsenmiento-Bajo'))
            .otherwise(pl.lit('Worsenmiento-Alto'))
        )
        .otherwise(pl.lit('Not Improve'))
        .alias('improvement_levels_hake_gain_v2')
    )

    return df

#########################################################################################################################################################