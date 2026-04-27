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
        pl.col(c) / MAX_PUNTUACION_TC
        for c in cols_tcc
    ]

def get_cols_tcc_int(df):

    return [
        c for c in df.columns if 'carga' in c and 'intrinseca' in c
    ] 

def get_exprs_tcc_int(cols_tcc):

    return [
        pl.col(c) / MAX_PUNTUACION_TC
        for c in cols_tcc
    ]


def get_cols_tcc_ext(df):

    return [
        c for c in df.columns if 'carga' in c and 'extrinseca' in c
    ] 

def get_exprs_tcc_ext(cols_tcc):

    return [
        pl.col(c) / MAX_PUNTUACION_TC
        for c in cols_tcc
    ]

#########################################################################################################################################################

def add_categorization(df):
    # Definimos las columnas que queremos categorizar
    cols_to_categorize = [
        'puntuacion_tc', 'puntuacion_ta', 
        'puntuacion_tc_retencion', 'puntuacion_tc_transferencia'
    ]
    
    # Añadimos las de carga cognitiva (estas existen en el pre-test como Null, y en el post-test con datos)
    if 'puntuacion_tcc_rel' in df.columns:
        cols_to_categorize.extend(['puntuacion_tcc_rel', 'puntuacion_tcc_int', 'puntuacion_tcc_ext'])

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
            pl.when(pl.col(col) <= q33).then(pl.lit("Bajo"))
            .when(pl.col(col) <= q67).then(pl.lit("Medio"))
            .otherwise(pl.lit("Alto"))
            .alias(f"{col}_cat")
        )

    # Generación de la Puntuación Final Sintética
    cat_mapping = {"Bajo": 1, "Medio": 2, "Alto": 3}
    
    df = df.with_columns(
        ((pl.col("puntuacion_tc_cat").replace(cat_mapping).cast(pl.Float32) + 
          pl.col("puntuacion_ta_cat").replace(cat_mapping).cast(pl.Float32)) / 2
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
            {c: c.lower().replace('retención', 'retencion').replace('extrínseca', 'extrinseca').replace('intrínseca', 'intrinseca') for c in df.columns}
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

                    pl.col("id").str.split("-").list.get(0).alias("centro"),

                    (pl.col('puntuación').str.splitn(" / ", 2).struct.field("field_0").cast(pl.Int64) / MAX_PUNTUACION_TC).alias('puntuacion_tc'),

                    (pl.sum_horizontal(exprs_ta) / len(cols_ta)).round(2).alias('puntuacion_ta'),

                    (pl.sum_horizontal(exprs_tcc_rel) / len(cols_tcc_rel)).round(2).alias('puntuacion_tcc_rel') if 'post' in raw_filename else pl.lit(None).alias('puntuacion_tcc_rel'),

                    (pl.sum_horizontal(exprs_tcc_int) / len(cols_tcc_int)).round(2).alias('puntuacion_tcc_int') if 'post' in raw_filename else pl.lit(None).alias('puntuacion_tcc_int'),

                    (pl.sum_horizontal(exprs_tcc_ext) / len(cols_tcc_ext)).round(2).alias('puntuacion_tcc_ext') if 'post' in raw_filename else pl.lit(None).alias('puntuacion_tcc_ext')
                ] + 
                [
                    (pl.sum_horizontal(exprs_tc_tipos[tc_tipo]) / MAX_PUNTUACION_TC_TIPO).alias(f'puntuacion_tc_{tc_tipo}') for tc_tipo in TC_TIPOS
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
        expr = (
            pl.when(pl.col(col_pre) == max_score)
            .then(pl.lit(0.0))
            .otherwise(
                (pl.col(col_post) - pl.col(col_pre)) / (max_score - pl.col(col_pre))
            )
            .alias(f"{metric}_hake_gain") # ej: puntuacion_tc_hake_gain
        )
        hake_exprs.append(expr)
        
    # Polars ejecuta todas estas expresiones en paralelo
    return df.with_columns(hake_exprs)

#########################################################################################################################################################