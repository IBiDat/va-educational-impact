import os
import polars as pl

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')

output_filename = 'processed_forms_interactions_data.parquet'
output_dir = os.path.join(project_path, 'data', 'combined')
output_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)

forms_data_filename = 'processed_pre_post_forms.parquet'
forms_data_path = os.path.join(project_path, 'data', 'forms', 'processed_data', forms_data_filename)

interactions_data_filename = 'interactions_processed_data.parquet'
interactions_data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', interactions_data_filename)

forms_df = pl.read_parquet(forms_data_path)
interactions_df = pl.read_parquet(interactions_data_path)

base_cols = ['id', 'centro', 'grupo']

metrics_pre_post = [
    'puntuacion_tc', 
    'puntuacion_ta', 
    'puntuacion_tc_retencion', 
    'puntuacion_tc_transferencia'
]

metrics_post = [
    'puntuacion_tcc_rel_post', 
    'puntuacion_tcc_int_post', 
    'puntuacion_tcc_ext_post', 
    'puntuacion_tcc_rel_cat_post', 
    'puntuacion_tcc_int_cat_post', 
    'puntuacion_tcc_ext_cat_post', 
]

hake_metrics = [
    'puntuacion_tc_hake_gain', 
    'puntuacion_tc_retencion_hake_gain', 
    'puntuacion_tc_transferencia_hake_gain',
    'puntuacion_tc_hake_gain_cat',
    'puntuacion_tc_retencion_hake_gain_cat',
    'puntuacion_tc_transferencia_hake_gain_cat'
]

forms_cols_analysis = base_cols + hake_metrics + [
    f"{metric}{cat_suffix}_{period}"
    for period in ['pre', 'post']
    for cat_suffix in ['', '_cat']
    for metric in metrics_pre_post
] + metrics_post

forms_df = forms_df[forms_cols_analysis]

forms_interactions_df = forms_df.join(interactions_df, how='left', on='id')

forms_interactions_df = forms_interactions_df.with_columns(
    pl.when(pl.col('grupo') == 'experimental').then(pl.col('experimental_type_freq_quality_v1'))
        .otherwise(pl.col('grupo'))
        .alias('grupo_segmented_v1')
)

forms_interactions_df = forms_interactions_df.with_columns(
    pl.when(pl.col('grupo') == 'experimental').then(pl.col('experimental_type_freq_quality_v2'))
        .otherwise(pl.col('grupo'))
        .alias('grupo_segmented_v2')
)

forms_interactions_df.write_parquet(output_path)