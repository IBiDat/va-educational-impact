###########################################################################################

# --- IMPORTS ---

import os
import sys
import logging
import polars as pl

###########################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

# Base paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
sys.path.append(project_path)

# Input data directories
forms_data_filename = 'processed_pre_post_forms.parquet'
forms_data_path = os.path.join(project_path, 'data', 'forms', 'processed_data', forms_data_filename)

interactions_data_filename = 'interactions_processed_data.parquet'
interactions_data_path = os.path.join(project_path, 'data', 'interactions', 'processed_data', interactions_data_filename)

centros_data_filename = 'centros_data.csv'
centros_data_path = os.path.join(project_path, 'data', 'centros', centros_data_filename)

# Output data directories
output_filename = 'processed_forms_interactions_data.parquet'
output_dir = os.path.join(project_path, 'data', 'combined')
output_path = os.path.join(output_dir, output_filename)


###########################################################################################

# --- LOCAL IMPORTS ---

from src.utils.educational_impact.forms_interactions_data_combination import (
    segment_groups,
    compute_time_interval_variables
)

###########################################################################################

CENTROS_IDS_MAP = {
    "IES Ramiro de Maeztu (Madrid)":                  "RamiroMaeztu",
    "IES Laguna de Joatzel (Getafe)":                 "Laguna",
    "Colegio Jesús María - García Noblejas (Madrid)": "JesusMaria",
    "IES José García Nieto (Las Rozas)":              "GarciaNieto",
}

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow for Combined Forms and Interactions Data.
    Filters form metrics, joins with interactions data, segments the experimental
    groups, enriches with centro socioeconomic data, and saves the final processed
    dataset as Parquet.
    """
    logging.info("▶️ STARTING COMBINED DATA PROCESSING PIPELINE")

    # 1. Load Data
    logging.info("STEP 1: Loading data files...\n")

    try:
        forms_df = pl.read_parquet(forms_data_path)
        logging.info(f" -> Loaded file: {forms_data_filename}")

        interactions_df = pl.read_parquet(interactions_data_path)
        logging.info(f" -> Loaded file: {interactions_data_filename}")

        centros_df = pl.read_csv(centros_data_path)
        logging.info(f" -> Loaded file: {centros_data_filename}\n")

    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Filter Form Columns
    logging.info("STEP 2: Selecting and filtering form columns...\n")

    try:
        BASE_COLS = ['id', 'centro', 'grupo']

        METRICS_PRE_POST = [
            'score_tc',
            'score_ta',
            'score_tc_retention',
            'score_tc_transfer'
        ]

        METRICS_POST = [
            'puntuacion_tcc_rel_post',
            'puntuacion_tcc_int_post',
            'puntuacion_tcc_ext_post',
            'puntuacion_tcc_rel_cat_post',
            'puntuacion_tcc_int_cat_post',
            'puntuacion_tcc_ext_cat_post',
        ]

        EXTRA_METRICS = [
            'score_tc_hake_gain',
            'score_tc_retention_hake_gain',
            'score_tc_transfer_hake_gain',
            'score_tc_hake_gain_cat',
            'score_tc_retention_hake_gain_cat',
            'score_tc_transfer_hake_gain_cat',
            'score_tc_units_hake_gain',
            'puntuacion_tc_retencion_units_hake_gain',
            'puntuacion_tc_transferencia_units_hake_gain',
            'score_tc_units_hake_gain_cat',
            'puntuacion_tc_retencion_units_hake_gain_cat',
            'puntuacion_tc_transferencia_units_hake_gain_cat',
            'puntuacion_ta_hake_gain',
            'puntuacion_ta_hake_gain_cat',
            'puntuacion_ta_units_hake_gain',
            'improvement_hake_gain',
            'improvement_hake_gain_v2',
            'mejora_units_hake_gain',
            "mejora_retencion_hake_gain",
            "mejora_transferencia_hake_gain",
            "mejora_autoconfianza_hake_gain",
            'niveles_improvement_hake_gain',
            'niveles_improvement_hake_gain_v2',
            'puntuacion_tc_cat_trad_scale_pre',
            'puntuacion_tc_cat_trad_scale_v2_pre',
            'puntuacion_tc_retencion_cat_trad_scale_v2_pre',
            'puntuacion_tc_transferencia_cat_trad_scale_v2_pre',
            'puntuacion_tc_cat_trad_scale_post',
            'puntuacion_tc_cat_trad_scale_v2_post',
            'puntuacion_tc_retencion_cat_trad_scale_v2_post',
            'puntuacion_tc_transferencia_cat_trad_scale_v2_post',
            'marca temporal_pre',
            'marca temporal_post'
        ]

        forms_cols_analysis = BASE_COLS + EXTRA_METRICS + [
            f"{metric}{cat_suffix}_{period}"
            for period in ['pre', 'post']
            for cat_suffix in ['', '_cat']
            for metric in METRICS_PRE_POST
        ] + METRICS_POST

        forms_df = forms_df.select(forms_cols_analysis)
        logging.info(f" -> Forms columns filtered successfully. Total columns: {len(forms_cols_analysis)}\n")

    except Exception as e:
        logging.error(f"Error during column filtering: {e}")
        sys.exit(1)

    # 3. Join Forms and Interactions
    logging.info("STEP 3: Joining forms and interactions dataframe...\n")

    try:
        forms_interactions_df = forms_df.join(
            interactions_df,
            how='left',
            on='id'
        )
        logging.info(" -> Join completed successfully\n")

    except Exception as e:
        logging.error(f"Error during dataframe join: {e}")
        sys.exit(1)

    # 4. Process and Segment Groups
    logging.info("STEP 4: Segmenting groups...\n")

    try:
        forms_interactions_df = segment_groups(forms_interactions_df)
        logging.info(" -> Experimental segmentation variables created successfully\n")

    except Exception as e:
        logging.error(f"Error during data segmentation: {e}")
        sys.exit(1)
    
    # 5. Include compute time interval variables
    logging.info("STEP 5: Include forms time interval variables ...\n")

    try:
        forms_interactions_df = compute_time_interval_variables(
            combined_df=forms_interactions_df
        )
        logging.info(" -> Forms Time Interval joined successfully\n")

    except Exception as e:
        logging.error(f"Error during forms time interval computation: {e}")
        sys.exit(1)

    # 6. Enrich with Centro Socioeconomic Data
    logging.info("STEP 6: Enriching with centro socioeconomic data...\n")

    try:
        centros_df = centros_df.with_columns(
            pl.col("nombre").replace(CENTROS_IDS_MAP).alias('centro')
        )
        forms_interactions_df = forms_interactions_df.join(
            centros_df,
            how='left',
            on='centro'
        )
        logging.info(" -> Centro data joined successfully\n")

    except Exception as e:
        logging.error(f"Error during centro data enrichment: {e}")
        sys.exit(1)

    # 7. Save Outputs
    logging.info("STEP 7: Saving results to Parquet...\n")

    try:
        os.makedirs(output_dir, exist_ok=True)
        forms_interactions_df.write_parquet(output_path)
        logging.info(f" -> Saved: {output_path}\n")

    except Exception as e:
        logging.error(f"Failed to save output files: {e}")
        sys.exit(1)

    logging.info("✅ PROCESSING PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()